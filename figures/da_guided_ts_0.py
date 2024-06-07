import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from fig_utils import show_skeletons, set_up_gui
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input, osim_states_to_knee_moments_in_percent_BW_BH, \
    linear_resample_data_as_num_of_dp
import nimblephysics as nimble
from model.utils import cross_product_2d, get_multi_body_loc_using_nimble, inverse_norm_cops
import pickle


class MotionDatasetManipulated(MotionDataset):
    def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
        trial_df['lumbar_bending'] = trial_df['lumbar_bending'] * 3
        self.manipulated_column_keywords = 'lumbar'
        return trial_df, mtp_r_vel, mtp_l_vel


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)

    model = MotionModel(opt, repr_dim)

    max_trial_num = 1
    trial_start_num = 0

    test_dataset_mani = MotionDatasetManipulated(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num
    )
    windows_manipulated = test_dataset_mani.get_all_wins_within_gait_cycle()

    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
    )
    windows_original = test_dataset.get_all_wins_within_gait_cycle()
    assert len(windows_original) == len(windows_manipulated)

    manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if test_dataset_mani.manipulated_column_keywords in col]

    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows_manipulated), opt.batch_size_inference):

        state_manipulated = [win.pose for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]]
        state_manipulated = torch.stack(state_manipulated)

        masks = torch.zeros_like(state_manipulated)      # 0 for masking, 1 for unmasking
        masks[:, :, manipulated_col_loc] = 1

        height_m_tensor = torch.tensor([win.height_m for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]]).unsqueeze(-1)

        value_diff_weight = torch.ones([len(opt.model_states_column_names)])
        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        value_diff_thd[:] = 0.1         # large value for no constraint
        value_diff_thd[manipulated_col_loc] = 0

        state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, value_diff_thd, value_diff_weight,
                                                num_of_generation_per_window=skel_num - 1)
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)

        for i_skel in range(skel_num-1):
            state_pred_list[i_skel] += state_pred_list_batch[i_skel]

    for i_skel in range(skel_num-1):
        assert len(state_pred_list[i_skel]) == len(windows_manipulated)

    names = ['baseline', '3x trunk sway']
    gui = set_up_gui()

    # TODO: only the first one is used, in the future make use all
    sub_bl_true, sub_bl_pred, height_m_all, weight_kg_all = {}, {}, {}, {}
    sub_ts_true, sub_ts_pred = {}, {}
    for i_win, (win, state_pred), in enumerate(zip(windows_original, state_pred_list[0])):
        trial_of_this_win = test_dataset.trials[win.trial_id]
        true_val = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win.pose.unsqueeze(0)), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], win.height_m).squeeze().numpy()

        dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
        skel_0 = test_dataset.skels[dset_sub_name]
        skel_1 = test_dataset_mani.skels[dset_sub_name]
        true_val = inverse_norm_cops(skel_0, true_val, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
        state_pred = inverse_norm_cops(skel_1, state_pred, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)

        true_moment, moment_names = osim_states_to_knee_moments_in_percent_BW_BH(true_val, skel_0, opt, trial_of_this_win.height_m)
        pred_moments, _ = osim_states_to_knee_moments_in_percent_BW_BH(state_pred, skel_1, opt, trial_of_this_win.height_m)

        gait_cycle_starts = np.where(win.gait_phase_label == 0)[0]
        gait_cycle_ends = np.where(win.gait_phase_label == 1000)[0]
        cycle_pairs = []
        for end_ in gait_cycle_ends:
            for start_ in reversed(gait_cycle_starts):
                if end_ > start_:
                    cycle_pairs.append((start_, end_))
                    break
        if len(cycle_pairs) == 0:
            continue

        if dset_sub_name not in sub_bl_true.keys():
            sub_bl_true[dset_sub_name], sub_bl_pred[dset_sub_name] = [], []
            sub_ts_true[dset_sub_name], sub_ts_pred[dset_sub_name] = [], []

        for pair_ in cycle_pairs:
            true_ = np.concatenate([true_val, true_moment], axis=-1)[pair_[0]:pair_[1]]
            true_ = linear_resample_data_as_num_of_dp(true_, 101)
            pred_ = np.concatenate([state_pred, pred_moments], axis=-1)[pair_[0]:pair_[1]]
            pred_ = linear_resample_data_as_num_of_dp(pred_, 101)
            if 'baseline' in trial_of_this_win.sub_and_trial_name:
                sub_bl_true[dset_sub_name].append(true_)
                sub_bl_pred[dset_sub_name].append(pred_)
            elif 'trunk_sway' in trial_of_this_win.sub_and_trial_name:
                sub_ts_true[dset_sub_name].append(true_)
                sub_ts_pred[dset_sub_name].append(pred_)

        height_m_all[dset_sub_name] = trial_of_this_win.height_m
        weight_kg_all[dset_sub_name] = trial_of_this_win.weight_kg

        name_states_dict = {names[0]: true_val, names[1]: state_pred.detach().numpy()}
        show_skeletons(opt, name_states_dict, gui, skel_0)

    pickle.dump([sub_bl_true, sub_bl_pred, None, None, opt.osim_dof_columns + moment_names,
                 None, height_m_all, weight_kg_all], open(f"results/da_guided_baseline.pkl", "wb"))
    pickle.dump([sub_ts_true, sub_ts_pred, None, None, opt.osim_dof_columns + moment_names,
                 None, height_m_all, weight_kg_all], open(f"results/da_guided_trunk_sway.pkl", "wb"))


li, camargo, carter, falisse, moore, tan2021, tan2022 = 'li', 'camargo', 'carter', 'falisse', 'moore', 'tan2021', 'tan2022'
uhlrich, santos, vanderzee, wang = 'uhlrich', 'santos', 'vanderzee', 'wang'
b3d_path = f'/mnt/d/Local/Data/MotionPriorData/{tan2022}_dset/'

""" To use this code,
1. in load_addb, manipulate channels
2. in this script, change manipulated_col_loc accordingly
"""

if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.guide_x_start_the_beginning_step = 1000

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'4925'}.pt"

    loop_all(opt)










