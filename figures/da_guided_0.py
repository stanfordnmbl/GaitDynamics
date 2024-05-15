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
        # # GRF perturbation
        # trial_gait_phase_label, _ = self.grf_to_trial_gait_phase_label(trial_df['calcn_r_force_vy'].values, self.window_len, self.target_sampling_rate)
        # strike_10_percent = np.where((20 < trial_gait_phase_label) & (trial_gait_phase_label < 40))[0]
        # strike_10_percent_nonrepeat = []
        # for item in strike_10_percent:
        #     if item - 1 not in strike_10_percent:
        #         strike_10_percent_nonrepeat.append(item)
        # # plt.figure()
        # # plt.plot(trial_df['calcn_r_force_vx'])
        # curve = np.array([1 - abs(i - 1) for i in np.arange(0, 2, 0.08)])
        # for i in strike_10_percent_nonrepeat:
        #     trial_df['calcn_r_force_vx'][i:i+25] += curve * 0.5
        # # plt.plot(trial_df['calcn_r_force_vx'])
        # # plt.show()
        # self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'calcn_r_force_vx' in col]
        # self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'normed_cop' in col]

        # # GRF reduction
        # trial_df[['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']] = trial_df[['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']] * 1.5
        # trial_df[['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']] = trial_df[['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']] * .5
        # self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if '_force_v' in col]
        # self.do_not_follow_col_loc = []

        # # vel increase
        # ratio = 1.3
        # trial_df['pelvis_tx'] = trial_df['pelvis_tx'] * ratio
        # mtp_r_vel[:, 0] = mtp_r_vel[:, 0] * ratio
        # mtp_l_vel[:, 0] = mtp_l_vel[:, 0] * ratio
        # self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'pelvis_tx' in col]
        # self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' in col]

        # GRF perturbation
        plt.figure()
        plt.plot(trial_df['calcn_r_force_normed_cop_x'])
        curve = np.array([1 - abs(i - 1) for i in np.arange(0, 2, 0.08)])
        for i in range(50, trial_df.shape[0], 50):
            trial_df['calcn_r_force_normed_cop_x'][i:i+25] += curve * 0.1
            trial_df['calcn_r_force_normed_cop_x'][i:i+25] += curve * 0.1
        plt.plot(trial_df['calcn_r_force_normed_cop_x'])
        plt.show()
        self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if '_normed_cop_' in col]
        self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if '_normed_cop_' not in col]

        return trial_df, mtp_r_vel, mtp_l_vel


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)

    model = MotionModel(opt, repr_dim)

    max_trial_num = 1
    trial_start_num = 1

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

    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows_manipulated), opt.batch_size_inference):

        state_manipulated = [win[0] for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]]
        state_manipulated = torch.stack(state_manipulated)

        masks = torch.zeros_like(state_manipulated)      # 0 for masking, 1 for unmasking
        masks[:, :, test_dataset_mani.manipulated_col_loc] = 1

        value_diff_weight = torch.ones([len(opt.model_states_column_names)])
        value_diff_weight[test_dataset_mani.do_not_follow_col_loc] = 0

        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        value_diff_thd[:] = 0.2         # large value for no constraint
        value_diff_thd[test_dataset_mani.manipulated_col_loc] = 0

        state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, value_diff_thd, value_diff_weight,
                                                num_of_generation_per_window=skel_num - 1)
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0])

        for i_skel in range(skel_num-1):
            state_pred_list[i_skel] += state_pred_list_batch[i_skel]

    for i_skel in range(skel_num-1):
        assert len(state_pred_list[i_skel]) == len(windows_manipulated)

    names = ['baseline', '3x trunk sway']
    gui = set_up_gui()

    # TODO: only the first one is used, in the future make use all
    sub_bl_true, sub_bl_pred, height_m_all, weights_kg_all = {}, {}, {}, {}
    for i_win, (win, state_pred), in enumerate(zip(windows_original, state_pred_list[0])):
        trial_of_this_win = test_dataset.trials[win[2]]
        true_val = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win[0].unsqueeze(0)), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, [0, 0, 0]).squeeze().numpy()

        dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
        skel_0 = test_dataset.skels[dset_sub_name]
        skel_1 = test_dataset_mani.skels[dset_sub_name]
        true_val = inverse_norm_cops(skel_0, true_val, opt, trial_of_this_win.weights_kg, trial_of_this_win.height_m)
        state_pred = inverse_norm_cops(skel_1, state_pred, opt, trial_of_this_win.weights_kg, trial_of_this_win.height_m)

        true_moment, moment_names = osim_states_to_knee_moments_in_percent_BW_BH(true_val, skel_0, opt, trial_of_this_win.height_m)
        pred_moments, _ = osim_states_to_knee_moments_in_percent_BW_BH(state_pred, skel_1, opt, trial_of_this_win.height_m)

        gait_cycle_starts = np.where(win[3] == 0)[0]
        gait_cycle_ends = np.where(win[3] == 1000)[0]
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

        for pair_ in cycle_pairs:
            true_ = np.concatenate([true_val, true_moment], axis=-1)[pair_[0]:pair_[1]]
            true_ = linear_resample_data_as_num_of_dp(true_, 101)
            pred_ = np.concatenate([state_pred, pred_moments], axis=-1)[pair_[0]:pair_[1]]
            pred_ = linear_resample_data_as_num_of_dp(pred_, 101)
            sub_bl_true[dset_sub_name].append(true_)
            sub_bl_pred[dset_sub_name].append(pred_)

        height_m_all[dset_sub_name] = trial_of_this_win.height_m
        weights_kg_all[dset_sub_name] = trial_of_this_win.weights_kg

        name_states_dict = {names[0]: true_val, names[1]: state_pred.detach().numpy()}
        show_skeletons(opt, name_states_dict, gui, [skel_0, skel_1])

    pickle.dump([sub_bl_true, sub_bl_pred, None, None, opt.osim_dof_columns + moment_names,
                 None, height_m_all, weights_kg_all], open(f"results/da_guided_.pkl", "wb"))


li, camargo, carter, falisse, moore, tan2021, tan2022 = 'li', 'camargo', 'carter', 'falisse', 'moore', 'tan2021', 'tan2022'
uhlrich, santos, vanderzee, wang = 'uhlrich', 'santos', 'vanderzee', 'wang'
b3d_path = f'/mnt/d/Local/Data/MotionPriorData/{santos}_dset/'

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





