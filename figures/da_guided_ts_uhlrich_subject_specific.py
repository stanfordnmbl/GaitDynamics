import numpy as np
import pandas as pd

from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from fig_utils import show_skeletons, set_up_gui
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input, osim_states_to_knee_moments_in_percent_BW_BH, \
    linear_resample_data_as_num_of_dp, convert_addb_state_to_model_input
from scipy.interpolate import interp1d
import nimblephysics as nimble
from model.utils import cross_product_2d, get_multi_body_loc_using_nimble, inverse_norm_cops
import pickle


def get_start_end_of_gait_cycle(grf_v):
    start_ = np.where((grf_v[:-1] < 1e-5) & (grf_v[1:] > 1e-5))[0]
    end_ = np.where((grf_v[:-1] > 1e-5) & (grf_v[1:] < 1e-5))[0]
    if len(start_) != 1 or len(end_) != 1:
        return None
    return start_[0], end_[0]


class MotionDatasetManipulated(MotionDataset):
    # def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
    #     delay_of_start = 0
    #     data_ = pickle.load(open(f"results/starts_ends.pkl", "rb"))
    #     selected_ts_trial = list(data_.keys())[-3]
    #     start_ts, end_ts, lumbar_bending_ts = data_[selected_ts_trial]
    #     start_, end_ = get_start_end_of_gait_cycle(trial_df['calcn_l_force_vy'].values)
    #
    #     k = (end_ts - start_ts) / (end_ - start_)
    #     b = start_ts - k * start_
    #     x = [k * x_ + b for x_ in range(trial_df.shape[0])]
    #     new_x = [x_ for x_ in range(lumbar_bending_ts.shape[0])]
    #     f = interp1d(x, trial_df.values, bounds_error=False, fill_value='extrapolate', axis=0)
    #     data_resampled = f(new_x)
    #     f = interp1d(x, mtp_r_vel, bounds_error=False, fill_value='extrapolate', axis=0)
    #     mtp_r_vel = f(new_x)
    #     f = interp1d(x, mtp_l_vel, bounds_error=False, fill_value='extrapolate', axis=0)
    #     mtp_l_vel = f(new_x)
    #
    #     trial_df = pd.DataFrame(data_resampled[delay_of_start:], columns=opt.osim_dof_columns)
    #
    #     trial_df['lumbar_bending'] = lumbar_bending_ts[delay_of_start:]
    #     self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'lumbar' in col]
    #     return trial_df, mtp_r_vel[delay_of_start:], mtp_l_vel[delay_of_start:]

    def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
        trial_df['lumbar_bending'] = trial_df['lumbar_bending'] * self.opt.x_times_lumbar_bending
        self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'lumbar' in col]
        return trial_df, mtp_r_vel, mtp_l_vel


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)

    model = MotionModel(opt, repr_dim)

    max_trial_num = 999
    trial_start_num = 0

    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        include_trials_shorter_than_window_len=True,
        trial_start_num=trial_start_num,
        specific_trial='walking'
    )
    windows_original = test_dataset.get_one_win_from_the_end_of_each_trial([0])
    # starts_ends_lumbar_bending = {}
    # for trial in test_dataset.trials:
    #     pose = inverse_convert_addb_state_to_model_input(
    #         model.normalizer.unnormalize(trial.converted_pose.unsqueeze(0)), opt.model_states_column_names,
    #         opt.joints_3d, opt.osim_dof_columns, [0, 0, 0]).squeeze().numpy()
    #     grf_v = pose[:, opt.osim_dof_columns.index('calcn_l_force_vy')]
    #     if get_start_end_of_gait_cycle(grf_v):
    #         start_, end_ = get_start_end_of_gait_cycle(grf_v)
    #         starts_ends_lumbar_bending[trial.sub_and_trial_name] = (start_, end_, pose[:, opt.osim_dof_columns.index('lumbar_bending')])
    #     else:
    #         continue
    # pickle.dump(starts_ends_lumbar_bending, open(f"results/starts_ends.pkl", "wb"))

    sub_bl_true, sub_bl_pred, height_m_all, weights_kg_all = {}, {}, {}, {}
    sub_ts_true, sub_ts_pred = {}, {}
    opt.x_times_lumbar_bending = 2.5
    test_dataset_mani = MotionDatasetManipulated(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        include_trials_shorter_than_window_len=True,
        trial_start_num=trial_start_num,
        specific_trial='walking'
    )
    windows_manipulated = test_dataset_mani.get_one_win_from_the_end_of_each_trial(test_dataset_mani.manipulated_col_loc)
    assert len(windows_original) == len(windows_manipulated)

    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows_manipulated), opt.batch_size_inference):

        state_manipulated = torch.stack([win[0] for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])
        masks = torch.stack([win[4] for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])

        value_diff_weight = masks.sum(dim=2).bool().float().unsqueeze(-1).repeat([1, 1, masks.shape[2]])
        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        value_diff_thd[:] = 0.6         # large value for no constraint
        value_diff_thd[test_dataset_mani.manipulated_col_loc] = 0

        state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, value_diff_thd, value_diff_weight,
                                                num_of_generation_per_window=skel_num - 1)
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0])

        for i_skel in range(skel_num-1):
            state_pred_list[i_skel] += state_pred_list_batch[i_skel]

    for i_skel in range(skel_num-1):
        assert len(state_pred_list[i_skel]) == len(windows_manipulated)

    gui = set_up_gui()

    # TODO: only the first one is used, in the future make use all
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

        l_grf_v = true_val[:, opt.osim_dof_columns.index('calcn_l_force_vy')] * windows_manipulated[i_win][4][:, test_dataset_mani.manipulated_col_loc[0]].numpy()
        if get_start_end_of_gait_cycle(l_grf_v):
            start_, end_ = get_start_end_of_gait_cycle(l_grf_v)
        else:
            continue

        true_ = np.concatenate([true_val, true_moment], axis=-1)[start_:end_]
        true_ = linear_resample_data_as_num_of_dp(true_, 101)
        pred_ = np.concatenate([state_pred, pred_moments], axis=-1)[start_:end_]
        pred_ = linear_resample_data_as_num_of_dp(pred_, 101)
        if ('walking' in trial_of_this_win.sub_and_trial_name) and ('ts' not in trial_of_this_win.sub_and_trial_name.lower()):
            if dset_sub_name not in sub_bl_true.keys():
                sub_bl_true[dset_sub_name] = []
                sub_bl_pred[dset_sub_name] = []
            sub_bl_true[dset_sub_name].append(true_)
            sub_bl_pred[dset_sub_name].append(pred_)
        elif 'ts' in trial_of_this_win.sub_and_trial_name.lower():
            if dset_sub_name not in sub_ts_true.keys():
                sub_ts_true[dset_sub_name] = []
                sub_ts_pred[dset_sub_name] = []
            sub_ts_true[dset_sub_name].append(true_)
            sub_ts_pred[dset_sub_name].append(pred_)

        height_m_all[dset_sub_name] = trial_of_this_win.height_m
        weights_kg_all[dset_sub_name] = trial_of_this_win.weights_kg

        name_states_dict = {'true': true_val, 'pred': state_pred.detach().numpy()}
        show_skeletons(opt, name_states_dict, gui, [skel_0, skel_1])

    pickle.dump([sub_bl_true, sub_bl_pred, None, None, opt.osim_dof_columns + moment_names,
                 None, height_m_all, weights_kg_all], open(f"results/da_guided_baseline.pkl", "wb"))
    pickle.dump([sub_ts_true, sub_ts_pred, None, None, opt.osim_dof_columns + moment_names,
                 None, height_m_all, weights_kg_all], open(f"results/da_guided_trunk_sway.pkl", "wb"))


uhlrich, tan2022, vanderzee, wang = 'uhlrich', 'tan2022', 'vanderzee', 'wang'
b3d_path = f'/mnt/d/Local/Data/MotionPriorData/{uhlrich}_dset/subject8/'

""" To use this code,

"""

if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.guide_x_start_the_beginning_step = 1000

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'5328'}.pt"

    loop_all(opt)










