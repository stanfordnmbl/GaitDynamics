import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import matplotlib.pyplot as plt
import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from fig_utils import show_skeletons, set_up_gui
from model.utils import inverse_convert_addb_state_to_model_input, linear_resample_data_as_num_of_dp, \
    osim_states_to_moments_in_percent_BW_BH_via_cross_product
from model.utils import inverse_norm_cops
import pickle
import time


def get_start_end_of_gait_cycle(grf_v):
    start_ = np.where((grf_v[:-1] < 1e-5) & (grf_v[1:] > 1e-5))[0]
    end_ = np.where((grf_v[:-1] > 1e-5) & (grf_v[1:] < 1e-5))[0]
    if len(start_) != 1 or len(end_) != 1:
        return None
    return start_[0], end_[0]


class MotionDatasetManipulated(MotionDataset):
    def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
        trial_df['lumbar_bending'] = trial_df['lumbar_bending'] * self.opt.x_times_lumbar_bending
        self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('lumbar' in col)]
        self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('_vel' in col) or ('force' in col)]
        return trial_df, mtp_r_vel, mtp_l_vel


def loop_all(opt):
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)
    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        include_trials_shorter_than_window_len=True,
        # align_moving_direction_flag=False,
        specific_trial='walking'
    )
    windows_original = test_dataset.get_one_win_from_the_end_of_each_trial([0])

    sub_bl_true, sub_bl_pred, height_m_all, weight_kg_all = {}, {}, {}, {}
    sub_ts_true, sub_ts_pred = {}, {}
    sub_bl_true_original_rate, sub_bl_pred_original_rate, sub_ts_true_original_rate, sub_ts_pred_original_rate = {}, {},  {}, {}

    gui = set_up_gui()

    for i_x_time, x_times_lumbar_bending in enumerate([1, 2, 2.5, 3]):
        print(f'x_times_lumbar_bending: {x_times_lumbar_bending}')
        opt.x_times_lumbar_bending = x_times_lumbar_bending
        test_dataset_mani = MotionDatasetManipulated(
            data_path=b3d_path,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            include_trials_shorter_than_window_len=True,
            # align_moving_direction_flag=False,
            specific_trial='walking'
        )
        windows_manipulated = test_dataset_mani.get_one_win_from_the_end_of_each_trial(test_dataset_mani.manipulated_col_loc)
        assert len(windows_original) == len(windows_manipulated)

        for i_win in range(0, len(windows_manipulated), opt.batch_size_inference):
            state_manipulated = torch.stack([win.pose for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])
            masks = torch.stack([win.mask for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])
            cond = torch.stack([win.cond for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])
            height_m_tensor = torch.tensor([win.height_m for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])

            value_diff_weight = masks.sum(dim=2).bool().float().unsqueeze(-1).repeat([1, 1, masks.shape[2]])
            value_diff_thd = torch.zeros([len(opt.model_states_column_names)])

            for i_dof in range(state_manipulated.shape[2]):
                thd = (state_manipulated[:, :, i_dof].max() - state_manipulated[:, :, i_dof].min()) * 0.1
                value_diff_thd[i_dof] = thd

            value_diff_thd[test_dataset_mani.manipulated_col_loc] = thd
            value_diff_thd[test_dataset_mani.do_not_follow_col_loc] = 999

            start_time = time.time()
            state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, value_diff_thd, value_diff_weight, cond=cond,
                                                    num_of_generation_per_window=1, mode='inpaint_ddim_guided')
            end_time = time.time()
            print('Took {:.2f} seconds'.format(end_time - start_time))
            state_pred_list_batch = inverse_convert_addb_state_to_model_input(
                state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)

        for i_win, (win, state_pred), in enumerate(zip(windows_original, state_pred_list_batch[0])):
            trial_of_this_win = test_dataset.trials[win.trial_id]
            true_val = inverse_convert_addb_state_to_model_input(
                model.normalizer.unnormalize(win.pose.unsqueeze(0)), opt.model_states_column_names,
                opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(win.height_m)).squeeze().numpy()

            dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
            test_name = f'_{x_times_lumbar_bending}'
            skel_0 = test_dataset.skels[dset_sub_name]
            skel_1 = test_dataset_mani.skels[dset_sub_name]

            true_val = inverse_norm_cops(skel_0, true_val, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
            state_pred = inverse_norm_cops(skel_1, state_pred, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)

            true_moment, moment_names = osim_states_to_moments_in_percent_BW_BH_via_cross_product(true_val, skel_0, opt, trial_of_this_win.height_m)
            pred_moments, _ = osim_states_to_moments_in_percent_BW_BH_via_cross_product(state_pred, skel_1, opt, trial_of_this_win.height_m)

            l_grf_v = true_val[:, opt.osim_dof_columns.index('calcn_l_force_vy')] * windows_manipulated[i_win].mask[:, test_dataset_mani.manipulated_col_loc[0]].numpy()
            if get_start_end_of_gait_cycle(l_grf_v):
                start_, end_ = get_start_end_of_gait_cycle(l_grf_v)
            else:
                continue

            true_original_rate = np.concatenate([true_val, true_moment], axis=-1)
            true_ = linear_resample_data_as_num_of_dp(true_original_rate[start_:end_], 101)
            pred_original_rate = np.concatenate([state_pred, pred_moments], axis=-1)
            pred_ = linear_resample_data_as_num_of_dp(pred_original_rate[start_:end_], 101)
            if ('walking' in trial_of_this_win.sub_and_trial_name) and ('ts' not in trial_of_this_win.sub_and_trial_name.lower()):
                if test_name not in sub_bl_true.keys():
                    sub_bl_true[test_name] = []
                    sub_bl_pred[test_name] = []
                    sub_bl_true_original_rate[test_name] = []
                    sub_bl_pred_original_rate[test_name] = []
                sub_bl_true[test_name].append(true_)
                sub_bl_pred[test_name].append(pred_)
                sub_bl_true_original_rate[test_name].append(true_original_rate)
                sub_bl_pred_original_rate[test_name].append(pred_original_rate)
            elif 'ts' in trial_of_this_win.sub_and_trial_name.lower():
                if test_name not in sub_ts_true.keys():
                    sub_ts_true[test_name] = []
                    sub_ts_pred[test_name] = []
                    sub_ts_true_original_rate[test_name] = []
                    sub_ts_pred_original_rate[test_name] = []
                sub_ts_true[test_name].append(true_)
                sub_ts_pred[test_name].append(pred_)
                sub_ts_true_original_rate[test_name].append(true_original_rate)
                sub_ts_pred_original_rate[test_name].append(pred_original_rate)

            if test_name not in height_m_all.keys():
                height_m_all[test_name] = []
                weight_kg_all[test_name] = []
            if ('walking' in trial_of_this_win.sub_and_trial_name) and ('ts' not in trial_of_this_win.sub_and_trial_name.lower()):
                height_m_all[test_name].append(trial_of_this_win.height_m)
                weight_kg_all[test_name].append(trial_of_this_win.weight_kg)

            # if x_times_lumbar_bending > 1:
            #     name_states_dict = {'true': true_val, 'pred': state_pred.detach().numpy()}
            #     show_skeletons(opt, name_states_dict, gui, skel_0)

    pickle.dump([sub_bl_true, sub_bl_pred, sub_bl_true_original_rate, sub_bl_pred_original_rate,
                 opt.osim_dof_columns + moment_names, None, height_m_all, weight_kg_all], open(f"results/da_guided_baseline.pkl", "wb"))
    pickle.dump([sub_ts_true, sub_ts_pred, sub_ts_true_original_rate, sub_ts_pred_original_rate,
                 opt.osim_dof_columns + moment_names, None, height_m_all, weight_kg_all], open(f"results/da_guided_trunk_sway.pkl", "wb"))


b3d_path = f'/mnt/d/Local/Data/MotionPriorData/uhlrich_dset/'


if __name__ == "__main__":
    opt = parse_opt()
    opt.n_guided_steps = 5
    opt.guidance_lr = 0.02
    opt.guide_x_start_the_beginning_step = 1000
    opt.guide_x_start_the_end_step = 0
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'2560_diffusion'}.pt"
    loop_all(opt)










