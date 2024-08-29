import copy

import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from fig_utils import show_skeletons, set_up_gui, extract_gait_parameters_from_osim_states, \
    extract_gait_parameters_from_osim_states_and_append
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input, osim_states_to_knee_moments_in_percent_BW_BH, \
    linear_resample_data_as_num_of_dp
import nimblephysics as nimble
from model.utils import cross_product_2d, get_multi_body_loc_using_nimble_by_body_names, inverse_norm_cops
import pickle


# class MotionDatasetManipulated(MotionDataset):
#     def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
#         # vel increase
#         ratio = 0.9
#         trial_df['pelvis_tx'] = trial_df['pelvis_tx'] * ratio
#         mtp_r_vel[:, 0] = mtp_r_vel[:, 0] * ratio
#         mtp_l_vel[:, 0] = mtp_l_vel[:, 0] * ratio
#         self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'pelvis_tx' in col]
#         self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' in col]
#
#         # plt.figure()
#         # plt.title(f'ratio: {ratio}')
#
#         return trial_df, mtp_r_vel, mtp_l_vel

def manipulate_vel_in_wins(windows, scaler, ratio):
    windows_manipulated_exp = copy.deepcopy(windows)
    for win in windows_manipulated_exp:
        speed_idx = opt.model_states_column_names.index('pelvis_tx')
        vel = (win.pose[0, speed_idx] - scaler.min_[0]) / scaler.scale_[0] * win.height_m
        win.pose[0, speed_idx] = vel * ratio / win.height_m * scaler.scale_[0] + scaler.min_[0]
    return windows_manipulated_exp


def loop_all(opt):
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)

    max_trial_num = 1
    trial_start_num = 0

    # test_dataset_mani = MotionDatasetManipulated(
    #     data_path=b3d_path,
    #     train=False,
    #     normalizer=model.normalizer,
    #     opt=opt,
    #     divide_jittery=False,
    #     max_trial_num=max_trial_num,
    #     trial_start_num=trial_start_num,
    #     specific_trial='400'
    # )
    # windows_manipulated_exp = test_dataset_mani.get_one_win_from_the_end_of_each_trial(
    #     [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'knee' not in col])

    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
        specific_trial='400'
    )
    windows_bl_exp = test_dataset.get_one_win_from_the_end_of_each_trial(
            [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'knee' not in col])

    windows_higher_syn = manipulate_vel_in_wins(windows_bl_exp, model.normalizer.scaler, 1.1)
    windows_lower_syn = manipulate_vel_in_wins(windows_bl_exp, model.normalizer.scaler, 0.9)
    # windows_manipulated_exp = copy.deepcopy(windows_bl_exp)
    # for win in windows_manipulated_exp:
    #     speed_idx = opt.model_states_column_names.index('pelvis_tx')
    #     vel = (win.pose[0, speed_idx] - model.normalizer.scaler.min_[0]) / model.normalizer.scaler.scale_[0] * win.height_m
    #     win.pose[0, speed_idx] = vel * 0.9 / win.height_m * model.normalizer.scaler.scale_[0] + model.normalizer.scaler.min_[0]

    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
        specific_trial='500'
    )
    windows_higher_exp = test_dataset.get_one_win_from_the_end_of_each_trial(
        [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'knee' not in col])

    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
        specific_trial='300'
    )
    windows_lower_exp = test_dataset.get_one_win_from_the_end_of_each_trial(
        [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('knee' not in col) or ('force' in col)])

    for win_bl, win_h_exp, win_l_exp in zip(windows_bl_exp, windows_higher_exp, windows_lower_exp):
        assert (test_dataset.trials[win_bl.trial_id].sub_and_trial_name.split('__')[0] ==
                test_dataset.trials[win_h_exp.trial_id].sub_and_trial_name.split('__')[0] ==
                test_dataset.trials[win_l_exp.trial_id].sub_and_trial_name.split('__')[0])

    state_pred_list_conditions = []
    for windows_manipulated in [windows_higher_syn, windows_bl_exp, windows_lower_syn]:
        state_pred_list = []
        for i_win in range(0, len(windows_manipulated), opt.batch_size_inference):
            state_manipulated = [win.pose for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]]
            state_manipulated = torch.stack(state_manipulated)

            masks = torch.stack([win.mask for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])
            cond = torch.stack([win.cond for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])
            height_m_tensor = torch.tensor([win.height_m for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]])

            # value_diff_weight = torch.ones([len(opt.model_states_column_names)])
            # value_diff_weight[test_dataset_mani.do_not_follow_col_loc] = 0
            #
            # value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
            # value_diff_thd[:] = 0.5         # large value for no constraint
            # value_diff_thd[test_dataset_mani.manipulated_col_loc] = 999
            # value_diff_thd[test_dataset_mani.do_not_follow_col_loc] = 999

            state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, cond=cond,
                                                    num_of_generation_per_window=skel_num - 1)
            state_pred_list_batch = inverse_convert_addb_state_to_model_input(
                state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)

            state_pred_list += [[state_pred_list_batch[i_skel, i_win] for i_skel in range(skel_num-1)] for i_win in range(state_pred_list_batch.shape[1])]
        state_pred_list_conditions.append(state_pred_list)

    gui = set_up_gui()
    param_dict_original_exp, param_dict_bl_exp, param_dict_syn = {'sub_name': []}, {}, {}
    for i_win, (win_higher_exp, win_bl_exp, win_lower_exp, win_higher_syn, win_bl_syn, win_lower_syn) in enumerate(
            zip(windows_higher_exp, windows_bl_exp, windows_lower_exp, *state_pred_list_conditions)):
        sub_name = test_dataset.trials[win_bl_exp.trial_id].sub_and_trial_name.split('__')[0]
        param_dict_original_exp['sub_name'].append(sub_name)
        trial_of_this_win = test_dataset.trials[win_bl_exp.trial_id]

        [win_higher_exp, win_bl_exp, win_lower_exp] = [
            inverse_convert_addb_state_to_model_input(model.normalizer.unnormalize(win.pose.unsqueeze(0)),
                                                      opt.model_states_column_names, opt.joints_3d,
                                                      opt.osim_dof_columns, [0, 0, 0],
                                                      torch.tensor(win.height_m)).squeeze().numpy() for win in
            [win_higher_exp, win_bl_exp, win_lower_exp]]

        # win_original_exp = inverse_convert_addb_state_to_model_input(
        #     model.normalizer.unnormalize(win_original_exp.pose.unsqueeze(0)), opt.model_states_column_names,
        #     opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(win_original_exp.height_m)).squeeze().numpy()
        # win_bl_exp = inverse_convert_addb_state_to_model_input(
        #     model.normalizer.unnormalize(win_bl_exp.pose.unsqueeze(0)), opt.model_states_column_names,
        #     opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(win_bl_exp.height_m)).squeeze().numpy()

        plt.plot(win_higher_exp[:, opt.osim_dof_columns.index('knee_angle_r')], label='higher')
        plt.plot(win_bl_exp[:, opt.osim_dof_columns.index('knee_angle_r')], label='bl')
        plt.plot(win_lower_exp[:, opt.osim_dof_columns.index('knee_angle_r')], label='lower')

        plt.plot(win_higher_syn[0][:, opt.osim_dof_columns.index('knee_angle_r')], '--', label='higher syn')
        plt.plot(win_bl_syn[0][:, opt.osim_dof_columns.index('knee_angle_r')], '--', label='bl syn')
        plt.plot(win_lower_syn[0][:, opt.osim_dof_columns.index('knee_angle_r')], '--', label='lower syn')

        # plt.plot(win_manipulated_syn_all_skel[0][:, opt.osim_dof_columns.index('knee_angle_r')], label='synthesized')
        plt.legend()
        plt.show()

        dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
        skel_0 = test_dataset.skels[dset_sub_name]
        win_original_exp = inverse_norm_cops(skel_0, win_original_exp, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
        param_dict_original_exp = extract_gait_parameters_from_osim_states_and_append(win_original_exp, skel_0, opt, param_dict_original_exp)
        win_bl_exp = inverse_norm_cops(skel_0, win_bl_exp, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
        param_dict_bl_exp = extract_gait_parameters_from_osim_states_and_append(win_bl_exp, skel_0, opt, param_dict_bl_exp)

        param_dict_syn_list_of_dict = []
        for win_manipulated_syn in win_manipulated_syn_all_skel:
            win_manipulated_syn = inverse_norm_cops(skel_0, win_manipulated_syn.detach().numpy(), opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
            param_dict_syn_list_of_dict.append(extract_gait_parameters_from_osim_states(win_manipulated_syn, skel_0, opt))
        for key_ in param_dict_original_exp.keys():
            if key_ not in ['hip_flexion_r', 'knee_angle_r', 'ankle_angle_r', 'sub_name']:
                if key_ not in param_dict_syn.keys():
                    param_dict_syn[key_] = []
                param_dict_syn[key_].append(np.mean([param_dict_syn_list_of_dict[i][key_] for i in range(len(param_dict_syn_list_of_dict))]))

        # name_states_dict = {
        #     'experiment 4 m/s': win_bl_exp,
        #     'synthesized 5 m/s': win_manipulated_syn_all_skel[0],     # only the first one is shown
        #     'experiment 5 m/s': win_original_exp}
        # for _ in range(5):
        #     show_skeletons(opt, name_states_dict, gui, skel_0)

    plt.figure(figsize=(6, 4))
    plt.ylabel('# of Subject decreased      # of Subject  increased  ', labelpad=13)
    plt.xlabel('Flexion angles')
    plt.yticks(np.arange(-10, 11, 2))
    plt.ylim([-10, 11])
    ax = plt.gca()
    for i_param, (param, param_name) in enumerate(zip(
            ['hip_flexion_r_max', 'knee_angle_r_max', 'ankle_angle_r_max', 'hip_flexion_r_min'],
            ['Hip max', 'Knee max', 'Ankle max', 'Hip min'])):
        print(param)
        delta_exp = np.array(param_dict_original_exp[param])-np.array(param_dict_bl_exp[param])
        delta_syn = np.array(param_dict_syn[param])-np.array(param_dict_bl_exp[param])
        increased_idx = delta_exp > 0
        increased_num = np.sum(increased_idx)
        increased_num_syn = np.sum(delta_syn[increased_idx] > 0)
        ax.bar(param_name, [increased_num, increased_num_syn], color=['gray', 'C0'], label=['', ''])
        decrease_num_syn = np.sum(delta_syn[~increased_idx] < 0)
        ax.bar(param_name, -np.array([delta_exp.shape[0] - increased_num, decrease_num_syn]), color=['gray', 'C0'])

        plt.figure(figsize=(6, 4))
        # plt.plot((np.array(param_dict_original_exp[param])-np.array(param_dict_bl_exp[param])) * 180 / np.pi,
        #          (np.array(param_dict_syn[param])-np.array(param_dict_bl_exp[param])) * 180 / np.pi, 'o', label=sub_name)
        for i_sub, sub_name in enumerate(param_dict_original_exp['sub_name']):
            plt.plot((param_dict_original_exp[param][i_sub]-param_dict_bl_exp[param][i_sub]) * 180 / np.pi,
                     (param_dict_syn[param][i_sub]-param_dict_bl_exp[param][i_sub]) * 180 / np.pi, 'o', label=sub_name)
        plt.xlabel(param_name + ' change - Experimental (deg)')
        plt.ylabel(param_name + ' change - Synthetic (deg)')
        plt.legend()
    ax.legend(['Experimental', 'Synthetic'])
    ax.plot([-0.5, i_param+0.5], [0, 0], 'black', linewidth=2)
    plt.show()


b3d_path = f'/mnt/d/Local/Data/MotionPriorData/hamner_dset/'

"""
This one is similar to the guided trunk sway.
"""

if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.n_guided_steps = 30
    opt.guidance_lr = 0.01

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'6993'}.pt"

    loop_all(opt)





