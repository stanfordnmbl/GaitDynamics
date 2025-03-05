import copy
import pickle
import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from data.osim_fk import get_model_offsets, forward_kinematics
from model.model import MotionModel
from data.addb_dataset import MotionDataset
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input, fix_seed
from model.utils import inverse_norm_cops
from fig_utils import show_skeletons, set_up_gui
import time


class MotionDatasetManipulated(MotionDataset):
    def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
        self.manipulated_col_loc = [opt.model_states_column_names.index(col) for col in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]
        self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('_vel' in col) or ('force' in col)]
        return trial_df, mtp_r_vel, mtp_l_vel


def profile_to_peak(states, columns, model_offsets):
    param_dict = {
        'hip_flexion_r_max': [], 'knee_angle_r_max': [], 'ankle_angle_r_max': [],
        'hip_flexion_r_min': [], 'knee_angle_r_min': [], 'ankle_angle_r_min': [],
        'peak_apgrf': [], 'peak_vgrf': [], 'vilr': [], 'stride_length': []
    }
    gait_phase_label = []

    v_grf_r = states[:, columns.index('calcn_r_force_vy')]
    stance_flag = np.abs(v_grf_r) > 2
    stance_flag = stance_flag.astype(int)
    start_end_indicator = np.diff(stance_flag)
    stance_starts_conservative = np.where(start_end_indicator == 1)[0]
    for i_start in range(len(stance_starts_conservative)-1):
        start_conservative = stance_starts_conservative[i_start]
        start_ = stance_starts_conservative[i_start]
        for i in range(start_conservative-1, 0, -1):
            if v_grf_r[i] < 1:          # 10% Body weight, roughly 15 N
                start_ = i
                break
            else:
                continue
        end_ = stance_starts_conservative[i_start + 1]
        for i in range(end_, 0, -1):
            if v_grf_r[i] < 1:
                end_ = i
                break
            else:
                continue

        vilr = np.diff(states[start_:end_, columns.index('calcn_r_force_vy')]).max() * 9.81 / 100
        peak_apgrf = states[start_:end_, columns.index('calcn_r_force_vx')].max() * 9.81 / 100
        peak_vgrf = states[start_:end_, columns.index('calcn_r_force_vy')].max() * 9.81 / 100
        # peak_apgrf2 = states[end_:, columns.index('calcn_r_force_vx')].max() * 9.81 / 100
        # peak_vgrf2 = states[end_:, columns.index('calcn_r_force_vy')].max() * 9.81 / 100
        # peak_apgrf = (peak_apgrf + peak_apgrf2) / 2     # each trial has two stance phases to use
        # peak_vgrf = (peak_vgrf + peak_vgrf2) / 2

        foot_locations, joint_locations, joint_names, _ = forward_kinematics(states[start_:end_, :23], model_offsets)

        # cal_r_loc = foot_locations[2, 0].cpu().numpy()
        # mtp_r_loc = foot_locations[3, 0].cpu().numpy()
        # cop_r_x = states[start_:end_, columns.index('calcn_r_force_normed_cop_x')]
        # # si_curve = (cop_r_x - cal_r_loc[:, 0]) / (mtp_r_loc[:, 0] - cal_r_loc[:, 0])
        # strike_index_pseudo = si_curve[4]

        stride_length = (foot_locations[2, 0, -1, 0] - foot_locations[2, 0, 0, 0]).cpu().numpy()

        param_dict['hip_flexion_r_max'].append(np.rad2deg(np.max(states[start_:end_, columns.index('hip_flexion_r')])))
        param_dict['knee_angle_r_max'].append(np.rad2deg(np.max(states[start_:end_, columns.index('knee_angle_r')])))
        param_dict['ankle_angle_r_max'].append(np.rad2deg(np.max(states[start_:end_, columns.index('ankle_angle_r')])))
        param_dict['hip_flexion_r_min'].append(np.rad2deg(np.min(states[start_:end_, columns.index('hip_flexion_r')])))
        param_dict['knee_angle_r_min'].append(np.rad2deg(np.min(states[start_:end_, columns.index('knee_angle_r')])))
        param_dict['ankle_angle_r_min'].append(np.rad2deg(np.min(states[start_:end_, columns.index('ankle_angle_r')])))
        param_dict['peak_apgrf'].append(peak_apgrf)
        param_dict['peak_vgrf'].append(peak_vgrf)
        param_dict['vilr'].append(vilr)
        # param_dict['strike_index'].append(strike_index_pseudo)
        param_dict['stride_length'].append(stride_length)

    return param_dict, gait_phase_label


def loop_one_sub(opt):
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)

    test_dataset_30 = MotionDatasetManipulated(
        data_path=subject_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        check_cop_to_calcn_distance=False,
        # align_moving_direction_flag=False,
        specific_trial='300'
    )
    skel = list(test_dataset_30.skels.values())[0]
    model_offsets = get_model_offsets(skel).float()
    windows_exp_30 = test_dataset_30.get_one_win_from_the_end_of_each_trial_with_offset([0], 10)

    test_dataset_40 = MotionDatasetManipulated(
        data_path=subject_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        check_cop_to_calcn_distance=False,
        # align_moving_direction_flag=False,
        specific_trial='400'
    )
    windows_exp_40 = test_dataset_40.get_one_win_from_the_end_of_each_trial_with_offset([0], 10)

    windows_syn_base = test_dataset_40.get_one_win_from_the_end_of_each_trial_with_offset(test_dataset_40.manipulated_col_loc, 10)
    windows_syn_dict = {}
    for syn_speed in range(30, 51, 4):
        windows_syn_current_speed = copy.deepcopy(windows_syn_base)
        for win in windows_syn_current_speed:
            pose = model.normalizer.unnormalize(win.pose.unsqueeze(0))
            vel_index = opt.model_states_column_names.index('pelvis_tx')
            pose[:, :, vel_index] = pose[:, :, vel_index] * (syn_speed / 40)
            win.pose = model.normalizer.normalize(pose.squeeze())
        windows_syn_dict[syn_speed] = windows_syn_current_speed

    test_dataset_50 = MotionDatasetManipulated(
        data_path=subject_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        check_cop_to_calcn_distance=False,
        # align_moving_direction_flag=False,
        specific_trial='500'
    )
    windows_exp_50 = test_dataset_50.get_one_win_from_the_end_of_each_trial_with_offset([0], 10)

    state_pred_dict = {}
    for syn_speed, windows_ in windows_syn_dict.items():
        print(f'Speed: {syn_speed}')
        state_syn = torch.stack([win.pose for win in windows_])
        masks = torch.stack([win.mask for win in windows_])
        cond = torch.stack([win.cond for win in windows_])
        height_m_tensor = torch.tensor([win.height_m for win in windows_])

        value_diff_weight = torch.ones([len(opt.model_states_column_names)])
        value_diff_weight[test_dataset_40.do_not_follow_col_loc] = 0

        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        for i_dof in range(state_syn.shape[2]):
            thd = (state_syn[:, :, i_dof].max() - state_syn[:, :, i_dof].min()) * 0.3
            value_diff_thd[i_dof] = thd
        value_diff_thd[test_dataset_40.manipulated_col_loc] = thd
        value_diff_thd[test_dataset_40.do_not_follow_col_loc] = 999

        fix_seed()
        state_pred = model.eval_loop(opt, state_syn, masks, value_diff_thd, value_diff_weight, cond=cond,
                                     num_of_generation_per_window=num_of_generation_per_window, mode='inpaint_ddim_guided')

        # # to test time
        # start_time = time.time()
        # state_pred = model.eval_loop(opt, state_syn.repeat(100, 1, 1), masks.repeat(100, 1, 1), value_diff_thd, value_diff_weight, cond=cond,
        #                              num_of_generation_per_window=num_of_generation_per_window, mode='inpaint_ddim_guided')
        # end_time = time.time()
        # print('Took {:.2f} seconds'.format(end_time - start_time))

        state_pred = inverse_convert_addb_state_to_model_input(
            state_pred, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)
        state_pred_dict[syn_speed] = state_pred

    trial_of_this_win = test_dataset_40.trials[0]
    dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
    skel = test_dataset_40.skels[dset_sub_name]

    param_dict_syn_dict = {}
    win_syn_sub_list = []
    for syn_speed, state_pred in state_pred_dict.items():
        # param_dict_syn_current_speed = {param: [] for param in param_dict_exp_50.keys()}
        param_dict_syn_current_speed = {}
        for i_repeated_pred in range(state_pred.shape[0]):
            win_syn_ = inverse_norm_cops(skel, state_pred[i_repeated_pred].squeeze().numpy(), opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
            win_syn_sub_list.append(win_syn_)
            param_syn_, _ = profile_to_peak(win_syn_, opt.osim_dof_columns, model_offsets)
            for param in param_syn_.keys():
                if param in param_dict_syn_current_speed.keys():
                    param_dict_syn_current_speed[param].extend(param_syn_[param])
                else:
                    param_dict_syn_current_speed[param] = param_syn_[param]
        param_dict_syn_dict[syn_speed] = param_dict_syn_current_speed

    param_dict_exp_dict = {}
    win_exp_list = []
    plt.plot(windows_exp_40[0].pose[:, opt.model_states_column_names.index('calcn_r_force_vy')])
    for speed, win_exp in zip([30, 40, 50], [windows_exp_30, windows_exp_40, windows_exp_50]):
        win_exp = torch.concatenate([win.pose for win in win_exp], dim=0).unsqueeze(0)
        win_exp = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win_exp), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(windows_exp_40[0].height_m)).squeeze().numpy()
        win_exp = inverse_norm_cops(skel, win_exp, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
        win_exp_list.append(win_exp)
        param_dict_exp, _ = profile_to_peak(win_exp, opt.osim_dof_columns, model_offsets)
        param_dict_exp_dict[speed] = param_dict_exp
    delta_exp = {param: np.mean(param_dict_exp_dict[50][param]) - np.mean(param_dict_exp_dict[40][param]) for param in param_dict_exp_dict[50].keys()}

    # gui = set_up_gui()
    # name_states_dict = {
    #     'experiment 4 m/s': win_exp_list[1],
    #     'synthetic 3 m/s': win_syn_sub_list[0],
    #     'synthetic 3.4 m/s': win_syn_sub_list[1],
    #     'synthetic 4.6 m/s': win_syn_sub_list[-2],
    #     'synthetic 5 m/s': win_syn_sub_list[-1]}
    # for _ in range(5):
    #     show_skeletons(opt, name_states_dict, gui, skel)

    delta_syn = {param: np.mean(param_dict_syn_dict[50][param]) - np.mean(param_dict_syn_dict[42][param]) for param in param_dict_syn_dict[50].keys()}
    return delta_exp, delta_syn, param_dict_syn_dict, param_dict_exp_dict, win_exp_list, win_syn_sub_list


def plot_and_save_direction_results(delta_exp_list, delta_syn_list, subject_list):
    delta_exp_dict, delta_syn_dict = {}, {}
    for key_ in delta_exp_list[0].keys():
        delta_exp_dict[key_] = np.array([delta_exp[key_] for delta_exp in delta_exp_list])
        delta_syn_dict[key_] = np.array([delta_syn[key_] for delta_syn in delta_syn_list])

    pickle.dump([delta_exp_dict, delta_syn_dict], open(f"results/da_run_faster.pkl", "wb"))

    plt.figure(figsize=(6, 4))
    plt.ylabel('# of Subject decreased      # of Subject  increased  ', labelpad=13)
    plt.xlabel('Flexion angles')
    plt.yticks(np.arange(-10, 11, 2))
    plt.ylim([-10, 11])
    ax = plt.gca()
    for i_param, (param, param_name) in enumerate(zip(
            ['hip_flexion_r_max', 'knee_angle_r_max', 'ankle_angle_r_max', 'hip_flexion_r_min'],
            ['Hip max', 'Knee max', 'Ankle max', 'Hip min'])):
        delta_exp = delta_exp_dict[param]
        delta_syn = delta_syn_dict[param]
        increased_idx = delta_exp > 0
        increased_num = np.sum(increased_idx)
        increased_num_syn = np.sum(delta_syn[increased_idx] > 0)
        ax.bar(param_name, [increased_num, increased_num_syn], color=['gray', 'C0'], label=['', ''])
        decrease_num_syn = np.sum(delta_syn[~increased_idx] < 0)
        ax.bar(param_name, -np.array([delta_exp.shape[0] - increased_num, decrease_num_syn]), color=['gray', 'C0'])

        plt.figure(figsize=(6, 4))
        for i_sub, sub_name in enumerate(subject_list):
            plt.plot(delta_exp_dict[param][i_sub], delta_syn_dict[param][i_sub], 'o', label=sub_name)
        plt.xlabel(param_name + ' change - Experimental (deg)')
        plt.ylabel(param_name + ' change - Synthetic (deg)')
        plt.legend()
    ax.legend(['Experimental', 'Synthetic'])
    ax.plot([-0.5, i_param+0.5], [0, 0], 'black', linewidth=2)


def plot_and_save_fine_grind_results(param_dict_syn_dict_list, param_dict_exp_dict_list):
    speed_param_syn, speed_param_exp = {}, {}
    for speed in param_dict_syn_dict_list[0].keys():
        speed_param_syn[speed] = {}
        for param in param_dict_syn_dict_list[0][speed].keys():
            speed_param_syn[speed][param] = np.array([np.mean(param_dict_subject[speed][param]) for param_dict_subject in param_dict_syn_dict_list])
    for speed in param_dict_exp_dict_list[0].keys():
        speed_param_exp[speed] = {}
        for param in param_dict_exp_dict_list[0][speed].keys():
            # Only take the fisrt gait cycle
            speed_param_exp[speed][param] = np.array([param_dict_subject[speed][param][0] for param_dict_subject in param_dict_exp_dict_list])
    pickle.dump([speed_param_syn, speed_param_exp], open(f"results/da_run_faster_speeds.pkl", "wb"))


def save_win_data(win_exp_dict, win_syn_dict):
    pickle.dump([win_exp_dict, win_syn_dict], open(f"results/da_run_faster_win.pkl", "wb"))


b3d_path = f'/mnt/d/Local/Data/MotionPriorData/hamner_dset/'
opt = parse_opt()
opt.n_guided_steps = 5
opt.guidance_lr = 0.02
opt.guide_x_start_the_beginning_step = 1000
opt.guide_x_start_the_end_step = 0
opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/{'train-2560_diffusion.pt'}"
num_of_generation_per_window = 1
"""
This one is similar to da_run_faster.py, but calibrated by a same-speed-simulation.
"""

if __name__ == "__main__":
    subject_list = os.listdir(b3d_path)[:]
    delta_exp_list, delta_syn_list, param_dict_syn_dict_list, param_dict_exp_dict_list = [], [], [], []
    win_exp_dict, win_syn_dict = [], []
    for subject in subject_list[:]:
        subject_path = os.path.join(b3d_path, subject)
        delta_exp, delta_syn, param_dict_syn_dict, param_dict_exp_dict, windows_exp_40, win_syn_sub_list = loop_one_sub(opt)
        win_exp_dict.append(windows_exp_40)
        win_syn_dict.append(win_syn_sub_list)
        delta_exp_list.append(delta_exp)
        delta_syn_list.append(delta_syn)
        param_dict_syn_dict_list.append(param_dict_syn_dict)
        param_dict_exp_dict_list.append(param_dict_exp_dict)
    save_win_data(win_exp_dict, win_syn_dict)
    plot_and_save_fine_grind_results(param_dict_syn_dict_list, param_dict_exp_dict_list)
    plot_and_save_direction_results(delta_exp_list, delta_syn_list, subject_list)
    plt.show()






