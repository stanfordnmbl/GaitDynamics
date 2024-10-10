import copy
import pickle
import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input, fix_seed
from model.utils import inverse_norm_cops
from fig_utils import show_skeletons, set_up_gui


class MotionDatasetManipulated(MotionDataset):
    def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
        self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'pelvis_tx' in col]
        self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' in col]
        return trial_df, mtp_r_vel, mtp_l_vel


def profile_to_peak(true_, columns):
    param_dict = {
        'hip_flexion_l_max': [], 'knee_angle_l_max': [], 'ankle_angle_l_max': [],
        'hip_flexion_l_min': [], 'knee_angle_l_min': [], 'ankle_angle_l_min': []
    }
    gait_phase_label = []

    v_grf = true_[:, columns.index('calcn_l_force_vy')]
    stance_flag = np.abs(v_grf) > 1
    stance_flag = stance_flag.astype(int)
    start_end_indicator = np.diff(stance_flag)
    stance_starts = np.where(start_end_indicator == 1)[0]
    for i_start in range(len(stance_starts) - 1):
        start_ = stance_starts[i_start]
        end_ = stance_starts[i_start + 1]

        param_dict['hip_flexion_l_max'].append(np.max(true_[start_:end_, columns.index('hip_flexion_l')]))
        param_dict['knee_angle_l_max'].append(np.max(true_[start_:end_, columns.index('knee_angle_l')]))
        param_dict['ankle_angle_l_max'].append(np.max(true_[start_:end_, columns.index('ankle_angle_l')]))
        param_dict['hip_flexion_l_min'].append(np.min(true_[start_:end_, columns.index('hip_flexion_l')]))
        param_dict['knee_angle_l_min'].append(np.min(true_[start_:end_, columns.index('knee_angle_l')]))
        param_dict['ankle_angle_l_min'].append(np.min(true_[start_:end_, columns.index('ankle_angle_l')]))

    return param_dict, gait_phase_label


def loop_one_sub(opt):
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)

    test_dataset_slow = MotionDatasetManipulated(
        data_path=subject_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        check_cop_to_calcn_distance=False,
        specific_trial='400'
    )
    windows_exp_slow = test_dataset_slow.get_all_wins([0], False)
    windows_syn_slow = test_dataset_slow.get_all_wins(test_dataset_slow.manipulated_col_loc, False)

    windows_syn_fast = copy.deepcopy(windows_syn_slow)
    for win in windows_syn_fast:
        pose = model.normalizer.unnormalize(win.pose.unsqueeze(0))
        vel_index = opt.model_states_column_names.index('pelvis_tx')
        pose[:, :, vel_index] = pose[:, :, vel_index] * 1.1
        win.pose = model.normalizer.normalize(pose.squeeze())

    test_dataset_fast = MotionDatasetManipulated(
        data_path=subject_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        check_cop_to_calcn_distance=False,
        specific_trial='500'
    )
    windows_exp_fast = test_dataset_fast.get_all_wins([0], False)

    list_for_win_syn_fast_and_slow = []
    for windows_ in [windows_syn_fast, windows_syn_slow]:
        state_syn_fast = torch.stack([win.pose for win in windows_])
        masks = torch.stack([win.mask for win in windows_])
        cond = torch.stack([win.cond for win in windows_])
        height_m_tensor = torch.tensor([win.height_m for win in windows_])

        value_diff_weight = torch.ones([len(opt.model_states_column_names)])
        value_diff_weight[test_dataset_slow.do_not_follow_col_loc] = 0

        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        for i_dof in range(state_syn_fast.shape[2]):
            thd = (state_syn_fast[:, :, i_dof].max() - state_syn_fast[:, :, i_dof].min()) * 0.2
            value_diff_thd[i_dof] = thd
        value_diff_thd[test_dataset_slow.manipulated_col_loc] = 999
        value_diff_thd[test_dataset_slow.do_not_follow_col_loc] = 999

        fix_seed()
        state_pred = model.eval_loop(opt, state_syn_fast, masks, value_diff_thd, value_diff_weight, cond=cond,
                                     num_of_generation_per_window=3, mode='guided_run_faster')
        state_pred = inverse_convert_addb_state_to_model_input(
            state_pred, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)
        list_for_win_syn_fast_and_slow.append(state_pred)

    # gui = set_up_gui()
    win_exp_fast = torch.concatenate([win.pose for win in windows_exp_fast], dim=0).unsqueeze(0)
    win_exp_slow = torch.concatenate([win.pose for win in windows_exp_slow], dim=0).unsqueeze(0)

    trial_of_this_win = test_dataset_slow.trials[0]
    win_exp_fast = inverse_convert_addb_state_to_model_input(
        model.normalizer.unnormalize(win_exp_fast), opt.model_states_column_names,
        opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(windows_exp_fast[0].height_m)).squeeze().numpy()
    win_exp_slow = inverse_convert_addb_state_to_model_input(
        model.normalizer.unnormalize(win_exp_slow), opt.model_states_column_names,
        opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(windows_exp_slow[0].height_m)).squeeze().numpy()

    dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
    skel = test_dataset_slow.skels[dset_sub_name]
    win_exp_fast = inverse_norm_cops(skel, win_exp_fast, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
    win_exp_slow = inverse_norm_cops(skel, win_exp_slow, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
    param_dict_exp_fast, _ = profile_to_peak(win_exp_fast, opt.osim_dof_columns)
    param_dict_exp_slow, _ = profile_to_peak(win_exp_slow, opt.osim_dof_columns)
    delta_exp = {param: np.mean(param_dict_exp_fast[param]) - np.mean(param_dict_exp_slow[param]) for param in param_dict_exp_fast.keys()}

    # if delta_exp['knee_angle_l_max'] < 0:
    #     plt.plot(win_exp_fast[:, opt.osim_dof_columns.index('knee_angle_l')], '--')
    #     plt.plot(win_exp_slow[:, opt.osim_dof_columns.index('knee_angle_l')])
    #     plt.title('knee_angle_l')
    #     plt.show()
    # if delta_exp['hip_flexion_l_max'] < 0:
    #     plt.plot(win_exp_fast[:, opt.osim_dof_columns.index('hip_flexion_l')], '--')
    #     plt.plot(win_exp_slow[:, opt.osim_dof_columns.index('hip_flexion_l')])
    #     plt.title('hip_flexion_l')
    #     plt.show()

    win_syn_fast_tensor, win_syn_slow_tensor = list_for_win_syn_fast_and_slow
    win_syn_fast_list, win_syn_slow_list = [], []
    param_dict_syn_fast = {param: [] for param in param_dict_exp_fast.keys()}
    param_dict_syn_slow = {param: [] for param in param_dict_exp_fast.keys()}
    for i_repeated_pred in range(win_syn_fast_tensor.shape[0]):
        win_syn_fast_list.append(inverse_norm_cops(skel, win_syn_fast_tensor[i_repeated_pred].squeeze().numpy(), opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m))
        param_syn_fast, _ = profile_to_peak(win_syn_fast_list[i_repeated_pred], opt.osim_dof_columns)
        win_syn_slow_list.append(inverse_norm_cops(skel, win_syn_slow_tensor[i_repeated_pred].squeeze().numpy(), opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m))
        param_syn_slow, _ = profile_to_peak(win_syn_slow_list[i_repeated_pred], opt.osim_dof_columns)
        for param in param_dict_syn_fast.keys():
            param_dict_syn_fast[param].extend(param_syn_fast[param])
            param_dict_syn_slow[param].extend(param_syn_slow[param])

    #     plt.plot(win_syn_fast_list[i_repeated_pred][:, opt.osim_dof_columns.index('knee_angle_l')], '--')
    #     plt.plot(win_syn_slow_list[i_repeated_pred][:, opt.osim_dof_columns.index('knee_angle_l')])
    # plt.show()

    # name_states_dict = {
    #     'experiment 4 m/s': win_exp_slow,
    #     'synthesized 5 m/s 1': win_syn_fast_list[0],
    #     'synthesized 5 m/s 2': win_syn_fast_list[1],
    #     'synthesized 5 m/s 3': win_syn_fast_list[2],
    #     'experiment 5 m/s': win_exp_fast}
    # for _ in range(5):
    #     show_skeletons(opt, name_states_dict, gui, skel)

    delta_syn = {param: np.mean(param_dict_syn_fast[param]) - np.mean(param_dict_syn_slow[param]) for param in param_dict_syn_fast.keys()}
    # print(param_dict_exp_slow)
    # print(param_dict_syn_slow)
    # print(param_dict_syn_fast)
    return delta_exp, delta_syn


def plot_and_save_results(delta_exp_list, delta_syn_list, subject_list):
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
            ['hip_flexion_l_max', 'knee_angle_l_max', 'ankle_angle_l_max', 'hip_flexion_l_min'],
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
            plt.plot(delta_exp_dict[param][i_sub] * 180 / np.pi, delta_syn_dict[param][i_sub] * 180 / np.pi, 'o', label=sub_name)
        plt.xlabel(param_name + ' change - Experimental (deg)')
        plt.ylabel(param_name + ' change - Synthetic (deg)')
        plt.legend()
    ax.legend(['Experimental', 'Synthetic'])
    ax.plot([-0.5, i_param+0.5], [0, 0], 'black', linewidth=2)
    plt.show()


b3d_path = f'/mnt/d/Local/Data/MotionPriorData/hamner_dset/'
opt = parse_opt()
opt.n_guided_steps = 3
opt.guidance_lr = 0.01
opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/{'train-7680_diffusion_no_hamner.pt'}"

"""
This one is similar to da_run_faster.py, but calibrated by a same-speed-simulation.
"""

if __name__ == "__main__":
    subject_list = os.listdir(b3d_path)[:]
    delta_exp_list, delta_syn_list = [], []
    for subject in subject_list:
        subject_path = os.path.join(b3d_path, subject)
        delta_exp, delta_syn = loop_one_sub(opt)
        delta_exp_list.append(delta_exp)
        delta_syn_list.append(delta_syn)
    plot_and_save_results(delta_exp_list, delta_syn_list, subject_list)






