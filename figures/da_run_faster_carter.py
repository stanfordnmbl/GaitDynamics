import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from figures.fig_utils import show_skeletons, set_up_gui, extract_gait_parameters_from_osim_states, \
    extract_gait_parameters_from_osim_states_and_append
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input
from model.utils import cross_product_2d, get_multi_body_loc_using_nimble_by_body_names, inverse_norm_cops


class MotionDatasetManipulated(MotionDataset):
    def customized_param_manipulation(self, trial_df, mtp_r_vel, mtp_l_vel):
        # vel increase
        ratio = 1.1
        trial_df['pelvis_tx'] = trial_df['pelvis_tx'] * ratio
        mtp_r_vel[:, 0] = mtp_r_vel[:, 0] * ratio
        mtp_l_vel[:, 0] = mtp_l_vel[:, 0] * ratio
        self.manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'pelvis_tx' in col]
        self.do_not_follow_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' in col]

        return trial_df, mtp_r_vel, mtp_l_vel


def get_diff_of_a_subject(sub):
    test_dataset_mani = MotionDatasetManipulated(
        data_path=carter_data_path+sub+'_split3',
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
        specific_trial='flatrun_norm'
    )
    if len(test_dataset_mani.trials) == 0:
        return
    windows_manipulated_exp = test_dataset_mani.get_all_wins_within_gait_cycle(test_dataset_mani.manipulated_col_loc)

    test_dataset = MotionDataset(
        data_path=carter_data_path+sub+'_split3',
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
        specific_trial='flatrun_norm'       # if doesn't work, try flatrun_norm flatrun_fixed
    )
    windows_bl_exp = test_dataset.get_all_wins_within_gait_cycle([0])

    test_dataset = MotionDataset(
        data_path=carter_data_path+sub+'_split1',
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
        specific_trial='flatrun_fast'
    )
    windows_original_exp = test_dataset.get_all_wins_within_gait_cycle([0])

    if len(windows_original_exp) * len(windows_bl_exp) * len(windows_manipulated_exp) == 0:
        return

    state_pred_list = []
    for i_win in range(0, len(windows_manipulated_exp), opt.batch_size_inference):
        state_manipulated = [win.pose for win in windows_manipulated_exp[i_win:i_win+opt.batch_size_inference]]
        state_manipulated = torch.stack(state_manipulated)

        masks = torch.stack([win.mask for win in windows_manipulated_exp[i_win:i_win+opt.batch_size_inference]])
        cond = torch.stack([win.cond for win in windows_manipulated_exp[i_win:i_win+opt.batch_size_inference]])
        height_m_tensor = torch.tensor([win.height_m for win in windows_manipulated_exp[i_win:i_win+opt.batch_size_inference]])

        value_diff_weight = torch.ones([len(opt.model_states_column_names)])
        value_diff_weight[test_dataset_mani.do_not_follow_col_loc] = 0

        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        value_diff_thd[:] = 0         # large value for no constraint
        value_diff_thd[test_dataset_mani.manipulated_col_loc] = 0

        state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, value_diff_thd, value_diff_weight, cond=cond,
                                                num_of_generation_per_window=skel_num - 1, mode='guided_run_faster')
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)

        state_pred_list += [[state_pred_list_batch[i_skel, i_win] for i_skel in range(skel_num-1)] for i_win in range(state_pred_list_batch.shape[1])]

    weight_kg = test_dataset.trials[0].weight_kg
    height_m = test_dataset.trials[0].height_m
    skel = list(test_dataset.skels.values())[0]

    param_dict_original_exp, param_dict_bl_exp, param_dict_syn = {}, {}, {}
    for i_type, (window_type, param_dict) in enumerate(zip([windows_original_exp, windows_bl_exp], [param_dict_original_exp, param_dict_bl_exp])):
        window_type = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(torch.stack([win.pose for win in window_type])), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(height_m))
        for i_win, win_original_exp in enumerate(window_type):
            win_original_exp = inverse_norm_cops(skel, win_original_exp, opt, weight_kg, height_m).squeeze().numpy()
            param_dict = extract_gait_parameters_from_osim_states_and_append(win_original_exp, skel, opt, param_dict, 999)

    for win_original_syn in state_pred_list:
        win_original_syn = inverse_norm_cops(skel, win_original_syn[0], opt, weight_kg, height_m).squeeze().numpy()
        param_dict_syn = extract_gait_parameters_from_osim_states_and_append(win_original_syn, skel, opt, param_dict_syn, 999)

    for param in param_to_plot:
        delta_exp[param].append(np.mean(param_dict_original_exp[param]) - np.mean(param_dict_bl_exp[param]))
        delta_syn[param].append(np.mean(param_dict_syn[param]) - np.mean(param_dict_bl_exp[param]))


def show_results():
    for i_param, (param, param_name) in enumerate(zip(param_to_plot, param_to_plot_names)):
        print(param)

        plt.figure(figsize=(6, 4))
        plt.plot(np.array(delta_exp[param]) * 180 / np.pi, np.array(delta_syn[param]) * 180 / np.pi, 'o')
        plt.xlabel(param_name + ' change - Experimental (deg)')
        plt.ylabel(param_name + ' change - Synthetic (deg)')
        plt.title(param_name)
        plt.savefig(f'{param_name}1.png')


max_trial_num = 99
trial_start_num = 0
skel_num = 2
param_to_plot = ['hip_flexion_r_max', 'knee_angle_r_max', 'ankle_angle_r_max', 'hip_flexion_r_min']
param_to_plot_names = ['Hip max', 'Knee max', 'Ankle max', 'Hip min']
gui = set_up_gui()

if __name__ == "__main__":
    opt = parse_opt()
    carter_data_path = '/dataNAS/people/alanttan/mfm/data/b3d_no_arm/train_cleaned/Carter2023_Formatted_No_Arm/'
    opt.checkpoint = f"/dataNAS/people/alanttan/mfm/code/runs/train/t_plus_cond/weights/train-{'6993'}.pt"

    # carter_data_path = '/mnt/d/Local/Data/MotionPriorData/b3d_no_arm/train_cleaned/Carter2023_Formatted_No_Arm/'
    # opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'6993'}.pt"

    set_with_arm_opt(opt, False)
    model = MotionModel(opt)

    subjects = list(sorted(set([x[0].split('Carter2023_Formatted_No_Arm/')[1].split('_split')[0]
                                for x in os.walk(carter_data_path)])))
    delta_exp, delta_syn = {param: [] for param in param_to_plot}, {param: [] for param in param_to_plot}
    for i_sub, sub_ in enumerate(subjects[:]):
        if 'P0' not in sub_:
            continue
        get_diff_of_a_subject(sub_)

    for i_param, param in enumerate(param_to_plot):
        show_results()


















