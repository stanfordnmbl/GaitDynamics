import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from fig_utils import show_skeletons, set_up_gui, extract_gait_parameters_from_osim_states_and_append
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input
from model.utils import inverse_norm_cops


def loop_all(opt):
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)
    model.diffusion.set_normalizer(model.normalizer)

    max_trial_num = 2
    trial_start_num = 0

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
    windows_bl_exp = test_dataset.get_one_win_from_the_end_of_each_trial([i for i, name in enumerate(opt.model_states_column_names) if i != 0 and 'vel' not in name])

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
    windows_original_exp = test_dataset.get_one_win_from_the_end_of_each_trial([i for i, name in enumerate(opt.model_states_column_names) if i != 0 and 'vel' not in name])

    state_pred_list = []
    for i_win in range(0, len(windows_bl_exp), 1):
        state_bl = [win.pose for win in windows_bl_exp[i_win:i_win+1]]
        state_bl = torch.stack(state_bl)
        masks = [win.mask for win in windows_bl_exp[i_win:i_win+1]]
        masks = torch.stack(masks)
        cond = [win.cond for win in windows_bl_exp[i_win:i_win+1]]
        cond = torch.stack(cond)

        height_m_tensor = torch.tensor([win.height_m for win in windows_bl_exp[i_win:i_win+1]])

        state_pred_list_batch = model.eval_loop(opt, state_bl, masks, cond=cond, num_of_generation_per_window=skel_num - 1, mode='run_faster')
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)

        state_pred_list += [[state_pred_list_batch[i_skel, i_win] for i_skel in range(skel_num-1)] for i_win in range(state_pred_list_batch.shape[1])]

    gui = set_up_gui()
    param_dict_original_exp, param_dict_bl_exp, param_dict_syn = {}, {}, {}
    for i_win, (win_original_exp, win_bl_exp, win_bl_syn_all_skel) in enumerate(zip(windows_original_exp, windows_bl_exp, state_pred_list)):
        trial_of_this_win = test_dataset.trials[win_original_exp.trial_id]
        win_original_exp = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win_original_exp.pose.unsqueeze(0)), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(win_original_exp.height_m)).squeeze().numpy()
        win_bl_exp = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win_bl_exp.pose.unsqueeze(0)), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(win_bl_exp.height_m)).squeeze().numpy()

        dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
        skel_ = test_dataset.skels[dset_sub_name]
        win_original_exp = inverse_norm_cops(skel_, win_original_exp, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
        param_dict_original_exp = extract_gait_parameters_from_osim_states_and_append(win_original_exp, skel_, opt, param_dict_original_exp)
        win_bl_exp = inverse_norm_cops(skel_, win_bl_exp, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
        param_dict_bl_exp = extract_gait_parameters_from_osim_states_and_append(win_bl_exp, skel_, opt, param_dict_bl_exp)
        win_bl_syn = inverse_norm_cops(skel_, win_bl_syn_all_skel[0], opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m).detach().numpy()
        param_dict_syn = extract_gait_parameters_from_osim_states_and_append(win_bl_syn, skel_, opt, param_dict_syn)

        # name_states_dict = {
        #     'experiment 4 m/s': win_bl_exp,
        #     'synthesized 5 m/s': win_bl_syn,
        #     # 'experiment 5 m/s': win_original_exp
        # }
        # for _ in range(3):
        #     show_skeletons(opt, name_states_dict, gui, test_dataset.skels[dset_sub_name])


    plt.figure()
    plt.ylabel('# of Subject with Decreased Parameter                                      # of Subject with Increased Parameter')
    ax = plt.gca()
    for i_param, param in enumerate(
            [
                # 'stride_length_r', 'stride_time_r', 'stance_time_r',
                'hip_flexion_r_max', 'hip_flexion_r_min',
                'knee_angle_r_max', 'knee_angle_r_min',
                'ankle_angle_r_max', 'ankle_angle_r_min']):
        print(param)
        print(param_dict_bl_exp[param])
        print(param_dict_syn[param])
        print(param_dict_original_exp[param])

        delta_exp = np.array(param_dict_original_exp[param])-np.array(param_dict_bl_exp[param])
        delta_syn = np.array(param_dict_syn[param])-np.array(param_dict_bl_exp[param])
        increased_idx = delta_exp > 0
        increased_num = np.sum(increased_idx)
        increased_num_syn = np.sum(delta_syn[increased_idx] > 0)
        ax.bar(param, [increased_num, increased_num_syn], color=['gray', 'C0'])
        decrease_num_syn = np.sum(delta_syn[~increased_idx] < 0)
        ax.bar(param, -np.array([delta_exp.shape[0] - increased_num, decrease_num_syn]), color=['gray', 'C0'])

        plt.figure()
        plt.plot(np.array(param_dict_original_exp[param])-np.array(param_dict_bl_exp[param]),
                 np.array(param_dict_syn[param])-np.array(param_dict_bl_exp[param]), 'o')
        plt.title(param)
    ax.plot([-0.5, i_param+0.5], [0, 0], 'black', linewidth=2)
    plt.show()


b3d_path = f'/mnt/d/Local/Data/MotionPriorData/hamner_dset/'

""" 
Gradient-based, which computes gradient of vel w.r.t. angles.
"""

if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'6993'}.pt"
    loop_all(opt)





