import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from fig_utils import show_skeletons, set_up_gui, extract_gait_parameters_from_osim_states, \
    extract_gait_parameters_from_osim_states_and_append
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input, linear_resample_data_as_num_of_dp
from model.utils import inverse_norm_cops


def select_the_best_generation(state_pred_list_batch, skel, trial_of_this_win):
    knee_angle_r_max, i_win_list = [], []
    for i_win, state_pred in enumerate(state_pred_list_batch):
        state_pred = inverse_norm_cops(skel, state_pred[0].numpy(), opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
        if len(extract_gait_parameters_from_osim_states(state_pred, skel, opt)['knee_angle_r_max']):
            knee_angle_r_max.append(extract_gait_parameters_from_osim_states(state_pred, skel, opt)['knee_angle_r_max'][0])
            i_win_list.append(i_win)

    i_median_gait = np.argsort(knee_angle_r_max)[len(knee_angle_r_max)//2]
    state_pred = inverse_norm_cops(skel, state_pred_list_batch[i_win_list[i_median_gait]][0], opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m).numpy()

    name_states_dict = {
        'generation_0': state_pred,
    }
    for _ in range(1):
        show_skeletons(opt, name_states_dict, gui, skel)

    return state_pred


def loop_all(opt):
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)
    for speed in [0.8, 1, 1.2, 1.4]:
        print('Generating speed: ', speed)
        test_dataset = MotionDataset(
            data_path=b3d_path,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            max_trial_num=1,
            # specific_trial=f'{speed}00'
        )
        windows_exp = test_dataset.get_one_win_from_the_end_of_each_trial([0])

        state_pred_list = []
        for win_exp in windows_exp:
            trial_of_this_win = test_dataset.trials[win_exp.trial_id]
            dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
            skel = test_dataset.skels[dset_sub_name]

            state_manipulated = torch.zeros([1, 150, 75])
            state_manipulated[:, :, 0] = speed / win_exp.height_m * model.normalizer.scaler.scale_[0] + model.normalizer.scaler.min_[0]
            masks = torch.zeros([1, 150, 75])
            masks[:, :, 0] = 1
            cond = torch.stack([win_exp.cond])
            height_m_tensor = torch.tensor([win_exp.height_m])
            state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, cond=cond,
                                                    num_of_generation_per_window=skel_num - 1, mode='guided_run_faster')

            state_pred_list_batch = inverse_convert_addb_state_to_model_input(
                state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)

            state_pred_list.append(select_the_best_generation(state_pred_list_batch, skel, trial_of_this_win))

        param_dict_exp, param_dict_syn = {}, {}
        for i_win, (win_exp, win_syn) in enumerate(zip(windows_exp, state_pred_list)):
            trial_of_this_win = test_dataset.trials[win_exp.trial_id]
            dset_sub_name = trial_of_this_win.dset_name + '_' + trial_of_this_win.sub_and_trial_name.split('__')[0]
            skel = test_dataset.skels[dset_sub_name]
            win_exp = inverse_convert_addb_state_to_model_input(
                model.normalizer.unnormalize(win_exp.pose.unsqueeze(0)), opt.model_states_column_names,
                opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], torch.tensor(win_exp.height_m)).squeeze().numpy()
            win_exp = inverse_norm_cops(skel, win_exp, opt, trial_of_this_win.weight_kg, trial_of_this_win.height_m)
            param_dict_exp = extract_gait_parameters_from_osim_states_and_append(win_exp, skel, opt, param_dict_exp)
            param_dict_syn = extract_gait_parameters_from_osim_states_and_append(win_syn, skel, opt, param_dict_syn)

        angle_resampled = [linear_resample_data_as_num_of_dp(item, 101) for item in param_dict_syn['knee_angle_r']]
        average_knee_angle_exp = np.mean(angle_resampled, axis=0)
        std_ = np.std(angle_resampled, axis=0)
        plt.plot(average_knee_angle_exp * 180 / np.pi, label='Experimental')
        plt.fill_between(range(101), (average_knee_angle_exp - std_) * 180 / np.pi, (average_knee_angle_exp + std_) * 180 / np.pi, alpha=0.3)
        plt.grid()
    plt.show()


b3d_path = f'/mnt/d/Local/Data/MotionPriorData/hamner_dset/'

"""
Pure-generation-based. The baseline trial is barely used.
"""

if __name__ == "__main__":
    skel_num = 10
    opt = parse_opt()
    opt.n_guided_steps = 0
    opt.guidance_lr = 0
    gui = set_up_gui()
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'7680_diffusion'}.pt"
    loop_all(opt)





