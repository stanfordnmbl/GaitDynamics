import pickle
from args import parse_opt, set_with_arm_opt
import torch
from model.model import MotionModel
from data.addb_dataset import MotionDataset
import numpy as np
from model.utils import inverse_convert_addb_state_to_model_input
import os
import matplotlib.pyplot as plt


def convert_overlapped_list_to_array(trial_len, win_list, s_, e_, fun=np.nanmedian):
    array_val_expand = np.full((len(win_list), trial_len, win_list[0].shape[1]), np.nan)
    for i_win, (win, s, e) in enumerate(zip(win_list, s_, e_)):
        array_val_expand[i_win, s:e] = win[:e-s]
    array_val = fun(array_val_expand, axis=0)
    std_val = np.nanstd(array_val_expand, axis=0)
    return array_val, std_val


def load_model(opt):
    if opt.use_server:
        opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'no_camargo_carter'}/weights/train-{'5994'}.pt"
    else:
        opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'2560_diffusion'}.pt"
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)
    model_key = 'diffusion'
    return model, model_key


def loop_trials(opt, trials):
    lumbar_dof = ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    params_of_interest_col_loc = [opt.osim_dof_columns.index(col) for col in lumbar_dof]
    camargo_reconstructed_dict = {}

    unmask_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('force' not in col and 'lumbar' not in col)]
    for i_trial, trial in enumerate(trials):
        print(trial.sub_and_trial_name)
        windows, s_list, e_list = test_dataset.get_overlapping_wins(unmask_col_loc, win_step_length, i_trial, i_trial+1, including_shorter_than_window_len=True)
        state_true = torch.stack([win.pose for win in windows])
        masks = torch.stack([win.mask for win in windows])
        cond = torch.stack([win.cond for win in windows])

        constraint = {'mask': masks, 'value': state_true.clone(), 'cond': cond}
        shape = (state_true.shape[0], state_true.shape[1], state_true.shape[2])
        samples = (model.diffusion.inpaint_ddim_loop(shape, constraint=constraint))
        samples = state_true * masks + (1.0 - masks) * samples.to(state_true.device)

        trial_reconstructed = convert_overlapped_list_to_array(trial.converted_pose.shape[0], [item for item in samples], s_list, e_list)[0]

        trial_reconstructed = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(torch.from_numpy(trial_reconstructed).unsqueeze(0))[0],
            opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, trial.pos_vec_for_pos_alignment, torch.tensor(trial.height_m))
        camargo_reconstructed_dict.update({trial.sub_and_trial_name: trial_reconstructed[:, params_of_interest_col_loc].cpu().numpy()})


        # import matplotlib.pyplot as plt
        # trial_original = inverse_convert_addb_state_to_model_input(
        #     model.normalizer.unnormalize(trial.converted_pose.unsqueeze(0))[0],
        #     opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, trial.pos_vec_for_pos_alignment, torch.tensor(trial.height_m))
        # plt.figure()
        # plt.plot(trial_original[:, opt.osim_dof_columns.index('knee_angle_r')].detach().cpu().numpy())
        # plt.plot(trial_reconstructed[:, opt.osim_dof_columns.index('knee_angle_r')])
        # plt.figure()
        # plt.plot(trial_original[:, opt.osim_dof_columns.index('lumbar_extension')].detach().cpu().numpy())
        # plt.plot(trial_reconstructed[:, opt.osim_dof_columns.index('lumbar_extension')])
        # plt.figure()
        # plt.plot(trial_original[:, opt.osim_dof_columns.index('lumbar_rotation')].detach().cpu().numpy())
        # plt.plot(trial_reconstructed[:, opt.osim_dof_columns.index('lumbar_rotation')])
        # plt.show()


    pickle.dump([camargo_reconstructed_dict, lumbar_dof], open(f"{data_path}/camargo_lumbar_reconstructed.pkl", "wb"))


opt = parse_opt()
win_step_length = 15

if __name__ == "__main__":
    model, model_key = load_model(opt)
    for data_path in [opt.data_path_train]:     # ,opt.data_path_test, opt.data_path_train
        test_dataset = MotionDataset(
            data_path=data_path,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            specific_dset='Camargo2021_Formatted_No_Arm',
            include_trials_shorter_than_window_len=True,
            use_camargo_lumbar_reconstructed=False,
            # max_trial_num=10,
        )
        loop_trials(opt, test_dataset.trials)







