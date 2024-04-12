from alant.args import parse_opt, set_with_arm_opt
import torch
import os
from alant.alan_consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE
from model.alan_model import MotionModel, MotionDataset
from model.utils import linear_resample_data_as_num_of_dp
from model.alan_model import inverse_convert_addb_state_to_model_input
import numpy as np
import pickle
import matplotlib.pyplot as plt


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, repr_dim == 56)

    model = MotionModel(opt, repr_dim)
    dset_list = DATASETS_NO_ARM

    for dset in dset_list:
        test_dataset = MotionDataset(
            data_path=opt.data_path_test,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            specific_dset=dset,
            # max_trial_num=1,     # !!!
            # trial_start_num=5,   # !!!
        )
        windows_compact = test_dataset.get_all_wins()
        if len(windows_compact) == 0:
            continue

        state_pred_list = [[] for _ in range(skel_num-1)]
        for i_win in range(0, len(windows_compact), opt.batch_size_inference):

            state_true = [win[0] for win in windows_compact[i_win:i_win+opt.batch_size_inference]]
            state_true = torch.stack(state_true)

            # For GRF estimation
            masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
            masks[:, :, :] = 1

            value_diff_thd = torch.zeros([len(opt.model_states_column_names)])     # TODO: for gait pattern manipulation, add soft boundaries and tune values
            value_diff_weight = torch.ones([len(opt.model_states_column_names)])
            col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'calcn_r_force' in col]

            value_diff_thd[:] = 1
            value_diff_thd[col_loc] = 0          # set tx ty tz to 0 for strict position constraint

            _, state_pred_list_batch = model.eval_loop(opt, state_true, masks, value_diff_thd, value_diff_weight,
                                                       num_of_generation_per_window=skel_num-1)
            for i_skel in range(skel_num-1):
                state_pred_list[i_skel] += state_pred_list_batch[i_skel]

        for i_skel in range(skel_num-1):
            assert len(state_pred_list[i_skel]) == len(windows_compact)
        state_pred_list_averaged = []
        for i_win in range(len(state_pred_list[0])):
            averaged = torch.zeros_like(state_pred_list[0][i_win])
            for i_skel in range(skel_num-1):
                averaged += state_pred_list[i_skel][i_win]
            averaged /= (skel_num-1)
            state_pred_list_averaged.append(averaged)

        for win, state_pred in zip(windows_compact, state_pred_list_averaged):
            true_val = inverse_convert_addb_state_to_model_input(
                model.normalizer.unnormalize(win[0].unsqueeze(0)),
                model_states_column_names=opt.model_states_column_names,
                joints_3d=opt.joints_3d, osim_dof_columns=opt.osim_dof_columns).squeeze().numpy()

            for param in ['calcn_l_force_vy', 'calcn_r_force_vy']:
            # for param in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz', 'calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']:
                # , 'subtalar_angle_r', 'mtp_angle_r', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                #           'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l'
                col_loc = opt.osim_dof_columns.index(param)
                plt.figure()
                plt.plot(true_val[:, col_loc])
                plt.plot(state_pred.detach().numpy()[:, col_loc])
                plt.title(param)
            plt.show()


if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.guide_x_start_the_beginning_step = 50

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'5000'}.pt"
    # opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'norm_all4'}/weights/train-{'5000'}.pt"

    da_to_test = 0
    loop_all(opt)










