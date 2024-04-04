from alant.args import parse_opt, set_with_arm_opt
import torch
import os
from alant.alan_consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE
from model.alan_model import MotionModel, MotionDataset
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
    results_true, results_pred, results_bl, sub_heights, sub_weights = {}, {}, {}, {}, {}
    is_output_label_array = torch.zeros([150, 29])

    for dset in dset_list:
        test_dataset = MotionDataset(
            data_path=opt.data_path_test,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            specific_dset=dset,
            # max_trial_num=1,     # !!!
            # trial_start_num=7,   # !!!
        )
        windows = test_dataset.get_all_wins_within_gait_cycle()
        if len(windows) == 0:
            continue

        baseline_returns = get_baseline_val(dset, windows, test_dataset.trials)
        if not baseline_returns:
            continue
        results_bl.update(get_baseline_val(dset, windows, test_dataset.trials))

        results_true.update({test_dataset.trials[0].dset_name: {}})
        results_pred.update({test_dataset.trials[0].dset_name: {}})

        sub_heights[dset] = {trial_.sub_and_trial_name: trial_.sub_height for trial_ in test_dataset.trials}
        sub_weights[dset] = {trial_.sub_and_trial_name: trial_.sub_weight for trial_ in test_dataset.trials}

        state_pred_list = [[] for _ in range(skel_num-1)]
        for i_win in range(0, len(windows), opt.batch_size_inference):

            state_true = [win[0] for win in windows[i_win:i_win+opt.batch_size_inference]]
            state_true = torch.stack(state_true)

            if da_to_test == 0:
                # For GRF estimation
                masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
                masks[:, :, kinematic_diffusion_col_loc] = 1
                is_output_label_array[:, grf_osim_col_loc] = 1
                _, state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
                for i_skel in range(skel_num-1):
                    state_pred_list[i_skel] += state_pred_list_batch[i_skel]
                save_name = 'downstream_grf'

            elif da_to_test == 1:
                # For future motion prediction
                masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
                masks[:, :, kinematic_diffusion_col_loc] = 1
                masks[:, end_of_known:, :] = 0
                is_output_label_array[end_of_known:, ] = 1

                _, state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
                for i_skel in range(skel_num-1):
                    state_pred_list[i_skel] += state_pred_list_batch[i_skel]
                save_name = f'downstream_future_motion_{end_of_known}'

            elif da_to_test == 2:
                # For reconstruct kinematics
                masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
                masks[:, :, cols_to_mask[mask_key]] = 1
                _, state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
                for i_skel in range(skel_num-1):
                    state_pred_list[i_skel] += state_pred_list_batch[i_skel]
                save_name = f'downstream_reconstruct_kinematics_{mask_key}'

        for i_skel in range(skel_num-1):
            assert len(state_pred_list[i_skel]) == len(windows)
        state_pred_list_averaged = []
        for i_win in range(len(state_pred_list[0])):
            averaged = torch.zeros_like(state_pred_list[0][i_win])
            for i_skel in range(skel_num-1):
                averaged += state_pred_list[i_skel][i_win]
            averaged /= (skel_num-1)
            state_pred_list_averaged.append(averaged)

        for win, state_pred in zip(windows, state_pred_list_averaged):
            # windows.append((trial_.converted_pose[i:i+self.window_len, ...], self.trials[i_trial].model_offsets, i_trial, gait_phase_label))
            trial = test_dataset.trials[win[2]]
            if trial.sub_and_trial_name not in results_true[trial.dset_name].keys():
                results_true[trial.dset_name].update({trial.sub_and_trial_name: []})
                results_pred[trial.dset_name].update({trial.sub_and_trial_name: []})
            true_val = inverse_convert_addb_state_to_model_input(
                model.normalizer.unnormalize(win[0].unsqueeze(0)),
                model_states_column_names=opt.model_states_column_names,
                joints_3d=opt.joints_3d, osim_dof_columns=opt.osim_dof_columns).squeeze().numpy()
            results_true[trial.dset_name][trial.sub_and_trial_name].append(true_val)
            results_pred[trial.dset_name][trial.sub_and_trial_name].append(state_pred)

    for dset in results_true.keys():
        for sub_and_trial in results_true[dset].keys():
            results_true[dset][sub_and_trial] = np.concatenate(results_true[dset][sub_and_trial], axis=0)
            results_pred[dset][sub_and_trial] = np.concatenate(results_pred[dset][sub_and_trial], axis=0)
            results_bl[dset][sub_and_trial] = np.concatenate(results_bl[dset][sub_and_trial], axis=0)

    pickle.dump([results_true, results_pred, results_bl, opt.osim_dof_columns, is_output_label_array,
                 sub_heights, sub_weights],
                open(f"figures/results/{save_name}.pkl", "wb"))


def save_average_gait(dset):
    train_dataset = MotionDataset(
        opt.data_path_train,
        train=True,
        specific_dset=dset,
        opt=opt,
        # max_trial_num=3,     # !!!
        # trial_start_num=2,   # !!!
    )
    if not train_dataset:
        return
    cycles_ = train_dataset.get_all_gait_cycles_and_set_gait_phase_label()
    if len(cycles_) == 0:
        return
    gait_cycles_resampled = [item_.gait_cycle_resampled for item_ in cycles_]
    average_gait_cycle = torch.mean(torch.stack(gait_cycles_resampled), dim=0)

    average_gait_cycle = inverse_convert_addb_state_to_model_input(
        train_dataset.normalizer.unnormalize(average_gait_cycle.unsqueeze(0)),
        model_states_column_names=opt.model_states_column_names,
        joints_3d=opt.joints_3d, osim_dof_columns=opt.osim_dof_columns).squeeze().numpy()

    pickle.dump(average_gait_cycle, open(f"figures/results/average_gait_cycle_{dset}.pkl", "wb"))


def get_baseline_val(dset, windows, trials):
    if not os.path.exists(f"figures/results/average_gait_cycle_{dset}.pkl"):
        return None
    average_gait_cycle = pickle.load(open(f"figures/results/average_gait_cycle_{dset}.pkl", "rb"))

    bl_pred = {dset: {}}

    for win in windows:
        i_trial = win[2]
        gait_phase_label = win[3]
        if trials[i_trial].sub_and_trial_name not in bl_pred[dset].keys():
            bl_pred[dset].update({trials[i_trial].sub_and_trial_name: []})
        bl_pred_current_win = average_gait_cycle[gait_phase_label]
        bl_pred_current_win[gait_phase_label == NOT_IN_GAIT_PHASE] = NOT_IN_GAIT_PHASE
        bl_pred[dset][trials[i_trial].sub_and_trial_name].append(bl_pred_current_win)

    return bl_pred


if __name__ == "__main__":
    skel_num = 4            # !!!
    opt = parse_opt()
    opt.guide_x_start_the_beginning_step = -10      # negative value means no guidance

    # opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'5000'}.pt"
    opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'hz100_smaller_model'}/weights/train-{'5000'}.pt"

    knee_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'knee' in col]
    ankle_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'ankle' in col]
    hip_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'hip' in col]
    kinematic_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' not in col]
    grf_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if 'force' in col]
    cols_to_mask = {
        'ankle': knee_diffusion_col_loc,
        'knee': knee_diffusion_col_loc,
        'hip': hip_diffusion_col_loc,
        'knee_ankle': knee_diffusion_col_loc + ankle_diffusion_col_loc,
        'knee_ankle_hip': knee_diffusion_col_loc + ankle_diffusion_col_loc + hip_diffusion_col_loc
    }

    # for dset in ['Han2023_Formatted_No_Arm']:
    #     save_average_gait(dset)

    da_to_test = 0
    loop_all(opt)

    # da_to_test = 1
    # for end_of_known in range(130, 151, 5):
    #     print('end_of_known: ', end_of_known)
    #     loop_all(opt)

    # da_to_test = 2
    # for mask_key in cols_to_mask.keys():
    #     print('mask_key: ', mask_key)
    #     loop_all(opt)















