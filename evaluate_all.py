from args import parse_opt, set_with_arm_opt
import torch
import os
from consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from model.model import inverse_convert_addb_state_to_model_input
import numpy as np
import pickle


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)

    model = MotionModel(opt, repr_dim)
    dset_list = DATASETS_NO_ARM
    results_true, results_pred, results_pred_std, results_bl, height_m_all, weight_kg_all = {}, {}, {}, {}, {}, {}
    is_output_label_array = torch.zeros([150, 35])

    for dset in dset_list:
        print(dset)
        test_dataset = MotionDataset(
            data_path=opt.data_path_test,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            specific_dset=dset,
        )
        windows = test_dataset.get_all_wins_within_gait_cycle([0])
        if len(windows) == 0:
            continue

        baseline_returns = get_baseline_val(dset, windows, test_dataset.trials)
        if not baseline_returns:
            continue
        results_bl.update(get_baseline_val(dset, windows, test_dataset.trials))

        results_true.update({test_dataset.trials[0].dset_name: {}})
        results_pred.update({test_dataset.trials[0].dset_name: {}})
        results_pred_std.update({test_dataset.trials[0].dset_name: {}})

        height_m_all[dset] = {trial_.sub_and_trial_name: trial_.height_m for trial_ in test_dataset.trials}
        weight_kg_all[dset] = {trial_.sub_and_trial_name: trial_.height_m for trial_ in test_dataset.trials}

        state_pred_list = [[] for _ in range(skel_num-1)]
        for i_win in range(0, len(windows), opt.batch_size_inference):

            state_true = [win.pose for win in windows[i_win:i_win+opt.batch_size_inference]]
            state_true = torch.stack(state_true)

            if da_to_test == 0:
                # For GRF estimation
                masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
                masks[:, :, opt.kinematic_diffusion_col_loc] = 1
                is_output_label_array[:, opt.grf_osim_col_loc] = 1
                save_name = 'downstream_grf'

            elif da_to_test == 1:
                # For future motion prediction
                masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
                masks[:, :, opt.kinematic_diffusion_col_loc] = 1
                masks[:, end_of_known:, :] = 0
                is_output_label_array[end_of_known:, ] = 1
                # state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
                # pos_vec = test_dataset.trials[windows[i_win][2]].pos_vec_for_pos_alignment
                # state_pred_list_batch = inverse_convert_addb_state_to_model_input(
                #     state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, pos_vec)
                #
                # for i_skel in range(skel_num-1):
                #     state_pred_list[i_skel] += state_pred_list_batch[i_skel]
                save_name = f'downstream_future_motion_{end_of_known}'

            elif da_to_test == 2:
                # For reconstruct kinematics
                masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
                masks[:, :, cols_to_mask[mask_key]] = 1
                save_name = f'downstream_reconstruct_kinematics_{mask_key}'
            else:
                raise ValueError('da_to_test should be 0, 1, or 2')

            state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
            pos_vec = np.array([test_dataset.trials[windows[i_win_vec].trial_id].pos_vec_for_pos_alignment
                                for i_win_vec in range(len(windows[i_win:i_win+opt.batch_size_inference]))])
            height_m_tensor = torch.tensor([win.height_m for win in windows[i_win:i_win+opt.batch_size_inference]])
            state_pred_list_batch = inverse_convert_addb_state_to_model_input(
                state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, pos_vec, height_m_tensor)
            for i_skel in range(skel_num-1):
                state_pred_list[i_skel] += state_pred_list_batch[i_skel]

        for i_skel in range(skel_num-1):
            assert len(state_pred_list[i_skel]) == len(windows)
        state_pred_list_averaged, state_pred_list_std = [], []
        for i_win in range(len(state_pred_list[0])):
            win_skels = [state_pred_list[i_skel][i_win] for i_skel in range(skel_num-1)]
            averaged = torch.mean(torch.stack(win_skels), dim=0)
            std = torch.std(torch.stack(win_skels), dim=0)
            state_pred_list_averaged.append(averaged)
            state_pred_list_std.append(std)

        for win, state_pred_mean, state_pred_std in zip(windows, state_pred_list_averaged, state_pred_list_std):
            trial = test_dataset.trials[win.trial_id]
            if trial.sub_and_trial_name not in results_true[trial.dset_name].keys():
                results_true[trial.dset_name].update({trial.sub_and_trial_name: []})
                results_pred[trial.dset_name].update({trial.sub_and_trial_name: []})
                results_pred_std[trial.dset_name].update({trial.sub_and_trial_name: []})
            true_val = inverse_convert_addb_state_to_model_input(
                model.normalizer.unnormalize(win.pose.unsqueeze(0)), opt.model_states_column_names,
                opt.joints_3d, opt.osim_dof_columns, trial.pos_vec_for_pos_alignment, torch.tensor(win.height_m)).squeeze().numpy()
            results_true[trial.dset_name][trial.sub_and_trial_name].append(true_val)
            results_pred[trial.dset_name][trial.sub_and_trial_name].append(state_pred_mean)
            results_pred_std[trial.dset_name][trial.sub_and_trial_name].append(state_pred_std)

    for dset in results_true.keys():
        for sub_and_trial in results_true[dset].keys():
            results_true[dset][sub_and_trial] = np.concatenate(results_true[dset][sub_and_trial], axis=0)
            results_pred[dset][sub_and_trial] = np.concatenate(results_pred[dset][sub_and_trial], axis=0)
            results_pred_std[dset][sub_and_trial] = np.concatenate(results_pred_std[dset][sub_and_trial], axis=0)
            results_bl[dset][sub_and_trial] = np.concatenate(results_bl[dset][sub_and_trial], axis=0)

    pickle.dump([results_true, results_pred, results_pred_std, results_bl, opt.osim_dof_columns,
                 is_output_label_array, height_m_all, weight_kg_all],
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
        train_dataset.normalizer.unnormalize(average_gait_cycle.unsqueeze(0)), opt.model_states_column_names,
        opt.joints_3d, opt.osim_dof_columns, pos_vec=[0, 0, 0]).squeeze().numpy()

    pickle.dump(average_gait_cycle, open(f"figures/results/average_gait_cycle_{dset}.pkl", "wb"))


def get_baseline_val(dset, windows, trials):
    if not os.path.exists(f"figures/results/average_gait_cycle_{dset}.pkl"):
        print(f"average_gait_cycle_{dset}.pkl does not exist")
        return None
    average_gait_cycle = pickle.load(open(f"figures/results/average_gait_cycle_{dset}.pkl", "rb"))

    bl_pred = {dset: {}}

    for win in windows:
        i_trial = win.trial_id
        gait_phase_label = win.gait_phase_label
        if trials[i_trial].sub_and_trial_name not in bl_pred[dset].keys():
            bl_pred[dset].update({trials[i_trial].sub_and_trial_name: []})
        bl_pred_current_win = average_gait_cycle[gait_phase_label]
        bl_pred_current_win[gait_phase_label == NOT_IN_GAIT_PHASE] = NOT_IN_GAIT_PHASE
        bl_pred[dset][trials[i_trial].sub_and_trial_name].append(bl_pred_current_win)

    return bl_pred


if __name__ == "__main__":
    skel_num = 4
    opt = parse_opt()

    # opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'4925'}.pt"
    opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'fixed_txtytz_vel'}/weights/train-{'6993'}.pt"

    cols_to_mask = {
        'ankle': opt.knee_diffusion_col_loc,
        'knee': opt.knee_diffusion_col_loc,
        'hip': opt.hip_diffusion_col_loc,
        'knee_ankle': opt.knee_diffusion_col_loc + opt.ankle_diffusion_col_loc,
        'knee_ankle_hip': opt.knee_diffusion_col_loc + opt.ankle_diffusion_col_loc + opt.hip_diffusion_col_loc
    }

    # for dset in ['Tan2021_Formatted_No_Arm']:
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















