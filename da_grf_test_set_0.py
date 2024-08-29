import pickle
from args import parse_opt, set_with_arm_opt
import torch
from consts import OSIM_DOF_ALL, KINETICS_ALL, WEIGHT_KG_OVERWRITE, DATASETS_NO_ARM
from model.model import MotionModel, BaselineModel, TransformerEncoderArchitecture, LstmArchitecture
from data.addb_dataset import MotionDataset, TrialData
import numpy as np
from model.utils import convert_addb_state_to_model_input, inverse_convert_addb_state_to_model_input, align_moving_direction
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
        opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'diffusion'}/weights/train-{'6993'}.pt"
    else:
        opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'7680_diffusion'}.pt"
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)
    model_key = 'diffusion'
    return model, model_key


def load_baseline_model(opt, model_to_test):
    if model_to_test == 1:
        model_architecture_class = TransformerEncoderArchitecture
        if opt.use_server:
            opt.checkpoint_bl = opt.data_path_parent + f"/../code/runs/train/{'baseline_tf'}/weights/train-{'7680'}.pt"
        else:
            opt.checkpoint_bl = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'7680_tf'}.pt"
        model_key = 'baseline_tf'
    elif model_to_test == 3:
        model_architecture_class = LstmArchitecture
        if opt.use_server:
            opt.checkpoint_bl = opt.data_path_parent + f"/../code/runs/train/{'baseline_lstm2'}/weights/train-{'7680'}.pt"
        else:
            opt.checkpoint_bl = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'7680_lstm'}.pt"
        model_key = 'baseline_lstm'

    set_with_arm_opt(opt, False)
    model = BaselineModel(opt, model_architecture_class, EMA=False)
    return model, model_key


def loop_all(opt, trials, windows, s_list, e_list):
    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows), opt.batch_size_inference):
        state_true = torch.stack([win.pose for win in windows[i_win:i_win+opt.batch_size_inference]])
        masks = torch.stack([win.mask for win in windows[i_win:i_win+opt.batch_size_inference]])
        cond = torch.stack([win.cond for win in windows[i_win:i_win+opt.batch_size_inference]])
        height_m_tensor = torch.tensor([win.height_m for win in windows[i_win:i_win+opt.batch_size_inference]])

        state_pred_list_batch = model.eval_loop(opt, state_true, masks, cond=cond, num_of_generation_per_window=skel_num-1)
        pos_vec = np.array([trials[windows[i_win_vec].trial_id].pos_vec_for_pos_alignment
                            for i_win_vec in range(len(windows[i_win:i_win+opt.batch_size_inference]))])
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

    results_true, results_pred, results_pred_std, results_s, results_e = {}, {}, {}, {}, {}
    for i_win, (win, state_pred_mean, state_pred_std) in enumerate(zip(windows, state_pred_list_averaged, state_pred_list_std)):
        trial = trials[win.trial_id]
        if trial.sub_and_trial_name not in results_true.keys():
            results_true.update({trial.sub_and_trial_name: []})
            results_pred.update({trial.sub_and_trial_name: []})
            results_pred_std.update({trial.sub_and_trial_name: []})
            results_s.update({trial.sub_and_trial_name: []})
            results_e.update({trial.sub_and_trial_name: []})

        true_val = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win.pose.unsqueeze(0)), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, trial.pos_vec_for_pos_alignment, torch.tensor(win.height_m)).squeeze().numpy()
        mask = win.mask.squeeze().numpy()
        true_val = true_val * np.bool_(mask.sum(axis=1)).repeat(35).reshape((150, -1))
        state_pred_mean = state_pred_mean * np.bool_(mask.sum(axis=1)).repeat(35).reshape((150, -1))

        results_true[trial.sub_and_trial_name].append(true_val)
        results_pred[trial.sub_and_trial_name].append(state_pred_mean.numpy())
        results_s[trial.sub_and_trial_name].append(s_list[i_win])
        results_e[trial.sub_and_trial_name].append(e_list[i_win])

    true_all, pred_all, pred_std_all = [], [], []
    for sub_and_trial in results_true.keys():
        trial_len = [trial.converted_pose.shape[0] for trial in trials if trial.sub_and_trial_name == sub_and_trial][0]
        results_true[sub_and_trial], _ = convert_overlapped_list_to_array(
            trial_len, results_true[sub_and_trial], results_s[sub_and_trial], results_e[sub_and_trial])
        results_pred[sub_and_trial], results_pred_std[sub_and_trial] = convert_overlapped_list_to_array(
            trial_len, results_pred[sub_and_trial], results_s[sub_and_trial], results_e[sub_and_trial])

        for trial in trials:
            if trial.sub_and_trial_name == sub_and_trial:
                probably_missing = trial.probably_missing
                break
        true_all.append(results_true[sub_and_trial][np.where(probably_missing==0)[0]])
        pred_all.append(results_pred[sub_and_trial][np.where(probably_missing==0)[0]])
        pred_std_all.append(results_pred_std[sub_and_trial][np.where(probably_missing==0)[0]])
    true_all = np.concatenate(true_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    pred_std_all = np.concatenate(pred_std_all, axis=0)
    return true_all, pred_all, pred_std_all


def loop_mask_conditions():
    n_split = 10
    results_dict = {}
    for mask_key, unmask_col_loc in cols_to_unmask.items():
        print(mask_key)
        true_sub_dict, pred_sub_dict, pred_std_sub_dict = {}, {}, {}
        for dset in test_dataset_dict.keys():
            if dset in dset_to_split:
                windows_splits, trials_splits, dset_names, s_list_splits, e_list_splits = [], [], [], [], []
                for i_split in range(n_split+1):
                    start_trial = i_split * (len(test_dataset_dict[dset].trials) // n_split)
                    end_trial = min(len(test_dataset_dict[dset].trials), (i_split + 1) * (len(test_dataset_dict[dset].trials) // n_split))
                    if end_trial == start_trial:
                        continue
                    windows, s_list, e_list = test_dataset_dict[dset].get_overlapping_wins(unmask_col_loc, win_step_length, start_trial, end_trial)
                    if len(windows) == 0:
                        continue
                    windows_splits.append(windows)
                    trials_splits.append(test_dataset_dict[dset].trials)
                    dset_names.append(dset + f'_{i_split}')
                    s_list_splits.append(s_list)
                    e_list_splits.append(e_list)

            else:
                dset_names = [dset]
                windows, s_list, e_list = test_dataset_dict[dset].get_overlapping_wins(unmask_col_loc, win_step_length)
                if len(windows) == 0:
                    continue
                windows_splits = [windows]
                trials_splits = [test_dataset_dict[dset].trials]
                s_list_splits = [s_list]
                e_list_splits = [e_list]

            # to speed up
            if speed_up and len(windows_splits) > 0 and len(windows_splits[-1]) > 200:
                windows_splits = [windows_splits[-1][:200]]
                trials_splits = [trials_splits[-1]]
                dset_names = [dset_names[-1]]
                s_list_splits = [s_list_splits[-1][:200]]
                e_list_splits = [e_list_splits[-1][:200]]

            for trials, windows, dset_name, s_list, e_list in zip(trials_splits, windows_splits, dset_names, s_list_splits, e_list_splits):
                if 'diffusion_model_for_filling' in locals() or 'diffusion_model_for_filling' in globals():
                    windows = fill_missing_with_diffusion(windows, diffusion_model_for_filling)
                true_sub, pred_sub, pred_std_sub = loop_all(opt, trials, windows, s_list, e_list)
                true_sub_dict[dset_name] = true_sub[:, params_of_interest_col_loc]
                pred_sub_dict[dset_name] = pred_sub[:, params_of_interest_col_loc]
                pred_std_sub_dict[dset_name] = pred_std_sub[:, params_of_interest_col_loc]
        results_dict.update({mask_key: [true_sub_dict, pred_sub_dict, pred_std_sub_dict, params_of_interest]})
    pickle.dump(results_dict, open(f"figures/results/addb_marker_based_{model_key}.pkl", "wb"))


def fill_missing_with_diffusion(windows, diffusion_model_for_filling):
    for i_win in range(0, len(windows), opt.batch_size_inference):
        state_true = torch.stack([win.pose for win in windows[i_win:i_win+opt.batch_size_inference]])
        masks = torch.stack([win.mask for win in windows[i_win:i_win+opt.batch_size_inference]])
        cond = torch.stack([win.cond for win in windows[i_win:i_win+opt.batch_size_inference]])

        constraint = {'mask': masks, 'value': state_true.clone(), 'cond': cond}
        shape = (state_true.shape[0], state_true.shape[1], state_true.shape[2])
        samples = (diffusion_model_for_filling.diffusion.inpaint_ddim_loop(shape, constraint=constraint))
        samples = state_true * masks + (1.0 - masks) * samples.to(state_true.device)
        samples[:, :, opt.kinetic_diffusion_col_loc] = state_true[:, :, opt.kinetic_diffusion_col_loc]      # reserve true GRF

        unmasked_samples_in_temporal_dim = (masks.sum(axis=2)).bool()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(samples[1, :, opt.model_states_column_names.index('knee_angle_r')].detach().cpu().numpy())
        # plt.plot(state_true[1, :, opt.model_states_column_names.index('knee_angle_r')].detach().cpu().numpy())
        # plt.figure()
        # plt.plot(samples[1, :, opt.model_states_column_names.index('calcn_r_force_vy')].detach().cpu().numpy())
        # plt.plot(state_true[1, :, opt.model_states_column_names.index('calcn_r_force_vy')].detach().cpu().numpy())
        # plt.show()

        for j_win in range(len(samples)):
            windows[j_win+i_win].pose = samples[j_win]
            updated_mask = windows[j_win+i_win].mask
            updated_mask[unmasked_samples_in_temporal_dim[j_win], :] = 1
            updated_mask[:, opt.kinetic_diffusion_col_loc] = 0
            windows[j_win+i_win].mask = updated_mask
    return windows


opt = parse_opt()
params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
params_of_interest_col_loc = [opt.osim_dof_columns.index(col) for col in params_of_interest]
cols_to_unmask = {
    'none': opt.kinematic_diffusion_col_loc,
    'velocity': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('force' not in col and 'pelvis_t' not in col)],
    'trunk': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('force' not in col and 'lumbar' not in col)],
    'pelvis': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('force' not in col and 'pelvis' not in col) or 'pelvis_t' in col],
    'hip': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('force' not in col and 'hip' not in col)],
    'knee': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('force' not in col and 'knee' not in col)],
    'ankle': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('force' not in col and 'ankle' not in col and 'subtalar' not in col)],
}
# cols_to_unmask.update({
#     'trunk_pelvis': list(set(cols_to_unmask['trunk']).intersection(cols_to_unmask['pelvis'])),
#     'trunk_hip': list(set(cols_to_unmask['trunk']).intersection(cols_to_unmask['hip'])),
#     'pelvis_hip': list(set(cols_to_unmask['pelvis']).intersection(cols_to_unmask['hip'])),
# })
# cols_to_unmask.update({
#     'trunk_pelvis_knee_ankle': list(list(set(cols_to_unmask['trunk_pelvis']).intersection(cols_to_unmask['knee']).intersection(cols_to_unmask['ankle']))),
#     'trunk_pelvis_hip_knee_ankle': list(list(set(cols_to_unmask['trunk_pelvis']).intersection(cols_to_unmask['knee']).intersection(cols_to_unmask['ankle']))),
# })
cols_to_unmask.update({
    'velocity_hip': list(set(cols_to_unmask['velocity']).intersection(cols_to_unmask['hip'])),
    'trunk_pelvis_knee_ankle': list(list(set(cols_to_unmask['trunk']).intersection(cols_to_unmask['pelvis']).intersection(cols_to_unmask['knee']).intersection(cols_to_unmask['ankle']))),
})
# cols_to_unmask = {key: cols_to_unmask[key] for key in ['none']}  # !!!

dset_to_split = ['Camargo2021_Formatted_No_Arm', 'Moore2015_Formatted_No_Arm', 'vanderZee2022_Formatted_No_Arm']
dset_to_skip = ['Santos2017_Formatted_No_Arm']
dset_specific_trial = {dset: None for dset in DATASETS_NO_ARM}
dset_specific_trial['Falisse2017_Formatted_No_Arm'] = 'Gait'
dset_specific_trial['Li2021_Formatted_No_Arm'] = 'Trial'
dset_specific_trial['Han2023_Formatted_No_Arm'] = 'walk'
dset_specific_trial['Uhlrich2023_Formatted_No_Arm'] = 'walking'
dset_specific_trial['Wang2023_Formatted_No_Arm'] = ['walk', 'run']
skel_num = 2
win_step_length = 15

if __name__ == "__main__":
    model_to_test = 0       # 0 for diffusion, 1 for diffusion+transformer, 2 for tcn, 3 for LSTM
    speed_up = False
    max_trial_num = None     # None for all trials

    if model_to_test == 0:
        model, model_key = load_model(opt)
    elif model_to_test == 1:
        diffusion_model_for_filling, _ = load_model(opt)
        model, model_key = load_baseline_model(opt, model_to_test=model_to_test)
    elif model_to_test == 2:
        model, model_key = load_baseline_model(opt, model_to_test=model_to_test)
        opt.batch_size = opt.batch_size*100
    elif model_to_test == 3:
        model, model_key = load_baseline_model(opt, model_to_test=model_to_test)

    test_dataset_dict = {}
    for dset in DATASETS_NO_ARM:
        if dset in dset_to_skip:
            continue
        print(dset)
        test_dataset = MotionDataset(
            data_path=opt.data_path_test,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            specific_dset=dset,
            specific_trial=dset_specific_trial[dset],
            include_trials_shorter_than_window_len=True,
            restrict_contact_bodies=False,
            max_trial_num=max_trial_num,
        )
        test_dataset_dict[dset] = test_dataset
    loop_mask_conditions()







