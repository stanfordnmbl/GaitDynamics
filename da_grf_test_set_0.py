import pickle
from args import parse_opt, set_with_arm_opt
import torch
from consts import OSIM_DOF_ALL, KINETICS_ALL, WEIGHT_KG_OVERWRITE, DATASETS_NO_ARM
from model.model import MotionModel
from data.addb_dataset import MotionDataset, TrialData
import numpy as np
from model.utils import convert_addb_state_to_model_input, inverse_convert_addb_state_to_model_input, align_moving_direction
import os


def convert_overlapped_list_to_array(trial_len, win_list, s_, e_, fun=np.nanmedian):
    array_val_expand = np.full((len(win_list), trial_len, win_list[0].shape[1]), np.nan)
    for i_win, (win, s, e) in enumerate(zip(win_list, s_, e_)):
        array_val_expand[i_win, s:e] = win[:e-s]
    array_val = fun(array_val_expand, axis=0)
    std_val = np.nanstd(array_val_expand, axis=0)
    return array_val, std_val


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
                break
        true_all.append(results_true[sub_and_trial])
        pred_all.append(results_pred[sub_and_trial])
        pred_std_all.append(results_pred_std[sub_and_trial])
    true_all = np.concatenate(true_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    pred_std_all = np.concatenate(pred_std_all, axis=0)
    return true_all, pred_all, pred_std_all


def load_model(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)
    model = MotionModel(opt, repr_dim)
    return model


def loop_mask_conditions():

    n_split = 10
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

            for trials, windows, dset_name, s_list, e_list in zip(trials_splits, windows_splits, dset_names, s_list_splits, e_list_splits):
                true_sub, pred_sub, pred_std_sub = loop_all(opt, trials, windows, s_list, e_list)
                true_sub_dict[dset_name] = true_sub
                pred_sub_dict[dset_name] = pred_sub
                pred_std_sub_dict[dset_name] = pred_std_sub

        pickle.dump([true_sub_dict, pred_sub_dict, pred_std_sub_dict, opt.osim_dof_columns], open(f"figures/results/addb_marker_based_{mask_key}.pkl", "wb"))


opt = parse_opt()
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
# cols_to_unmask = {key: cols_to_unmask[key] for key in ['velocity', 'trunk', 'trunk_pelvis_knee_ankle', 'velocity_hip']}  # !!!

dset_to_split = ['Camargo2021_Formatted_No_Arm', 'Moore2015_Formatted_No_Arm', 'vanderZee2022_Formatted_No_Arm']
dset_to_skip = ['Santos2017_Formatted_No_Arm', 'Li2021_Formatted_No_Arm', 'Fregly2012_Formatted_No_Arm', 'Uhlrich2023_Formatted_No_Arm', 'Han2023_Formatted_No_Arm']
dset_specific_trial = {dset: None for dset in DATASETS_NO_ARM}
dset_specific_trial['Uhlrich2023_Formatted_No_Arm'] = 'walking'
dset_specific_trial['Wang2023_Formatted_No_Arm'] = ['walk', 'run']

if __name__ == "__main__":
    skel_num = 2
    win_step_length = 15

    """ Test subjects, use marker-based kinematics """
    # opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'small_model'}/weights/train-{'6993'}.pt"
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'6993'}.pt"

    model = load_model(opt)
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
            # max_trial_num=3
        )
        test_dataset_dict[dset] = test_dataset
    loop_mask_conditions()







