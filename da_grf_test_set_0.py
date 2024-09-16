import copy
import pickle
import pandas as pd
from args import parse_opt, set_with_arm_opt
import torch
from consts import DATASETS_NO_ARM
from model.model import MotionModel, BaselineModel, TransformerEncoderArchitecture
from model_baseline.grf_baseline import GroundLinkArchitecture, SugaiNetArchitecture
from data.addb_dataset import MotionDataset
import numpy as np
from model.utils import inverse_convert_addb_state_to_model_input, inverse_norm_cops, \
    osim_states_to_moments_in_percent_BW_BH_via_cross_product, model_states_osim_dof_conversion
import os
import matplotlib.pyplot as plt


def convert_overlapped_list_to_array(trial_len, win_list, s_, e_, fun=np.nanmedian):
    array_val_expand = np.full((len(win_list), trial_len, win_list[0].shape[1]), np.nan)
    for i_win, (win, s, e) in enumerate(zip(win_list, s_, e_)):
        array_val_expand[i_win, s:e] = win[:e-s]
    array_val = fun(array_val_expand, axis=0)
    std_val = np.nanstd(array_val_expand, axis=0)
    return array_val, std_val


def load_model(model_to_test):
    if model_to_test == 0:
        model, model_key = load_diffusion_model(opt)
    elif model_to_test == 1:
        model, model_key = load_baseline_model(opt, model_to_test=model_to_test)
    elif model_to_test == 2:
        model, model_key = load_baseline_model(opt, model_to_test=model_to_test)
    elif model_to_test == 3:
        model, model_key = load_baseline_model(opt, model_to_test=model_to_test)
    else:
        raise ValueError('Invalid model_to_test')
    print(model_key)
    return model, model_key


def load_diffusion_model(opt):
    if opt.use_server:
        opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'remove_wrong_cop_diffusion'}/weights/train-{'7680'}.pt"
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
            opt.checkpoint_bl = opt.data_path_parent + f"/../code/runs/train/{'remove_wrong_cop_tf'}/weights/train-{'7680'}.pt"
        else:
            opt.checkpoint_bl = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'7680_tf'}.pt"
        model_key = 'tf'
    elif model_to_test == 2:
        model_architecture_class = GroundLinkArchitecture
        if opt.use_server:
            opt.checkpoint_bl = opt.data_path_parent + f"/../code/runs/train/{'remove_wrong_cop_groundlink'}/weights/train-{'7680'}.pt"
        else:
            opt.checkpoint_bl = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'7680_groundlink'}.pt"
        model_key = 'groundlink'
    elif model_to_test == 3:
        model_architecture_class = SugaiNetArchitecture
        if opt.use_server:
            opt.checkpoint_bl = opt.data_path_parent + f"/../code/runs/train/{'remove_wrong_cop_sugainet'}/weights/train-{'999'}.pt"
        else:
            opt.checkpoint_bl = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'1998_sugainet'}.pt"
        model_key = 'sugainet'

    set_with_arm_opt(opt, False)
    model = BaselineModel(opt, model_architecture_class, EMA=True)
    return model, model_key


def loop_all(model, opt, skels, trials, windows, windows_reconstructed, s_list, e_list):
    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows_reconstructed), opt.batch_size_inference):
        state_true = torch.stack([win.pose for win in windows_reconstructed[i_win:i_win+opt.batch_size_inference]])
        masks = torch.stack([win.mask for win in windows_reconstructed[i_win:i_win+opt.batch_size_inference]])
        cond = torch.stack([win.cond for win in windows_reconstructed[i_win:i_win+opt.batch_size_inference]])

        state_pred_list_batch = model.eval_loop(opt, state_true, masks, cond=cond, num_of_generation_per_window=skel_num-1)
        for i_skel in range(skel_num-1):
            state_pred_list[i_skel] += state_pred_list_batch[i_skel]

    for i_skel in range(skel_num-1):
        assert len(state_pred_list[i_skel]) == len(windows_reconstructed)
    state_pred_list_averaged, state_pred_list_std = [], []
    for i_win in range(len(state_pred_list[0])):
        win_skels = [state_pred_list[i_skel][i_win] for i_skel in range(skel_num-1)]
        averaged = torch.mean(torch.stack(win_skels), dim=0)
        std = torch.std(torch.stack(win_skels), dim=0)
        state_pred_list_averaged.append(averaged)
        state_pred_list_std.append(std)

    results_true, results_pred, results_pred_std, results_s, results_e = {}, {}, {}, {}, {}
    heights_weights = {}
    for i_win, (win, state_pred_mean, state_pred_std) in enumerate(zip(windows, state_pred_list_averaged, state_pred_list_std)):
        trial = trials[win.trial_id]
        if trial.sub_and_trial_name not in results_true.keys():
            results_true.update({trial.sub_and_trial_name: []})
            results_pred.update({trial.sub_and_trial_name: []})
            results_pred_std.update({trial.sub_and_trial_name: []})
            results_s.update({trial.sub_and_trial_name: []})
            results_e.update({trial.sub_and_trial_name: []})
            heights_weights[trial.sub_and_trial_name] = win.height_m, win.weight_kg

        true_val = model.normalizer.unnormalize(win.pose.unsqueeze(0))[0].numpy()
        mask = win.mask.squeeze().numpy()
        true_val = true_val * np.bool_(mask.sum(axis=1)).repeat(true_val.shape[1]).reshape((150, -1))
        state_pred_mean = state_pred_mean * np.bool_(mask.sum(axis=1)).repeat(true_val.shape[1]).reshape((150, -1))

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

        sub_name = sub_and_trial.split('__')[0]
        skel_list = [skel for dset_sub_name, skel in skels.items() if sub_name == dset_sub_name[-len(sub_name):]]
        assert len(skel_list) == 1
        skel = skel_list[0]

        height_m_tensor = torch.tensor([heights_weights[sub_and_trial][0]])
        for results_ in [results_true, results_pred, results_pred_std]:
            results_[sub_and_trial] = inverse_convert_addb_state_to_model_input(
                torch.from_numpy(results_[sub_and_trial]).unsqueeze(0), opt.model_states_column_names,
                opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)[0].numpy()
            pelvis_col_loc = [opt.osim_dof_columns.index(col) for col in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]
            results_[sub_and_trial][:, pelvis_col_loc] = np.diff(results_[sub_and_trial][:, pelvis_col_loc], prepend=0, axis=0) * opt.target_sampling_rate
            results_[sub_and_trial] = inverse_norm_cops(skel, results_[sub_and_trial], opt, heights_weights[sub_and_trial][1], heights_weights[sub_and_trial][0])
            params, param_columns = osim_states_to_moments_in_percent_BW_BH_via_cross_product(results_[sub_and_trial], skel, opt, heights_weights[sub_and_trial][0])
            results_[sub_and_trial] = np.concatenate([results_[sub_and_trial], params], axis=-1)

        for trial in trials:
            if trial.sub_and_trial_name == sub_and_trial:
                probably_missing = trial.probably_missing
                break
        true_all.append(results_true[sub_and_trial][np.where(probably_missing==0)[0]])
        pred_all.append(results_pred[sub_and_trial][np.where(probably_missing==0)[0]])
        pred_std_all.append(results_pred_std[sub_and_trial][np.where(probably_missing==0)[0]])
    column_names = opt.osim_dof_columns + param_columns
    return true_all, pred_all, pred_std_all, column_names


def loop_mask_segment_conditions(model, model_key, test_dataset_dict):
    win_step_length = 15
    n_split = 10
    diffusion_model_for_filling, _ = load_diffusion_model(opt)
    for mask_key, unmask_col_loc in cols_to_unmask.items():
        print(mask_key)
        masked_state_names = [opt.model_states_column_names[i] for i in opt.kinematic_diffusion_col_loc if i not in unmask_col_loc]
        masked_osim_dofs = model_states_osim_dof_conversion(masked_state_names, opt)
        for filling_method in [MedianFilling(), DiffusionFilling()]:
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
                    if mask_key == 'none':
                        windows_reconstructed = windows
                    else:
                        windows_reconstructed = filling_method.fill_param(windows, diffusion_model_for_filling)
                    true_sub, pred_sub, pred_std_sub, column_names = loop_all(
                        model, opt, test_dataset_dict[dset].skels, trials, windows, windows_reconstructed, s_list, e_list)
                    if mask_key == 'none':
                        col_name_to_save = column_names
                    else:
                        col_name_to_save = params_of_interest+masked_osim_dofs
                    col_loc_to_save = [column_names.index(col) for col in col_name_to_save]
                    true_sub_dict[dset_name] = [trial_[:, col_loc_to_save] for trial_ in true_sub]
                    pred_sub_dict[dset_name] = [trial_[:, col_loc_to_save] for trial_ in pred_sub]
                    pred_std_sub_dict[dset_name] = [trial_[:, col_loc_to_save] for trial_ in pred_std_sub]
            results_ = [true_sub_dict, pred_sub_dict, pred_std_sub_dict, col_name_to_save]
            pickle.dump(results_, open(f"figures/results/{folder}/{model_key}_{mask_key}_{filling_method}.pkl", "wb"))


def loop_drop_temporal_conditions(model, model_key, test_dataset_dict):
    n_split = 10
    diffusion_model_for_filling, _ = load_diffusion_model(opt)
    filling_methods = [InterpoFilling(), DiffusionFilling(), MedianFilling()]
    for drop_frame_num in drop_frame_num_range:
        print('drop_frame_num:', drop_frame_num)
        true_sub_dict = {method: {} for method in filling_methods}
        pred_sub_dict = {method: {} for method in filling_methods}
        pred_std_sub_dict = {method: {} for method in filling_methods}
        for dset in test_dataset_dict.keys():
            if dset in dset_to_split:
                windows_splits, trials_splits, dset_names, s_list_splits, e_list_splits = [], [], [], [], []
                for i_split in range(n_split+1):
                    start_trial = i_split * (len(test_dataset_dict[dset].trials) // n_split)
                    end_trial = min(len(test_dataset_dict[dset].trials), (i_split + 1) * (len(test_dataset_dict[dset].trials) // n_split))
                    if end_trial == start_trial:
                        continue
                    windows, s_list, e_list = test_dataset_dict[dset].get_overlapping_wins(opt.kinematic_diffusion_col_loc, opt.window_len, start_trial, end_trial)
                    if len(windows) == 0:
                        continue

                    windows_splits.append(windows)
                    trials_splits.append(test_dataset_dict[dset].trials)
                    dset_names.append(dset + f'_{i_split}')
                    s_list_splits.append(s_list)
                    e_list_splits.append(e_list)
            else:
                dset_names = [dset]
                windows, s_list, e_list = test_dataset_dict[dset].get_overlapping_wins(opt.kinematic_diffusion_col_loc, opt.window_len)
                if len(windows) == 0:
                    continue
                windows_splits = [windows]
                trials_splits = [test_dataset_dict[dset].trials]
                s_list_splits = [s_list]
                e_list_splits = [e_list]

            for trials, windows, dset_name, s_list, e_list in zip(trials_splits, windows_splits, dset_names, s_list_splits, e_list_splits):
                rand_start = [np.random.randint(1, win.mask.shape[0] - drop_frame_num - 1) for win in windows]
                mask_original = [win.mask.clone() for win in windows]
                for i_win, (win, start) in enumerate(zip(windows, rand_start)):
                    win.mask[start:start+drop_frame_num, :] = 0

                # check if temporal dimension is complete
                for filling_method in filling_methods:
                    windows_reconstructed = filling_method.fill_temporal(windows, diffusion_model_for_filling, mask_original)

                    # if filling_method == filling_methods[0]:
                    #     plt.figure()
                    #     plt.plot(windows_reconstructed[0].pose[:, opt.model_states_column_names.index('knee_angle_l')], 'C0')
                    #     plt.plot(range(rand_start[0], rand_start[0]+drop_frame_num),
                    #              windows_reconstructed[0].pose[rand_start[0]:rand_start[0]+drop_frame_num, opt.model_states_column_names.index('knee_angle_l')], 'C1')

                    true_sub, pred_sub, pred_std_sub, column_names = loop_all(
                        model, opt, test_dataset_dict[dset].skels, trials, windows, windows_reconstructed, s_list, e_list)

                    col_loc_to_save = [column_names.index(col) for col in params_of_interest]
                    true_sub_dict[filling_method][dset_name] = [trial_[:, col_loc_to_save] for trial_ in true_sub]
                    pred_sub_dict[filling_method][dset_name] = [trial_[:, col_loc_to_save] for trial_ in pred_sub]
                    pred_std_sub_dict[filling_method][dset_name] = [trial_[:, col_loc_to_save] for trial_ in pred_std_sub]
            for filling_method in filling_methods:
                results_ = [true_sub_dict[filling_method], pred_sub_dict[filling_method], pred_std_sub_dict[filling_method], params_of_interest]
                pickle.dump(results_, open(f"figures/results/{folder}/{model_key}_{drop_frame_num}_{filling_method}.pkl", "wb"))


class FillingBase:
    def fill_param(self, windows, diffusion_model_for_filling):
        return self.filling(windows, diffusion_model_for_filling, self.update_kinematics_and_masks_for_masking_column)

    def fill_temporal(self, windows, diffusion_model_for_filling, mask_original):
        self.mask_original = mask_original
        return self.filling(windows, diffusion_model_for_filling, self.update_kinematics_and_masks_for_masking_temporal)

    @staticmethod
    def update_kinematics_and_masks_for_masking_column(windows, samples, i_win, masks):
        unmasked_samples_in_temporal_dim = (masks.sum(axis=2)).bool()
        for j_win in range(len(samples)):
            windows[j_win+i_win].pose = samples[j_win]
            updated_mask = windows[j_win+i_win].mask
            updated_mask[unmasked_samples_in_temporal_dim[j_win], :] = 1
            updated_mask[:, opt.kinetic_diffusion_col_loc] = 0
            windows[j_win+i_win].mask = updated_mask
        return windows

    def update_kinematics_and_masks_for_masking_temporal(self, windows, samples, i_win, masks):
        for j_win in range(len(samples)):
            windows[j_win+i_win].pose = samples[j_win]
            windows[j_win+i_win].mask = self.mask_original[j_win+i_win]
        return windows

    @staticmethod
    def filling(windows, diffusion_model_for_filling, windows_update_func):
        raise NotImplementedError


class DiffusionFilling(FillingBase):
    @staticmethod
    def filling(windows, diffusion_model_for_filling, windows_update_func):
        windows = copy.deepcopy(windows)
        for i_win in range(0, len(windows), opt.batch_size_inference):
            state_true = torch.stack([win.pose for win in windows[i_win:i_win+opt.batch_size_inference]])
            masks = torch.stack([win.mask for win in windows[i_win:i_win+opt.batch_size_inference]])
            cond = torch.stack([win.cond for win in windows[i_win:i_win+opt.batch_size_inference]])

            constraint = {'mask': masks, 'value': state_true.clone(), 'cond': cond}
            shape = (state_true.shape[0], state_true.shape[1], state_true.shape[2])
            samples = (diffusion_model_for_filling.diffusion.inpaint_ddim_loop(shape, constraint=constraint))
            samples = state_true * masks + (1.0 - masks) * samples.to(state_true.device)
            samples[:, :, opt.kinetic_diffusion_col_loc] = state_true[:, :, opt.kinetic_diffusion_col_loc]      # reset GRF to true values

            windows = windows_update_func(windows, samples, i_win, masks)
        return windows

    def __str__(self):
        return 'diffusion_filling'


class MedianFilling(FillingBase):
    @staticmethod
    def filling(windows, diffusion_model_for_filling, windows_update_func):
        windows = copy.deepcopy(windows)
        for i_win in range(0, len(windows), opt.batch_size_inference):
            state_true = torch.stack([win.pose for win in windows[i_win:i_win+opt.batch_size_inference]])
            masks = torch.stack([win.mask for win in windows[i_win:i_win+opt.batch_size_inference]])
            samples = state_true.clone()
            samples[~masks.bool()] = 0
            samples[:, :, opt.kinetic_diffusion_col_loc] = state_true[:, :, opt.kinetic_diffusion_col_loc]      # reset GRF to true values
            windows = windows_update_func(windows, samples, i_win, masks)
        return windows

    def __str__(self):
        return 'zero_filling'       # change to median_filling


class InterpoFilling(FillingBase):
    @staticmethod
    def filling(windows, diffusion_model_for_filling, windows_update_func):
        windows = copy.deepcopy(windows)
        for i_win in range(0, len(windows)):
            state_true = torch.stack([win.pose for win in windows[i_win:i_win+1]])
            masks = torch.stack([win.mask for win in windows[i_win:i_win+1]])
            samples = pd.DataFrame(state_true[0].clone().numpy())

            samples[(~masks.bool()[0]).numpy()] = np.nan
            samples = samples.interpolate(method='linear', axis=0)
            samples = torch.from_numpy(samples.values).unsqueeze(0)
            samples[:, :, opt.kinetic_diffusion_col_loc] = state_true[:, :, opt.kinetic_diffusion_col_loc]      # reset GRF to true values
            windows = windows_update_func(windows, samples, i_win, masks)
        return windows

    def __str__(self):
        return 'interpo_filling'


def load_test_dataset_dict():
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
    return test_dataset_dict


opt = parse_opt()
opt.batch_size = opt.batch_size*5
params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
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

drop_frame_num_range = range(5, 51, 5)
dset_to_split = ['Camargo2021_Formatted_No_Arm', 'Moore2015_Formatted_No_Arm', 'vanderZee2022_Formatted_No_Arm']
dset_to_skip = ['Santos2017_Formatted_No_Arm']
dset_specific_trial = {dset: None for dset in DATASETS_NO_ARM}
dset_specific_trial['Falisse2017_Formatted_No_Arm'] = 'Gait'
dset_specific_trial['Li2021_Formatted_No_Arm'] = ['Trial25', 'Trial26']     # Other trials do not have valid pelvis angles
dset_specific_trial['Han2023_Formatted_No_Arm'] = 'walk'
dset_specific_trial['Uhlrich2023_Formatted_No_Arm'] = 'walking'
dset_specific_trial['Wang2023_Formatted_No_Arm'] = ['walk', 'run']
skel_num = 2

if __name__ == "__main__":
    # 0: diffusion, 1: tf, 2: groundlink, 3: Sugai LSTM
    model_to_test = 1
    max_trial_num = None     # None for all trials

    folder = 'full' if max_trial_num is None else 'fast'
    model, model_key = load_model(model_to_test)
    test_dataset_dict = load_test_dataset_dict()
    loop_mask_segment_conditions(model, model_key, test_dataset_dict)
    # loop_drop_temporal_conditions(model, model_key, test_dataset_dict)








