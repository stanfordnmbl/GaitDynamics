import os
from args import parse_opt, set_with_arm_opt
import torch
import pandas as pd
from model.model import MotionModel
from data.addb_dataset import MotionDataset, TrialData
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import collections
from model.utils import inverse_convert_addb_state_to_model_input, inverse_norm_cops
from fig_utils import show_skeletons, set_up_gui
import pickle


def get_sub_meta(sub, meta_table):
    sub_meta = meta_table[meta_table['ID'] == sub]
    assert len(sub_meta) == 1
    sub_meta_dict = sub_meta.iloc[0].to_dict()
    if np.isnan(sub_meta_dict['FAC']):
        sub_meta_dict['FAC'] = 0
    return sub_meta_dict
    # try:
    #     assert len(sub_meta) == 1
    #     return sub_meta.iloc[0].to_dict()
    # except AssertionError:
    #     return {'FAC': 6}


def metric_smoothness(pred_array, diff_array, param_col_loc):
    mean_smoothness = []
    for i_T in range(pred_array.shape[0]):
        pred_ = pred_array[i_T, :, np.array(param_col_loc)].swapaxes(0, 1)
        pred_array_normed = (pred_ - pred_.mean(axis=0)) / pred_.std(axis=0)
        a = pred_array_normed[:-2]
        b = pred_array_normed[1:-1]
        c = pred_array_normed[2:]
        smoothness = np.abs(a - 2*b + c)
        mean_smoothness.append(np.array([np.median(smoothness, axis=0)]))
    return np.mean(mean_smoothness)


def metric_mae(pred_array, diff_array, param_col_loc):
    mae_list = []
    for i_T in range(diff_array.shape[0]):
        diff_ = diff_array[i_T, :, np.array(param_col_loc)].swapaxes(0, 1)
        mae = np.abs(diff_).mean(axis=0)
        mae_list.append(mae)
    if save_z_score:
        return mae_list
    else:
        mae_array = (np.array(mae_list) - mean_) / std_
        return mae_array.mean()


def show_pca(data_transformed, trial_lens, fac_of_trials):
    i_current = 0
    plt.figure()
    for i_trial in range(len(trial_lens)):
        data_transformed_trial = data_transformed[i_current:i_current + trial_lens[i_trial]]
        i_current += trial_lens[i_trial]
        trial_pc_average = np.mean(data_transformed_trial, axis=0)
        plt.scatter(trial_pc_average[0], trial_pc_average[1], c=colors[fac_of_trials[i_trial]],
                    label=fac_of_trials[i_trial])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = collections.OrderedDict(sorted(dict(zip(labels, handles)).items()))
    plt.legend(by_label.values(), by_label.keys())


def loop_all():
    meta_table = pd.read_csv('/mnt/g/Shared drives/NMBL Shared Data/datasets/VanCriekinge2023/meta_data.csv')
    meta_table.columns = [col.split('\n')[0] for col in meta_table.columns]
    data_path = '/mnt/d/Local/Data/MotionPriorData/vancriek_dset/'
    stroke_subjects = [data_path + 'stroke/' + sub_name + '/' for sub_name in os.listdir(data_path + 'stroke')]
    healhty_subjects = [data_path + 'healthy/' + sub_name + '/' for sub_name in os.listdir(data_path + 'healthy')]
    all_subjects = stroke_subjects[:10] + healhty_subjects[:10]  # !!!

    fac_of_subs, ood_metric, baseline_metric = [], [], []
    save_data_for_z_score = []
    for subject_path in all_subjects:
        test_dataset = MotionDataset(
            data_path=subject_path,
            train=False,
            normalizer=model.normalizer,
            include_trials_shorter_than_window_len=True,  # Otherwise kinematics got removed
            opt=opt,
            divide_jittery=False,
            trial_start_num=-2,
        )
        test_dataset.trials = [trial for trial in test_dataset.trials if
                               trial.sub_and_trial_name not in bad_sub_and_trial_names]
        test_dataset.trials = [trial for trial in test_dataset.trials if        # (0) is a standing trial
                               '(0)' not in trial.sub_and_trial_name]
        if len(test_dataset.trials) == 0:
            continue
        fac_of_subs.append(
            int(get_sub_meta(test_dataset.trials[0].sub_and_trial_name.split('__')[0], meta_table)['FAC']))
        windows = test_dataset.get_all_wins_including_shorter_than_window_len(opt.kinematic_diffusion_col_loc)
        windows = windows[:-1]  # remove the last incomplete window

        mean_speed = np.mean([model.normalizer.unnormalize(trial_.converted_pose.unsqueeze(0))[:, :,
                              opt.model_states_column_names.index('pelvis_tx')].mean() for trial_ in
                              test_dataset.trials])
        baseline_metric.append(mean_speed * test_dataset.trials[0].height_m)

        pred_list_sub, diff_list_sub = [], []
        for i_win in range(0, len(windows), opt.batch_size_inference):
            state_true = torch.stack([win.pose for win in windows[i_win:i_win + opt.batch_size_inference]])
            masks = torch.stack([win.mask for win in windows[i_win:i_win + opt.batch_size_inference]])
            cond = torch.stack([win.cond for win in windows[i_win:i_win + opt.batch_size_inference]])

            constraint = {'mask': masks, 'value': state_true.clone(), 'cond': cond}
            # fill missing GRF
            state_true_filled = model.diffusion.inpaint_ddim_loop(
                shape=(state_true.shape[0], model.horizon, model.repr_dim),
                constraint=constraint).cpu()
            state_true_filled = state_true*masks + state_true_filled*(1-masks)

            constraint = {'value': state_true_filled.clone(), 'cond': cond, 'total_timesteps': 500}
            state_pred_list = model.diffusion.noise_denoise_at_each_t_ddpm(
                shape=(state_true_filled.shape[0], model.horizon, model.repr_dim),
                constraint=constraint)

            pred_list_sub.append([item.cpu().numpy() for item in state_pred_list])
            diff = [((state_true - state_pred.cpu()) * masks).numpy() for state_pred in state_pred_list]
            diff_list_sub.append(diff)

            # # [DEBUG]
            # state_pred_to_show = [model.normalizer.unnormalize(state_pred).cpu() for state_pred in state_pred_list]
            # for j_win in range(i_win, i_win+state_true.shape[0]):
            #     height_m_tensor = torch.tensor([win.height_m for win in windows[j_win:j_win+1]])
            #     name_states_dict = {}
            #     for i_generation, states in enumerate([model.normalizer.unnormalize(state_true)] + state_pred_to_show[::]):
            #         states = inverse_convert_addb_state_to_model_input(
            #             states, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)
            #         skel_ = list(test_dataset.skels.values())[0]
            #         win_osim = inverse_norm_cops(skel_, states[j_win], opt, windows[j_win].weight_kg, windows[j_win].height_m).detach().numpy()
            #         if i_generation == 0:
            #             name = 'True'
            #         else:
            #             name = f'Generation {i_generation}'
            #         name_states_dict.update({name: win_osim})
            #     for _ in range(1):
            #         show_skeletons(opt, name_states_dict, gui, skel_)

        pred_array = np.array(pred_list_sub).swapaxes(0, 1)
        pred_array = pred_array.reshape(pred_array.shape[0], -1, pred_array.shape[-1])
        diff_array = np.array(diff_list_sub).swapaxes(0, 1)
        diff_array = diff_array.reshape(diff_array.shape[0], -1, diff_array.shape[-1])

        knee_ankle_col_loc = [opt.model_states_column_names.index(col) for col in ['knee_angle_r', 'ankle_angle_r', 'knee_angle_l', 'ankle_angle_l']]
        non_vel_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('vel' not in col) and ('force' not in col)]
        vel_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('vel' in col) and ('force' not in col)]

        if save_z_score:
            save_data_for_z_score.append(np.array(metric_mae(pred_array, diff_array, knee_ankle_col_loc)))

        ood_metric.append(np.array([metric_mae(pred_array, diff_array, knee_ankle_col_loc),
                                    metric_smoothness(pred_array, diff_array, knee_ankle_col_loc),
                                    ]))

    if save_z_score:
        save_data_for_z_score = np.array(save_data_for_z_score)
        mean_, std_ = save_data_for_z_score.mean(axis=0), save_data_for_z_score.std(axis=0)
        pickle.dump([mean_, std_], open("results/for_z_score.pkl", "wb"))

    plt.figure()
    for i_sub in range(len(fac_of_subs)):
        plt.scatter(baseline_metric[i_sub], ood_metric[i_sub][0], c=colors[fac_of_subs[i_sub]], label=fac_of_subs[i_sub])
    plt.xlabel('Baseline speed')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = collections.OrderedDict(sorted(dict(zip(labels, handles)).items()))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylabel('OOD score')

    plt.figure()
    for i_sub in range(len(fac_of_subs)):
        plt.scatter(ood_metric[i_sub][0], ood_metric[i_sub][1], c=colors[fac_of_subs[i_sub]], label=fac_of_subs[i_sub])
    plt.show()


""" noise-denoise OOD, following paper
"Denoising Diffusion Models for Out-of-Distribution Detection"
"""
bad_sub_and_trial_names = ['TVC23__BWA5_segment_0']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
save_z_score = False
if not save_z_score:
    mean_, std_ = pickle.load(open("results/for_z_score.pkl", "rb"))
# gui = set_up_gui()

if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'6993_gait_only'}.pt"
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)
    model = MotionModel(opt, repr_dim)
    loop_all()
