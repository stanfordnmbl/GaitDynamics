from args import parse_opt, set_with_arm_opt
import torch
import os
from consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE
from figures.fig_utils import vancriek_bad_sub_and_trial_names
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from model.model import inverse_convert_addb_state_to_model_input
import numpy as np
import pickle


def loop_all(opt):
    set_with_arm_opt(opt, False)
    model = MotionModel(opt)
    results_true, results_pred, results_pred_std, results_bl, height_m_all, weight_kg_all = {}, {}, {}, {}, {}, {}

    stroke_subjects = [data_path + 'stroke/' + sub_name + '/' for sub_name in os.listdir(data_path + 'stroke')]
    healhty_subjects = [data_path + 'healthy/' + sub_name + '/' for sub_name in os.listdir(data_path + 'healthy')]
    all_subjects = stroke_subjects[:] + healhty_subjects[:]

    for subject_path in all_subjects:
        test_dataset = MotionDataset(
            data_path=subject_path,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            include_trials_shorter_than_window_len=True
        )
        test_dataset.trials = [trial for trial in test_dataset.trials if
                               trial.sub_and_trial_name not in vancriek_bad_sub_and_trial_names]
        test_dataset.trials = [trial for trial in test_dataset.trials if        # (0) is a standing trial
                               '(0)' not in trial.sub_and_trial_name]
        windows = test_dataset.get_all_wins([0], including_shorter_than_window_len=False)

        if len(windows) == 0:
            continue

        height_m_all = {trial_.sub_and_trial_name: trial_.height_m for trial_ in test_dataset.trials}
        weight_kg_all = {trial_.sub_and_trial_name: trial_.height_m for trial_ in test_dataset.trials}

        state_pred_list = [[] for _ in range(skel_num-1)]
        for i_win in range(0, len(windows), opt.batch_size_inference):

            state_true = [win.pose for win in windows[i_win:i_win+opt.batch_size_inference]]
            state_true = torch.stack(state_true)

            # For reconstruct kinematics
            masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
            masks[:, :, cols_to_mask[mask_key]] = 1
            save_name = f'downstream_reconstruct_kinematics_{mask_key}'

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
            if trial.sub_and_trial_name not in results_true.keys():
                results_true.update({trial.sub_and_trial_name: []})
                results_pred.update({trial.sub_and_trial_name: []})
                results_pred_std.update({trial.sub_and_trial_name: []})
            true_val = inverse_convert_addb_state_to_model_input(
                model.normalizer.unnormalize(win.pose.unsqueeze(0)), opt.model_states_column_names,
                opt.joints_3d, opt.osim_dof_columns, trial.pos_vec_for_pos_alignment, torch.tensor(win.height_m)).squeeze().numpy()
            results_true[trial.sub_and_trial_name].append(true_val)
            results_pred[trial.sub_and_trial_name].append(state_pred_mean)
            results_pred_std[trial.sub_and_trial_name].append(state_pred_std)

    for sub_and_trial in results_true.keys():
        results_true[sub_and_trial] = np.concatenate(results_true[sub_and_trial], axis=0)
        results_pred[sub_and_trial] = np.concatenate(results_pred[sub_and_trial], axis=0)
        results_pred_std[sub_and_trial] = np.concatenate(results_pred_std[sub_and_trial], axis=0)

        # plt.figure()
        # col_loc = opt.osim_dof_columns.index('knee_angle_r')
        # plt.plot(results_true[sub_and_trial][:, col_loc])
        # plt.plot(results_pred[sub_and_trial][:, col_loc])
        # plt.fill_between(range(len(results_true[sub_and_trial])), results_pred[sub_and_trial][:, col_loc] - results_pred_std[sub_and_trial][:, col_loc],
        #                  results_pred[sub_and_trial][:, col_loc] + results_pred_std[sub_and_trial][:, col_loc], color='C1', alpha=0.5)

    pickle.dump([results_true, results_pred, results_pred_std, None, opt.osim_dof_columns,
                 None, height_m_all, weight_kg_all],
                open(f"results/{save_name}.pkl", "wb"))


if __name__ == "__main__":
    skel_num = 4
    opt = parse_opt()

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'6993'}.pt"
    # opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'fixed_txtytz_vel'}/weights/train-{'6993'}.pt"

    data_path = '/mnt/d/Local/Data/MotionPriorData/vancriek_dset/'

    cols_to_mask = {
        # 'ankle': opt.knee_diffusion_col_loc,
        # 'knee': opt.knee_diffusion_col_loc,
        # 'hip': opt.hip_diffusion_col_loc,
        # 'knee_ankle': opt.knee_diffusion_col_loc + opt.ankle_diffusion_col_loc,
        # 'knee_ankle_hip': opt.knee_diffusion_col_loc + opt.ankle_diffusion_col_loc + opt.hip_diffusion_col_loc
        'pelvis_hip': opt.hip_diffusion_col_loc + [i_item for i_item, item in enumerate(opt.model_states_column_names) if 'pelvis' in item and '_t' not in item],
    }

    for mask_key in cols_to_mask.keys():
        print('mask_key: ', mask_key)
        loop_all(opt)










