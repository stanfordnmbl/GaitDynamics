from args import parse_opt, set_with_arm_opt
import torch
import os
import pandas as pd
from consts import OSIM_DOF_ALL, KINETICS_ALL
from model.model import MotionModel
from data.addb_dataset import MotionDataset, TrialData
import numpy as np
from fig_utils import get_scores
import matplotlib.pyplot as plt
import nimblephysics as nimble
from data.osim_fk import get_model_offsets, forward_kinematics
from model.utils import from_foot_loc_to_foot_vel, linear_resample_data
from model.utils import convert_addb_state_to_model_input, inverse_convert_addb_state_to_model_input, align_moving_direction


class DatasetOpenCap(MotionDataset):
    def load_addb(self, opt, max_trial_num):
        subject_paths = []
        if os.path.isdir(self.data_path + 'Kinematics/'):
            for root, dirs, files in os.walk(self.data_path + 'Kinematics/'):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(".mot") and 'walking' in file_path:
                        subject_paths.append(file_path)

        customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
            self.data_path + 'Model/LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjustedMA.osim',
            '/mnt/d/Local/AddBiom/vmu-suit/ml_and_simulation/data/Geometry/')
        skel = customOsim.skeleton
        unscaledSkeletonMass = skel.getMass()

        self.trials = []
        for i_sub, subject_path in enumerate(subject_paths):
            model_offsets = get_model_offsets(skel).float()
            subject_name = subject_path.split('/')[-1].split('.')[0]
            dset_name = subject_path.split('/')[-3]
            if dset_name == '':
                dset_name = subject_name.split('_')[0]

            poses_df = pd.read_csv(subject_path, sep='\t', skiprows=10)
            force_df = pd.read_csv(subject_path.replace('Kinematics', 'Forces').replace('OpenCap', 'Mocap'), sep='\t', skiprows=5)
            force_df = force_df.iloc[::20, :]
            force_df.iloc[:, 1:4] /= unscaledSkeletonMass
            force_df.iloc[:, 10:13] /= unscaledSkeletonMass
            assert force_df.iloc[0, 0] == poses_df.iloc[0, 0]

            col_list = list(poses_df.columns)
            col_loc = [col_list.index(col) for col in OSIM_DOF_ALL[:23]]
            angle_col_loc = [col_list.index(col) for col in OSIM_DOF_ALL[:23] if col not in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]

            poses_df.iloc[:, angle_col_loc] = np.deg2rad(poses_df.iloc[:, angle_col_loc])
            poses = poses_df.values[:, col_loc]
            force_df = force_df[[f'{side}_ground_force_{variable}' for side in ['R', 'L'] for variable in ['vx', 'vy', 'vz', 'px', 'py', 'pz']]]
            if force_df.shape[0] > poses.shape[0]:
                forces = force_df.iloc[:poses.shape[0], :]
            else:
                forces = np.zeros((poses.shape[0], 12))
                forces[:forces.shape[0], :] = force_df.values

            poses, forces = np.array(poses), np.array(forces)
            states = np.concatenate([poses, forces], axis=1)

            if self.align_moving_direction_flag:
                states, rot_mat = align_moving_direction(states, opt.osim_dof_columns)
            else:
                rot_mat = np.eye(3)

            sampling_rate = 100
            if self.target_sampling_rate != sampling_rate:
                states = linear_resample_data(states, sampling_rate, self.target_sampling_rate)

            foot_locations, _, _, _ = forward_kinematics(states[:, :-len(KINETICS_ALL)], model_offsets)
            mtp_r_loc, mtp_l_loc = foot_locations[1].squeeze().cpu().numpy(), foot_locations[3].squeeze().cpu().numpy()
            mtp_r_vel = from_foot_loc_to_foot_vel(mtp_r_loc, states[:, -len(KINETICS_ALL):][:, KINETICS_ALL.index('calcn_r_force_vy')], self.target_sampling_rate)
            mtp_l_vel = from_foot_loc_to_foot_vel(mtp_l_loc, states[:, -len(KINETICS_ALL):][:, KINETICS_ALL.index('calcn_l_force_vy')], self.target_sampling_rate)

            states_df = pd.DataFrame(states, columns=opt.osim_dof_columns)
            states_df, pos_vec = convert_addb_state_to_model_input(states_df, opt.joints_3d, self.target_sampling_rate)
            assert list(states_df.columns) == opt.model_states_column_names

            converted_states = torch.tensor(states_df.values).float()
            probably_missing = [False] * converted_states.shape[0]
            self.trials.append(TrialData(
                converted_states, probably_missing, model_offsets, [], subject_path.split('/')[-1][:-4],
                i_sub, 0., unscaledSkeletonMass, dset_name, rot_mat, pos_vec, poses.shape[0], mtp_r_vel, mtp_l_vel))
            self.dset_set.add(dset_name)

            print('Current trial num: {}'.format(len(self.trials)))


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)

    model = MotionModel(opt, repr_dim)

    """ Use marker-based kinematics """
    test_dataset = MotionDataset(
        data_path='/mnt/d/Local/Data/MotionPriorData/uhlrich_dset/',
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        include_trials_shorter_than_window_len=True,
        specific_trial='walking'
    )
    windows = test_dataset.get_all_wins_including_shorter_than_window_len(opt.kinematic_diffusion_col_loc)
    trials = test_dataset.trials

    """ Use opencap-based kinematics """
    # trials, windows = [], []
    # for sub_num in [2, 3, 4, 6, 7, 8, 9, 11]:
    #     test_dataset = DatasetOpenCap(
    #         data_path=f'/mnt/g/Shared drives/NMBL Shared Data/datasets/Uhlrich2023/Raw/4CVPR/Data/subject{sub_num}/OpenCap/',
    #         train=False,
    #         normalizer=model.normalizer,
    #         opt=opt,
    #         divide_jittery=False,
    #         # max_trial_num=1,
    #     )
    #     windows_sub = test_dataset.get_all_wins_including_shorter_than_window_len(opt.kinematic_diffusion_col_loc)
    #     windows.extend(windows_sub)
    #     trials.extend(test_dataset.trials)

    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows), opt.batch_size_inference):

        state_true = torch.stack([win[0] for win in windows[i_win:i_win+opt.batch_size_inference]])
        masks = torch.stack([win[4] for win in windows[i_win:i_win+opt.batch_size_inference]])

        state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
        pos_vec = np.array([trials[windows[i_win_vec][2]].pos_vec_for_pos_alignment
                            for i_win_vec in range(len(windows[i_win:i_win+opt.batch_size_inference]))])
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, pos_vec)
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

    results_true, results_pred, scores = {}, {}, {}
    for win, state_pred_mean, state_pred_std in zip(windows, state_pred_list_averaged, state_pred_list_std):
        trial = trials[win[2]]
        if trial.sub_and_trial_name not in results_true.keys():
            results_true.update({trial.sub_and_trial_name: []})
            results_pred.update({trial.sub_and_trial_name: []})

        true_val = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win[0].unsqueeze(0)), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, trial.pos_vec_for_pos_alignment).squeeze().numpy()
        mask = win[4].squeeze().numpy()
        true_val = true_val * mask[:, 0].repeat(35).reshape((150, -1))
        state_pred_mean = state_pred_mean * mask[:, 0].repeat(35).reshape((150, -1))
        results_true[trial.sub_and_trial_name].append(true_val)
        results_pred[trial.sub_and_trial_name].append(state_pred_mean)

    params_of_interest = ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']
    params_of_interest_col_loc = [opt.osim_dof_columns.index(col) for col in params_of_interest]
    true_all, pred_all = [], []
    for sub_and_trial in results_true.keys():
        results_true[sub_and_trial] = np.concatenate(results_true[sub_and_trial], axis=0)
        results_pred[sub_and_trial] = np.concatenate(results_pred[sub_and_trial], axis=0)

        stance_phase = np.abs(results_true[sub_and_trial][:, params_of_interest_col_loc[0]]) > 1e-5
        true_all.append(results_true[sub_and_trial][stance_phase])
        pred_all.append(results_pred[sub_and_trial][stance_phase])

    true_all = np.concatenate(true_all, axis=0)
    pred_all = np.concatenate(pred_all, axis=0)
    for i_param, param in enumerate(params_of_interest):
        idx = opt.osim_dof_columns.index(param)
        scores = get_scores(true_all[:, idx:idx+1], pred_all[:, idx:idx+1], [param], None)
        print(f'{param}: {scores[0]["mae"]:.2f}')
        plt.figure()
        plt.plot(true_all[:, idx:idx+1], label='True')
        plt.plot(pred_all[:, idx:idx+1], label='Pred')
    plt.show()


if __name__ == "__main__":
    skel_num = 3
    opt = parse_opt()
    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'5328'}.pt"
    loop_all(opt)















