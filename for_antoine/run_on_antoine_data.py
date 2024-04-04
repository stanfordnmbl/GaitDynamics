from alant.args import parse_opt, set_with_arm_opt
import torch
import os
from alant.alan_consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE, KINETICS_ALL, OSIM_DOF_ALL
from model.alan_model import MotionModel, MotionDataset, TrialData, convert_addb_state_to_model_input
from model.utils import linear_resample_data_as_num_of_dp, from_foot_loc_to_foot_vel, linear_resample_data
from alant.alan_osim_fk import get_model_offsets, forward_kinematics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nimblephysics as nimble


class DatasetAntoine(MotionDataset):
    def load_addb(self, opt, max_trial_num):
        subject_paths = []
        if os.path.isdir(self.data_path):
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(".mot"):
                        subject_paths.append(file_path)

        customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
            self.data_path + 'LaiUhlrich2022_scaled_adjusted.osim')
        skel = customOsim.skeleton
        unscaledSkeletonMass = skel.getMass()

        self.trials = []
        for i_sub, subject_path in enumerate(subject_paths):

            # skel = subject.readSkel(0, geometryFolder=os.path.dirname(os.path.realpath(__file__)) + "/../../data/Geometry/")
            model_offsets = get_model_offsets(skel).float()
            subject_name = subject_path.split('/')[-1].split('.')[0]
            dset_name = subject_path.split('/')[-3]
            if dset_name == '':
                dset_name = subject_name.split('_')[0]

            poses_df = pd.read_csv(subject_path, sep='\t', skiprows=10)
            col_list = list(poses_df.columns)
            col_loc = [col_list.index(col) for col in OSIM_DOF_ALL[:23]]
            angle_col_loc = [col_list.index(col) for col in OSIM_DOF_ALL[:23] if col not in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]

            poses_df.iloc[:, angle_col_loc] = np.deg2rad(poses_df.iloc[:, angle_col_loc])
            poses = poses_df.values[:, col_loc]
            forces = np.zeros((poses.shape[0], 6))

            poses, forces = np.array(poses), np.array(forces)
            states = np.concatenate([poses, forces], axis=1)

            if self.align_moving_direction_flag:
                states, rot_mat = self.align_moving_direction(states, opt.osim_dof_columns)

            sampling_rate = 100
            if self.target_sampling_rate != sampling_rate:
                states = linear_resample_data(states, sampling_rate, self.target_sampling_rate)

            foot_locations, _, _ = forward_kinematics(states[:, :-len(KINETICS_ALL)], model_offsets)
            mtp_r_loc, mtp_l_loc = foot_locations[1].squeeze().cpu().numpy(), foot_locations[3].squeeze().cpu().numpy()
            mtp_r_vel = from_foot_loc_to_foot_vel(mtp_r_loc, states[:, -len(KINETICS_ALL):][:, 1], self.target_sampling_rate)
            mtp_l_vel = from_foot_loc_to_foot_vel(mtp_l_loc, states[:, -len(KINETICS_ALL):][:, 4], self.target_sampling_rate)

            states_df = pd.DataFrame(states, columns=opt.osim_dof_columns)
            states_df = convert_addb_state_to_model_input(states_df, opt.joints_3d)
            self.converted_column_names = list(states_df.columns)

            converted_states = torch.tensor(states_df.values).float()

            self.trials.append(TrialData(converted_states, model_offsets, [], subject_path.split('/')[-1][:-4],
                                         poses.shape[0], unscaledSkeletonMass, dset_name, rot_mat, mtp_r_vel, mtp_l_vel))
            self.dset_set.add(dset_name)

            print('Current trial num: {}'.format(len(self.trials)))


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, repr_dim == 56)

    model = MotionModel(opt, repr_dim)
    results_true, results_pred, results_bl, sub_heights, sub_weights = {}, {}, {}, {}, {}
    is_output_label_array = torch.zeros([150, 29])

    test_dataset = DatasetAntoine(
        data_path=os.path.dirname(os.path.realpath(__file__)) + '/subject2/',
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        # max_trial_num=2,     # !!!
        # trial_start_num=5,   # !!!
    )
    windows = test_dataset.get_all_wins_regardless_gait_cycle()

    results_true.update({test_dataset.trials[0].dset_name: {}})
    results_pred.update({test_dataset.trials[0].dset_name: {}})

    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows), opt.batch_size_inference):

        state_true = [win[0] for win in windows[i_win:i_win+opt.batch_size_inference]]
        state_true = torch.stack(state_true)

        # For GRF estimation
        masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
        masks[:, :, kinematic_diffusion_col_loc] = 1
        is_output_label_array[:, grf_osim_col_loc] = 1
        _, state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
        for i_skel in range(skel_num-1):
            state_pred_list[i_skel] += state_pred_list_batch[i_skel]
        # save_name = 'downstream_grf'

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
        # true_val = inverse_convert_addb_state_to_model_input(
        #     model.normalizer.unnormalize(win[0].unsqueeze(0)),
        #     model_states_column_names=opt.model_states_column_names,
        #     joints_3d=opt.joints_3d, osim_dof_columns=opt.osim_dof_columns).squeeze().numpy()
        # results_true[trial.dset_name][trial.sub_and_trial_name].append(true_val)
        results_pred[trial.dset_name][trial.sub_and_trial_name].append(state_pred[-win[3]:])

    # plt.figure()
    # plt.plot(results_pred[:, 0])
    # plt.show()

    for dset in results_true.keys():
        for i_trial, sub_and_trial in enumerate(results_true[dset].keys()):
            # results_true[dset][sub_and_trial] = np.concatenate(results_true[dset][sub_and_trial], axis=0)
            results_pred[dset][sub_and_trial] = np.concatenate(results_pred[dset][sub_and_trial], axis=0)
            grf_results = torch.from_numpy(results_pred[dset][sub_and_trial][:, -6:] * test_dataset.trials[i_trial].sub_weight)

            # plt.figure()
            # plt.plot(grf_results)
            # plt.title(sub_and_trial)

            # grf_results[:, :3] = test_dataset.trials[i_trial].rot_mat_for_moving_direction_alignment
            mat_ = test_dataset.trials[i_trial].rot_mat_for_moving_direction_alignment.T
            for i in [0, 3]:
                grf_results[:, i:i+3] = torch.matmul(mat_, grf_results[:, i:i+3].unsqueeze(2)).squeeze(2)
            # plt.figure()
            # plt.plot(grf_results)
            # plt.title(sub_and_trial)

            trial_original_len = test_dataset.trials[i_trial].sub_height
            grf_results_resampled = linear_resample_data_as_num_of_dp(grf_results.numpy(), trial_original_len)

            # plt.figure()
            # plt.plot(grf_results_resampled)
            # plt.title(sub_and_trial)

            grf_df = pd.DataFrame(grf_results_resampled, columns=KINETICS_ALL)
            grf_df.to_csv(f"subject2/grf_preds/{sub_and_trial}.csv", index=False)
        plt.show()


if __name__ == "__main__":
    skel_num = 4
    opt = parse_opt()
    opt.guide_x_start_the_beginning_step = -10      # negative value means no guidance

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'5000'}.pt"
    # opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'vel_as_first_three2'}/weights/train-{'5000'}.pt"

    kinematic_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' not in col]
    grf_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if 'force' in col]

    loop_all(opt)















