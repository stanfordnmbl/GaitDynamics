from alant.args import parse_opt, set_with_arm_opt
import torch
from tqdm import tqdm
import more_itertools as mit
import os
from alant.alan_consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE
from model.alan_model import MotionModel, MotionDataset
from model.alan_model import inverse_convert_addb_state_to_model_input
import numpy as np
from alant.alan_osim_fk import get_model_offsets, get_knee_rotation_coefficients, forward_kinematics
import matplotlib.pyplot as plt
import nimblephysics as nimble
from model.utils import extract, make_beta_schedule, linear_resample_data, update_d_dd, fix_seed, nan_helper, \
    from_foot_loc_to_foot_vel, moving_average_filtering, convert_addb_state_to_model_input, identity, maybe_wrap, \
    inverse_convert_addb_state_to_model_input, align_moving_direction, inverse_align_moving_direction


# class MotionDatasetForCompeletion(MotionDataset):
#     def load_addb(self, opt, max_trial_num):
#         subject_paths = []
#         if os.path.isdir(self.data_path):
#             for root, dirs, files in os.walk(self.data_path):
#                 for file in files:
#                     file_path = os.path.join(root, file)
#                     if file.endswith(".b3d"):
#                         subject_paths.append(file_path)
#
#         # self.trials = []
#         self.trial_protos = {}
#         for i_sub, subject_path in enumerate(subject_paths):
#             # Add the skeleton to the list of skeletons
#
#             if self.specific_dset and self.specific_dset not in subject_path:
#                 continue
#
#             subject = nimble.biomechanics.SubjectOnDisk(subject_path)
#             self.trial_protos[i_sub] = subject.getHeaderProto().getTrials()
#
#             sub_mass = subject.getMassKg()
#             sub_height = subject.getHeightM()
#             contact_bodies = subject.getGroundForceBodies()
#             print(f'Loading subject: {subject_path}')
#
#             skel = subject.readSkel(0, geometryFolder=os.path.dirname(os.path.realpath(__file__)) + "/../../data/Geometry/")
#             model_offsets = get_model_offsets(skel).float()
#             subject_name = subject_path.split('/')[-1].split('.')[0]
#             dset_name = subject_path.split('/')[-3]
#             if dset_name == '':
#                 dset_name = subject_name.split('_')[0]
#
#             trial_start_num_ = self.trial_start_num if self.trial_start_num >= 0 else max(0, subject.getNumTrials() + self.trial_start_num)
#             for trial_index in tqdm(range(trial_start_num_, subject.getNumTrials())):
#                 sampling_rate = int(1 / subject.getTrialTimestep(trial_index))
#                 sub_and_trial_name = subject_name + '_' + subject.getTrialName(trial_index)
#                 trial_length = subject.getTrialLength(trial_index)
#                 probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
#                                                 in subject.getMissingGRF(trial_index)]
#
#                 frames: nimble.biomechanics.FrameList = subject.readFrames(trial_index, 0, trial_length,
#                                                                            includeSensorData=False,
#                                                                            includeProcessingPasses=True)
#                 try:
#                     first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]
#                 except IndexError:
#                     print(f'{subject_name}, {trial_index} has no processing passes, skipping')
#                     continue
#                 poses = [frame.pos for frame in first_passes]
#                 forces = [frame.groundContactForce for frame in first_passes]
#                 if len(forces[0]) != 6:
#                     continue        # only include data with 2 contact bodies.
#
#                 # This is to compensate an AddBiomechanics bug that first GRF is always 0.
#                 if (forces[0] == 0).all():
#                     print(f'Compensating an AddB bug, {subject_name}, {trial_index} has 0 GRF at the first frame.', end='')
#                     forces[0] = forces[1]
#                 forces = np.array(forces)
#
#                 cops = np.array([frame.groundContactCenterOfPressure for frame in first_passes])
#                 foot_bodies = [skel.getBodyNode(name) for name in subject.getGroundForceBodies()]
#                 foot_loc = []
#                 for i_frame in range(len(cops)):
#                     skel.setPositions(poses[i_frame])
#                     foot_loc.append(np.concatenate([foot_body.getWorldTransform().translation() for foot_body in foot_bodies]))
#                 foot_loc = np.array(foot_loc)
#                 assert foot_loc.shape[1] == 6
#
#                 force_v0, force_v1 = forces[:, :3], forces[:, 3:]
#                 force_p0, force_p1 = (cops - foot_loc)[:, :3], (cops - foot_loc)[:, 3:]
#                 force_moment0, force_moment1 = np.cross(force_p0, force_v0), np.cross(force_p1, force_v1)
#
#                 # normal by weight or weight * height
#                 force_v0, force_v1 = force_v0 / sub_mass, force_v1 / sub_mass
#                 force_moment0, force_moment1 = force_moment0 / (sub_mass * sub_height), force_moment1 / (sub_mass * sub_height)
#
#                 # plt.figure()
#                 # plt.plot(cops)
#                 # for i_axis in range(3):
#                 #     # plt.plot(torque_and_forces[:, i_axis], color=f'C{i_axis}')
#                 #     # plt.plot(torque_wrt_body[:, i_axis] * 100, '--', color=f'C{i_axis}')
#                 #     plt.plot(force_moment1[:, i_axis])
#                 #     plt.title(subject_path.split('/')[-2])
#                 # plt.show()
#
#                 if np.max(np.abs([force_moment0, force_moment1])) > 15:
#                     print(f'{subject_name}, {trial_index} has abnormally large moment.')
#                     continue
#
#                 states = np.concatenate([np.array(poses), force_v0, force_moment0, force_v1, force_moment1], axis=1)
#                 if not self.is_lumbar_rotation_reasonable(np.array(states), opt.osim_dof_columns):
#                     continue
#
#                 grf_flag_counts = list(mit.run_length.encode(probably_missing))
#
#                 if len(grf_flag_counts) > 30:
#                     print(f'Compensating an AddB bug, {dset_name} - {subject_name} - {trial_index} notMissingGRF flag abnormally'
#                           f' flipped {len(grf_flag_counts)} times, thus setting all to True.', end='')
#                     probably_missing = [False] * len(probably_missing)
#                     # grf_flag_counts = list(mit.run_length.encode(probably_missing))
#
#                 # max_has_grf_count = -1
#                 # max_has_grf_idx = -1
#                 # for idx, (val, count) in enumerate(grf_flag_counts):
#                 #     if not val and max_has_grf_count < count:
#                 #         max_has_grf_count = count
#                 #         max_has_grf_idx = idx
#                 # elems_before_idx = sum((idx[1] for idx in grf_flag_counts[:max_has_grf_idx]))
#                 # states = states[elems_before_idx:elems_before_idx+max_has_grf_count, :]
#
#                 if self.align_moving_direction_flag:
#                     states, rot_mot = align_moving_direction(states, opt.osim_dof_columns)
#                 else:
#                     rot_mot = torch.eye(3).float()
#
#                 if (states.shape[0] / sampling_rate * self.target_sampling_rate) < self.window_len + 2:
#                     continue
#                 if sampling_rate != self.target_sampling_rate:
#                     states = linear_resample_data(states, sampling_rate, self.target_sampling_rate)
#                     probably_missing = linear_resample_data(np.array(probably_missing).astype(float), sampling_rate, self.target_sampling_rate).astype(bool)
#
#                 foot_locations, _, _ = forward_kinematics(states[:, :-len(KINETICS_ALL)], model_offsets)
#                 mtp_r_loc, mtp_l_loc = foot_locations[1].squeeze().cpu().numpy(), foot_locations[3].squeeze().cpu().numpy()
#                 mtp_r_vel = from_foot_loc_to_foot_vel(mtp_r_loc, states[:, -len(KINETICS_ALL):][:, 1], self.target_sampling_rate)
#                 mtp_l_vel = from_foot_loc_to_foot_vel(mtp_l_loc, states[:, -len(KINETICS_ALL):][:, 4], self.target_sampling_rate)
#
#                 states_df = pd.DataFrame(states, columns=opt.osim_dof_columns)
#                 states_df = convert_addb_state_to_model_input(states_df, opt.joints_3d, self.target_sampling_rate)
#                 assert self.opt.model_states_column_names == list(states_df.columns)
#
#                 converted_states = torch.tensor(states_df.values).float()
#
#                 assert converted_states.shape[0] == len(frames)
#                 trial_data = TrialData(converted_states, probably_missing, model_offsets, contact_bodies,
#                                        sub_and_trial_name, subject.getHeightM(), subject.getMassKg(),
#                                        dset_name, rot_mot, self.window_len, mtp_r_vel, mtp_l_vel)
#                 self.trials.append(trial_data)
#                 self.dset_set.add(dset_name)
#
#                 if max_trial_num is not None and len(self.trials) >= max_trial_num:
#                     return
#             print('Current trial num: {}'.format(len(self.trials)))


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, repr_dim == 56)

    model = MotionModel(opt, repr_dim)
    dset_list = DATASETS_NO_ARM
    results_true, results_pred, sub_heights, sub_weights = {}, {}, {}, {}
    is_output_label_array = torch.zeros([150, 35])

    for dset in dset_list:
        print(dset)
        dataset_to_process = MotionDataset(
            data_path=opt.data_path_train,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            specific_dset=dset,
            max_trial_num=2,
            # trial_start_num=6,
        )

        minimal_has_grf_num = 0.5 * opt.window_len
        check_sample_len = int(opt.window_len / 5)

        subject_paths = []
        if os.path.isdir(dataset_to_process.data_path):
            for root, dirs, files in os.walk(dataset_to_process.data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(".b3d"):
                        subject_paths.append(file_path)

        # trial_protos = {}
        # i_loaded_trial = 0
        subject_loaded = []
        for i_sub, subject_path in enumerate(subject_paths):
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            subject_name = subject_path.split('/')[-1].split('.')[0]
            assert subject_name not in subject_loaded
            subject_loaded.append(subject_name)
            trial_protos = subject.getHeaderProto().getTrials()
            trial_forceplate_dict = {}

            # if dataset_to_process.trials[i_loaded_trial].sub_and_trial_name.split('_')[0] == subject_name:

            windows_to_complete = []
            windows_probably_missing = []
            windows_start_time_sample = []
            windows_trial_id = []
            for trial_loaded in dataset_to_process.trials:
                if not trial_loaded.sub_and_trial_name.split('__')[0] == subject_name:
                    continue

                poses = trial_loaded.converted_pose
                probably_missing = trial_loaded.probably_missing
                for i in range(0, poses.shape[0] - opt.window_len, check_sample_len):

                    # debug
                    print(sum(probably_missing[i:i+opt.window_len]))
                    print(sum(probably_missing[i:i+opt.window_len]) <= minimal_has_grf_num)
                    if sum(probably_missing[i:i+opt.window_len]) and sum(probably_missing[i:i+opt.window_len]) <= minimal_has_grf_num:
                        windows_to_complete.append(trial_loaded.converted_pose[i:i+opt.window_len, ...])
                        windows_probably_missing.append(probably_missing[i:i+opt.window_len])
                        windows_start_time_sample.append(i)
                        windows_trial_id.append(trial_loaded.trial_id)

                trial_forceplate_dict[trial_loaded.trial_id] = trial_protos[trial_loaded.trial_id].getForcePlates()

            if len(windows_to_complete) == 0:
                continue

            # state_pred_list = [[] for _ in range(skel_num-1)]
            for i_win_chunk in range(0, len(windows_to_complete), opt.batch_size_inference):

                state_true = windows_to_complete[i_win_chunk:i_win_chunk+opt.batch_size_inference]
                state_true = torch.stack(state_true)

                # For GRF estimation
                masks = torch.ones_like(state_true)      # 0 for masking, 1 for unmasking

                for i_win in range(i_win_chunk, min(i_win_chunk+opt.batch_size_inference, len(windows_to_complete))):
                    win_grf_mask = masks[i_win, :, grf_and_moment_model_col_loc]
                    win_grf_mask[windows_probably_missing[i_win]] = 0
                    masks[i_win, :, grf_and_moment_model_col_loc] = win_grf_mask

                _, state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
                # for i_skel in range(skel_num-1):
                #     state_pred_list[i_skel] += state_pred_list_batch[i_skel]

                averaged = torch.zeros_like(state_pred_list_batch[0])
                for i_skel in range(skel_num-1):
                    averaged += state_pred_list_batch[i_skel]
                averaged /= (skel_num-1)
                averaged_grf = averaged[:, :, grf_model_col_loc].cpu().numpy()
                averaged_moment = averaged[:, :, grf_and_moment_model_col_loc].cpu().numpy()
                cop_x = averaged_moment[:, :, 2] / averaged_grf[:, :, 1]
                cop_z = - averaged_moment[:, :, 0] / averaged_grf[:, :, 1]
                # averaged_cop = np.concatenate([cop_x, cop_z], axis=1)

                for i_win in range(i_win_chunk, min(i_win_chunk+opt.batch_size_inference, len(windows_to_complete))):
                    trial_id = windows_trial_id[i_win]
                    start_sample = windows_start_time_sample[i_win]
                    for i_sample in range(start_sample, start_sample+opt.window_len):
                        if windows_probably_missing[i_win][i_sample-start_sample]:
                            idx_0 = i_win%opt.batch_size_inference
                            idx_1 = i_sample-start_sample

                            # [debug] from here, wait until no force issue fixed
                            trial_forceplate_dict[trial_id][i_sample].forces = averaged_grf[idx_0, idx_1, :]
                            trial_forceplate_dict[trial_id][i_sample].centersOfPressure = np.array(
                                [cop_x[idx_0, idx_1], trial_forceplate_dict[trial_id][i_sample].centersOfPressure[1], cop_z[idx_0, idx_1]])

            for trial_loaded in dataset_to_process.trials:
                trial_id = trial_loaded.trial_id
                trial_protos[trial_id].setForcePlates(trial_forceplate_dict[trial_id])

                # raw_force_plates = trial_protos[trial].getForcePlates()
                # for f in range(len(raw_force_plates)):
                #     raw_force_plates[f].centersOfPressure = cops[f]
                #     raw_force_plates[f].forces = forces[f]
                # trial_protos[0].setForcePlates()

            subject_path_new = subject_path.replace('train_cleaned', 'train_cleaned_completed')
            nimble.biomechanics.writeB3D(subject_path_new, subject)


if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.guide_x_start_the_beginning_step = -10      # negative value means no guidance

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'3000'}.pt"
    # opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'test_data_loader3'}/weights/train-{'3000'}.pt"

    kinematic_model_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' not in col]
    grf_model_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' in col and 'moment' not in col]
    moment_model_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'moment' in col]
    grf_and_moment_model_col_loc = grf_model_col_loc + moment_model_col_loc
    loop_all(opt)