import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from data.preprocess import increment_path, Normalizer
import os
import torch
import random
from model.utils import linear_resample_data, update_d_dd, fix_seed, nan_helper, from_foot_loc_to_foot_vel, \
    convert_addb_state_to_model_input, align_moving_direction, data_filter, norm_cops
from typing import Any, List
import nimblephysics as nimble
from consts import *
from data.quaternion import rotation_6d_to_matrix, matrix_to_rotation_6d
from data.osim_fk import get_model_offsets, forward_kinematics
from nimblephysics import NimbleGUI
import time
import more_itertools as mit
from scipy.interpolate import interp1d


fix_seed()


class MotionDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            train: bool,
            opt,
            align_moving_direction_flag: bool = True,
            normalizer: Any = None,
            trial_start_num=0,
            max_trial_num=None,
            divide_jittery=True,
            specific_dset=None,
            specific_trial=None,
            include_trials_shorter_than_window_len=False,
    ):
        self.data_path = data_path
        self.trial_start_num = trial_start_num
        self.target_sampling_rate = opt.target_sampling_rate
        self.window_len = opt.window_len
        self.align_moving_direction_flag = align_moving_direction_flag
        self.opt = opt
        self.divide_jittery = divide_jittery
        self.specific_dset = specific_dset
        self.specific_trial = specific_trial
        self.skels = {}

        self.train = train
        self.name = "Train" if self.train else "Test"
        self.include_trials_shorter_than_window_len = include_trials_shorter_than_window_len

        self.dset_set = set()

        print("Loading dataset...")
        self.load_addb(opt, max_trial_num)

        # functions for cleaning up the data
        if train and self.divide_jittery:
            self.trials = self.divide_jittery_trials(self.trials, opt.model_states_column_names)
        self.guess_vel_and_replace_txtytz()

        self.trial_num = len(self.trials)
        if self.trial_num == 0:
            print("No trials loaded")
            return None
        total_hour = sum([trial.length for trial in self.trials]) / self.target_sampling_rate / 60 / 60
        total_clip_num = sum([trial.length for trial in self.trials]) / self.target_sampling_rate / 3
        print(f"In total, {self.trial_num} trials, {total_hour} hours, {total_clip_num} clips not considering overlapping")

        if train:
            data_concat = torch.cat([trial_.converted_pose for trial_ in self.trials], dim=0)
            print("normalizing training data")

            self.normalizer = Normalizer(data_concat, range(data_concat.shape[1]))            # Norm center and force

            # # [debug] check data continuity    # TODO: Might need to check if 1D joints are continuous
            # import matplotlib.pyplot as plt
            # for trial in self.trials:
            #     plt.figure()
            #     # plt.plot(trial.converted_pose[:, 8])
            #     # plt.plot(trial.converted_pose[:, 6])
            #     plt.plot(np.abs(trial.converted_pose[1:, 8] - trial.converted_pose[:-1, 8]))
            #     plt.plot(np.abs(trial.converted_pose[1:, 6] - trial.converted_pose[:-1, 6]))
            # plt.show()
            #
            # sub_and_trial_names, dset_names = self.get_attributes_of_trials()
            #
            # num_of_sample_to_check = 100
            # sampled_data = [self[0] for _ in range(num_of_sample_to_check)]
            #
            # data_concat_check = [item_[0] for item_ in sampled_data]
            # sub_and_trial_names_data_concat = [item_[2] for item_ in sampled_data]
            # data_concat_check = torch.cat(data_concat_check, dim=0)
            #
            # # data_concat_check = self.normalizer.normalize(data_concat_check)
            # data_concat_check = data_concat_check.reshape([num_of_sample_to_check, -1, 30])
            # values = (data_concat_check[:, 1:] - data_concat_check[:, :-1]).abs()
            # print('{:.3f}'.format(values.mean()), end=' +- ')
            # print('{:.3f}'.format(values.std()))
            # print('{:.3f}'.format(values.max()))
            #
            # # print('{:.3f}'.format(values[:, :, 14].max()))
            #
            # values_np = values.cpu().numpy()
            #
            # max_index = np.where(values_np > values_np.max() - 0.1)
            # print(max_index)

        else:
            self.normalizer = normalizer
        for i in range(self.trial_num):
            self.trials[i].converted_pose = self.normalizer.normalize(self.trials[i].converted_pose).clone().detach().float()

        # # [debug]
        # data_concat = torch.cat([trial_.converted_pose for trial_ in self.trials], dim=0)
        # for i_col, col_name in enumerate(opt.model_states_column_names):
        #     print(f'{col_name}, mean: {data_concat[:, i_col].mean()}, std: {data_concat[:, i_col].std()},'
        #           f' max: {data_concat[:, i_col].max()}, min: {data_concat[:, i_col].min()}')

    def guess_vel_and_replace_txtytz(self):
        def get_vel_and_reset(r_foot_vel_buffer_, l_foot_vel_buffer_, vel_from_t_, i_trial):
            r_foot_vel = np.concatenate(r_foot_vel_buffer_, axis=0)
            l_foot_vel = np.concatenate(l_foot_vel_buffer_, axis=0)

            for vel in [r_foot_vel, l_foot_vel]:
                nans, x = nan_helper(vel)
                if not np.sum(nans[:, 0]) == vel.shape[0]:
                    for axis in range(3):
                        vel[nans[:, axis], axis] = np.interp(x(nans[:, axis]), x(~nans[:, axis]), vel[~nans[:, axis], axis])

            if np.min(r_foot_vel[:, 0]) > 0.6:
                print('Warning, found r_foot_vel > 0.6 m/s, not possible unless backward walking on a treadmill')
            if np.min(l_foot_vel[:, 0]) > 0.6:
                print('Warning, found l_foot_vel > 0.6 m/s, not possible unless backward walking on a treadmill')

            r_foot_vel[np.isnan(r_foot_vel)] = l_foot_vel[np.isnan(r_foot_vel)]
            l_foot_vel[np.isnan(l_foot_vel)] = r_foot_vel[np.isnan(l_foot_vel)]
            average_foot_vel = (r_foot_vel + l_foot_vel) / 2

            vel_from_t = np.concatenate(vel_from_t_, axis=0)
            if np.min(vel_from_t[:, 0]) < -0.6:
                print('Warning, found tx_vel < -0.6 m/s, not possible for gait')

            average_foot_vel[np.isnan(average_foot_vel)] = 0
            walking_vel = vel_from_t
            # Treadmill only operates in AP direction after reseting orientation, thus only use foot_vel in ap direction
            walking_vel[:, 0] = walking_vel[:, 0] - average_foot_vel[:, 0]
            # walking_vel = data_filter(walking_vel, 0.5, self.target_sampling_rate).astype(np.float32)
            walking_vel = torch.from_numpy(walking_vel)

            current_index = walking_vel.shape[0]
            current_trial = i_trial
            pelvis_pos_loc = [self.opt.model_states_column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
            while current_index > 0:
                self.trials[current_trial].converted_pose[:, pelvis_pos_loc] = walking_vel[current_index-self.trials[current_trial].length:current_index]
                current_index -= self.trials[current_trial].length
                current_trial -= 1

            assert current_index == 0

            # plt.figure()
            # plt.plot(walking_vel)
            # # plt.figure()
            # # plt.plot(vel_from_t[:, 0])
            # # plt.plot(r_foot_vel[:, 0])
            # # plt.plot(l_foot_vel[:, 0])
            # plt.title(self.trials[i_trial].dset_name + self.trials[i_trial].sub_and_trial_name)
            # plt.show()
            return [], [], []

        for i_trial, trial in enumerate(self.trials):
            if i_trial == 0:
                current_sub_trial_name = trial.sub_and_trial_name.split('_segment_')[0]
                r_foot_vel_buffer, l_foot_vel_buffer, vel_from_t = [], [], []

            if not trial.sub_and_trial_name.split('_segment_')[0] == current_sub_trial_name:
                r_foot_vel_buffer, l_foot_vel_buffer, vel_from_t = get_vel_and_reset(r_foot_vel_buffer, l_foot_vel_buffer, vel_from_t, i_trial-1)
                current_sub_trial_name = trial.sub_and_trial_name.split('_segment_')[0]

            print(trial.sub_and_trial_name)
            r_foot_vel_buffer.append(trial.mtp_r_vel)
            l_foot_vel_buffer.append(trial.mtp_l_vel)
            body_center = trial.converted_pose[:, 0:3]
            body_center = data_filter(body_center, 10, self.target_sampling_rate).astype(np.float32)
            vel_from_t_trial = np.diff(body_center, axis=0) * self.target_sampling_rate
            vel_from_t_trial = np.concatenate([vel_from_t_trial, vel_from_t_trial[-1][None, :]], axis=0)
            vel_from_t.append(vel_from_t_trial)

        if len(self.trials) > 0:
            get_vel_and_reset(r_foot_vel_buffer, l_foot_vel_buffer, vel_from_t, i_trial)

    def __len__(self):
        return self.opt.pseudo_dataset_len

    def __getitem__(self, _):
        i_trial = torch.randint(0, self.trial_num, (1,)).item()        # a random trial regardless of its length
        slice_index = random.sample(self.trials[i_trial].available_win_start, 1)[0]

        converted_pose = self.trials[i_trial].converted_pose[slice_index:slice_index+self.window_len, ...]
        if np.sum(self.trials[i_trial].probably_missing[slice_index:slice_index+self.window_len]):
            print('Error: probably missing')
            print(self.trials[i_trial].dset_name)
            print(self.trials[i_trial].sub_and_trial_name)
            print(self.trials[i_trial].probably_missing[slice_index:slice_index+self.window_len])
            print(slice_index)
            raise ValueError('probably missing')
        return (converted_pose, self.trials[i_trial].model_offsets, i_trial)

    @staticmethod
    def apply_random_moving_direction(converted_pose, converted_column_names, angle=np.random.rand()*2*np.pi):
        pelvis_orientation_col_loc = [converted_column_names.index(col) for col in [f'pelvis_{x}' for x in range(6)]]
        p_pos_col_loc = [converted_column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
        r_grf_col_loc = [converted_column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
        l_grf_col_loc = [converted_column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]

        if len(pelvis_orientation_col_loc) != 6 or len(p_pos_col_loc) != 3 or len(r_grf_col_loc) != 3 or len(l_grf_col_loc) != 3:
            raise ValueError('check column names')

        pelvis_orientation = converted_pose[:, pelvis_orientation_col_loc]
        pelvis_orientation = rotation_6d_to_matrix(pelvis_orientation)
        p_pos = converted_pose[:, p_pos_col_loc]
        r_grf = converted_pose[:, r_grf_col_loc]
        l_grf = converted_pose[:, l_grf_col_loc]

        rot_mat = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]).float()

        pelvis_orientation_rotated = torch.matmul(rot_mat, pelvis_orientation)
        p_pos_rotated = torch.matmul(rot_mat, p_pos.unsqueeze(2)).squeeze(2)
        r_grf_rotated = torch.matmul(rot_mat, r_grf.unsqueeze(2)).squeeze(2)
        l_grf_rotated = torch.matmul(rot_mat, l_grf.unsqueeze(2)).squeeze(2)

        # converted_pose = copy.deepcopy(converted_pose)
        converted_pose[:, pelvis_orientation_col_loc] = matrix_to_rotation_6d(pelvis_orientation_rotated.float())
        converted_pose[:, p_pos_col_loc] = p_pos_rotated.float()
        converted_pose[:, r_grf_col_loc] = r_grf_rotated.float()
        converted_pose[:, l_grf_col_loc] = l_grf_rotated.float()

        return converted_pose

    def get_all_wins_within_gait_cycle(self):
        windows = []
        self.get_all_gait_cycles_and_set_gait_phase_label()
        for i_trial, trial_ in enumerate(self.trials):
            trial_len = trial_.length
            for i in range(0, trial_len - self.window_len + 1, self.window_len):
                gait_phase_label = trial_.trial_gait_phase_label[i:i+self.window_len]
                if (gait_phase_label != NOT_IN_GAIT_PHASE).any():
                    windows.append((trial_.converted_pose[i:i+self.window_len, ...], self.trials[i_trial].model_offsets, i_trial, gait_phase_label))
        return windows

    def get_all_wins_regardless_gait_cycle(self):
        windows = []
        for i_trial, trial_ in enumerate(self.trials):
            trial_len = trial_.length
            for i in range(0, trial_len - self.window_len + 1, self.window_len):
                windows.append((trial_.converted_pose[i:i+self.window_len, ...], self.trials[i_trial].model_offsets, i_trial, self.window_len))
            # The last window is probably overlapping with the previous one.
            windows.append((trial_.converted_pose[trial_len-self.window_len:, ...], self.trials[i_trial].model_offsets, i_trial, trial_len - self.window_len - i))
        return windows

    def get_all_wins_including_shorter_than_window_len(self, col_loc_to_unmask):
        windows = []
        for i_trial, trial_ in enumerate(self.trials):
            trial_len = trial_.length
            i = - self.opt.window_len
            for i in range(0, trial_len - self.opt.window_len + 1, self.opt.window_len):
                mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
                mask[:, col_loc_to_unmask] = 1
                windows.append((trial_.converted_pose[i:i+self.opt.window_len, ...], trial_.model_offsets, i_trial, self.opt.window_len, mask))
            # The last window is incomplete
            window = torch.zeros([self.opt.window_len, trial_.converted_pose.shape[1]])
            window[:trial_len-i-self.opt.window_len, ...] = trial_.converted_pose[i+self.opt.window_len:i+2*self.opt.window_len, ...]
            mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            mask[:trial_len-i-self.opt.window_len, col_loc_to_unmask] = 1      # 0 for masking, 1 for unmasking
            windows.append((window, trial_.model_offsets, i_trial, trial_len - self.opt.window_len - i, mask))
        return windows

    def get_one_win_from_the_end_of_each_trial(self, col_loc_to_unmask):
        windows = []
        for i_trial, trial_ in enumerate(self.trials):
            len_ = min(self.opt.window_len, trial_.length)
            mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            mask[-len_:, col_loc_to_unmask] = 1
            pose = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            pose[-len_:] = trial_.converted_pose[-len_:]
            windows.append((pose, trial_.model_offsets, i_trial, self.opt.window_len, mask))
        return windows

    def get_all_gait_cycles_and_set_gait_phase_label(self):
        cycles_ = []
        for i_trial, trial_ in enumerate(self.trials):
            unnormalized_poses = self.normalizer.unnormalize(trial_.converted_pose.clone().unsqueeze(0))
            r_v_grf = unnormalized_poses[0, :, self.opt.model_states_column_names.index('calcn_r_force_vy')]
            r_v_grf = r_v_grf.cpu().numpy()

            trial_gait_phase_label, stance_start_valid = self.grf_to_trial_gait_phase_label(
                r_v_grf, self.window_len, self.opt.target_sampling_rate)

            for i_start in range(len(stance_start_valid)-1):
                gait_cycle_converted = trial_.converted_pose[stance_start_valid[i_start]:stance_start_valid[i_start+1]]
                cycles_.append(GaitCycles(
                    gait_cycle_converted, (stance_start_valid[i_start], stance_start_valid[i_start+1]),
                    trial_.sub_and_trial_name, trial_.dset_name))
            self.trials[i_trial].trial_gait_phase_label = trial_gait_phase_label
        return cycles_

    @staticmethod
    def grf_to_trial_gait_phase_label(v_grf, window_len, target_sampling_rate):
        stance_vgrf_thd = 1    # 100% of body mass. Needs to be large because some datasets are noisy.
        stance_len_thds = [int(target_sampling_rate * 0.1), int(window_len * 0.8)]      # 0.1 s to 1 s
        cycle_len_thds = [int(target_sampling_rate * 0.2), int(window_len)]      # 0.2 s to 1.5 s

        trial_gait_phase_label = np.full([v_grf.shape[0]], NOT_IN_GAIT_PHASE)       # shape x
        stance_start_valid = []

        stance_flag = np.abs(v_grf) > stance_vgrf_thd
        stance_flag = stance_flag.astype(int)
        start_end_indicator = np.diff(stance_flag)
        stance_start = np.where(start_end_indicator == 1)[0]
        stance_end = np.where(start_end_indicator == -1)[0]
        for i_start in range(1, len(stance_start)-1):
            end_ = stance_end[(stance_start[i_start] < stance_end) & (stance_end < stance_start[i_start+1])]
            # Exclusion criteria
            if len(end_) != 1:
                continue
            if not stance_len_thds[0] < (end_ - stance_start[i_start]) < stance_len_thds[1]:
                continue
            if not cycle_len_thds[0] < (stance_start[i_start + 1] - stance_start[i_start]) < cycle_len_thds[1]:
                continue
            if stance_start[i_start+1] - window_len < 0:
                continue
            trial_gait_phase_label[stance_start[i_start]:stance_start[i_start+1]] = np.linspace(0, 1000, stance_start[i_start+1]-stance_start[i_start])
            stance_start_valid.append(stance_start[i_start])
        return trial_gait_phase_label, stance_start_valid

    def get_attributes_of_trials(self):
        attributes = [trial.get_attributes() for trial in self.trials]
        sub_and_trial_names = [attribute['sub_and_trial_name'] for attribute in attributes]
        dset_names = [attribute['dset_name'] for attribute in attributes]
        return sub_and_trial_names, dset_names

    def load_addb(self, opt, max_trial_num):
        subject_paths = []
        if os.path.isdir(self.data_path):
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(".b3d"):
                        subject_paths.append(file_path)

        self.trials = []
        for i_sub, subject_path in enumerate(subject_paths):
            # Add the skeleton to the list of skeletons

            if self.specific_dset and self.specific_dset not in subject_path:
                continue

            try:
                subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            except RuntimeError:
                print(f'Loading subject {subject_path} failed, skipping')
                continue
            sub_mass = subject.getMassKg()
            height_m = subject.getHeightM()
            contact_bodies = subject.getGroundForceBodies()
            print(f'Loading subject: {subject_path}')

            subject_name = subject_path.split('/')[-1].split('.')[0]
            dset_name = subject_path.split('/')[-3]

            skel = subject.readSkel(0, geometryFolder=os.path.dirname(os.path.realpath(__file__)) + "/../../data/Geometry/")
            self.skels[dset_name+'_'+subject_name] = skel

            model_offsets = get_model_offsets(skel).float()
            if dset_name == '':
                dset_name = subject_name.split('_')[0]

            trial_start_num_ = self.trial_start_num if self.trial_start_num >= 0 else max(0, subject.getNumTrials() + self.trial_start_num)
            for trial_id in tqdm(range(trial_start_num_, subject.getNumTrials())):
                sampling_rate = int(1 / subject.getTrialTimestep(trial_id))
                sub_and_trial_name = subject_name + '__' + subject.getTrialName(trial_id)
                if self.specific_trial and self.specific_trial not in sub_and_trial_name:
                    continue

                trial_length = subject.getTrialLength(trial_id)
                probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                                in subject.getMissingGRF(trial_id)]

                frames: nimble.biomechanics.FrameList = subject.readFrames(trial_id, 0, trial_length,
                                                                           includeSensorData=False,
                                                                           includeProcessingPasses=True)
                try:
                    first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]
                except IndexError:
                    print(f'{subject_name}, {trial_id} has no processing passes, skipping')
                    continue
                poses = [frame.pos for frame in first_passes]
                forces = [frame.groundContactForce for frame in first_passes]
                if len(forces[0]) != 6:
                    continue        # only include data with 2 contact bodies.
                forces = np.array(forces)
                cops = np.array([frame.groundContactCenterOfPressure for frame in first_passes])
                # This is to compensate an AddBiomechanics bug that first GRF is always 0.
                if (forces[0] == 0).all() or (cops[0] == 0).all():
                    forces[0] = forces[1]
                    cops[0] = cops[1]
                if (forces[-1] == 0).all() or (cops[-1] == 0).all():
                    print(f'Compensating an AddB bug, {subject_name}, {trial_id} has 0 GRF at the first frame.', end='')
                    forces[-1] = forces[-2]
                    cops[-1] = cops[-2]

                force_v0, force_v1 = forces[:, :3] / sub_mass, forces[:, 3:] / sub_mass
                states = np.concatenate([np.array(poses), force_v0, cops[:, :3], force_v1, cops[:, 3:]], axis=1)
                if not self.is_lumbar_rotation_reasonable(np.array(states), opt.osim_dof_columns):
                    continue

                grf_flag_counts = list(mit.run_length.encode(probably_missing))

                if len(grf_flag_counts) > 30:
                    print(f'Compensating an AddB bug, {dset_name} - {subject_name} - {trial_id} notMissingGRF flag abnormally'
                          f' flipped {len(grf_flag_counts)} times, thus setting all to True.', end='')
                    probably_missing = [False] * len(probably_missing)

                if self.align_moving_direction_flag:
                    states, rot_mat = align_moving_direction(states, opt.osim_dof_columns)
                else:
                    rot_mat = torch.eye(3).float()
                states = norm_cops(skel, states, opt, sub_mass, height_m)

                if (not self.include_trials_shorter_than_window_len) and (states.shape[0] / sampling_rate * self.target_sampling_rate) < self.window_len + 2:
                    continue
                if sampling_rate != self.target_sampling_rate:
                    states = linear_resample_data(states, sampling_rate, self.target_sampling_rate)
                    probably_missing = linear_resample_data(np.array(probably_missing).astype(float), sampling_rate, self.target_sampling_rate).astype(bool)

                foot_locations, _, _, _ = forward_kinematics(states[:, :-len(KINETICS_ALL)], model_offsets)
                mtp_r_loc, mtp_l_loc = foot_locations[1].squeeze().cpu().numpy(), foot_locations[3].squeeze().cpu().numpy()

                mtp_r_loc = data_filter(mtp_r_loc, 10, self.target_sampling_rate)
                mtp_l_loc = data_filter(mtp_l_loc, 10, self.target_sampling_rate)
                mtp_r_vel = from_foot_loc_to_foot_vel(mtp_r_loc, states[:, -len(KINETICS_ALL):][:, KINETICS_ALL.index('calcn_r_force_vy')], self.target_sampling_rate)
                mtp_l_vel = from_foot_loc_to_foot_vel(mtp_l_loc, states[:, -len(KINETICS_ALL):][:, KINETICS_ALL.index('calcn_l_force_vy')], self.target_sampling_rate)
                mtp_r_vel, mtp_l_vel = mtp_r_vel.astype(np.float32), mtp_l_vel.astype(np.float32)

                states_df = pd.DataFrame(states, columns=opt.osim_dof_columns)
                states_df, mtp_r_vel, mtp_l_vel = self.customized_param_manipulation(states_df, mtp_r_vel, mtp_l_vel)
                states_df, pos_vec = convert_addb_state_to_model_input(states_df, opt.joints_3d, self.target_sampling_rate)

                assert self.opt.model_states_column_names == list(states_df.columns)
                converted_states = torch.tensor(states_df.values).float()

                trial_data = TrialData(converted_states, probably_missing, model_offsets, contact_bodies,
                                       sub_and_trial_name, trial_id, subject.getHeightM(), subject.getMassKg(),
                                       dset_name, rot_mat, pos_vec, self.window_len, mtp_r_vel, mtp_l_vel)
                if self.include_trials_shorter_than_window_len or len(trial_data.available_win_start) > 0:
                    self.trials.append(trial_data)
                self.dset_set.add(dset_name)

                if max_trial_num is not None and len(self.trials) >= max_trial_num:
                    return
            print('Current trial num: {}'.format(len(self.trials)))
        # self.trial_length_probability = torch.tensor([1000 / trial.length for trial in self.trials]).float()

    def customized_param_manipulation(self, states_df, mtp_r_vel, mtp_l_vel):
        """ This is only for guided diffusion"""
        return states_df, mtp_r_vel, mtp_l_vel

    @staticmethod
    def is_lumbar_rotation_reasonable(states, column_names):
        lumbar_rotation_col_loc = column_names.index('lumbar_rotation')
        if np.abs(np.mean(states[:, lumbar_rotation_col_loc])) > np.deg2rad(45):
            return False
        else:
            return True

    def divide_jittery_trials(self, trials, converted_column_names):
        """ If one sample have abnormal acceleration or angular velocity, divide the trial into two from this sample."""
        linear_acc_limit = 320          # 320 m/s^2
        angular_v_limit = np.deg2rad(2000)          # 2000 deg/s
        rot_mat_angular_v_limit = angular_v_limit               # use the same limit because sin(x) ~= x for small x

        pos_col = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        pos_col_loc = [converted_column_names.index(col) for col in pos_col]
        rot_mat_col = [col for col in converted_column_names if '_vel' not in col and
                       ('_0' in col or '_1' in col or '_2' in col or '_3' in col or '_4' in col or '_5' in col)]
        rot_mat_col_loc = [converted_column_names.index(col) for col in rot_mat_col]
        euler_angle_col = [col for col in converted_column_names if '_vel' not in col and
                           ('angle' in col or 'elbow' in col or 'pro_sup' in col or 'wrist' in col)]
        euler_angle_col_loc = [converted_column_names.index(col) for col in euler_angle_col]

        trial_data_list = []
        for current_trial in trials:
            vel, acc = update_d_dd(current_trial.converted_pose, 1 / self.target_sampling_rate)
            pos_break_point = np.where(acc[:, pos_col_loc] > linear_acc_limit)[0]
            euler_angle_break_point = np.where(vel[:, euler_angle_col_loc] > angular_v_limit)[0]
            rot_mat_break_point = np.where(vel[:, rot_mat_col_loc] > rot_mat_angular_v_limit)[0]
            break_point_list = list(set(np.concatenate([pos_break_point, euler_angle_break_point, rot_mat_break_point]).flatten()))
            break_point_list.sort()

            if len(break_point_list) > 0:
                break_point_list = [0] + break_point_list + [current_trial.length]
                for i_clip in range(len(break_point_list) - 1):
                    start_, end_ = break_point_list[i_clip] + 2, break_point_list[i_clip+1] - 2
                    trial_data = TrialData(current_trial.converted_pose[start_:end_],
                                           current_trial.probably_missing[start_:end_],
                                           *current_trial.get_attributes_for_reinitialization(),
                                           current_trial.mtp_r_vel[start_:end_],
                                           current_trial.mtp_l_vel[start_:end_])
                    if len(trial_data.available_win_start) > 0:
                        trial_data_list.append(trial_data)

                print('Divided {} {} into {} trials, {} due to linear acc, {} due to euler angle vel, {} due to rot mat vel'.format(
                    current_trial.dset_name, current_trial.sub_and_trial_name,
                    len(trial_data_list), len(pos_break_point), len(euler_angle_break_point), len(rot_mat_break_point)))
            else:
                trial_data_list.append(current_trial)
        return trial_data_list


class GaitCycles:
    def __init__(self, gait_cycle_converted, index_pair, sub_and_trial_name, dset_name):
        self.sub_and_trial_name = sub_and_trial_name
        self.dset_name = dset_name
        self.gait_cycle = gait_cycle_converted
        # self.gait_cycle_raw = gait_cycle_raw
        self.gait_cycle_len = gait_cycle_converted.shape[0]
        self.index_pair = index_pair
        self.gait_cycle_resampled = torch.from_numpy(self.resample_to_1000_timesteps(gait_cycle_converted))

    @staticmethod
    def resample_to_1000_timesteps(data_raw):
        x, step = np.linspace(0., 1., data_raw.shape[0], retstep=True)
        new_x = np.linspace(0., .99, 1001)
        f = interp1d(x, data_raw, axis=0)
        data_resampled = f(new_x)
        return data_resampled


class TrialData:
    def __init__(self, converted_states, probably_missing, model_offsets, contact_bodies, sub_and_trial_name, trial_id,
                 height_m, weights_kg, dset_name, rot_mat_for_moving_direction_alignment, pos_vec_for_pos_alignment,
                 window_len, mtp_r_vel, mtp_l_vel, trial_gait_phase_label=None):
        self.converted_pose = converted_states
        self.probably_missing = probably_missing
        self.model_offsets = model_offsets
        self.contact_bodies = contact_bodies
        self.sub_and_trial_name = sub_and_trial_name
        self.height_m = height_m
        self.weights_kg = weights_kg
        self.dset_name = dset_name
        self.length = converted_states.shape[0]
        self.mtp_r_vel = mtp_r_vel
        self.mtp_l_vel = mtp_l_vel
        assert self.mtp_r_vel.shape[0] == self.length
        assert self.mtp_l_vel.shape[0] == self.length
        self.trial_gait_phase_label = trial_gait_phase_label
        self.rot_mat_for_moving_direction_alignment = rot_mat_for_moving_direction_alignment
        self.pos_vec_for_pos_alignment = pos_vec_for_pos_alignment
        self.window_len = window_len
        self.available_win_start = self.set_available_win_start(window_len)
        self.trial_id = trial_id

    def set_available_win_start(self, window_len):
        grf_flag_counts = list(mit.run_length.encode(self.probably_missing))
        available_win_start = []
        for idx, (val, count) in enumerate(grf_flag_counts):
            if not val and count >= window_len:
                elems_before_idx = sum((idx[1] for idx in grf_flag_counts[:idx]))
                available_win_start.extend([i for i in range(elems_before_idx, elems_before_idx+count-window_len+1)])
        return available_win_start

    def get_attributes(self):
        return {'sub_and_trial_name': self.sub_and_trial_name, 'dset_name': self.dset_name,
                'height_m': self.height_m, 'weights_kg': self.weights_kg}

    def get_attributes_for_reinitialization(self):
        return [
            self.model_offsets,
            self.contact_bodies,
            self.sub_and_trial_name,
            self.trial_id,
            self.height_m,
            self.weights_kg,
            self.dset_name,
            self.rot_mat_for_moving_direction_alignment,
            self.pos_vec_for_pos_alignment,
            self.window_len
        ]
