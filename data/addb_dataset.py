import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from data.preprocess import Normalizer
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
            dset_keyworks_to_exclude=(),
            include_trials_shorter_than_window_len=False,
            restrict_contact_bodies=True,
            use_camargo_lumbar_reconstructed=False,
            check_cop_to_calcn_distance=True,
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
        self.dset_keyworks_to_exclude = dset_keyworks_to_exclude
        self.restrict_contact_bodies = restrict_contact_bodies
        self.use_camargo_lumbar_reconstructed = use_camargo_lumbar_reconstructed
        self.check_cop_to_calcn_distance = check_cop_to_calcn_distance
        self.skels = {}
        self.num_of_excluded_trials = {'contact_body_num': 0, 'trial_length': 0, 'lumbar_rotation': 0, 'wrong_cop': 0,
                                       'large_moving_direction_change': 0, 'jittery_sample': 0}

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

        if len(self.trials) == 0:
            print("No trials loaded")
            return None
        self.dset_and_trials = {}
        for i_trial, trial in enumerate(self.trials):
            if trial.dset_name not in self.dset_and_trials.keys():
                self.dset_and_trials[trial.dset_name] = [i_trial]
            else:
                self.dset_and_trials[trial.dset_name].append(i_trial)
        self.dset_num = len(self.dset_and_trials.keys())
        for dset_name in self.dset_and_trials.keys():
            print(f"{dset_name}: {len(self.dset_and_trials[dset_name])} trials")
        total_hour = sum([trial.length for trial in self.trials]) / self.target_sampling_rate / 60 / 60
        total_clip_num = sum([trial.length for trial in self.trials]) / self.target_sampling_rate / 3
        print(f"In total, {len(self.trials)} trials, {total_hour} hours, {total_clip_num} clips not considering overlapping")
        print(f"Removed trials: {self.num_of_excluded_trials}")

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
        for i in range(len(self.trials)):
            self.trials[i].converted_pose = self.normalizer.normalize(self.trials[i].converted_pose).clone().detach().float()

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
            walking_vel = torch.from_numpy(walking_vel)

            current_index = walking_vel.shape[0]
            current_trial = i_trial
            pelvis_pos_loc = [self.opt.model_states_column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
            while current_index > 0:
                self.trials[current_trial].converted_pose[:, pelvis_pos_loc] = walking_vel[current_index - self.trials[
                    current_trial].length:current_index] / self.trials[current_trial].height_m
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
        dset_name = random.sample(self.dset_and_trials.keys(), 1)[0]
        i_trial = random.sample(self.dset_and_trials[dset_name], 1)[0]
        slice_index = random.sample(self.trials[i_trial].available_win_start, 1)[0]
        converted_pose = self.trials[i_trial].converted_pose[slice_index:slice_index+self.window_len, ...]
        return (converted_pose, self.trials[i_trial].model_offsets, i_trial, slice_index,
                self.trials[i_trial].height_m, self.trials[i_trial].cond)

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

        converted_pose[:, pelvis_orientation_col_loc] = matrix_to_rotation_6d(pelvis_orientation_rotated.float())
        converted_pose[:, p_pos_col_loc] = p_pos_rotated.float()
        converted_pose[:, r_grf_col_loc] = r_grf_rotated.float()
        converted_pose[:, l_grf_col_loc] = l_grf_rotated.float()

        return converted_pose

    def get_all_wins_within_gait_cycle(self, col_loc_to_unmask):
        windows = []
        self.get_all_gait_cycles_and_set_gait_phase_label()
        for i_trial, trial_ in enumerate(self.trials):
            trial_len = trial_.length
            for i in range(0, trial_len - self.window_len + 1, self.window_len):
                gait_phase_label = trial_.trial_gait_phase_label[i:i+self.window_len]
                if (gait_phase_label != NOT_IN_GAIT_PHASE).any():
                    mask = torch.zeros([self.window_len, len(self.opt.model_states_column_names)])
                    mask[:, col_loc_to_unmask] = 1
                    windows.append(WindowData(trial_.converted_pose[i:i+self.window_len, ...], self.trials[i_trial].model_offsets,
                                              i_trial, gait_phase_label, mask, trial_.height_m, trial_.weight_kg, trial_.cond))
        return windows

    def get_all_wins(self, col_loc_to_unmask, including_shorter_than_window_len=True):
        windows = []
        for i_trial, trial_ in enumerate(self.trials):
            trial_len = trial_.converted_pose.shape[0]
            i = - self.opt.window_len
            for i in range(0, trial_len - self.opt.window_len + 1, self.opt.window_len):
                mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
                mask[:, col_loc_to_unmask] = 1
                windows.append(WindowData(trial_.converted_pose[i:i+self.opt.window_len, ...], trial_.model_offsets, i_trial,
                                          None, mask, trial_.height_m, trial_.weight_kg, trial_.cond))
            if including_shorter_than_window_len:
                # The last pose is incomplete
                pose = torch.zeros([self.opt.window_len, trial_.converted_pose.shape[1]])
                mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])

                pose[:trial_len-i-self.opt.window_len, ...] = trial_.converted_pose[i+self.opt.window_len:i+2*self.opt.window_len, ...]
                mask[:trial_len-i-self.opt.window_len, col_loc_to_unmask] = 1      # 0 for masking, 1 for unmasking
                windows.append(WindowData(pose, trial_.model_offsets, i_trial, None, mask, trial_.height_m, trial_.weight_kg, trial_.cond))
        return windows

    def get_overlapping_wins(self, col_loc_to_unmask, step_len, start_trial=0, end_trial=None, including_shorter_than_window_len=False):
        if end_trial is None:
            end_trial = len(self.trials)
        windows, s_list, e_list = [], [], []
        for i_trial in range(start_trial, end_trial):
            trial_ = self.trials[i_trial]
            trial_len = trial_.converted_pose.shape[0]
            if including_shorter_than_window_len:
                e_of_trial = trial_len
            else:
                e_of_trial = trial_len - self.opt.window_len + step_len
            for i in range(0, e_of_trial, step_len):
                s = max(0, i)
                e = min(trial_len, i+self.opt.window_len)
                s_list.append(s)
                e_list.append(e)
                mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
                mask[:, col_loc_to_unmask] = 1
                mask[e-s:, :] = 0
                data_ = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
                data_[:e-s] = trial_.converted_pose[s:e, ...]
                windows.append(WindowData(data_, trial_.model_offsets, i_trial,
                                          None, mask, trial_.height_m, trial_.weight_kg, trial_.cond))
        return windows, s_list, e_list

    def get_one_win_from_the_end_of_each_trial(self, col_loc_to_unmask):
        windows = []
        for i_trial, trial_ in enumerate(self.trials):
            len_ = min(self.opt.window_len, trial_.length)
            mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            mask[-len_:, col_loc_to_unmask] = 1
            pose = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            pose[-len_:] = trial_.converted_pose[-len_:]
            windows.append(WindowData(pose, trial_.model_offsets, i_trial, None, mask, trial_.height_m,
                                      trial_.weight_kg, trial_.cond))
        return windows

    def get_one_win_from_the_end_of_each_trial_with_offset(self, col_loc_to_unmask, end_offset):
        windows = []
        for i_trial, trial_ in enumerate(self.trials):
            len_ = min(self.opt.window_len, trial_.length-end_offset)
            mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            mask[-len_:, col_loc_to_unmask] = 1
            pose = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            pose[-len_:] = trial_.converted_pose[-len_-end_offset:-end_offset]
            windows.append(WindowData(pose, trial_.model_offsets, i_trial, None, mask, trial_.height_m,
                                      trial_.weight_kg, trial_.cond))
        return windows

    def get_all_gait_cycles_and_set_gait_phase_label(self):
        cycles_ = []
        for i_trial, trial_ in enumerate(self.trials):
            unnormalized_poses = self.normalizer.unnormalize(trial_.converted_pose.clone().unsqueeze(0))
            r_v_grf = unnormalized_poses[0, :, self.opt.model_states_column_names.index('calcn_r_force_vy')]
            r_v_grf = r_v_grf.cpu().numpy()

            trial_gait_phase_label, stance_start_valid, _ = self.grf_to_trial_gait_phase_label(
                r_v_grf, self.window_len, self.opt.target_sampling_rate)

            for i_start in range(len(stance_start_valid)-1):
                gait_cycle_converted = trial_.converted_pose[stance_start_valid[i_start]:stance_start_valid[i_start+1]]
                cycles_.append(GaitCycles(
                    gait_cycle_converted, (stance_start_valid[i_start], stance_start_valid[i_start+1]),
                    trial_.sub_and_trial_name, trial_.dset_name))
            self.trials[i_trial].trial_gait_phase_label = trial_gait_phase_label
        return cycles_

    @staticmethod
    def grf_to_trial_gait_phase_label(v_grf, window_len, target_sampling_rate, stance_len_thds=None, cycle_len_thds=None):
        stance_vgrf_thd = 1    # 100% of body mass. Needs to be large because some datasets are noisy.
        if stance_len_thds is None:
            stance_len_thds = [int(target_sampling_rate * 0.1), int(target_sampling_rate * 1.5)]      # 0.1 s to 1.5 s
        if cycle_len_thds is None:
            cycle_len_thds = [int(target_sampling_rate * 0.2), int(target_sampling_rate * 2)]      # 0.2 s to 2 s

        trial_gait_phase_label = np.full([v_grf.shape[0]], NOT_IN_GAIT_PHASE)       # shape x
        stance_start_valid, stance_end_valid = [], []

        stance_flag = np.abs(v_grf) > stance_vgrf_thd
        stance_flag = stance_flag.astype(int)
        start_end_indicator = np.diff(stance_flag)
        stance_start = np.where(start_end_indicator == 1)[0]
        stance_end = np.where(start_end_indicator == -1)[0]
        for i_start in range(0, len(stance_start)-1):
            end_ = stance_end[(stance_start[i_start] < stance_end) & (stance_end < stance_start[i_start+1])]
            # Exclusion criteria
            if len(end_) != 1:
                continue
            if not stance_len_thds[0] < (end_ - stance_start[i_start]) < stance_len_thds[1]:
                continue
            if not cycle_len_thds[0] < (stance_start[i_start + 1] - stance_start[i_start]) < cycle_len_thds[1]:
                continue
            trial_gait_phase_label[stance_start[i_start]:stance_start[i_start+1]] = np.linspace(0, 1000, stance_start[i_start+1]-stance_start[i_start])
            stance_start_valid.append(stance_start[i_start])
            stance_end_valid.append(end_[0])
        return trial_gait_phase_label, stance_start_valid, stance_end_valid

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
            if sum([keyword in subject_path for keyword in self.dset_keyworks_to_exclude]):
                continue

            try:
                subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            except RuntimeError:
                print(f'Failed loading subject {subject_path}, skipping')
                continue
            contact_bodies = subject.getGroundForceBodies()
            print(f'Loading subject: {subject_path}')

            subject_name = subject_path.split('/')[-1].split('.')[0]
            dset_name = subject_path.split('/')[-3]

            if f'{dset_name}_{subject_name}' in WEIGHT_KG_OVERWRITE.keys():
                weight_kg = WEIGHT_KG_OVERWRITE[f'{dset_name}_{subject_name}']
                print(f'Overwriting {dset_name}_{subject_name}\'s weight to {weight_kg} kg')
            else:
                weight_kg = subject.getMassKg()
            height_m = subject.getHeightM()

            skel = subject.readSkel(0, geometryFolder=os.path.dirname(os.path.realpath(__file__)) + "/../../data/Geometry/")
            self.skels[dset_name+'_'+subject_name] = skel

            # neutral_pose = np.zeros([1, skel.getNumDofs()])
            # body_nodes = [skel.getBodyNode(i) for i in range(skel.getNumBodyNodes())]
            # body_locs = get_multi_body_loc_using_nimble_by_body_nodes(body_nodes, skel, neutral_pose)
            # cond = body_loc_to_cond(body_locs).tolist()[0] + [subject.getHeightM(), subject.getMassKg() / 50]
            # cond = torch.tensor(cond).float()
            cond = torch.zeros([6]).float()

            model_offsets = get_model_offsets(skel).float()
            if dset_name == '':
                dset_name = subject_name.split('_')[0]

            trial_start_num_ = self.trial_start_num if self.trial_start_num >= 0 else max(0, subject.getNumTrials() + self.trial_start_num)

            if self.use_camargo_lumbar_reconstructed and 'Camargo2021' in subject_path:
                camargo_reconstructed_dict, lumbar_dof = pickle.load( open(f"{self.data_path}/camargo_lumbar_reconstructed.pkl", "rb" ))
                lumbar_col_loc = [opt.osim_dof_columns.index(col) for col in lumbar_dof]

            for trial_id in tqdm(range(trial_start_num_, subject.getNumTrials())):
                sampling_rate = int(1 / subject.getTrialTimestep(trial_id))
                sub_and_trial_name = subject_name + '__' + subject.getTrialName(trial_id)
                if self.specific_trial:
                    if isinstance(self.specific_trial, str) and self.specific_trial not in sub_and_trial_name:
                        continue
                    if isinstance(self.specific_trial, list) and not sum([trial in sub_and_trial_name for trial in self.specific_trial]):
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
                forces = np.array([frame.groundContactForce for frame in first_passes])

                if 'Carter' not in subject_path:
                    cops = np.array([frame.groundContactCenterOfPressure for frame in first_passes])
                else:
                    plates = subject.readForcePlates(trial_id)
                    if len(plates) == 1:
                        cops = np.concatenate([np.array(plates[0].centersOfPressure), np.array(plates[0].centersOfPressure)], axis=1)
                    elif len(plates) == 2:
                        cops = np.concatenate([np.array(plates[1].centersOfPressure), np.array(plates[0].centersOfPressure)], axis=1)
                    else:
                        print(f'{subject_name}, {trial_id} has {len(plates)} force plates, skipping')
                        continue

                if self.restrict_contact_bodies and len(forces[0]) != 6:
                    self.num_of_excluded_trials['contact_body_num'] += 1
                    continue        # only include data with 2 contact bodies.
                else:
                    r_foot_idx, l_foot_idx = contact_bodies.index('calcn_r'), contact_bodies.index('calcn_l')
                    foot_idx = [r_foot_idx*3, r_foot_idx*3+1, r_foot_idx*3+2, l_foot_idx*3, l_foot_idx*3+1, l_foot_idx*3+2]
                    contact_bodies = ['calcn_r', 'calcn_l']
                    forces = forces[:, foot_idx]
                    cops = cops[:, foot_idx]

                if len(poses) < 10:
                    self.num_of_excluded_trials['trial_length'] += 1
                    continue
                    # This is to compensate an AddBiomechanics bug that first GRF is always 0.
                if (forces[0] == 0).all() or (cops[0] == 0).all():
                    poses = poses[1:]
                    forces = forces[1:]
                    cops = cops[1:]
                    probably_missing = probably_missing[1:]
                if (forces[-1] == 0).all() or (cops[-1] == 0).all():
                    # print(f'Compensating an AddB bug, {subject_name}, {trial_id} has 0 GRF at the first frame.', end='')
                    poses = poses[:-1]
                    forces = forces[:-1]
                    cops = cops[:-1]
                    probably_missing = probably_missing[:-1]

                force_v0, force_v1 = forces[:, :3] / weight_kg, forces[:, 3:] / weight_kg
                states = np.concatenate([np.array(poses), force_v0, cops[:, :3], force_v1, cops[:, 3:]], axis=1)
                if not self.is_lumbar_rotation_reasonable(np.array(states), opt.osim_dof_columns):
                    self.num_of_excluded_trials['lumbar_rotation'] += 1
                    continue

                grf_flag_counts = list(mit.run_length.encode(probably_missing))

                if len(grf_flag_counts) > 30:
                    # print(f'Compensating an AddB bug, {dset_name} - {subject_name} - {trial_id} notMissingGRF flag abnormally'
                    #       f' flipped {len(grf_flag_counts)} times, thus setting all to True.', end='')
                    probably_missing = [False] * len(probably_missing)

                states = norm_cops(skel, states, opt, weight_kg, height_m, self.check_cop_to_calcn_distance)
                if states is False:
                    print(f'{sub_and_trial_name} has CoP far away from foot, skipping')
                    self.num_of_excluded_trials['wrong_cop'] += 1
                    continue

                if self.align_moving_direction_flag:
                    states, rot_mat = align_moving_direction(states, opt.osim_dof_columns)
                else:
                    rot_mat = torch.eye(3).float()
                if states is False:
                    print(f'{sub_and_trial_name} moving direction changed by more than 45 deg, skipping')
                    self.num_of_excluded_trials['large_moving_direction_change'] += 1
                    continue

                if (not self.include_trials_shorter_than_window_len) and (states.shape[0] / sampling_rate * self.target_sampling_rate) < self.window_len + 2:
                    self.num_of_excluded_trials['trial_length'] += 1
                    continue
                if states.shape[0] < 20:        # need to be longer than 20 frames for filtering
                    self.num_of_excluded_trials['trial_length'] += 1
                    continue
                if sampling_rate != self.target_sampling_rate:
                    print(f'{dset_name} is collected at {sampling_rate} Hz, resampling to {self.target_sampling_rate} Hz')
                    states = linear_resample_data(states, sampling_rate, self.target_sampling_rate)
                    probably_missing = linear_resample_data(np.array(probably_missing).astype(float), sampling_rate, self.target_sampling_rate).astype(bool)
                probably_missing = np.array(probably_missing).astype(np.float64)

                if 'camargo_reconstructed_dict' in locals():
                    if sub_and_trial_name in camargo_reconstructed_dict.keys():
                        assert states.shape[0] == camargo_reconstructed_dict[sub_and_trial_name].shape[0]
                        states[:, lumbar_col_loc] = camargo_reconstructed_dict[sub_and_trial_name]

                foot_locations, _, _, _ = forward_kinematics(states[:, :-len(KINETICS_ALL)], model_offsets)
                mtp_r_loc, mtp_l_loc = foot_locations[1].squeeze().cpu().numpy(), foot_locations[3].squeeze().cpu().numpy()

                mtp_r_loc = data_filter(mtp_r_loc, 10, self.target_sampling_rate)
                mtp_l_loc = data_filter(mtp_l_loc, 10, self.target_sampling_rate)

                # vancriek dataset has one two foot on one plate issue.
                if sum([keyword in subject_path.lower() for keyword in OVERGROUND_DSETS]) or ('camargo' in subject_path.lower() and '_split5' not in subject_path.lower()):
                    mtp_r_vel, mtp_l_vel = np.zeros_like(mtp_r_loc), np.zeros_like(mtp_l_loc)
                else:
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
                                       dset_name, rot_mat, pos_vec, self.window_len, cond, mtp_r_vel, mtp_l_vel)
                if self.include_trials_shorter_than_window_len or len(trial_data.available_win_start) > 0:
                    self.trials.append(trial_data)
                    print(f'{sub_and_trial_name} loaded')
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
                self.num_of_excluded_trials['jittery_sample'] += len(break_point_list)
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


class WindowData:
    def __init__(self, pose, model_offsets, trial_id, gait_phase_label, mask, height_m, weight_kg, cond):
        self.pose = pose
        self.model_offsets = model_offsets
        self.trial_id = trial_id
        self.gait_phase_label = gait_phase_label
        self.mask = mask
        self.height_m = height_m
        self.weight_kg = weight_kg
        self.cond = cond


class TrialData:
    def __init__(self, converted_states, probably_missing, model_offsets, contact_bodies, sub_and_trial_name, trial_id,
                 height_m, weight_kg, dset_name, rot_mat_for_moving_direction_alignment, pos_vec_for_pos_alignment,
                 window_len, cond, mtp_r_vel, mtp_l_vel, trial_gait_phase_label=None):
        self.converted_pose = converted_states
        self.probably_missing = probably_missing
        self.model_offsets = model_offsets
        self.contact_bodies = contact_bodies
        self.sub_and_trial_name = sub_and_trial_name
        self.height_m = height_m
        self.weight_kg = weight_kg
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
        self.cond = cond
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
                'height_m': self.height_m, 'weight_kg': self.weight_kg}

    def get_attributes_for_reinitialization(self):
        return [
            self.model_offsets,
            self.contact_bodies,
            self.sub_and_trial_name,
            self.trial_id,
            self.height_m,
            self.weight_kg,
            self.dset_name,
            self.rot_mat_for_moving_direction_alignment,
            self.pos_vec_for_pos_alignment,
            self.window_len,
            self.cond
        ]
