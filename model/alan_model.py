import math
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from alant.preprocess import increment_path, Normalizer
from data.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
from model.adan import Adan
from model.dance_decoder import DanceDecoder
from alant.vis import SMPLSkeleton
import os
import torch
import copy
import torch.nn as nn
from model.utils import extract, make_beta_schedule, linear_resample_data, update_d_dd
from typing import Any, List
import nimblephysics as nimble
from alant.alan_consts import *
from alant.quaternion import euler_from_6v, euler_to_6v, rotation_6d_to_matrix, matrix_to_rotation_6d
from alant.alan_osim_fk import get_model_offsets, get_knee_rotation_coefficients, forward_kinematics
from nimblephysics import NimbleGUI
import time
import more_itertools as mit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def convert_addb_state_to_model_input(pose_df, joints_3d):
    # shift root position to start in (x,y) = (0,0)
    pose_df['pelvis_tx'] = pose_df['pelvis_tx'] - pose_df['pelvis_tx'][0]
    pose_df['pelvis_ty'] = pose_df['pelvis_ty'] - pose_df['pelvis_ty'][0]
    pose_df['pelvis_tz'] = pose_df['pelvis_tz'] - pose_df['pelvis_tz'][0]

    # remove frozen dof
    for frozen_col in FROZEN_DOFS:
        if frozen_col in pose_df.columns:
            pose_df = pose_df.drop(frozen_col, axis=1)

    # convert euler to 6v
    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_6v = euler_to_6v(torch.tensor(pose_df[joints_with_3_dof].values), "ZXY").numpy()
        for joints_euler_name in joints_with_3_dof:
            pose_df = pose_df.drop(joints_euler_name, axis=1)
        for i in range(6):
            pose_df[joint_name + '_' + str(i)] = joint_6v[:, i]

    return pose_df


def inverse_convert_addb_state_to_model_input(model_states, model_states_column_names, joints_3d, osim_dof_columns):
    model_states_dict = {col: model_states[..., i] for i, col in enumerate(model_states_column_names) if col in osim_dof_columns}

    # convert 6v to euler
    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_name_6v = [joint_name + '_' + str(i) for i in range(6)]
        index_ = [model_states_column_names.index(joint_name_6v[i]) for i in range(6)]
        joint_euler = euler_from_6v(model_states[..., index_], "ZXY")

        for i, joints_euler_name in enumerate(joints_with_3_dof):
            model_states_dict[joints_euler_name] = joint_euler[..., i]

    # add frozen dof back
    for frozen_col in FROZEN_DOFS:
        model_states_dict[frozen_col] = torch.zeros(model_states.shape[:2]).to(model_states.device)

    osim_states = torch.stack([model_states_dict[col] for col in osim_dof_columns], dim=2).float()
    return osim_states


def identity(t, *args, **kwargs):
    return t


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class MotionDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            train: bool,
            opt,
            reset_moving_direction_flag: bool = True,
            normalizer: Any = None,
            trial_start_num=0,
            max_trial_num=None,
            divide_jittery=True,
            specific_dset=None
    ):
        self.data_path = data_path
        self.trial_start_num = trial_start_num
        self.target_sampling_rate = opt.target_sampling_rate
        self.window_len = int(opt.target_sampling_rate * opt.window_len)     # 1.5 seconds for each window
        self.reset_moving_direction_flag = reset_moving_direction_flag
        self.opt = opt
        self.divide_jittery = divide_jittery
        self.specific_dset = specific_dset

        self.train = train
        self.name = "Train" if self.train else "Test"

        self.dset_set = set()

        print("Loading dataset...")
        self.load_addb(opt, max_trial_num)

        # self.visualize_loaded(opt, self.trials)       # !!!

        self.trial_num = len(self.trials)
        if self.trial_num == 0:
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
            self.trials[i].converted_pose = torch.tensor(self.normalizer.normalize(self.trials[i].converted_pose)).clone().detach().float()

        # # [debug]
        # data_concat = torch.cat([trial_.converted_pose for trial_ in self.trials], dim=0)
        # for i_col, col_name in enumerate(opt.model_states_column_names):
        #     print(f'{col_name}, mean: {data_concat[:, i_col].mean()}, std: {data_concat[:, i_col].std()},'
        #           f' max: {data_concat[:, i_col].max()}, min: {data_concat[:, i_col].min()}')

    def __len__(self):
        return self.opt.pseudo_dataset_len

    def __getitem__(self, _):
        i_trial = torch.randint(0, self.trial_num, (1,)).item()        # a random trial regardless of its length
        trial_length = self.trials[i_trial].length
        slice_index = torch.randint(0, trial_length - self.window_len+1, (1,)).item()

        converted_pose = self.trials[i_trial].converted_pose[slice_index:slice_index+self.window_len, ...]
        return (converted_pose, self.trials[i_trial].model_offsets, i_trial)

    @staticmethod
    def reset_moving_direction(poses, column_names):
        converted_pose_clone = torch.from_numpy(poses).clone().float()
        pelvis_orientation_col_loc = [column_names.index(col) for col in JOINTS_3D_ALL['pelvis']]
        p_pos_col_loc = [column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
        r_grf_col_loc = [column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
        l_grf_col_loc = [column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]

        if len(pelvis_orientation_col_loc) != 3 or len(p_pos_col_loc) != 3 or len(r_grf_col_loc) != 3 or len(l_grf_col_loc) != 3:
            raise ValueError('check column names')

        pelvis_orientation = converted_pose_clone[:, pelvis_orientation_col_loc]
        pelvis_orientation = euler_angles_to_matrix(pelvis_orientation, "ZXY")
        p_pos = converted_pose_clone[:, p_pos_col_loc]
        r_grf = converted_pose_clone[:, r_grf_col_loc]
        l_grf = converted_pose_clone[:, l_grf_col_loc]

        angle = math.atan2(- pelvis_orientation[0][0, 2], pelvis_orientation[0][2, 2])
        rot_mat = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]).float()

        pelvis_orientation_rotated = torch.matmul(rot_mat, pelvis_orientation)
        p_pos_rotated = torch.matmul(rot_mat, p_pos.unsqueeze(2)).squeeze(2)
        r_grf_rotated = torch.matmul(rot_mat, r_grf.unsqueeze(2)).squeeze(2)
        l_grf_rotated = torch.matmul(rot_mat, l_grf.unsqueeze(2)).squeeze(2)

        converted_pose_clone[:, pelvis_orientation_col_loc] = matrix_to_euler_angles(pelvis_orientation_rotated.float(), "ZXY")
        converted_pose_clone[:, p_pos_col_loc] = p_pos_rotated.float()
        converted_pose_clone[:, r_grf_col_loc] = r_grf_rotated.float()
        converted_pose_clone[:, l_grf_col_loc] = l_grf_rotated.float()

        return converted_pose_clone

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

    @staticmethod
    def augment_one_trial_to_4_moving_directions(trials, converted_column_names):
        trials_augmented = []
        rot_mat_to_apply = [
            torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]).float()
            for angle in [0.5*np.pi, np.pi, 1.5*np.pi]
        ]

        pelvis_orientation_col_loc = [converted_column_names.index(col) for col in [f'pelvis_{x}' for x in range(6)]]
        pelvis_position_col_loc = [converted_column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
        r_grf_col_loc = [converted_column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
        l_grf_col_loc = [converted_column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]

        if len(pelvis_orientation_col_loc) != 6:
            raise ValueError('pelvis column number incorrect')
        for trial_ in trials:
            trials_augmented.append(trial_)

            pelvis_orientation = trial_.converted_pose[:, pelvis_orientation_col_loc]
            pelvis_orientation = rotation_6d_to_matrix(pelvis_orientation)
            pelvis_position = trial_.converted_pose[:, pelvis_position_col_loc]
            r_grf = trial_.converted_pose[:, r_grf_col_loc]
            l_grf = trial_.converted_pose[:, l_grf_col_loc]

            for rot_mat in rot_mat_to_apply:
                pelvis_orientation_rotated = torch.matmul(rot_mat, pelvis_orientation)
                pelvis_position_rotated = torch.matmul(rot_mat, pelvis_position.unsqueeze(2)).squeeze(2)
                r_grf_rotated = torch.matmul(rot_mat, r_grf.unsqueeze(2)).squeeze(2)
                l_grf_rotated = torch.matmul(rot_mat, l_grf.unsqueeze(2)).squeeze(2)

                converted_pose = copy.deepcopy(trial_.converted_pose)
                converted_pose[:, pelvis_orientation_col_loc] = matrix_to_rotation_6d(pelvis_orientation_rotated.float())
                converted_pose[:, pelvis_position_col_loc] = pelvis_position_rotated.float()
                converted_pose[:, r_grf_col_loc] = r_grf_rotated.float()
                converted_pose[:, l_grf_col_loc] = l_grf_rotated.float()

                augmented_trial = TrialData(converted_pose,
                                            trial_.raw_states,
                                            trial_.model_offsets,
                                            trial_.contact_bodies,
                                            trial_.sub_and_trial_name,
                                            trial_.dset_name)

                trials_augmented.append(augmented_trial)
        return trials_augmented

    def get_all_wins(self):
        windows = []
        _ = self.get_all_gait_cycles_and_set_gait_phase_label()
        for i_trial, trial_ in enumerate(self.trials):
            trial_len = trial_.length
            for i in range(0, trial_len - self.window_len + 1, self.window_len):
                gait_phase_label = trial_.trial_gait_phase_label[i:i+self.window_len]
                if (gait_phase_label != NOT_IN_GAIT_PHASE).any():
                    windows.append((trial_.converted_pose[i:i+self.window_len, ...], self.trials[i_trial].model_offsets, i_trial, gait_phase_label))
            # # The last window is probably overlapping with the previous one.
            # windows.append((trial_.converted_pose[trial_len-self.window_len:, ...], self.trials[i_trial].model_offsets, i_trial))
        return windows

    def get_all_gait_cycles_and_set_gait_phase_label(self):
        stance_vgrf_thd = 1    # 100% of body mass. Needs to be large because some datasets are noisy.
        stance_len_thds = [int(self.opt.target_sampling_rate * 0.1), int(self.opt.target_sampling_rate * self.opt.window_len * 0.8)]      # 0.1 s to 1 s
        cycle_len_thds = [int(self.opt.target_sampling_rate * 0.2), int(self.opt.target_sampling_rate * self.opt.window_len)]      # 0.2 s to 1.5 s

        cycles_ = []
        for i_trial, trial_ in enumerate(self.trials):
            unnormalized_poses = self.normalizer.unnormalize(trial_.converted_pose.clone().unsqueeze(0))
            trial_gait_phase_label = np.full([unnormalized_poses.shape[1]], NOT_IN_GAIT_PHASE)
            r_v_grf = unnormalized_poses[0, :, self.converted_column_names.index('calcn_r_force_vy')]
            r_v_grf = r_v_grf.cpu().numpy()

            stance_flag = np.abs(r_v_grf) > stance_vgrf_thd
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
                if stance_start[i_start+1] - self.window_len < 0:
                    continue
                gait_cycle_converted = trial_.converted_pose[stance_start[i_start]:stance_start[i_start+1]]
                gait_cycle_raw = trial_.raw_states[stance_start[i_start]:stance_start[i_start+1]]
                trial_gait_phase_label[stance_start[i_start]:stance_start[i_start+1]] = np.linspace(0, 1000, stance_start[i_start+1]-stance_start[i_start])
                cycles_.append(GaitCycles(
                    gait_cycle_converted, gait_cycle_raw, (stance_start[i_start], stance_start[i_start+1]),
                    trial_.sub_and_trial_name, trial_.dset_name))
            self.trials[i_trial].trial_gait_phase_label = trial_gait_phase_label
        return cycles_

    def visualize_loaded(self, opt, trials):
        pose_converted = [trial.converted_pose.reshape([1, -1, trial.converted_pose.shape[1]]) for trial in trials]

        pose_col_loc = [i_dof for i_dof, dof in enumerate(opt.osim_dof_columns) if '_force_' not in dof]
        force_col_loc = [i_dof for i_dof, dof in enumerate(opt.osim_dof_columns) if '_force_' in dof]

        pose_list = [inverse_convert_addb_state_to_model_input(
            pose, self.opt.model_states_column_names, self.opt.joints_3d, self.opt.osim_dof_columns) for pose in pose_converted]

        geometry = '/mnt/d/Local/AddBiom/vmu-suit/ml_and_simulation/data/Geometry/'
        print('Using Geometry folder: '+geometry)
        geometry = os.path.abspath(geometry)
        if not geometry.endswith('/'):
            geometry += '/'
            print(' > Converted to absolute path: '+geometry)

        customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
            '/mnt/d/Local/Data/MotionPriorData/model_and_geometry/unscaled_generic_no_arm.osim')
        skel = customOsim.skeleton

        world = nimble.simulation.World()
        gui = NimbleGUI(world)
        gui.serve(8090)

        while True:
            gui.nativeAPI().createText('trial num', 'Trial: ', [1200, 200], [250, 50])
            for i_trial in range(len(pose_list)):
                gui.nativeAPI().setTextContents('trial num', 'Trial: ' + trials[i_trial].sub_and_trial_name)
                states = pose_list[i_trial]
                for i in range(states.shape[1]):
                    poses = states[0, i, pose_col_loc]
                    skel.setPositions(poses.reshape([-1, 1]))
                    for i_f, contact_body in enumerate(['calcn_r', 'calcn_l']):
                        body_pos = skel.getBodyNode(contact_body).getWorldTransform().translation()
                        forces = states[0, i, force_col_loc[3 * i_f:3 * (i_f + 1)]]
                        gui.nativeAPI().createLine(f'line_{i_f}', [body_pos, body_pos + 0.1 * forces.numpy()], color=[1, 0., 0., 1])
                    gui.nativeAPI().renderSkeleton(skel)
                    time.sleep(0.005)
            # for i in range(pose_list[0].shape[1]):        # visualize multiple skeletons
            #     for j in range(len(pose_list)):
            #         q = pose_list[j][0, i]
            #         skel.setPositions(q[pose_col_loc])
            #         for i_f, contact_body in enumerate(['calcn_r', 'calcn_l']):
            #             body_pos = skel.getBodyNode(contact_body).getWorldTransform().translation()
            #             forces = q[force_col_loc[3 * i_f:3 * (i_f + 1)]]
            #             gui.nativeAPI().createLine(f'line_{j}_{i_f}', [body_pos, body_pos + 0.1 * forces.numpy()], color=[1, 0., 0., 1])
            #         gui.nativeAPI().renderSkeleton(skel, str(j))
            #         time.sleep(0.007)

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
            subject_mass = subject.getMassKg()
            contact_bodies = subject.getGroundForceBodies()
            print(f'Loading subject: {subject_path}, mass: {subject_mass} kg')

            skel = subject.readSkel(0, geometryFolder=os.path.dirname(os.path.realpath(__file__)) + "/../../data/Geometry/")
            model_offsets = get_model_offsets(skel).float()
            subject_name = subject_path.split('/')[-1].split('.')[0]
            dset_name = subject_path.split('/')[-3]
            if dset_name == '':
                dset_name = subject_name.split('_')[0]

            trial_start_num_ = self.trial_start_num if self.trial_start_num >= 0 else max(0, subject.getNumTrials() + self.trial_start_num)
            for trial_index in tqdm(range(trial_start_num_, subject.getNumTrials())):
                sub_and_trial_name = subject_name + '_' + subject.getTrialName(trial_index)
                trial_length = subject.getTrialLength(trial_index)
                probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                                in subject.getMissingGRF(trial_index)]

                frames: nimble.biomechanics.FrameList = subject.readFrames(trial_index, 0, trial_length,
                                                                           includeSensorData=False,
                                                                           includeProcessingPasses=True)
                try:
                    first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]
                except IndexError:
                    print(f'{subject_name}, {trial_index} has no processing passes, skipping')
                    continue
                poses = [frame.pos for frame in first_passes]

                forces = [frame.groundContactForce / subject_mass for frame in first_passes]
                if len(forces[0]) != 6:
                    continue        # only include data with 2 contact bodies.
                states = np.concatenate([np.array(poses), np.array(forces)], axis=1)

                grf_flag_counts = list(mit.run_length.encode(probably_missing))
                max_has_grf_count = -1
                max_has_grf_idx = -1
                for idx, (val, count) in enumerate(grf_flag_counts):
                    if not val and max_has_grf_count < count:
                        max_has_grf_count = count
                        max_has_grf_idx = idx
                elems_before_idx = sum((idx[1] for idx in grf_flag_counts[:max_has_grf_idx]))
                states = states[elems_before_idx:elems_before_idx+max_has_grf_count, :]

                if self.reset_moving_direction_flag:
                    states = self.reset_moving_direction(states, opt.osim_dof_columns)

                sampling_rate = 1 / subject.getTrialTimestep(trial_index)
                if (states.shape[0] / sampling_rate * self.target_sampling_rate) < self.window_len + 2:
                    # print(f'Subject {subject_name} trial {sub_and_trial_name} has {round(len(states) / sampling_rate, 1)} s data (less than 3 s), skipping')
                    continue
                states = linear_resample_data(states, sampling_rate, self.target_sampling_rate)
                if not self.is_lumbar_rotation_reasonable(states, opt.osim_dof_columns):
                    continue

                states_df = pd.DataFrame(states, columns=opt.osim_dof_columns)
                states_df = convert_addb_state_to_model_input(states_df, opt.joints_3d)
                self.converted_column_names = list(states_df.columns)

                converted_states = torch.tensor(states_df.values).float()

                current_trials = [TrialData(converted_states, states, model_offsets, contact_bodies, sub_and_trial_name, dset_name)]

                # functions for cleaning up the data
                if self.divide_jittery:
                    current_trials = self.divide_jittery_trials(current_trials, self.converted_column_names)

                for trial_ in current_trials:
                    self.trials.append(trial_)
                    self.dset_set.add(dset_name)

                if max_trial_num is not None and len(self.trials) >= max_trial_num:
                    return
            print('Current trial num: {}'.format(len(self.trials)))
        # self.trial_length_probability = torch.tensor([1000 / trial.length for trial in self.trials]).float()

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

        pos_col = [col for col in converted_column_names if '_tx' in col or '_ty' in col or '_tz' in col]
        pos_col_loc = [converted_column_names.index(col) for col in pos_col]
        rot_mat_col = [col for col in converted_column_names if '_0' in col or '_1' in col or '_2' in col or '_3'
                       in col or '_4' in col or '_5' in col]
        rot_mat_col_loc = [converted_column_names.index(col) for col in rot_mat_col]
        euler_angle_col = [col for col in converted_column_names if 'angle' in col or 'elbow' in col or 'pro_sup' in
                           col or 'wrist' in col]
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
                    if end_ - start_ < self.window_len:
                        continue
                    trial_data_list.append(
                        TrialData(current_trial.converted_pose[start_:end_],
                                  current_trial.raw_states,
                                  current_trial.model_offsets,
                                  current_trial.contact_bodies,
                                  current_trial.sub_and_trial_name,
                                  current_trial.dset_name)
                    )

                print('Divided {} {} into {} trials, {} due to linear acc, {} due to euler angle vel, {} due to rot mat vel'.format(
                    current_trial.dset_name, current_trial.sub_and_trial_name,
                    len(trial_data_list), len(pos_break_point), len(euler_angle_break_point), len(rot_mat_break_point)))
            else:
                trial_data_list = [current_trial]
        return trial_data_list


class GaitCycles:
    def __init__(self, gait_cycle_converted, gait_cycle_raw, index_pair, sub_and_trial_name, dset_name):
        self.sub_and_trial_name = sub_and_trial_name
        self.dset_name = dset_name
        self.gait_cycle = gait_cycle_converted
        self.gait_cycle_raw = gait_cycle_raw
        self.gait_cycle_len = gait_cycle_converted.shape[0]
        self.index_pair = index_pair
        self.gait_cycle_raw_resampled = torch.from_numpy(self.resample_to_100_steps(gait_cycle_raw))

    @staticmethod
    def resample_to_100_steps(data_raw):
        x, step = np.linspace(0., 1., data_raw.shape[0], retstep=True)
        new_x = np.linspace(0., .99, 100)
        f = interp1d(x, data_raw, axis=0)
        data_resampled = f(new_x)
        return data_resampled


class TrialData:
    def __init__(self, converted_states, raw_states, model_offsets, contact_bodies, sub_and_trial_name, dset_name, trial_gait_phase_label=None):
        self.converted_pose = converted_states
        self.raw_states = raw_states
        self.model_offsets = model_offsets
        self.contact_bodies = contact_bodies
        self.sub_and_trial_name = sub_and_trial_name
        self.dset_name = dset_name
        self.length = converted_states.shape[0]
        self.trial_gait_phase_label = trial_gait_phase_label

    def get_attributes(self):
        return {'sub_and_trial_name': self.sub_and_trial_name, 'dset_name': self.dset_name}


class MotionModel:
    def __init__(
            self,
            opt,
            repr_dim,
            normalizer=None,
            EMA=True,
            learning_rate=4e-4,
            weight_decay=0.02,
    ):
        self.opt = opt
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = opt.feature_type == "baseline"

        self.repr_dim = repr_dim

        feature_dim = 35 if use_baseline_feats else 4800
        self.horizon = horizon = 90     # [!] sampling rate * win size in seconds

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if opt.checkpoint != "":
            checkpoint = torch.load(
                opt.checkpoint, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            opt,
            # schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,          # predict epsilon is usually True for diffusion models right???
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if opt.checkpoint != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        if opt.log_with_wandb:
            wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, dir="wandb_logs")
        train_dataset = MotionDataset(
            data_path=opt.data_path_train,
            train=True,
            # trial_start_num=-2,
            # max_trial_num=10,            # !!!
            opt=opt,
        )
        # set normalizer
        self.normalizer = train_dataset.normalizer
        self.diffusion.set_normalizer(self.normalizer)

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            # num_workers=min(int(num_cpus * 0.75), 32),            # causes error on slurm
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")     # , miniters=int(223265/100)
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        sub_and_trial_names, dset_names = train_dataset.get_attributes_of_trials()
        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            print(f'epoch: {epoch} / {opt.epochs}')
            avg_loss_simple = 0
            avg_loss_vel = 0
            avg_loss_fk = 0
            avg_loss_drift = 0
            avg_loss_slide = 0

            joint_loss = np.zeros([opt.model_states_column_names.__len__()])
            dataset_loss_record = np.zeros([train_dataset.trial_num])

            # train
            self.train()            # switch to train mode.

            for step, x in enumerate(load_loop(train_data_loader)):
                total_loss, losses = self.diffusion(x, None, t_override=None)
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                if opt.log_with_wandb:
                    # ema update and train loss update only on main
                    if self.accelerator.is_main_process:
                        avg_loss_simple += losses[0].detach().cpu().numpy()
                        avg_loss_vel += losses[1].detach().cpu().numpy()
                        avg_loss_fk += losses[2].detach().cpu().numpy()
                        avg_loss_drift += losses[3].detach().cpu().numpy()
                        avg_loss_slide += losses[4].detach().cpu().numpy()

                        joint_loss += losses[5].detach().mean(axis=0).mean(axis=0).cpu().numpy()
                        dataset_loss_record[x[2].cpu()] += losses[5].detach().mean(axis=1).mean(axis=1).cpu().numpy()

                        if step % opt.ema_interval == 0:
                            self.diffusion.ema.update_model_average(self.diffusion.master_model, self.diffusion.model)

            self.accelerator.wait_for_everyone()

            if epoch > -1:       # [!]
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }
                    if opt.log_with_wandb:
                        loss_val_name_pairs_terms = [
                            (avg_loss_simple / len(train_data_loader), 'simple'), (avg_loss_vel / len(train_data_loader), 'vel'),
                            (avg_loss_fk / len(train_data_loader), 'forward kinematics'), (avg_loss_drift / len(train_data_loader), 'drift'),
                            (avg_loss_slide / len(train_data_loader), 'slide')]

                        loss_val_name_pairs_joints = [(loss_, joint_) for joint_, loss_ in zip(opt.model_states_column_names, joint_loss)]

                        loss_dset = {dset: [0, 0] for dset in train_dataset.dset_set}
                        for dset_name, trial_loss in zip(dset_names, dataset_loss_record):
                            if trial_loss == 0:
                                continue        # if 0, then the trial is not used for training, thus skipping
                            loss_dset[dset_name] = [loss_dset[dset_name][0] + trial_loss, loss_dset[dset_name][1] + 1]

                        loss_val_name_pairs_dset = [(loss_count[0] / loss_count[1], dset_name)
                                                    for dset_name, loss_count in loss_dset.items()]

                        log_dict = {name: loss for loss, name in loss_val_name_pairs_dset + loss_val_name_pairs_joints + loss_val_name_pairs_terms}
                        wandb.log(log_dict)

            # if epoch % 100 == 0 or epoch == opt.epochs:
            if (epoch % 200) == 0:
                torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process and opt.log_with_wandb:
            wandb.run.finish()

    def eval_loop(self, opt, state_true, masks, num_of_generation_per_window=1):
        self.eval()
        constraint = {'mask': masks, 'value': state_true.clone()}

        shape = (state_true.shape[0], self.horizon, self.repr_dim)
        cond = torch.ones(state_true.shape[0])
        cond = cond.to(self.accelerator.device)
        state_pred_list = [self.diffusion.generate_samples(
            shape,
            cond,
            self.normalizer,
            opt,
            mode="inpaint",
            constraint=constraint)
            for _ in range(num_of_generation_per_window)]

        state_true = self.normalizer.unnormalize(state_true)
        state_true = state_true.detach().cpu()

        state_true = inverse_convert_addb_state_to_model_input(state_true, opt.model_states_column_names,
                                                               opt.joints_3d, opt.osim_dof_columns)
        return state_true, state_pred_list


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
                current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,      # this is DanceDecoder
            horizon,
            repr_dim,
            smpl,
            opt,
            n_timestep=1000,
            schedule="linear",
            loss_type="l1",
            clip_denoised=False,
            predict_epsilon=True,
            guidance_weight=3,
            use_p2=False,
            cond_drop_prob=0.2,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        self.opt = opt

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module
        self.smpl = smpl

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
            )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, cond, t, weight=None, clip_x_start=False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, t, weight)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(          # Only used during inference
            self,
            shape,
            cond,
            noise=None,
            constraint=None,
            return_diffusion=False,
            start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        return x

    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        if batch == 1:
            return self.ddim_sample(shape, cond)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def inpaint_ddim_loop(self, shape, cond, noise=None, constraint=None, return_diffusion=False, start_point=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)

            if time_next < 0:
                x = x_start
                return x

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            timesteps = torch.full((batch,), time_next, device=device, dtype=torch.long)
            value_ = self.q_sample(value, timesteps) if (time > 0) else x
            x = value_ * mask + (1.0 - mask) * x
        return x

    @torch.no_grad()
    def inpaint_loop(
            self,
            shape,
            cond,
            noise=None,
            constraint=None,
            return_diffusion=False,
            start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
            self,
            shape,
            cond,
            noise=None,
            constraint=None,
            return_diffusion=False,
            start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
            self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):     # blend noise into state variables
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x, cond, t):
        x_start, model_offsets, _ = x
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss_simple = self.loss_fn(model_out, target, reduction="none")
        loss_simple = loss_simple * extract(self.p2_loss_weight, t, loss_simple.shape)

        loss_vel = self.loss_fn(model_out[:, 1:, 3:] - model_out[:, :-1, 3:], target[:, 1:, 3:] - target[:, :-1, 3:], reduction="none")
        loss_vel = loss_vel * extract(self.p2_loss_weight, t, loss_vel.shape)

        loss_drift = self.loss_fn(model_out[..., :3], target[..., :3], reduction="none")
        loss_drift = loss_drift * extract(self.p2_loss_weight, t, loss_drift.shape)

        osim_states_pred = self.normalizer.unnormalize(model_out)
        osim_states_pred = inverse_convert_addb_state_to_model_input(osim_states_pred, self.opt.model_states_column_names, self.opt.joints_3d, self.opt.osim_dof_columns)
        foot_locations_pred, joint_locations_pred, segment_orientations_pred = forward_kinematics(osim_states_pred, model_offsets)
        osim_states_true = self.normalizer.unnormalize(target)
        osim_states_true = inverse_convert_addb_state_to_model_input(osim_states_true, self.opt.model_states_column_names, self.opt.joints_3d, self.opt.osim_dof_columns)
        foot_locations_true, joint_locations_true, segment_orientations_true = forward_kinematics(osim_states_true, model_offsets)

        loss_fk = self.loss_fn(joint_locations_pred, joint_locations_true, reduction="none")
        # loss_fk = reduce(loss_fk, "b ... -> b (...)", "mean")
        loss_fk = loss_fk * extract(self.p2_loss_weight, t, loss_fk.shape[1:])

        # loss_floor_penetration = self.loss_fn(foot_locations_pred[..., 1], foot_locations_true[..., 1], reduction="none")
        # loss_floor_penetration = loss_floor_penetration * extract(self.p2_loss_weight, t, loss_floor_penetration.shape)

        foot_acc_pred = (foot_locations_pred[..., 2:, :] - 2 * foot_locations_pred[..., 1:-1, :] + foot_locations_pred[..., :-2, :]).abs() * 10000
        stance_based_on_foot_vel = (torch.norm(foot_acc_pred, dim=-1) < 0.3)[..., None].expand(-1, -1, -1, 3)       # TODO tune this. (0.3 m/s2)
        foot_acc_pred[~stance_based_on_foot_vel] = 0
        loss_slide = self.loss_fn(foot_acc_pred, foot_acc_pred * 0, reduction="none")
        loss_slide = loss_slide * extract(self.p2_loss_weight, t, loss_slide.shape[1:])

        losses = [
            1. * loss_simple.mean(),         # TODO tune this
            0 * loss_vel.mean(),
            1. * loss_fk.mean(),
            1. * loss_drift.mean(),
            # 1. * loss_floor_penetration.mean(),
            100. * loss_slide.mean()]
        return sum(losses), losses + [loss_simple]

    def loss(self, x, cond, t_override=None):
        batch_size = len(x[0])
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x[0].device).long()        # randomly select a timestep to train
        else:
            t = torch.full((batch_size,), t_override, device=x[0].device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def generate_samples(
            self,
            shape,
            cond,
            normalizer,
            opt,
            mode="normal",
            noise=None,
            constraint=None,
            start_point=None,
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_ddim_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples.detach().cpu())
        samples = samples.detach().cpu()

        samples = inverse_convert_addb_state_to_model_input(samples, opt.model_states_column_names,
                                                            opt.joints_3d, opt.osim_dof_columns)
        return samples














