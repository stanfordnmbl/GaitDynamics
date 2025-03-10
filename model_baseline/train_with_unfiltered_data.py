import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from args import parse_opt
from model.model import MotionModel, BaselineModel, TransformerEncoderArchitecture
from data.addb_dataset import MotionDataset
from torch.nn import functional as F
import wandb
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
from model.utils import linear_resample_data, align_moving_direction, from_foot_loc_to_foot_vel, \
    convert_addb_state_to_model_input, data_filter, norm_cops
from typing import Any, List
import nimblephysics as nimble
from consts import *
from data.osim_fk import get_model_offsets, forward_kinematics
import more_itertools as mit
from data.addb_dataset import TrialData
from data.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles


class MotionDatasetUnfiltered(MotionDataset):
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
                # if not self.is_lumbar_rotation_reasonable(np.array(states), opt.osim_dof_columns):
                #     self.num_of_excluded_trials['lumbar_rotation'] += 1
                #     continue

                grf_flag_counts = list(mit.run_length.encode(probably_missing))

                if len(grf_flag_counts) > 30:
                    # print(f'Compensating an AddB bug, {dset_name} - {subject_name} - {trial_id} notMissingGRF flag abnormally'
                    #       f' flipped {len(grf_flag_counts)} times, thus setting all to True.', end='')
                    probably_missing = [False] * len(probably_missing)

                states = norm_cops(skel, states, opt, weight_kg, height_m, self.check_cop_to_calcn_distance, 10)
                if states is False:
                    print(f'{sub_and_trial_name} has CoP far away from foot, skipping')
                    self.num_of_excluded_trials['wrong_cop'] += 1
                    continue

                if self.align_moving_direction_flag:
                    states, rot_mat = align_moving_direction(states, opt.osim_dof_columns, exclude_trial_with_large_angle_diff=False)
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


def train(opt):
    # model = MotionModel(opt)
    model = BaselineModel(opt, TransformerEncoderArchitecture)
    if opt.log_with_wandb:
        wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, dir="wandb_logs")
        wandb.watch(model.diffusion, F.mse_loss, log='all', log_freq=200)
    train_dataset = MotionDatasetUnfiltered(
        data_path=opt.data_path_train,
        train=True,
        # trial_start_num=-1,
        # max_trial_num=1,
        dset_keyworks_to_exclude=['Fregly2012', 'Uhlrich2023', 'Hamner2013', 'Han2023'],
        opt=opt,
        divide_jittery=False,
    )
    model.train_loop(opt, train_dataset)


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)






















