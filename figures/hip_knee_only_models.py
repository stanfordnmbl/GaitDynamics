import os
import inspect, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from args import parse_opt
from consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE
from consts import WEIGHT_KG_OVERWRITE, KINETICS_ALL, OVERGROUND_DSETS
from data.osim_fk import get_model_offsets
from data.preprocess import Normalizer
from figures.da_grf_test_set_tab import combine_splits, dset_data_profile_to_peak
from model.adan import Adan
from model.model import BaselineModel
from data.addb_dataset import MotionDataset, TrialData, WindowData
from model.utils import *
import numpy as np
import nimblephysics as nimble
import torch, pickle
from tqdm import tqdm
from typing import List, Any
import more_itertools as mit
import wandb
from torch.nn import functional as F
from inspect import isfunction
from math import log, pi
from einops import rearrange, repeat
from torch import einsum, nn


def exists(val):
    return val is not None


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert (
            rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                    theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.cache = dict()

        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def rotate_queries_or_keys(self, t, seq_dim=-2):
        device = t.device
        seq_len = t.shape[seq_dim]
        freqs = self.forward(
            lambda: torch.arange(seq_len, device=device), cache_key=seq_len
        )
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=512, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.rotary = RotaryEmbedding(dim=d_model)
        self.use_rotary = self.rotary is not None

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        attn_output, _ = self.self_attn(qk, qk, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoderArchitecture(nn.Module):
    def __init__(self, repr_dim, opt, nlayers=6):
        super(TransformerEncoderArchitecture, self).__init__()
        self.input_dim = len(opt.kinematic_diffusion_col_loc)
        self.output_dim = repr_dim - self.input_dim
        embedding_dim = 192
        self.input_to_embedding = nn.Linear(self.input_dim, embedding_dim)
        self.encoder_layers = nn.Sequential(*[EncoderLayer(embedding_dim) for _ in range(nlayers)])
        self.embedding_to_output = nn.Linear(embedding_dim, self.output_dim)
        self.opt = opt
        self.input_col_loc = opt.kinematic_diffusion_col_loc
        self.output_col_loc = [i for i in range(repr_dim) if i not in self.input_col_loc]

    def get_optimizer(self):
        return Adan(self.parameters(), lr=4e-4, weight_decay=0.02)

    def loss_fun(self, output_pred, output_true):
        return F.mse_loss(output_pred, output_true, reduction='none')

    def end_to_end_prediction(self, x):
        input = x[0][:, :, self.input_col_loc]
        sequence = self.input_to_embedding(input)
        sequence = self.encoder_layers(sequence)
        output_pred = self.embedding_to_output(sequence)
        return output_pred

    def predict_samples(self, x, constraint):
        x[0] = x[0] * constraint['mask']
        output_pred = self.end_to_end_prediction(x)
        x[0][:, :, self.output_col_loc] = output_pred
        return x[0]

    def __str__(self):
        return 'tf'


class DatasetOnlyKnee(MotionDataset):
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
        # self.guess_vel_and_replace_txtytz()

        self.trial_num = len(self.trials)
        if self.trial_num == 0:
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
        print(f"In total, {self.trial_num} trials, {total_hour} hours, {total_clip_num} clips not considering overlapping")

        if train:
            data_concat = torch.cat([trial_.converted_pose for trial_ in self.trials], dim=0)
            print("normalizing training data")

            self.normalizer = Normalizer(data_concat, range(data_concat.shape[1]))            # Norm center and force

        else:
            self.normalizer = normalizer
        for i in range(self.trial_num):
            self.trials[i].converted_pose = self.normalizer.normalize(self.trials[i].converted_pose).clone().detach().float()

    def divide_jittery_trials(self, trials, converted_column_names):
        """ If one sample have abnormal acceleration or angular velocity, divide the trial into two from this sample."""
        angular_v_limit = np.deg2rad(2000)          # 2000 deg/s
        rot_mat_angular_v_limit = angular_v_limit               # use the same limit because sin(x) ~= x for small x

        rot_mat_col = [col for col in converted_column_names if '_vel' not in col and
                       ('_0' in col or '_1' in col or '_2' in col or '_3' in col or '_4' in col or '_5' in col)]
        rot_mat_col_loc = [converted_column_names.index(col) for col in rot_mat_col]
        euler_angle_col = [col for col in converted_column_names if '_vel' not in col and
                           ('angle' in col or 'elbow' in col or 'pro_sup' in col or 'wrist' in col)]
        euler_angle_col_loc = [converted_column_names.index(col) for col in euler_angle_col]

        trial_data_list = []
        for current_trial in trials:
            vel, acc = update_d_dd(current_trial.converted_pose, 1 / self.target_sampling_rate)
            euler_angle_break_point = np.where(vel[:, euler_angle_col_loc] > angular_v_limit)[0]
            rot_mat_break_point = np.where(vel[:, rot_mat_col_loc] > rot_mat_angular_v_limit)[0]
            break_point_list = list(set(np.concatenate([euler_angle_break_point, rot_mat_break_point]).flatten()))
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

                print('Divided {} {} into {} trials, {} due to euler angle vel, {} due to rot mat vel'.format(
                    current_trial.dset_name, current_trial.sub_and_trial_name,
                    len(trial_data_list), len(euler_angle_break_point), len(rot_mat_break_point)))
            else:
                trial_data_list.append(current_trial)
        return trial_data_list


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
                cops = np.array([frame.groundContactCenterOfPressure for frame in first_passes])
                if self.restrict_contact_bodies and len(forces[0]) != 6:
                    continue        # only include data with 2 contact bodies.
                else:
                    r_foot_idx, l_foot_idx = contact_bodies.index('calcn_r'), contact_bodies.index('calcn_l')
                    foot_idx = [r_foot_idx*3, r_foot_idx*3+1, r_foot_idx*3+2, l_foot_idx*3, l_foot_idx*3+1, l_foot_idx*3+2]
                    contact_bodies = ['calcn_r', 'calcn_l']
                    forces = forces[:, foot_idx]
                    cops = cops[:, foot_idx]

                # This is to compensate an AddBiomechanics bug that first GRF is always 0.
                if len(poses) < 10:
                    continue
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
                    continue

                grf_flag_counts = list(mit.run_length.encode(probably_missing))

                if len(grf_flag_counts) > 30:
                    # print(f'Compensating an AddB bug, {dset_name} - {subject_name} - {trial_id} notMissingGRF flag abnormally'
                    #       f' flipped {len(grf_flag_counts)} times, thus setting all to True.', end='')
                    probably_missing = [False] * len(probably_missing)

                states = norm_cops(skel, states, opt, weight_kg, height_m, sampling_rate)
                if states is False:
                    print(f'{sub_and_trial_name} has CoP far away from foot, skipping')
                    continue

                if self.align_moving_direction_flag:
                    states, rot_mat = align_moving_direction(states, opt.osim_dof_columns)
                else:
                    rot_mat = torch.eye(3).float()

                if (not self.include_trials_shorter_than_window_len) and (states.shape[0] / sampling_rate * self.target_sampling_rate) < self.window_len + 2:
                    continue
                if states.shape[0] < 20:        # need to be longer than 20 frames for filtering
                    continue
                if sampling_rate != self.target_sampling_rate:
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

                states_df = states_df[[ele for ele in columns_to_keep if '_vel' not in ele]]

                vel_col_loc = [i for i, col in enumerate(states_df.columns) if not np.sum([term in col for term in ['force']])]
                vel_col_names = [f'{col}_vel' for i, col in enumerate(states_df.columns) if not np.sum([term in col for term in ['force']])]
                kinematics_np = states_df.iloc[:, vel_col_loc].to_numpy().copy()
                kinematics_np_filtered = data_filter(kinematics_np, 15, opt.target_sampling_rate, 4)
                kinematics_vel = np.stack([spline_fitting_1d(kinematics_np_filtered[:, i_col], range(kinematics_np_filtered.shape[0]), 1).ravel()
                                           for i_col in range(kinematics_np_filtered.shape[1])]).T
                states_df = pd.DataFrame(np.concatenate([states_df.values, kinematics_vel], axis=1), columns=list(states_df.columns)+vel_col_names)


                assert self.opt.model_states_column_names == list(states_df.columns)
                converted_states = torch.tensor(states_df.values).float()

                trial_data = TrialData(converted_states, probably_missing, model_offsets, contact_bodies,
                                       sub_and_trial_name, trial_id, subject.getHeightM(), subject.getMassKg(),
                                       dset_name, rot_mat, [0, 0, 0], self.window_len, cond, mtp_r_vel, mtp_l_vel)
                if self.include_trials_shorter_than_window_len or len(trial_data.available_win_start) > 0:
                    self.trials.append(trial_data)
                self.dset_set.add(dset_name)

                if max_trial_num is not None and len(self.trials) >= max_trial_num:
                    return
            print('Current trial num: {}'.format(len(self.trials)))
        # self.trial_length_probability = torch.tensor([1000 / trial.length for trial in self.trials]).float()

    def get_wins_step_len_one_sample(self, col_loc_to_unmask):
        windows = []
        for i_trial in range(len(self.trials)):
            trial_ = self.trials[i_trial]
            trial_len = trial_.converted_pose.shape[0]
            for i_sample in range(0, trial_len - self.opt.window_len + 1):
                mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
                mask[:, col_loc_to_unmask] = 1
                data_ = trial_.converted_pose[i_sample:i_sample+self.opt.window_len, ...]
                windows.append(WindowData(data_, trial_.model_offsets, i_trial,
                                          None, mask, trial_.height_m, trial_.weight_kg, trial_.cond))
        return windows


def inverse_convert_addb_state_to_model_input(model_states, model_states_column_names, joints_3d, osim_dof_columns, pos_vec, height_m, sampling_fre=100):
    model_states_dict = {col: model_states[..., i] for i, col in enumerate(model_states_column_names) if
                         col in osim_dof_columns}
    osim_states = torch.stack([model_states_dict[col] for col in osim_dof_columns], dim=len(model_states.shape)-1).float()
    return osim_states


def train_model(columns_to_keep):
    # for key_, columns_to_keep in columns_to_keep_dict.items():
    model = BaselineModel(opt, TransformerEncoderArchitecture)

    if opt.log_with_wandb:
        wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, dir="wandb_logs")
        wandb.watch(model.diffusion, F.mse_loss, log='all', log_freq=200)

    train_dataset = DatasetOnlyKnee(
        data_path=opt.data_path_train,
        train=True,
        # trial_start_num=-1,
        # max_trial_num=1,            # !!!
        dset_keyworks_to_exclude=['Fregly2012', 'Uhlrich2023', 'Han2023'],
        # dset_keyworks_to_exclude=['Carter2023', 'Fregly2012', 'Falisse2017', 'Hamner2013', 'Han2023', 'Li2021', 'Santos2017', 'Tan2021', 'Uhlrich2023', 'Wang2023'],
        opt=opt,
    )
    model.train_loop(opt, train_dataset)


def loop_all(model, opt, skels, trials, windows):
    state_pred_list = []
    for i_win in range(0, len(windows), opt.batch_size_inference):
        state_true = torch.stack([win.pose for win in windows[i_win:i_win+opt.batch_size_inference]])
        masks = torch.stack([win.mask for win in windows[i_win:i_win+opt.batch_size_inference]])
        cond = torch.stack([win.cond for win in windows[i_win:i_win+opt.batch_size_inference]])

        state_pred_list_batch = model.eval_loop(opt, state_true, masks, cond=cond, num_of_generation_per_window=1)
        state_pred_list += state_pred_list_batch[0]

    # state_pred_list_averaged, state_pred_list_std = [], []
    # for i_win in range(len(state_pred_list[0])):
    #     win_skels = [state_pred_list[i_skel][i_win] for i_skel in range(skel_num-1)]
    #     averaged = torch.mean(torch.stack(win_skels), dim=0)
    #     std = torch.std(torch.stack(win_skels), dim=0)
    #     state_pred_list_averaged.append(averaged)
    #     state_pred_list_std.append(std)

    results_true, results_pred, results_pred_std = {}, {}, {}
    heights_weights = {}
    for i_win, (win, state_pred) in enumerate(zip(windows, state_pred_list)):
        trial = trials[win.trial_id]
        true_val = model.normalizer.unnormalize(win.pose.unsqueeze(0))[0].numpy()
        mask = win.mask.squeeze().numpy()
        true_val = true_val * np.bool_(mask.sum(axis=1)).repeat(true_val.shape[1]).reshape((150, -1))
        state_pred = state_pred * np.bool_(mask.sum(axis=1)).repeat(true_val.shape[1]).reshape((150, -1))

        if trial.sub_and_trial_name not in results_true.keys():
            results_true.update({trial.sub_and_trial_name: []})
            results_pred.update({trial.sub_and_trial_name: []})
            # results_pred_std.update({trial.sub_and_trial_name: []})
            heights_weights[trial.sub_and_trial_name] = win.height_m, win.weight_kg
            results_true[trial.sub_and_trial_name].append(true_val)
            results_pred[trial.sub_and_trial_name].append(state_pred.numpy())
        else:
            results_true[trial.sub_and_trial_name].append(true_val[-1:])
            results_pred[trial.sub_and_trial_name].append(state_pred.numpy()[-1:])

    true_all, pred_all = [], []
    for sub_and_trial in results_true.keys():
        # trial_len = [trial.converted_pose.shape[0] for trial in trials if trial.sub_and_trial_name == sub_and_trial][0]
        # results_true[sub_and_trial], _ = convert_overlapped_list_to_array(
        #     trial_len, results_true[sub_and_trial], results_s[sub_and_trial], results_e[sub_and_trial])
        # results_pred[sub_and_trial], results_pred_std[sub_and_trial] = convert_overlapped_list_to_array(
        #     trial_len, results_pred[sub_and_trial], results_s[sub_and_trial], results_e[sub_and_trial])
        results_true[sub_and_trial] = np.concatenate(results_true[sub_and_trial], axis=0)
        results_pred[sub_and_trial] = np.concatenate(results_pred[sub_and_trial], axis=0)
        # results_pred_std[sub_and_trial] = np.concatenate(results_pred_std[sub_and_trial], axis=0)

        sub_name = sub_and_trial.split('__')[0]
        skel_list = [skel for dset_sub_name, skel in skels.items() if sub_name == dset_sub_name[-len(sub_name):]]
        assert len(skel_list) == 1
        skel = skel_list[0]

        # height_m_tensor = torch.tensor([heights_weights[sub_and_trial][0]])
        for results_ in [results_true, results_pred]:
            # results_[sub_and_trial] = inverse_convert_addb_state_to_model_input(
            #     torch.from_numpy(results_[sub_and_trial]).unsqueeze(0), opt.model_states_column_names,
            #     opt.joints_3d, opt.osim_dof_columns, [0, 0, 0], height_m_tensor)[0].numpy()
            # pelvis_col_loc = [opt.osim_dof_columns.index(col) for col in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]
            # results_[sub_and_trial][:, pelvis_col_loc] = np.diff(results_[sub_and_trial][:, pelvis_col_loc], prepend=0, axis=0) * opt.target_sampling_rate

            osim_state_list = []
            for _, dof in enumerate(opt.osim_dof_columns):
                if dof in opt.model_states_column_names:
                    data_ = results_[sub_and_trial][:, opt.model_states_column_names.index(dof)]
                    osim_state_list.append(data_)
                else:
                    # osim_states = np.zeros([results_[sub_and_trial].shape[0], 1])
                    osim_state_list.append(np.zeros([results_[sub_and_trial].shape[0]]))
            results_[sub_and_trial] = np.stack(osim_state_list, axis=-1)

            results_[sub_and_trial] = inverse_norm_cops(skel, results_[sub_and_trial], opt, heights_weights[sub_and_trial][1], heights_weights[sub_and_trial][0])
            # params, param_columns = osim_states_to_moments_in_percent_BW_BH_via_cross_product(results_[sub_and_trial], skel, opt, heights_weights[sub_and_trial][0])
            # results_[sub_and_trial] = np.concatenate([results_[sub_and_trial], params], axis=-1)

        for trial in trials:
            if trial.sub_and_trial_name == sub_and_trial:
                probably_missing = trial.probably_missing
                break
        true_all.append(results_true[sub_and_trial][np.where(probably_missing==0)[0]])
        pred_all.append(results_pred[sub_and_trial][np.where(probably_missing==0)[0]])
    # column_names = opt.osim_dof_columns
    return true_all, pred_all, opt.osim_dof_columns


def load_test_dataset_dict(model):
    dset_specific_trial = {dset: None for dset in DATASETS_NO_ARM}
    dset_specific_trial['Falisse2017_Formatted_No_Arm'] = 'Gait'
    dset_specific_trial['Li2021_Formatted_No_Arm'] = ['Trial25', 'Trial26']     # Other trials do not have valid pelvis angles
    dset_specific_trial['Han2023_Formatted_No_Arm'] = 'walk'
    dset_specific_trial['Uhlrich2023_Formatted_No_Arm'] = 'walking'
    dset_specific_trial['Wang2023_Formatted_No_Arm'] = ['walk', 'run']

    test_dataset_dict = {}
    for dset in DATASETS_NO_ARM:
        if dset in ['Santos2017_Formatted_No_Arm']:
            continue
        print(dset)
        test_dataset = DatasetOnlyKnee(
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


def evaluate(key_):
    if key_ == 'with_adduction':
        model_folder = 'with_adduction'
    else:
        model_folder = 'no_adduction'

    if opt.use_server:
        opt.checkpoint_bl = opt.data_path_parent + f"/../code/runs/train/{model_folder}/weights/{'train-7680_tf.pt'}"
    else:
        opt.checkpoint_bl = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/{'train-6912_tf.pt'}"

    model = BaselineModel(opt, TransformerEncoderArchitecture, EMA=True)
    test_dataset_dict = load_test_dataset_dict(model)

    true_sub_dict, pred_sub_dict, pred_std_sub_dict = {}, {}, {}
    for dset, test_dataset in test_dataset_dict.items():
        windows = test_dataset.get_wins_step_len_one_sample(opt.kinematic_diffusion_col_loc)
        if len(windows) == 0:
            continue
        true_sub, pred_sub, column_names = loop_all(model, opt, test_dataset_dict[dset].skels, test_dataset.trials, windows)

        col_loc_to_save = [column_names.index(col) for col in params_of_interest]
        true_sub_dict[dset] = [trial_[:, col_loc_to_save] for trial_ in true_sub]
        pred_sub_dict[dset] = [trial_[:, col_loc_to_save] for trial_ in pred_sub]
    results_ = [true_sub_dict, pred_sub_dict, None, params_of_interest]
    pickle.dump(results_, open(f"/dataNAS/people/alanttan/mfm/code/figures/results/{folder}/{key_}.pkl", "wb"))


def get_all_the_metrics(model_key):
    results_ = pickle.load(open(f"/dataNAS/people/alanttan/mfm/code/figures/results/{model_key}.pkl", "rb"))
    results_ = combine_splits(results_)
    true_, pred_, _, columns = results_
    dset_list = list(true_.keys())
    param_pattern_and_ratio = {'calcn_l_force_v': 100 / 9.81, 'calcn_l_force_normed_cop': 100, 'moment': 1}
    params_of_interest_profile = [
        'calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz',
    ]
    [params_of_interest_profile.extend([param]) for param in
     ['calcn_l_force_normed_cop_x', 'calcn_l_force_normed_cop_z', 'knee_moment_l_x', 'knee_moment_l_z'] if
     param in columns]

    metric_all_dsets = {'dset_short': []}
    for i_dset, dset_short in enumerate(dset_list):
        metric_dset = {}
        dset = dset_short + '_Formatted_No_Arm'
        metric_all_dsets['dset_short'].append(dset_short)
        param_true_dict, param_pred_dict, gait_phase_label = dset_data_profile_to_peak(true_[dset_short], pred_[dset_short], columns, dset_short)

        for param_col in param_true_dict.keys():
            ratio = [v for k, v in param_pattern_and_ratio.items() if k in param_col]
            assert len(ratio) == 1
            # metric_mean = np.sqrt(np.mean((np.array(param_true_dict[param_col]) - np.array(param_pred_dict[param_col]))**2)) * ratio[0]
            metric_mean = np.mean(np.abs(np.array(param_true_dict[param_col]) - np.array(param_pred_dict[param_col]))) * ratio[0]
            metric_dset[param_col] = metric_mean

        true_concat = np.concatenate(true_[dset_short], axis=0)
        pred_concat = np.concatenate(pred_[dset_short], axis=0)
        gait_phase_label_concat = np.concatenate(gait_phase_label, axis=0)
        for i_param, param_col in enumerate(params_of_interest_profile):
            param_col_loc = columns.index(param_col)
            ratio = [v for k, v in param_pattern_and_ratio.items() if k in param_col]
            assert len(ratio) == 1
            within_gait_cycle = (gait_phase_label_concat != NOT_IN_GAIT_PHASE)
            # metric_mean = np.sqrt(np.mean((true_concat[within_gait_cycle, param_col_loc] - pred_concat[within_gait_cycle, param_col_loc])**2)) * ratio[0]
            metric_mean = np.mean(np.abs(true_concat[within_gait_cycle, param_col_loc] - pred_concat[within_gait_cycle, param_col_loc])) * ratio[0]

            if 'normed_cop' in param_col or 'moment' in param_col:
                stance_phase = (np.abs(true_concat[:, columns.index('calcn_l_force_normed_cop_x')]) > 1e-10) & (
                        np.abs(pred_concat[:, columns.index('calcn_l_force_normed_cop_x')]) > 1e-10)
                metric_mean = np.mean(np.abs(true_concat[stance_phase & within_gait_cycle, param_col_loc] -
                                             pred_concat[stance_phase & within_gait_cycle, param_col_loc])) * ratio[0]

            metric_dset[param_col] = metric_mean
        # print(dset_short)
        for param_col, metric_mean in metric_dset.items():
            # print(f'{param_col}, {metric_mean:.2f}')
            if param_col not in metric_all_dsets.keys():
                metric_all_dsets[param_col] = [metric_mean]
            else:
                metric_all_dsets[param_col].append(metric_mean)
    return metric_all_dsets


def print_table(key_):
    results_ = pickle.load(open(f"/dataNAS/people/alanttan/mfm/code/figures/results/{folder}/{key_}.pkl", "rb"))
    true_, pred_, _, columns = results_
    metric_all_dsets = get_all_the_metrics(f'{folder}/{key_}')
    dset_list = list(true_.keys())

    params_to_print = ['calcn_l_force_vy_max', 'calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    print('\t\tmax_vF\t\tvF\t\tapF\t\tmlF')

    results_array = [[] for _ in range(len(dset_list))]
    for i_dset, dset in enumerate(dset_list):
        dset_short = dset.split('_Formatted_No_Arm')[0]
        dset_index = metric_all_dsets['dset_short'].index(dset_short)
        print(dset[:9], end='\t')
        for i_param, param_col in enumerate(params_to_print):
            print(f'{metric_all_dsets[param_col][dset_index]:.1f}', end='\t\t')
            results_array[i_dset].append(metric_all_dsets[param_col][dset_index])
        print()
    results_average = np.mean(np.array(results_array), axis=0)
    print('Average', end='\t\t')
    [print(round(element, 1), end='\t\t') for element in results_average]
    print()


opt = parse_opt()
columns_to_keep_dict = {
    'with_adduction': ['hip_flexion_r', 'hip_adduction_r', 'knee_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'knee_angle_l'] + KINETICS_ALL +
                      ['hip_flexion_r_vel', 'hip_adduction_r_vel', 'knee_angle_r_vel', 'hip_flexion_l_vel', 'hip_adduction_l_vel', 'knee_angle_l_vel'],
    'no_adduction': ['hip_flexion_r', 'knee_angle_r', 'hip_flexion_l', 'knee_angle_l'] + KINETICS_ALL +
                    ['hip_flexion_r_vel', 'knee_angle_r_vel', 'hip_flexion_l_vel', 'knee_angle_l_vel'],
}
params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
max_trial_num = None     # None for all trials
folder = 'full_kee' if max_trial_num is None else 'fast_kee'
if __name__ == '__main__':
    for key_ in columns_to_keep_dict.keys():
        columns_to_keep = columns_to_keep_dict[key_]
        opt.model_states_column_names = columns_to_keep
        opt.kinematic_diffusion_col_loc = [columns_to_keep.index(col) for col in columns_to_keep if 'force' not in col]

        # train_model(columns_to_keep)

        # evaluate(key_)

        print_table(key_)



























