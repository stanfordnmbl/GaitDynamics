import copy
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.interpolate import interp1d
import scipy.interpolate as interpo
import random
from scipy.signal import filtfilt, butter
from data.quaternion import euler_from_6v, euler_to_6v
from consts import JOINTS_3D_ALL, FROZEN_DOFS
from data.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles
import matplotlib.pyplot as plt
from data.osim_fk import get_model_offsets, forward_kinematics


def cross_product_2d(a, b):
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("Input vectors must be 3-dimensional.")

    return np.array([a[:, 1]*b[:, 2] - a[:, 2]*b[:, 1],
                     a[:, 2]*b[:, 0] - a[:, 0]*b[:, 2],
                     a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]]).T


def identity(t, *args, **kwargs):
    return t


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


def get_multi_body_loc_using_nimble(body_names, skel, poses):
    body_ids = [skel.getBodyNode(name) for name in body_names]
    body_loc = []
    for i_frame in range(len(poses)):
        skel.setPositions(poses[i_frame])
        body_loc.append(np.concatenate([body_id.getWorldTransform().translation() for body_id in body_ids]))
    body_loc = np.array(body_loc)
    return body_loc


def get_multi_joint_loc_using_tom_fk(joint_names, skel, poses):
    model_offsets = get_model_offsets(skel)
    _, joint_locations_pred, joint_names_full, _ = forward_kinematics(poses, model_offsets)
    body_loc = []
    for joint_name in joint_names:
        if joint_name not in joint_names_full:
            raise ValueError(f'Joint name {joint_name} not found in the dictionary.')
        body_loc.append(joint_locations_pred[joint_names_full.index(joint_name), 0].cpu())
    body_loc = torch.concatenate(body_loc, axis=-1)
    return body_loc


def inverse_norm_cops(skel, states, opt, sub_mass, height_m):
    poses = states[:, opt.kinematic_osim_col_loc]
    forces = states[:, opt.grf_osim_col_loc]
    normed_cops = states[:, opt.cop_osim_col_loc]
    foot_loc = get_multi_body_loc_using_nimble(('calcn_r', 'calcn_l'), skel, poses)

    for i_plate in range(2):
        force_v = forces[:, 3*i_plate:3*(i_plate+1)]
        force_v[force_v == 0] = 1e-6
        vector = normed_cops[:, 3 * i_plate:3 * (i_plate + 1)] / force_v[:, 1:2] * height_m
        vector = np.nan_to_num(vector, posinf=0, neginf=0)
        vector.clip(min=-0.4, max=0.4, out=vector)      # CoP should be within 0.4 m from the foot
        cops = vector + foot_loc[:, 3*i_plate:3*(i_plate+1)]
        if isinstance(states, torch.Tensor):
            cops = torch.from_numpy(cops).to(states.dtype)
        else:
            cops = cops.astype(states.dtype)
        states[:, opt.cop_osim_col_loc[3*i_plate:3*(i_plate+1)]] = cops
    return states


def norm_cops(skel, states, opt, sub_mass, height_m):
    states = torch.from_numpy(states)
    poses = states[:, opt.kinematic_osim_col_loc]
    forces = states[:, opt.grf_osim_col_loc]
    cops = states[:, opt.cop_osim_col_loc]

    heel_centers = torch.from_numpy(get_multi_body_loc_using_nimble(('calcn_r', 'calcn_l'), skel, poses)).to(states.dtype)
    toe_centers = torch.from_numpy(get_multi_body_loc_using_nimble(('toes_r', 'toes_l'), skel, poses)).to(states.dtype)
    for i_plate in range(2):
        force_v = forces[:, 3*i_plate:3*(i_plate+1)]

        cop_to_toe = (toe_centers - cops)[:, 3*i_plate:3*(i_plate+1)]
        cop_to_heel = (heel_centers - cops)[:, 3*i_plate:3*(i_plate+1)]

        distance_to_toe = torch.norm(cop_to_toe, dim=-1)
        distance_to_heel = torch.norm(cop_to_heel, dim=-1)
        stance_phase = force_v[:, 1] > 0.5
        use_toe = distance_to_toe < distance_to_heel

        cop_to_foot = torch.zeros_like(cop_to_toe, dtype=cop_to_toe.dtype)
        cop_to_foot[use_toe] = cop_to_toe[use_toe]
        cop_to_foot[~use_toe] = cop_to_heel[~use_toe]

        cop_to_foot[:, [0, 2]] = torch.clip(cop_to_foot[:, [0, 2]], -0.1, 0.1)

        cops[use_toe&stance_phase, 3*i_plate:3*(i_plate+1)] = toe_centers[use_toe&stance_phase, 3*i_plate:3*(i_plate+1)] - cop_to_foot[use_toe&stance_phase]
        cops[(~use_toe)&stance_phase, 3*i_plate:3*(i_plate+1)] = heel_centers[(~use_toe)&stance_phase, 3*i_plate:3*(i_plate+1)] - cop_to_foot[(~use_toe)&stance_phase]

        vector = (cops - heel_centers)[:, 3*i_plate:3*(i_plate+1)]
        normed_vector = vector * force_v[:, 1:2] / height_m
        states[:, opt.cop_osim_col_loc[3*i_plate:3*(i_plate+1)]] = normed_vector.to(states.dtype)
    return states


def convert_addb_state_to_model_input(pose_df, joints_3d, sampling_fre):
    # shift root position to start in (x,y) = (0,0)
    pos_vec = [pose_df['pelvis_tx'][0], pose_df['pelvis_ty'][0], pose_df['pelvis_tz'][0]]
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

    vel_col_loc = [i for i, col in enumerate(pose_df.columns) if 'force' not in col and 'pelvis_t' not in col]
    vel_col_names = [f'{col}_vel' for i, col in enumerate(pose_df.columns) if 'force' not in col and 'pelvis_t' not in col]
    kinematics_np = pose_df.iloc[:, vel_col_loc].to_numpy().copy()
    kinematics_np_filtered = data_filter(kinematics_np, 15, sampling_fre, 4)
    kinematics_vel = np.stack([spline_fitting_1d(kinematics_np_filtered[:, i_col], range(kinematics_np_filtered.shape[0]), 1).ravel()
                               for i_col in range(kinematics_np_filtered.shape[1])]).T

    pose_vel_df = pd.DataFrame(np.concatenate([pose_df.values, kinematics_vel], axis=1), columns=list(pose_df.columns)+vel_col_names)
    return pose_vel_df, pos_vec


def inverse_convert_addb_state_to_model_input(model_states, model_states_column_names, joints_3d, osim_dof_columns, pos_vec, height_m, sampling_fre=100):
    model_states_dict = {col: model_states[..., i] for i, col in enumerate(model_states_column_names) if
                         col in osim_dof_columns}

    for i_col, col in enumerate(['pelvis_tx', 'pelvis_ty', 'pelvis_tz']):
        model_states_dict[col] = torch.cumsum(model_states_dict[col], dim=-1) / sampling_fre
    model_states_dict['pelvis_ty'] = model_states_dict['pelvis_ty'] * height_m.unsqueeze(-1).expand(model_states_dict['pelvis_ty'].shape)

    # convert 6v to euler
    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_name_6v = [joint_name + '_' + str(i) for i in range(6)]
        index_ = [model_states_column_names.index(joint_name_6v[i]) for i in range(6)]
        joint_euler = euler_from_6v(model_states[..., index_], "ZXY")

        for i, joints_euler_name in enumerate(joints_with_3_dof):
            model_states_dict[joints_euler_name] = joint_euler[..., i]

    # add frozen dof back
    for frozen_col in FROZEN_DOFS:
        model_states_dict[frozen_col] = torch.zeros(model_states.shape[:len(model_states.shape)-1]).to(model_states.device)

    pos_vec = torch.tensor(pos_vec)
    pos_vec_torch = pos_vec.unsqueeze(-1).repeat(*[1 for _ in range(len(pos_vec.shape))], model_states.shape[-2]).to(model_states.device)
    for i_col, col in enumerate(['pelvis_tx', 'pelvis_ty', 'pelvis_tz']):
        model_states_dict[col] += pos_vec_torch[..., i_col, :]

    osim_states = torch.stack([model_states_dict[col] for col in osim_dof_columns], dim=len(model_states.shape)-1).float()
    return osim_states


def osim_states_to_knee_moments_in_percent_BW_BH(osim_states, skel, opt, height_m):
    if isinstance(osim_states, torch.Tensor):
        osim_states = osim_states.numpy()
    forces = osim_states[:, opt.grf_osim_col_loc]
    cops = osim_states[:, opt.cop_osim_col_loc]
    poses = osim_states[:, opt.kinematic_osim_col_loc]
    knee_loc = get_multi_joint_loc_using_tom_fk(['knee_r', 'knee_l'], skel, poses)
    vector = (knee_loc - cops).numpy()
    knee_moment_r = cross_product_2d(vector[:, :3], forces[:, :3]) / (height_m * 9.81) * 100
    knee_moment_l = cross_product_2d(vector[:, 3:], forces[:, 3:]) / (height_m * 9.81) * 100
    knee_moment_r[:, 0] = -knee_moment_r[:, 0]
    knee_moment_l[:, 0] = knee_moment_l[:, 0]
    moments = np.concatenate([knee_moment_r, knee_moment_l], axis=-1)
    moment_names = ['knee_moment_r_x', 'knee_moment_r_y', 'knee_moment_r_z', 'knee_moment_l_x', 'knee_moment_l_y', 'knee_moment_l_z']
    return moments, moment_names


def align_moving_direction(poses, column_names):
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses)
    pose_clone = poses.clone().float()
    pelvis_orientation_col_loc = [column_names.index(col) for col in JOINTS_3D_ALL['pelvis']]
    p_pos_col_loc = [column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
    r_grf_col_loc = [column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
    l_grf_col_loc = [column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]
    r_cop_col_loc = [column_names.index(col) for col in [f'calcn_r_force_normed_cop_{x}' for x in ['x', 'y', 'z']]]
    l_cop_col_loc = [column_names.index(col) for col in [f'calcn_l_force_normed_cop_{x}' for x in ['x', 'y', 'z']]]

    if len(pelvis_orientation_col_loc) != 3 or len(p_pos_col_loc) != 3 or len(r_grf_col_loc) != 3 or len(
            l_grf_col_loc) != 3:
        raise ValueError('check column names')

    pelvis_orientation = pose_clone[:, pelvis_orientation_col_loc]
    pelvis_orientation = euler_angles_to_matrix(pelvis_orientation, "ZXY")
    p_pos = pose_clone[:, p_pos_col_loc]
    r_grf = pose_clone[:, r_grf_col_loc]
    l_grf = pose_clone[:, l_grf_col_loc]
    r_cop = pose_clone[:, r_cop_col_loc]
    l_cop = pose_clone[:, l_cop_col_loc]

    angle = math.atan2(- pelvis_orientation[0][0, 2], pelvis_orientation[0][2, 2])
    rot_mat = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]).float()

    pelvis_orientation_rotated = torch.matmul(rot_mat, pelvis_orientation)
    p_pos_rotated = torch.matmul(rot_mat, p_pos.unsqueeze(2)).squeeze(2)
    r_grf_rotated = torch.matmul(rot_mat, r_grf.unsqueeze(2)).squeeze(2)
    l_grf_rotated = torch.matmul(rot_mat, l_grf.unsqueeze(2)).squeeze(2)
    r_cop_rotated = torch.matmul(rot_mat, r_cop.unsqueeze(2)).squeeze(2)
    l_cop_rotated = torch.matmul(rot_mat, l_cop.unsqueeze(2)).squeeze(2)

    pose_clone[:, pelvis_orientation_col_loc] = matrix_to_euler_angles(pelvis_orientation_rotated.float(), "ZXY")
    pose_clone[:, p_pos_col_loc] = p_pos_rotated.float()
    pose_clone[:, r_grf_col_loc] = r_grf_rotated.float()
    pose_clone[:, l_grf_col_loc] = l_grf_rotated.float()
    pose_clone[:, r_cop_col_loc] = r_cop_rotated.float()
    pose_clone[:, l_cop_col_loc] = l_cop_rotated.float()

    return pose_clone, rot_mat


def inverse_align_moving_direction(poses, column_names, rot_mat_to_reset):
    """ Has not been tested yet! """
    pose_clone = torch.from_numpy(poses).clone().float()
    pelvis_orientation_col_loc = [column_names.index(col) for col in JOINTS_3D_ALL['pelvis']]
    p_pos_col_loc = [column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
    r_grf_col_loc = [column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
    l_grf_col_loc = [column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]
    r_cop_col_loc = [column_names.index(col) for col in ['calcn_r_normed_cop_x', 'calcn_r_normed_cop_y', 'calcn_r_normed_cop_z']]
    l_cop_col_loc = [column_names.index(col) for col in ['calcn_l_normed_cop_x', 'calcn_l_normed_cop_y', 'calcn_l_normed_cop_z']]

    if len(pelvis_orientation_col_loc) != 3 or len(p_pos_col_loc) != 3 or len(r_grf_col_loc) != 3 or len(
            l_grf_col_loc) != 3:
        raise ValueError('check column names')

    pelvis_orientation = pose_clone[:, pelvis_orientation_col_loc]
    pelvis_orientation = euler_angles_to_matrix(pelvis_orientation, "ZXY")
    p_pos = pose_clone[:, p_pos_col_loc]
    r_grf = pose_clone[:, r_grf_col_loc]
    l_grf = pose_clone[:, l_grf_col_loc]
    r_cop = pose_clone[:, r_cop_col_loc]
    l_cop = pose_clone[:, l_cop_col_loc]

    pelvis_orientation_rotated = torch.matmul(rot_mat_to_reset, pelvis_orientation)
    p_pos_rotated = torch.matmul(rot_mat_to_reset, p_pos.unsqueeze(2)).squeeze(2)
    r_grf_rotated = torch.matmul(rot_mat_to_reset, r_grf.unsqueeze(2)).squeeze(2)
    l_grf_rotated = torch.matmul(rot_mat_to_reset, l_grf.unsqueeze(2)).squeeze(2)
    r_cop_rotated = torch.matmul(rot_mat_to_reset, r_cop.unsqueeze(2)).squeeze(2)
    l_cop_rotated = torch.matmul(rot_mat_to_reset, l_cop.unsqueeze(2)).squeeze(2)

    pose_clone[:, pelvis_orientation_col_loc] = matrix_to_euler_angles(pelvis_orientation_rotated.float(), "ZXY")
    pose_clone[:, p_pos_col_loc] = p_pos_rotated.float()
    pose_clone[:, r_grf_col_loc] = r_grf_rotated.float()
    pose_clone[:, l_grf_col_loc] = l_grf_rotated.float()
    pose_clone[:, r_cop_col_loc] = r_cop_rotated.float()
    pose_clone[:, l_cop_col_loc] = l_cop_rotated.float()

    return pose_clone


def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered


def fix_seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def moving_average_filtering(x, N):
    x_padded = np.pad(x, (N // 2, N - 1 - N // 2), mode='edge')
    cumsum = np.cumsum(np.insert(x_padded, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def from_foot_loc_to_foot_vel(mtp_loc, foot_grf, sampling_rate, grf_thd=2):         # 2 times of body mass Kg
    foot_vel = np.diff(mtp_loc, axis=0) * sampling_rate
    foot_vel = np.concatenate([foot_vel, foot_vel[-1][None, :]], axis=0)
    low_grf_loc = foot_grf < grf_thd
    foot_vel[low_grf_loc, :] = np.nan
    return foot_vel


def linear_resample_data(trial_data, original_fre, target_fre):
    x, step = np.linspace(0., 1., trial_data.shape[0], retstep=True)
    new_x = np.arange(0., 1., step * original_fre / target_fre)
    f = interp1d(x, trial_data, axis=0)
    trial_data_resampled = f(new_x)
    return trial_data_resampled


def linear_resample_data_as_num_of_dp(data_raw, target_dp):
    x, step = np.linspace(0., 1., data_raw.shape[0], retstep=True)
    new_x = np.linspace(0., 1., target_dp)
    f = interp1d(x, data_raw, axis=0)
    data_resampled = f(new_x)
    return data_resampled


def update_d_dd(q, dt):
    dq = np.zeros(q.shape)
    dq[1:] = (q[1:] - q[:-1]) / dt
    ddq = np.zeros(q.shape)
    ddq[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) / (dt * dt)
    return dq, ddq


def spline_fitting_1d(data_, step_to_resample, der=0):
    assert len(data_.shape) == 1
    data_ = data_.reshape(1, -1)
    tck, step = interpo.splprep(data_, u=range(data_.shape[1]), s=0)
    data_resampled = interpo.splev(step_to_resample, tck, der=der)
    data_resampled = np.column_stack(data_resampled)
    return data_resampled


# absolute positional embedding used for vanilla transformer sequential data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


# very similar positional embedding used for diffusion timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# dropout mask
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()
