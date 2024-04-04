import math
import numpy as np
import torch
from torch import nn
from scipy.interpolate import interp1d
import scipy.interpolate as interpo
import random
from scipy.signal import filtfilt, butter
from alant.quaternion import euler_from_6v, euler_to_6v
from alant.alan_consts import JOINTS_3D_ALL, FROZEN_DOFS
from data.rotation_conversions import euler_angles_to_matrix, matrix_to_euler_angles


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


def align_moving_direction(poses, column_names):
    pose_clone = torch.from_numpy(poses).clone().float()
    pelvis_orientation_col_loc = [column_names.index(col) for col in JOINTS_3D_ALL['pelvis']]
    p_pos_col_loc = [column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
    r_grf_col_loc = [column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
    l_grf_col_loc = [column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]

    if len(pelvis_orientation_col_loc) != 3 or len(p_pos_col_loc) != 3 or len(r_grf_col_loc) != 3 or len(l_grf_col_loc) != 3:
        raise ValueError('check column names')

    pelvis_orientation = pose_clone[:, pelvis_orientation_col_loc]
    pelvis_orientation = euler_angles_to_matrix(pelvis_orientation, "ZXY")
    p_pos = pose_clone[:, p_pos_col_loc]
    r_grf = pose_clone[:, r_grf_col_loc]
    l_grf = pose_clone[:, l_grf_col_loc]

    angle = math.atan2(- pelvis_orientation[0][0, 2], pelvis_orientation[0][2, 2])
    rot_mat = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]).float()

    pelvis_orientation_rotated = torch.matmul(rot_mat, pelvis_orientation)
    p_pos_rotated = torch.matmul(rot_mat, p_pos.unsqueeze(2)).squeeze(2)
    r_grf_rotated = torch.matmul(rot_mat, r_grf.unsqueeze(2)).squeeze(2)
    l_grf_rotated = torch.matmul(rot_mat, l_grf.unsqueeze(2)).squeeze(2)

    pose_clone[:, pelvis_orientation_col_loc] = matrix_to_euler_angles(pelvis_orientation_rotated.float(), "ZXY")
    pose_clone[:, p_pos_col_loc] = p_pos_rotated.float()
    pose_clone[:, r_grf_col_loc] = r_grf_rotated.float()
    pose_clone[:, l_grf_col_loc] = l_grf_rotated.float()

    return pose_clone, rot_mat


def inverse_align_moving_direction(poses, column_names, rot_mat_to_reset):
    pose_clone = torch.from_numpy(poses).clone().float()
    pelvis_orientation_col_loc = [column_names.index(col) for col in JOINTS_3D_ALL['pelvis']]
    p_pos_col_loc = [column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
    r_grf_col_loc = [column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
    l_grf_col_loc = [column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]

    if len(pelvis_orientation_col_loc) != 3 or len(p_pos_col_loc) != 3 or len(r_grf_col_loc) != 3 or len(l_grf_col_loc) != 3:
        raise ValueError('check column names')

    pelvis_orientation = pose_clone[:, pelvis_orientation_col_loc]
    pelvis_orientation = euler_angles_to_matrix(pelvis_orientation, "ZXY")
    p_pos = pose_clone[:, p_pos_col_loc]
    r_grf = pose_clone[:, r_grf_col_loc]
    l_grf = pose_clone[:, l_grf_col_loc]

    pelvis_orientation_rotated = torch.matmul(rot_mat_to_reset, pelvis_orientation)
    p_pos_rotated = torch.matmul(rot_mat_to_reset, p_pos.unsqueeze(2)).squeeze(2)
    r_grf_rotated = torch.matmul(rot_mat_to_reset, r_grf.unsqueeze(2)).squeeze(2)
    l_grf_rotated = torch.matmul(rot_mat_to_reset, l_grf.unsqueeze(2)).squeeze(2)

    pose_clone[:, pelvis_orientation_col_loc] = matrix_to_euler_angles(pelvis_orientation_rotated.float(), "ZXY")
    pose_clone[:, p_pos_col_loc] = p_pos_rotated.float()
    pose_clone[:, r_grf_col_loc] = r_grf_rotated.float()
    pose_clone[:, l_grf_col_loc] = l_grf_rotated.float()

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
    x_padded = np.pad(x, (N//2, N-1-N//2), mode='edge')
    cumsum = np.cumsum(np.insert(x_padded, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def from_foot_loc_to_foot_vel(mtp_loc, foot_grf, sampling_rate):
    grf_thd = 2           # 2 times of body mass Kg
    foot_vel = np.diff(mtp_loc, axis=0) * sampling_rate
    foot_vel = np.concatenate([foot_vel, foot_vel[-1][None, :]], axis=0)
    low_grf_loc = foot_grf < grf_thd
    foot_vel[low_grf_loc, :] = np.nan
    # nans, x = nan_helper(foot_vel)
    # if not sum(nans[:, 0]) == foot_vel.shape[0]:
    #     foot_vel[nans] = np.interp(x(nans), x(~nans), foot_vel[~nans])
    return foot_vel


def linear_resample_data(trial_data, original_fre, target_fre):
    x, step = np.linspace(0., 1., trial_data.shape[0], retstep=True)
    new_x = np.arange(0., 1., step*original_fre/target_fre)
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


def resample_via_spline_fitting(data_, step_to_resample):
    tck, step = interpo.splprep(data_[:, :].T, u=data_[:, 0], s=0)
    data_resampled = interpo.splev(step_to_resample, tck, der=0)
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
