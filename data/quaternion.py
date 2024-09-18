import torch
from data.rotation_conversions import (axis_angle_to_matrix, matrix_to_axis_angle, euler_angles_to_matrix,
                                       matrix_to_quaternion, matrix_to_rotation_6d, matrix_to_euler_angles,
                                       quaternion_to_matrix, rotation_6d_to_matrix)
import numpy as np


def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def euler_to_6v(q, convention="XYZ"):
    assert q.shape[-1] == 3
    mat = euler_angles_to_matrix(q, convention)
    mat = matrix_to_rotation_6d(mat)
    return mat


def euler_to_angular_velocity_lie_algebra(q, sampling_fre, convention="XYZ"):
    """ A slower version of euler_to_angular_velocity but more accurate """
    assert q.shape[-1] == 3
    mat = euler_angles_to_matrix(q, convention)

    angular_velocity = []
    for i_sample in range(mat.shape[0]-1):
        R_one_sample = torch.mm(mat[i_sample, :, :], mat[i_sample+1, :, :].T)

        cos_value = torch.clamp((torch.trace(R_one_sample) - 1) / 2, -1, 1)
        theta = torch.acos(cos_value)
        a, b = torch.linalg.eig(R_one_sample)
        for i_eig in range(a.__len__()):
            if abs(a[i_eig].imag) < 1e-12:
                rotation_vec_in_world_frame = b[:, i_eig].real
                break
            if i_eig == a.__len__():
                raise RuntimeError('no eig')

        # check the direction of the rotation axis
        if (R_one_sample[2, 1] - R_one_sample[1, 2]) * rotation_vec_in_world_frame[0] < 0:
            rotation_vec_in_world_frame = -rotation_vec_in_world_frame
        angular_velocity.append(theta * rotation_vec_in_world_frame)
    angular_velocity.append(angular_velocity[-1])
    angular_velocity = torch.stack(angular_velocity, dim=0) * sampling_fre
    return angular_velocity


def euler_to_angular_velocity(q, sampling_fre, convention="XYZ"):
    assert q.shape[-1] == 3
    mat = euler_angles_to_matrix(q, convention)
    mat_diff = mat.clone()
    mat_diff[:-1, :, :] = mat_diff[:-1, :, :] - mat_diff[1:, :, :]
    mat_diff[-1, :, :] = mat_diff[-2, :, :]

    angular_velocity = torch.stack([mat_diff[:, 2, 1] - mat_diff[:, 1, 2],
                                    mat_diff[:, 0, 2] - mat_diff[:, 2, 0],
                                    mat_diff[:, 1, 0] - mat_diff[:, 0, 1]], dim=1) * sampling_fre * 0.5
    return angular_velocity


def angular_velocity_to_euler(angular_velocity, sampling_fre, convention="XYZ"):
    R = torch.eye(3)
    mat = [R]
    for i_sample in range(angular_velocity.shape[0] - 1):
        R_one_frame = torch.tensor([[0, -angular_velocity[i_sample, 2], angular_velocity[i_sample, 1]],
                                    [angular_velocity[i_sample, 2], 0, -angular_velocity[i_sample, 0]],
                                    [-angular_velocity[i_sample, 1], angular_velocity[i_sample, 0], 0]])
        R = R - R_one_frame / sampling_fre
        mat.append(R)
    mat = torch.stack(mat, dim=0)
    euler = matrix_to_euler_angles(mat, convention)
    return euler


def euler_from_6v(q, convention="XYZ"):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    eul = matrix_to_euler_angles(mat, convention)
    return eul


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 4)
    :param y: quaternion tensor (N, S, J, 4)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """
    len = torch.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y

    return res
