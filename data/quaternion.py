import torch
from data.rotation_conversions import (euler_angles_to_matrix, matrix_to_rotation_6d, 
                                       matrix_to_euler_angles, rotation_6d_to_matrix)


def euler_to_6v(q, convention="XYZ"):
    assert q.shape[-1] == 3
    mat = euler_angles_to_matrix(q, convention)
    mat = matrix_to_rotation_6d(mat)
    return mat


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


def euler_from_6v(q, convention="XYZ"):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    eul = matrix_to_euler_angles(mat, convention)
    return eul


