import numpy
import data.rotation_conversions as rotation_conversions
import torch


def second_order_poly(coeff, x):
    y = coeff[...,0] * x**2 + coeff[...,1] * x + coeff[...,2]
    return y


def batch_identity(batch_shape, size):
    batch_identity = torch.eye(size)
    output_shape = batch_shape.copy()
    output_shape.append(size)
    output_shape.append(size)
    batch_identity_out = batch_identity.view(*(1,) * (len(output_shape) - batch_identity.ndim),*batch_identity.shape).expand(output_shape)
    return batch_identity_out.clone()


def get_knee_rotation_coefficients():

    knee_Z_rotation_function = numpy.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                             1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                            [0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779,
                                             0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782]]).T
    polyfit_knee_Z_rotation = numpy.polyfit(knee_Z_rotation_function[:,0], knee_Z_rotation_function[:,1], deg=2, full = True)
    coefficients_knee_Z_rotation = polyfit_knee_Z_rotation[0]

    knee_Y_rotation_function = numpy.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                             1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                            [0, 0.059461, 0.109399, 0.150618, 0.18392, 0.210107, 0.229983, 0.24435, 0.254012, 0.25977, 0.262428, 0.262788, 0.261654]]).T
    polyfit_knee_Y_rotation = numpy.polyfit(knee_Y_rotation_function[:, 0], knee_Y_rotation_function[:, 1], deg=2, full=True)
    coefficients_knee_Y_rotation = polyfit_knee_Y_rotation[0]

    knee_X_translation_function = numpy.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                                1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                               [0, 5.3e-05, 0.000188, 0.000378, 0.000597, 0.000825, 0.001045, 0.001247, 0.00142, 0.001558, 0.001661, 0.001728, 0.00176]]).T
    polyfit_knee_X_translation = numpy.polyfit(knee_X_translation_function[:, 0], knee_X_translation_function[:, 1], deg=2,
                                               full=True)
    coefficients_knee_X_translation = polyfit_knee_X_translation[0]


    knee_Y_translation_function = numpy.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                                1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                               [0, 0.000301, 0.000143, -0.000401, -0.001233, -0.002243, -0.003316, -0.004346, -0.005239, -0.005924, -0.006361, -0.006539, -0.00648]]).T
    polyfit_knee_Y_translation = numpy.polyfit(knee_Y_translation_function[:, 0], knee_Y_translation_function[:, 1], deg=2,
                                               full=True)
    coefficients_knee_Y_translation = polyfit_knee_Y_translation[0]

    knee_Z_translation_function = numpy.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                                1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                               [0, 0.001055, 0.002061, 0.00289, 0.003447, 0.003676, 0.003559, 0.00311, 0.002373, 0.001418, 0.000329, -0.000805, -0.001898]]).T
    polyfit_knee_Z_translation = numpy.polyfit(knee_Z_translation_function[:, 0], knee_Z_translation_function[:, 1], deg=2,
                                               full=True)
    coefficients_knee_Z_translation = polyfit_knee_Z_translation[0]

    walker_knee_coefficients = numpy.stack((coefficients_knee_Y_rotation, coefficients_knee_Z_rotation, coefficients_knee_X_translation, coefficients_knee_Y_translation, coefficients_knee_Z_translation), axis=1)

    return walker_knee_coefficients


walker_knee_coefficients = get_knee_rotation_coefficients()
walker_knee_coefficients = torch.tensor(walker_knee_coefficients).to(torch.device('cuda:0'))         # Bugprone


def forward_kinematics(pose, offsets, with_arm=False):
    """
    Pose indices
    0-5: pelvis orientation + translation
    6-8: hip_r
    9: knee_r
    10: ankle_r
    11: subtalar_r
    12: mtp_r
    13-15: hip_l
    16: knee_l
    17: ankle_l
    18: subtalar_l
    19: mtp_l
    20-22: lumbar
    23-25: shoulder_r
    26: elbow_r
    27: radioulnar
    28-30: shoulder_l
    31: elbow_l
    32: radioulnar_l
    """
    offsets = offsets[:, None, ...]
    batch_shape = pose.shape[:-1]
    batch_shape_list = []
    for i in range(pose.dim()-1):
        batch_shape_list.append(int(batch_shape[i]))
    batch_shape = batch_shape_list
    if batch_shape == ():
        batch_shape = (1,)

    coefficients_knee_Y_rotation = walker_knee_coefficients[..., 0]
    coefficients_knee_Z_rotation = walker_knee_coefficients[..., 1]
    coefficients_knee_X_translation = walker_knee_coefficients[..., 2]
    coefficients_knee_Y_translation = walker_knee_coefficients[..., 3]
    coefficients_knee_Z_translation = walker_knee_coefficients[..., 4]

    knee_r_Y_rot = second_order_poly(coefficients_knee_Y_rotation, pose[..., 9])
    knee_r_Z_rot = second_order_poly(coefficients_knee_Z_rotation, pose[..., 9])
    knee_r_X_trans = second_order_poly(coefficients_knee_X_translation, pose[..., 9])
    knee_r_Y_trans = second_order_poly(coefficients_knee_Y_translation, pose[..., 9])
    knee_r_Z_trans = second_order_poly(coefficients_knee_Z_translation, pose[..., 9])

    knee_l_Y_rot = second_order_poly(coefficients_knee_Y_rotation, pose[..., 16])
    knee_l_Z_rot = second_order_poly(coefficients_knee_Z_rotation, pose[..., 16])
    knee_l_X_trans = second_order_poly(coefficients_knee_X_translation, pose[..., 16])
    knee_l_Y_trans = second_order_poly(coefficients_knee_Y_translation, pose[..., 16])
    knee_l_Z_trans = second_order_poly(coefficients_knee_Z_translation, pose[..., 16])

    # Pelvis
    pelvis_transform = batch_identity(batch_shape, 4).to(pose.device)
    pelvis_transform[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(pose[..., 0:3], 'ZXY')
    pelvis_transform[..., :3, 3] = pose[..., 3:6].clone().detach()

    # Get offsets (model and model scaling dependent)
    offset_hip_pelvis_r = offsets[..., 0]
    femur_offset_in_hip_r = offsets[..., 1]
    knee_offset_in_femur_r = offsets[..., 2]
    tibia_offset_in_knee_r = offsets[..., 3]
    ankle_offset_in_tibia_r = offsets[..., 4]
    talus_offset_in_ankle_r = offsets[..., 5]
    subtalar_offset_in_talus_r = offsets[..., 6]
    calcaneus_offset_in_subtalar_r = offsets[..., 7]
    mtp_offset_in_calcaneus_r = offsets[..., 8]
    offset_hip_pelvis_l = offsets[..., 9]
    femur_offset_in_hip_l = offsets[..., 10]
    knee_offset_in_femur_l = offsets[..., 11]
    tibia_offset_in_knee_l = offsets[..., 12]
    ankle_offset_in_tibia_l = offsets[..., 13]
    talus_offset_in_ankle_l = offsets[..., 14]
    subtalar_offset_in_talus_l = offsets[..., 15]
    calcaneus_offset_in_subtalar_l = offsets[..., 16]
    mtp_offset_in_calcaneus_l = offsets[..., 17]
    lumbar_offset_in_pelvis = offsets[..., 18]
    torso_offset_in_lumbar = offsets[..., 19]
    if with_arm:
        shoulder_offset_in_torso_r = offsets[..., 20]
        humerus_offset_in_shoulder_r = offsets[..., 21]
        elbow_offset_in_humerus_r = offsets[..., 22]
        ulna_offset_in_elbow_r = offsets[..., 23]
        radioulnar_offset_in_radius_r = offsets[..., 24]
        radius_offset_in_radioulnar_r = offsets[..., 25]
        wrist_offset_in_radius_r = offsets[..., 26]
        hand_offset_in_wrist_r = offsets[..., 27]
        shoulder_offset_in_torso_l = offsets[..., 28]
        humerus_offset_in_shoulder_l = offsets[..., 29]
        elbow_offset_in_humerus_l = offsets[..., 30]
        ulna_offset_in_elbow_l = offsets[..., 31]
        radioulnar_offset_in_radius_l = offsets[..., 32]
        radius_offset_in_radioulnar_l = offsets[..., 33]
        wrist_offset_in_radius_l = offsets[..., 34]
        hand_offset_in_wrist_l = offsets[..., 35]

    # Coordinates to transformation matrix
    hip_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    knee_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    ankle_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    subtalar_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    mtp_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    hip_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    knee_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    ankle_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    subtalar_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    mtp_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    lumbar_coordinates_transform = batch_identity(batch_shape, 4).to(pose.device)
    if with_arm:
        shoulder_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        elbow_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        radioulnar_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        wrist_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        shoulder_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
        elbow_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
        radioulnar_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
        wrist_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)

    # Knee axis translation
    knee_coordinates_transform_r[..., :3, -1] = torch.stack((knee_r_X_trans, knee_r_Y_trans, knee_r_Z_trans), dim=-1)
    knee_coordinates_transform_l[..., :3, -1] = torch.stack((knee_l_X_trans, knee_l_Y_trans, -knee_l_Z_trans), dim=-1)

    # Joint rotations
    zero_2_shape = batch_shape.copy()
    zero_2_shape.append(2)
    zero_2 = torch.zeros(tuple(zero_2_shape), device=pose.device)
    zero_3_shape = batch_shape.copy()
    zero_3_shape.append(3)
    zero_3 = torch.zeros(tuple(zero_3_shape), device=pose.device)
    hip_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix((pose[..., 6:9]), 'ZXY')
    knee_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.stack((pose[..., 9], knee_r_Y_rot, knee_r_Z_rot), dim=-1), 'XYZ')
    ankle_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat((pose[..., 10:11], zero_2), dim=-1), 'ZXY')
    subtalar_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat((pose[..., 11:12], zero_2), dim=-1), 'ZXY')
    mtp_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat((pose[..., 12:13], zero_2), dim=-1), 'ZXY')
    hip_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat(([pose[..., 13:14], -pose[..., 14:15], -pose[..., 15:16]]), dim=-1), 'ZXY')
    knee_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.stack((-pose[..., 16], -knee_l_Y_rot, knee_l_Z_rot), dim=-1), 'XYZ')
    ankle_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat((pose[..., 17:18], zero_2), dim=-1), 'ZXY')
    subtalar_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat((pose[..., 18:19], zero_2), dim=-1), 'ZXY')
    mtp_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat((pose[..., 19:20], zero_2), dim=-1), 'ZXY')
    lumbar_coordinates_transform[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
        torch.cat((pose[..., 20:21], pose[..., 21:22], pose[..., 22:23]), dim=-1), 'ZXY')
    if with_arm:
        shoulder_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            torch.cat((pose[..., 23:24], pose[..., 24:25], pose[..., 25:26]), dim=-1), 'ZXY')
        elbow_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            torch.cat((pose[..., 26:27], zero_2), dim=-1), 'ZXY')
        radioulnar_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            torch.cat((pose[..., 27:28], zero_2), dim=-1), 'ZXY')
        wrist_coordinates_transform_r[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            zero_3, 'ZXY')
        shoulder_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            torch.cat((pose[..., 28:29], -pose[..., 29:30], -pose[..., 30:31]), dim=-1), 'ZXY')
        elbow_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            torch.cat((pose[..., 31:32], zero_2), dim=-1), 'ZXY')
        radioulnar_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            torch.cat((pose[..., 32:33], zero_2), dim=-1), 'ZXY')
        wrist_coordinates_transform_l[..., :3, :3] = rotation_conversions.euler_angles_to_matrix(
            zero_3, 'ZXY')

    # Forward kinematics for the lower body
    hip_transform_r = torch.matmul(torch.matmul(pelvis_transform, offset_hip_pelvis_r), hip_coordinates_transform_r)
    femur_transform_r = torch.matmul(hip_transform_r, femur_offset_in_hip_r)
    knee_transform_r = torch.matmul(torch.matmul(femur_transform_r, knee_offset_in_femur_r),
                                    knee_coordinates_transform_r)
    tibia_transform_r = torch.matmul(knee_transform_r, tibia_offset_in_knee_r)
    ankle_transform_r = torch.matmul(torch.matmul(tibia_transform_r, ankle_offset_in_tibia_r),
                                     ankle_coordinates_transform_r)
    talus_transform_r = torch.matmul(ankle_transform_r, talus_offset_in_ankle_r)
    subtalar_transform_r = torch.matmul(torch.matmul(talus_transform_r, subtalar_offset_in_talus_r),
                                        subtalar_coordinates_transform_r)
    calcaneus_transform_r = torch.matmul(subtalar_transform_r, calcaneus_offset_in_subtalar_r)
    mtp_offset_transform_r = torch.matmul(torch.matmul(calcaneus_transform_r, mtp_offset_in_calcaneus_r),
                                          mtp_coordinates_transform_r)

    hip_transform_l = torch.matmul(torch.matmul(pelvis_transform, offset_hip_pelvis_l), hip_coordinates_transform_l)
    femur_transform_l = torch.matmul(hip_transform_l, femur_offset_in_hip_l)

    knee_transform_l = torch.matmul(torch.matmul(femur_transform_l, knee_offset_in_femur_l),
                                    knee_coordinates_transform_l)
    tibia_transform_l = torch.matmul(knee_transform_l, tibia_offset_in_knee_l)

    ankle_transform_l = torch.matmul(torch.matmul(tibia_transform_l, ankle_offset_in_tibia_l),
                                     ankle_coordinates_transform_l)
    talus_transform_l = torch.matmul(ankle_transform_l, talus_offset_in_ankle_l)

    subtalar_transform_l = torch.matmul(torch.matmul(talus_transform_l, subtalar_offset_in_talus_l),
                                        subtalar_coordinates_transform_l)
    calcaneus_transform_l = torch.matmul(subtalar_transform_l, calcaneus_offset_in_subtalar_l)

    mtp_offset_transform_l = torch.matmul(torch.matmul(calcaneus_transform_l, mtp_offset_in_calcaneus_l),
                                          mtp_coordinates_transform_l)

    # Forward kinematics for the upper body
    lumbar_transform = torch.matmul(torch.matmul(pelvis_transform, lumbar_offset_in_pelvis), lumbar_coordinates_transform)
    torso_transform = torch.matmul(lumbar_transform, torso_offset_in_lumbar)
    if with_arm:
        shoulder_transform_r = torch.matmul(torch.matmul(torso_transform, shoulder_offset_in_torso_r), shoulder_coordinates_transform_r)
        humerus_transform_r = torch.matmul(shoulder_transform_r, humerus_offset_in_shoulder_r)
        elbow_transform_r = torch.matmul(torch.matmul(humerus_transform_r, elbow_offset_in_humerus_r), elbow_coordinates_transform_r)
        ulna_transform_r = torch.matmul(elbow_transform_r, ulna_offset_in_elbow_r)
        radioulnar_transform_r = torch.matmul(torch.matmul(ulna_transform_r, radioulnar_offset_in_radius_r), radioulnar_coordinates_transform_r)
        radius_transform_r = torch.matmul(radioulnar_transform_r, radius_offset_in_radioulnar_r)
        wrist_transform_r = torch.matmul(torch.matmul(ulna_transform_r, wrist_offset_in_radius_r), wrist_coordinates_transform_r)
        hand_transform_r = torch.matmul(wrist_transform_r, hand_offset_in_wrist_r)
        shoulder_transform_l = torch.matmul(torch.matmul(torso_transform, shoulder_offset_in_torso_l), shoulder_coordinates_transform_l)
        humerus_transform_l = torch.matmul(shoulder_transform_l, humerus_offset_in_shoulder_l)
        elbow_transform_l = torch.matmul(torch.matmul(humerus_transform_l, elbow_offset_in_humerus_l), elbow_coordinates_transform_l)
        ulna_transform_l = torch.matmul(elbow_transform_l, ulna_offset_in_elbow_l)
        radioulnar_transform_l = torch.matmul(torch.matmul(ulna_transform_l, radioulnar_offset_in_radius_l), radioulnar_coordinates_transform_l)
        radius_transform_l = torch.matmul(radioulnar_transform_l, radius_offset_in_radioulnar_l)
        wrist_transform_l = torch.matmul(torch.matmul(ulna_transform_l, wrist_offset_in_radius_l), wrist_coordinates_transform_l)
        hand_transform_l = torch.matmul(wrist_transform_l, hand_offset_in_wrist_l)

    joint_locations = torch.stack((pelvis_transform[..., :3, 3],
                                   hip_transform_r[..., :3, 3], knee_transform_r[..., :3, 3],
                                   ankle_transform_r[..., :3, 3], calcaneus_transform_r[..., :3, 3],
                                   mtp_offset_transform_r[..., :3, 3],
                                   hip_transform_l[..., :3, 3], knee_transform_l[..., :3, 3],
                                   ankle_transform_l[..., :3, 3], calcaneus_transform_l[..., :3, 3],
                                   mtp_offset_transform_l[..., :3, 3]))
    if with_arm:
        joint_locations = torch.stack((*[joint_locations[i] for i in range(joint_locations.shape[0])],
                                       lumbar_transform[..., :3, 3],
                                       shoulder_transform_r[..., :3, 3],
                                       elbow_transform_r[..., :3, 3], wrist_transform_r[..., :3, 3],
                                       shoulder_transform_l[..., :3, 3],
                                       elbow_transform_l[..., :3, 3], wrist_transform_l[..., :3, 3]))
    if torch.isnan(joint_locations).any():
        print('NAN in joint locations')

    foot_locations = torch.stack((calcaneus_transform_r[..., :3, 3], mtp_offset_transform_r[..., :3, 3],
                                  calcaneus_transform_l[..., :3, 3], mtp_offset_transform_l[..., :3, 3]))

    segment_orientations = torch.stack((pelvis_transform[..., :3, :3],
                                        femur_transform_r[..., :3, :3], tibia_transform_r[..., :3, :3],
                                        talus_transform_r[..., :3, :3], calcaneus_transform_r[..., :3, :3],
                                        femur_transform_l[..., :3, :3], tibia_transform_l[..., :3, :3],
                                        talus_transform_l[..., :3, :3], calcaneus_transform_l[..., :3, :3],
                                        torso_transform[..., :3, :3]))
    if with_arm:
        segment_orientations = torch.stack((*[segment_orientations[i] for i in range(segment_orientations.shape[0])],
                                            humerus_transform_r[..., :3, :3], ulna_transform_r[..., :3, :3],
                                            radius_transform_r[..., :3, :3],
                                            humerus_transform_l[..., :3, :3], ulna_transform_l[..., :3, :3],
                                            radius_transform_l[..., :3, :3]))

    return foot_locations, joint_locations, segment_orientations


def get_model_offsets(skeleton, with_arm=False):
    pelvis = skeleton.getBodyNode(0)
    hip_r_joint = pelvis.getChildJoint(0)
    femur_r = hip_r_joint.getChildBodyNode()
    knee_r_joint = femur_r.getChildJoint(0)
    tibia_r = knee_r_joint.getChildBodyNode()
    ankle_r_joint = tibia_r.getChildJoint(0)
    talus_r = ankle_r_joint.getChildBodyNode()
    subtalar_r_joint = talus_r.getChildJoint(0)
    calcn_r = subtalar_r_joint.getChildBodyNode()
    mtp_r_joint = calcn_r.getChildJoint(0)

    hip_l_joint = pelvis.getChildJoint(1)
    femur_l = hip_l_joint.getChildBodyNode()
    knee_l_joint = femur_l.getChildJoint(0)
    tibia_l = knee_l_joint.getChildBodyNode()
    ankle_l_joint = tibia_l.getChildJoint(0)
    talus_l = ankle_l_joint.getChildBodyNode()
    subtalar_l_joint = talus_l.getChildJoint(0)
    calcn_l = subtalar_l_joint.getChildBodyNode()
    mtp_l_joint = calcn_l.getChildJoint(0)

    lumbar_joint = pelvis.getChildJoint(2)
    torso = lumbar_joint.getChildBodyNode()

    if with_arm:
        shoulder_r_joint = torso.getChildJoint(0)
        humerus_r = shoulder_r_joint.getChildBodyNode()
        elbow_r_joint = humerus_r.getChildJoint(0)
        ulna_r = elbow_r_joint.getChildBodyNode()
        radioulnar_r_joint = ulna_r.getChildJoint(0)
        radius_r = radioulnar_r_joint.getChildBodyNode()
        wrist_r_joint = radius_r.getChildJoint(0)
        hand_r = wrist_r_joint.getChildBodyNode()

        shoulder_l_joint = torso.getChildJoint(1)
        humerus_l = shoulder_l_joint.getChildBodyNode()
        elbow_l_joint = humerus_l.getChildJoint(0)
        ulna_l = elbow_l_joint.getChildBodyNode()
        radioulnar_l_joint = ulna_l.getChildJoint(0)
        radius_l = radioulnar_l_joint.getChildBodyNode()
        wrist_l_joint = radius_l.getChildJoint(0)
        hand_l = wrist_l_joint.getChildBodyNode()

    # hip offset
    hip_offset_r = torch.eye(4)
    hip_offset_r[:3, :3] = torch.tensor(hip_r_joint.getTransformFromParentBodyNode().rotation())
    hip_offset_r[:3, 3] = torch.tensor(hip_r_joint.getTransformFromParentBodyNode().translation())
    hip_offset_l = torch.eye(4)
    hip_offset_l[:3, :3] = torch.tensor(hip_l_joint.getTransformFromParentBodyNode().rotation())
    hip_offset_l[:3, 3] = torch.tensor(hip_l_joint.getTransformFromParentBodyNode().translation())

    # femur offset
    femur_offset_to_knee_in_femur_r = -torch.tensor(hip_r_joint.getTransformFromChildBodyNode().translation())
    femur_rotation_to_knee_in_femur_r = torch.inverse(torch.tensor(hip_r_joint.getTransformFromChildBodyNode().rotation()))
    femur_offset_rotation_r = torch.eye(4)
    femur_offset_rotation_r[:3, :3] = femur_rotation_to_knee_in_femur_r
    femur_offset_translation_r = torch.eye(4)
    femur_offset_translation_r[:3, 3] = femur_offset_to_knee_in_femur_r
    femur_offset_r = torch.matmul(femur_offset_rotation_r, femur_offset_translation_r)

    femur_offset_to_knee_in_femur_l = -torch.tensor(hip_l_joint.getTransformFromChildBodyNode().translation())
    femur_rotation_to_knee_in_femur_l = torch.inverse(torch.tensor(hip_l_joint.getTransformFromChildBodyNode().rotation()))
    femur_offset_rotation_l = torch.eye(4)
    femur_offset_rotation_l[:3, :3] = femur_rotation_to_knee_in_femur_l
    femur_offset_translation_l = torch.eye(4)
    femur_offset_translation_l[:3, 3] = femur_offset_to_knee_in_femur_l
    femur_offset_l = torch.matmul(femur_offset_rotation_l, femur_offset_translation_l)

    # knee offset
    knee_offset_r = torch.eye(4)
    knee_offset_r[:3, :3] = torch.tensor(knee_r_joint.getTransformFromParentBodyNode().rotation())
    knee_offset_r[:3, 3] = torch.tensor(knee_r_joint.getTransformFromParentBodyNode().translation())

    knee_offset_l = torch.eye(4)
    knee_offset_l[:3, :3] = torch.tensor(knee_l_joint.getTransformFromParentBodyNode().rotation())
    knee_offset_l[:3, 3] = torch.tensor(knee_l_joint.getTransformFromParentBodyNode().translation())

    # tibia offset
    tibia_offset_to_knee_in_tibia_r = -torch.tensor(knee_r_joint.getTransformFromChildBodyNode().translation())
    tibia_rotation_to_knee_in_tibia_r = torch.inverse(torch.tensor(knee_r_joint.getTransformFromChildBodyNode().rotation()))
    tibia_offset_rotation_r = torch.eye(4)
    tibia_offset_rotation_r[:3, :3] = tibia_rotation_to_knee_in_tibia_r
    tibia_offset_translation_r = torch.eye(4)
    tibia_offset_translation_r[:3, 3] = tibia_offset_to_knee_in_tibia_r
    tibia_offset_r = torch.matmul(tibia_offset_rotation_r, tibia_offset_translation_r)

    tibia_offset_to_knee_in_tibia_l = -torch.tensor(knee_l_joint.getTransformFromChildBodyNode().translation())
    tibia_rotation_to_knee_in_tibia_l = torch.inverse(torch.tensor(knee_l_joint.getTransformFromChildBodyNode().rotation()))
    tibia_offset_rotation_l = torch.eye(4)
    tibia_offset_rotation_l[:3, :3] = tibia_rotation_to_knee_in_tibia_l
    tibia_offset_translation_l = torch.eye(4)
    tibia_offset_translation_l[:3, 3] = tibia_offset_to_knee_in_tibia_l
    tibia_offset_l = torch.matmul(tibia_offset_rotation_l, tibia_offset_translation_l)

    # ankle offset
    ankle_offset_r = torch.eye(4)
    ankle_offset_r[:3,:3] = torch.tensor(ankle_r_joint.getTransformFromParentBodyNode().rotation())
    ankle_offset_r[:3, 3] = torch.tensor(ankle_r_joint.getTransformFromParentBodyNode().translation())

    ankle_offset_l = torch.eye(4)
    ankle_offset_l[:3, :3] = torch.tensor(ankle_l_joint.getTransformFromParentBodyNode().rotation())
    ankle_offset_l[:3, 3] = torch.tensor(ankle_l_joint.getTransformFromParentBodyNode().translation())

    # talus offset
    talus_offset_to_ankle_in_talus_r = -torch.tensor(ankle_r_joint.getTransformFromChildBodyNode().translation())
    talus_rotation_to_ankle_in_talus_r = torch.inverse(torch.tensor(ankle_r_joint.getTransformFromChildBodyNode().rotation()))
    talus_offset_rotation_r = torch.eye(4)
    talus_offset_rotation_r[:3, :3] = talus_rotation_to_ankle_in_talus_r
    talus_offset_translation_r = torch.eye(4)
    talus_offset_translation_r[:3, 3] = talus_offset_to_ankle_in_talus_r
    talus_offset_r = torch.matmul(talus_offset_rotation_r, talus_offset_translation_r)

    talus_offset_to_ankle_in_talus_l = -torch.tensor(ankle_l_joint.getTransformFromChildBodyNode().translation())
    talus_rotation_to_ankle_in_talus_l = torch.inverse(torch.tensor(ankle_l_joint.getTransformFromChildBodyNode().rotation()))
    talus_offset_rotation_l = torch.eye(4)
    talus_offset_rotation_l[:3, :3] = talus_rotation_to_ankle_in_talus_l
    talus_offset_translation_l = torch.eye(4)
    talus_offset_translation_l[:3, 3] = talus_offset_to_ankle_in_talus_l
    talus_offset_l = torch.matmul(talus_offset_rotation_l, talus_offset_translation_l)

    # subtalar offset
    subtalar_offset_r = torch.eye(4)
    subtalar_offset_r[:3,:3] = torch.tensor(subtalar_r_joint.getTransformFromParentBodyNode().rotation())
    subtalar_offset_r[:3, 3] = torch.tensor(subtalar_r_joint.getTransformFromParentBodyNode().translation())

    subtalar_offset_l = torch.eye(4)
    subtalar_offset_l[:3, :3] = torch.tensor(subtalar_l_joint.getTransformFromParentBodyNode().rotation())
    subtalar_offset_l[:3, 3] = torch.tensor(subtalar_l_joint.getTransformFromParentBodyNode().translation())

    # calcaneus offset
    calcaneus_offset_to_subtalar_in_calcaneus_r = -torch.tensor(subtalar_r_joint.getTransformFromChildBodyNode().translation())
    calcaneus_rotation_to_subtalar_in_calcaneus_r = torch.inverse(torch.tensor(subtalar_r_joint.getTransformFromChildBodyNode().rotation()))
    calcaneus_offset_rotation_r = torch.eye(4)
    calcaneus_offset_rotation_r[:3, :3] = calcaneus_rotation_to_subtalar_in_calcaneus_r
    calcaneus_offset_translation_r = torch.eye(4)
    calcaneus_offset_translation_r[:3, 3] = calcaneus_offset_to_subtalar_in_calcaneus_r
    calcaneus_offset_r = torch.matmul(calcaneus_offset_rotation_r, calcaneus_offset_translation_r)

    calcaneus_offset_to_subtalar_in_calcaneus_l = -torch.tensor(subtalar_l_joint.getTransformFromChildBodyNode().translation())
    calcaneus_rotation_to_subtalar_in_calcaneus_l = torch.inverse(torch.tensor(subtalar_l_joint.getTransformFromChildBodyNode().rotation()))
    calcaneus_offset_rotation_l = torch.eye(4)
    calcaneus_offset_rotation_l[:3, :3] = calcaneus_rotation_to_subtalar_in_calcaneus_l
    calcaneus_offset_translation_l = torch.eye(4)
    calcaneus_offset_translation_l[:3, 3] = calcaneus_offset_to_subtalar_in_calcaneus_l
    calcaneus_offset_l = torch.matmul(calcaneus_offset_rotation_l, calcaneus_offset_translation_l)

    # mtp offset
    mtp_offset_r = torch.eye(4)
    mtp_offset_r[:3, :3] = torch.tensor(mtp_r_joint.getTransformFromParentBodyNode().rotation())
    mtp_offset_r[:3, 3] = torch.tensor(mtp_r_joint.getTransformFromParentBodyNode().translation())

    mtp_offset_l = torch.eye(4)
    mtp_offset_l[:3, :3] = torch.tensor(mtp_l_joint.getTransformFromParentBodyNode().rotation())
    mtp_offset_l[:3, 3] = torch.tensor(mtp_l_joint.getTransformFromParentBodyNode().translation())

    # toes offset
    toes_offset_to_mtp_in_toes_r = -torch.tensor(mtp_r_joint.getTransformFromChildBodyNode().translation())
    toes_rotation_to_mtp_in_toes_r = torch.inverse(torch.tensor(mtp_r_joint.getTransformFromChildBodyNode().rotation()))
    toes_offset_rotation_r = torch.eye(4)
    toes_offset_rotation_r[:3, :3] = toes_rotation_to_mtp_in_toes_r
    toes_offset_translation_r = torch.eye(4)
    toes_offset_translation_r[:3, 3] = toes_offset_to_mtp_in_toes_r
    toes_offset_r = torch.matmul(toes_offset_rotation_r, toes_offset_translation_r)

    toes_offset_to_mtp_in_toes_l = -torch.tensor(mtp_l_joint.getTransformFromChildBodyNode().translation())
    toes_rotation_to_mtp_in_toes_l = torch.inverse(torch.tensor(mtp_l_joint.getTransformFromChildBodyNode().rotation()))
    toes_offset_rotation_l = torch.eye(4)
    toes_offset_rotation_l[:3, :3] = toes_rotation_to_mtp_in_toes_l
    toes_offset_translation_l = torch.eye(4)
    toes_offset_translation_l[:3, 3] = toes_offset_to_mtp_in_toes_l
    toes_offset_l = torch.matmul(toes_offset_rotation_l, toes_offset_translation_l)


    # lumbar offset
    lumbar_offset = torch.eye(4)
    lumbar_offset[:3, :3] = torch.tensor(lumbar_joint.getTransformFromParentBodyNode().rotation())
    lumbar_offset[:3, 3] = torch.tensor(lumbar_joint.getTransformFromParentBodyNode().translation())

    # torso offset
    torso_offset_to_lumbar_in_torso = -torch.tensor(lumbar_joint.getTransformFromChildBodyNode().translation())
    torso_offset_rotation_to_lumbar_in_torso = torch.inverse(torch.tensor(lumbar_joint.getTransformFromChildBodyNode().rotation()))
    torso_offset_rotation = torch.eye(4)
    torso_offset_rotation[:3, :3] = torso_offset_rotation_to_lumbar_in_torso
    torso_offset_translation = torch.eye(4)
    torso_offset_translation[:3, 3] = torso_offset_to_lumbar_in_torso
    torso_offset = torch.matmul(torso_offset_rotation, torso_offset_translation)

    if with_arm:
        # shoulder offset
        shoulder_offset_r = torch.eye(4)
        shoulder_offset_r[:3, :3] = torch.tensor(shoulder_r_joint.getTransformFromParentBodyNode().rotation())
        shoulder_offset_r[:3, 3] = torch.tensor(shoulder_r_joint.getTransformFromParentBodyNode().translation())

        shoulder_offset_l = torch.eye(4)
        shoulder_offset_l[:3, :3] = torch.tensor(shoulder_l_joint.getTransformFromParentBodyNode().rotation())
        shoulder_offset_l[:3, 3] = torch.tensor(shoulder_l_joint.getTransformFromParentBodyNode().translation())

        # humerus offset
        humerus_offset_to_shoulder_in_humerus_r = -torch.tensor(shoulder_r_joint.getTransformFromChildBodyNode().translation())
        humerus_offset_rotation_to_shoulder_in_humerus_r = torch.inverse(torch.tensor(shoulder_r_joint.getTransformFromChildBodyNode().rotation()))
        humerus_offset_rotation_r = torch.eye(4)
        humerus_offset_rotation_r[:3, :3] = humerus_offset_rotation_to_shoulder_in_humerus_r
        humerus_offset_translation_r = torch.eye(4)
        humerus_offset_translation_r[:3, 3] = humerus_offset_to_shoulder_in_humerus_r
        humerus_offset_r = torch.matmul(humerus_offset_rotation_r, humerus_offset_translation_r)

        humerus_offset_to_shoulder_in_humerus_l = -torch.tensor(shoulder_l_joint.getTransformFromChildBodyNode().translation())
        humerus_offset_rotation_to_shoulder_in_humerus_l = torch.inverse(torch.tensor(shoulder_l_joint.getTransformFromChildBodyNode().rotation()))
        humerus_offset_rotation_l = torch.eye(4)
        humerus_offset_rotation_l[:3, :3] = humerus_offset_rotation_to_shoulder_in_humerus_l
        humerus_offset_translation_l = torch.eye(4)
        humerus_offset_translation_l[:3, 3] = humerus_offset_to_shoulder_in_humerus_l
        humerus_offset_l = torch.matmul(humerus_offset_rotation_l, humerus_offset_translation_l)

        # elbow offset
        elbow_offset_r = torch.eye(4)
        elbow_offset_r[:3, :3] = torch.tensor(elbow_r_joint.getTransformFromParentBodyNode().rotation())
        elbow_offset_r[:3, 3] = torch.tensor(elbow_r_joint.getTransformFromParentBodyNode().translation())

        elbow_offset_l = torch.eye(4)
        elbow_offset_l[:3, :3] = torch.tensor(elbow_l_joint.getTransformFromParentBodyNode().rotation())
        elbow_offset_l[:3, 3] = torch.tensor(elbow_l_joint.getTransformFromParentBodyNode().translation())

        # ulna offset
        ulna_offset_to_elbow_in_ulna_r = -torch.tensor(elbow_r_joint.getTransformFromChildBodyNode().translation())
        ulna_offset_rotation_to_elbow_in_ulna_r = torch.inverse(torch.tensor(elbow_r_joint.getTransformFromChildBodyNode().rotation()))
        ulna_offset_rotation_r = torch.eye(4)
        ulna_offset_rotation_r[:3, :3] = ulna_offset_rotation_to_elbow_in_ulna_r
        ulna_offset_translation_r = torch.eye(4)
        ulna_offset_translation_r[:3, 3] = ulna_offset_to_elbow_in_ulna_r
        ulna_offset_r = torch.matmul(ulna_offset_rotation_r, ulna_offset_translation_r)

        ulna_offset_to_elbow_in_ulna_l = -torch.tensor(elbow_l_joint.getTransformFromChildBodyNode().translation())
        ulna_offset_rotation_to_elbow_in_ulna_l = torch.inverse(torch.tensor(elbow_l_joint.getTransformFromChildBodyNode().rotation()))
        ulna_offset_rotation_l = torch.eye(4)
        ulna_offset_rotation_l[:3, :3] = ulna_offset_rotation_to_elbow_in_ulna_l
        ulna_offset_translation_l = torch.eye(4)
        ulna_offset_translation_l[:3, 3] = ulna_offset_to_elbow_in_ulna_l
        ulna_offset_l = torch.matmul(ulna_offset_rotation_l, ulna_offset_translation_l)

        # radioulnar offset
        radioulnar_offset_r = torch.eye(4)
        radioulnar_offset_r[:3, :3] = torch.tensor(radioulnar_r_joint.getTransformFromParentBodyNode().rotation())
        radioulnar_offset_r[:3, 3] = torch.tensor(radioulnar_r_joint.getTransformFromParentBodyNode().translation())

        radioulnar_offset_l = torch.eye(4)
        radioulnar_offset_l[:3, :3] = torch.tensor(radioulnar_l_joint.getTransformFromParentBodyNode().rotation())
        radioulnar_offset_l[:3, 3] = torch.tensor(radioulnar_l_joint.getTransformFromParentBodyNode().translation())

        # radius offset
        radius_offset_to_radioulnar_in_radius_r = -torch.tensor(radioulnar_r_joint.getTransformFromChildBodyNode().translation())
        radius_offset_rotation_to_radioulnar_in_radius_r = torch.inverse(torch.tensor(radioulnar_r_joint.getTransformFromChildBodyNode().rotation()))
        radius_offset_rotation_r = torch.eye(4)
        radius_offset_rotation_r[:3, :3] = radius_offset_rotation_to_radioulnar_in_radius_r
        radius_offset_translation_r = torch.eye(4)
        radius_offset_translation_r[:3, 3] = radius_offset_to_radioulnar_in_radius_r
        radius_offset_r = torch.matmul(radius_offset_rotation_r, radius_offset_translation_r)

        radius_offset_to_radioulnar_in_radius_l = -torch.tensor(radioulnar_l_joint.getTransformFromChildBodyNode().translation())
        radius_offset_rotation_to_radioulnar_in_radius_l = torch.inverse(torch.tensor(radioulnar_l_joint.getTransformFromChildBodyNode().rotation()))
        radius_offset_rotation_l = torch.eye(4)
        radius_offset_rotation_l[:3, :3] = radius_offset_rotation_to_radioulnar_in_radius_l
        radius_offset_translation_l = torch.eye(4)
        radius_offset_translation_l[:3, 3] = radius_offset_to_radioulnar_in_radius_l
        radius_offset_l = torch.matmul(radius_offset_rotation_l, radius_offset_translation_l)

        # wrist offset
        wrist_offset_r = torch.eye(4)
        wrist_offset_r[:3, :3] = torch.tensor(wrist_r_joint.getTransformFromParentBodyNode().rotation())
        wrist_offset_r[:3, 3] = torch.tensor(wrist_r_joint.getTransformFromParentBodyNode().translation())

        wrist_offset_l = torch.eye(4)
        wrist_offset_l[:3, :3] = torch.tensor(wrist_l_joint.getTransformFromParentBodyNode().rotation())
        wrist_offset_l[:3, 3] = torch.tensor(wrist_l_joint.getTransformFromParentBodyNode().translation())

        # hand offset
        hand_offset_to_wrist_in_hand_r = -torch.tensor(wrist_r_joint.getTransformFromChildBodyNode().translation())
        hand_offset_rotation_to_wrist_in_hand_r = torch.inverse(torch.tensor(wrist_r_joint.getTransformFromChildBodyNode().rotation()))
        hand_offset_rotation_r = torch.eye(4)
        hand_offset_rotation_r[:3, :3] = hand_offset_rotation_to_wrist_in_hand_r
        hand_offset_translation_r = torch.eye(4)
        hand_offset_translation_r[:3, 3] = hand_offset_to_wrist_in_hand_r
        hand_offset_r = torch.matmul(hand_offset_rotation_r, hand_offset_translation_r)

        hand_offset_to_wrist_in_hand_l = -torch.tensor(wrist_l_joint.getTransformFromChildBodyNode().translation())
        hand_offset_rotation_to_wrist_in_hand_l = torch.inverse(torch.tensor(wrist_l_joint.getTransformFromChildBodyNode().rotation()))
        hand_offset_rotation_l = torch.eye(4)
        hand_offset_rotation_l[:3, :3] = hand_offset_rotation_to_wrist_in_hand_l
        hand_offset_translation_l = torch.eye(4)
        hand_offset_translation_l[:3, 3] = hand_offset_to_wrist_in_hand_l
        hand_offset_l = torch.matmul(hand_offset_rotation_l, hand_offset_translation_l)

    offsets = torch.stack((hip_offset_r, femur_offset_r, knee_offset_r, tibia_offset_r,
                           ankle_offset_r, talus_offset_r, subtalar_offset_r,
                           calcaneus_offset_r, mtp_offset_r,
                           hip_offset_l, femur_offset_l, knee_offset_l, tibia_offset_l,
                           ankle_offset_l, talus_offset_l, subtalar_offset_l,
                           calcaneus_offset_l, mtp_offset_l,
                           lumbar_offset, torso_offset), dim=2)
    if with_arm:
        offsets = torch.stack((*[offsets[i] for i in range(offsets.shape[0])],
                               shoulder_offset_r, humerus_offset_r, elbow_offset_r, ulna_offset_r,
                               radioulnar_offset_r, radius_offset_r, wrist_offset_r, hand_offset_r,
                               shoulder_offset_l, humerus_offset_l, elbow_offset_l, ulna_offset_l,
                               radioulnar_offset_l, radius_offset_l, wrist_offset_l, hand_offset_l,toes_offset_r, toes_offset_l), dim=2)

    return offsets
