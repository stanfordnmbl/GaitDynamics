
JOINTS_3D_ALL = {'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
                 'hip_r': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r'],
                 'hip_l': ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'],
                 'arm_r': ['arm_flex_r', 'arm_add_r', 'arm_rot_r'],
                 'arm_l': ['arm_flex_l', 'arm_add_l', 'arm_rot_l']}

OSIM_DOF_ALL = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'hip_flexion_r',
    'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
    'mtp_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
    'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l',
    'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l']


MODEL_STATES_COLUMN_NAMES_NO_ARM = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'knee_angle_l',
    'ankle_angle_l', 'subtalar_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'pelvis_0',
    'pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4', 'pelvis_5', 'hip_r_0', 'hip_r_1', 'hip_r_2', 'hip_r_3', 'hip_r_4',
    'hip_r_5', 'hip_l_0', 'hip_l_1', 'hip_l_2', 'hip_l_3', 'hip_l_4', 'hip_l_5', 'arm_r_0', 'arm_r_1', 'arm_r_2',
    'arm_r_3', 'arm_r_4', 'arm_r_5', 'arm_l_0', 'arm_l_1', 'arm_l_2', 'arm_l_3', 'arm_l_4', 'arm_l_5']

MODEL_STATES_COLUMN_NAMES_WITH_ARM = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'knee_angle_l',
    'ankle_angle_l', 'subtalar_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'elbow_flex_r',
    'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l', 'pelvis_0',
    'pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4', 'pelvis_5', 'hip_r_0', 'hip_r_1', 'hip_r_2', 'hip_r_3', 'hip_r_4',
    'hip_r_5', 'hip_l_0', 'hip_l_1', 'hip_l_2', 'hip_l_3', 'hip_l_4', 'hip_l_5', 'arm_r_0', 'arm_r_1', 'arm_r_2',
    'arm_r_3', 'arm_r_4', 'arm_r_5', 'arm_l_0', 'arm_l_1', 'arm_l_2', 'arm_l_3', 'arm_l_4', 'arm_l_5']

FROZEN_DOFS = ['mtp_angle_r', 'mtp_angle_l']
