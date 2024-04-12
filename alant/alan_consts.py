import copy

JOINTS_3D_ALL = {'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
                 'hip_r': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r'],
                 'hip_l': ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'],
                 'lumbar': ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation'],
                 'arm_r': ['arm_flex_r', 'arm_add_r', 'arm_rot_r'],
                 'arm_l': ['arm_flex_l', 'arm_add_l', 'arm_rot_l']}

OSIM_DOF_ALL = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'hip_flexion_r',
    'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
    'mtp_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
    'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l',
    'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l']


KINETICS_ALL = [body + modality + axis for body in ['calcn_r', 'calcn_l'] for modality in ['_force_v', '_force_moment'] for axis in ['x', 'y', 'z']]

MODEL_STATES_COLUMN_NAMES_WITH_ARM = [
    'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'knee_angle_r', 'ankle_angle_r', 'knee_angle_l', 'ankle_angle_l',
    'elbow_flex_r', 'pro_sup_r', 'elbow_flex_l', 'pro_sup_l'        # only for with arm
                                     ] + KINETICS_ALL + [
    'pelvis_0', 'pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4', 'pelvis_5',
    'hip_r_0', 'hip_r_1', 'hip_r_2', 'hip_r_3', 'hip_r_4', 'hip_r_5',
    'hip_l_0', 'hip_l_1', 'hip_l_2', 'hip_l_3', 'hip_l_4', 'hip_l_5',
    'lumbar_0', 'lumbar_1', 'lumbar_2', 'lumbar_3', 'lumbar_4', 'lumbar_5',
    'arm_r_0', 'arm_r_1', 'arm_r_2', 'arm_r_3', 'arm_r_4', 'arm_r_5',        # only for with arm
    'arm_l_0', 'arm_l_1', 'arm_l_2', 'arm_l_3', 'arm_l_4', 'arm_l_5'        # only for with arm
]

MODEL_STATES_COLUMN_NAMES_NO_ARM = copy.deepcopy(MODEL_STATES_COLUMN_NAMES_WITH_ARM)
for name_ in ['elbow_flex_r', 'pro_sup_r', 'elbow_flex_l', 'pro_sup_l', 'arm_r_0', 'arm_r_1', 'arm_r_2', 'arm_r_3',
              'arm_r_4', 'arm_r_5', 'arm_l_0', 'arm_l_1', 'arm_l_2', 'arm_l_3', 'arm_l_4', 'arm_l_5']:
    MODEL_STATES_COLUMN_NAMES_NO_ARM.remove(name_)

FROZEN_DOFS = ['subtalar_angle_r', 'mtp_angle_r', 'subtalar_angle_l', 'mtp_angle_l',
               'wrist_flex_r', 'wrist_dev_r', 'wrist_flex_l', 'wrist_dev_l']

# 'Falisse2017' has 3 contact bodies
DSET_SHORT_NAMES = ['Camargo2021', 'Carter2023', 'Fregly2012', 'Falisse2017', 'Hammer2013', 'Han2023', 'Li2021', 'Moore2015',
                    'Santos2017', 'Tan2021', 'Tan2022', 'Tiziana2019', 'Uhlrich2023', 'vanderZee2022', 'Wang2023']
DATASETS_NO_ARM = [name_ + '_Formatted_No_Arm' for name_ in DSET_SHORT_NAMES]

NOT_IN_GAIT_PHASE = -1000

# Carter GRF are too noisy
# Hammer2013 only have 2 windows, too sensitive.
# Han is not a pure gait dataset.
EXCLUDE_FROM_ASB = ['Carter2023_Formatted_No_Arm', 'Hammer2013_Formatted_No_Arm', 'Han2023_Formatted_No_Arm']

























