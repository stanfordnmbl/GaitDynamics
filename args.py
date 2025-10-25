import argparse
import json
import os
from consts import *


def parse_opt():
    machine_specific_config = json.load(open(os.path.dirname(os.path.realpath(__file__)) + '/machine_specific_config.json', 'r'))
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="no_data_filter", help="save to project/name")
    parser.add_argument("--with_arm", type=bool, default=False, help="whether osim model has arm DoFs")
    parser.add_argument("--with_kinematics_vel", type=bool, default=True, help="whether to include 1st derivative of kinematics")
    parser.add_argument("--log_with_wandb", type=bool, default=machine_specific_config['log_with_wandb'], help="log with wandb")
    parser.add_argument("--epochs", type=int, default=7680)
    parser.add_argument("--target_sampling_rate", type=int, default=100)
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--guide_x_start_the_beginning_step", type=int, default=-10)      # negative value means no guidance

    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="dataset_backups/",
        help="Dataset backup path",
    )

    parser.add_argument(
        "--wandb_pj_name", type=str, default="MotionModel", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=machine_specific_config['batch_size'], help="batch size")
    parser.add_argument("--batch_size_inference", type=int, default=128, help="batch size during inference")
    parser.add_argument("--pseudo_dataset_len", type=int, default=machine_specific_config['pseudo_dataset_len'], help="pseudo dataset length")
    # parser.add_argument(
    #     "--save_interval",
    #     type=int,
    #     default=50,
    #     help='Log model after every "save_period" epoch',
    # )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--checkpoint_bl", type=str, default="", help="trained checkpoint path (optional)"
    )
    opt = parser.parse_args()
    opt.data_path_parent = machine_specific_config['b3d_path']
    opt.use_server = machine_specific_config['use_server']


    opt.with_arm = False
    opt.osim_dof_columns = copy.deepcopy(OSIM_DOF_ALL[:23] + KINETICS_ALL)
    opt.joints_3d = {key_: value_ for key_, value_ in JOINTS_3D_ALL.items() if key_ in ['pelvis', 'hip_r', 'hip_l', 'lumbar']}
    data_path = opt.data_path_parent + '/b3d_no_arm/'
    opt.data_path_osim_model = opt.data_path_parent + 'osim_model/unscaled_generic_no_arm.osim'
    opt.model_states_column_names = copy.deepcopy(MODEL_STATES_COLUMN_NAMES_NO_ARM)

    for joint_name, joints_with_3_dof in opt.joints_3d.items():
        opt.model_states_column_names = opt.model_states_column_names + [
            joint_name + '_' + axis + '_angular_vel' for axis in ['x', 'y', 'z']]

    if opt.with_kinematics_vel:
        opt.model_states_column_names = opt.model_states_column_names + [
            f'{col}_vel' for i_col, col in enumerate(opt.model_states_column_names)
            if not sum([term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5']])]

    opt.data_path_train = data_path + 'train_cleaned/'
    opt.data_path_test = data_path + 'test_cleaned/'

    opt.knee_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'knee' in col]
    opt.ankle_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'ankle' in col]
    opt.hip_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'hip' in col]
    opt.kinematic_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' not in col]
    opt.kinetic_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if i_col not in opt.kinematic_diffusion_col_loc]
    opt.grf_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if 'force' in col and '_cop_' not in col]
    opt.cop_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if '_cop_' in col]
    opt.kinematic_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if 'force' not in col]

    return opt
