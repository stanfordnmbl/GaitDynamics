import argparse
import copy
import json
import os
from alant.alan_consts import *
import numpy as np


def parse_opt():
    machine_specific_config = json.load(open(os.path.dirname(os.path.realpath(__file__)) + '/machine_specific_config.json', 'r'))
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="check_txtytz", help="save to project/name")
    parser.add_argument("--with_arm", type=bool, default=False, help="whether osim model has arm DoFs")
    parser.add_argument("--with_kinematics_vel", type=bool, default=True, help="whether to include 1st derivative of kinematics")
    parser.add_argument("--log_with_wandb", type=bool, default=True, help="log with wandb")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--target_sampling_rate", type=int, default=100)
    parser.add_argument("--window_len", type=int, default=150)

    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="dataset_backups/",
        help="Dataset backup path",
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="MotionModel", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=machine_specific_config['batch_size'], help="batch size")
    parser.add_argument("--batch_size_inference", type=int, default=128, help="batch size during inference")
    parser.add_argument("--pseudo_dataset_len", type=int, default=machine_specific_config['pseudo_dataset_len'], help="pseudo dataset length")
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=50,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    opt = parser.parse_args()
    opt.data_path_parent = machine_specific_config['b3d_path']
    set_with_arm_opt(opt, opt.with_arm)
    return opt


def set_with_arm_opt(opt, with_arm):
    if with_arm:
        opt.with_arm = True
        opt.osim_dof_columns = copy.deepcopy(OSIM_DOF_ALL + KINETICS_ALL)
        opt.joints_3d = JOINTS_3D_ALL
        data_path = opt.data_path_parent + '/b3d_with_arm/'
        opt.model_states_column_names = copy.deepcopy(MODEL_STATES_COLUMN_NAMES_WITH_ARM)
    else:
        opt.with_arm = False
        opt.osim_dof_columns = copy.deepcopy(OSIM_DOF_ALL[:23] + KINETICS_ALL)
        opt.joints_3d = {key_: value_ for key_, value_ in JOINTS_3D_ALL.items() if key_ in ['pelvis', 'hip_r', 'hip_l', 'lumbar']}
        data_path = opt.data_path_parent + '/b3d_no_arm/'
        opt.model_states_column_names = copy.deepcopy(MODEL_STATES_COLUMN_NAMES_NO_ARM)

    if opt.with_kinematics_vel:
        opt.model_states_column_names = opt.model_states_column_names + [
            f'{col}_vel' for i_col, col in enumerate(opt.model_states_column_names) if col not in KINETICS_ALL and 'pelvis_t' not in col]

    opt.data_path_train = data_path + 'train_cleaned/'
    opt.data_path_test = data_path + 'test_cleaned/'


