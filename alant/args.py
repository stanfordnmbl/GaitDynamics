import argparse
import json
import os
from alant.alan_consts import *


def parse_train_opt():
    machine_specific_config = json.load(open(os.path.dirname(os.path.realpath(__file__)) + '/machine_specific_config.json', 'r'))
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_arm", type=bool, default=False, help="whether osim model has arm DoFs")
    parser.add_argument("--log_with_wandb", type=bool, default=True, help="log with wandb")
    parser.add_argument("--epochs", type=int, default=1000)

    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="MotionModel", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=machine_specific_config['batch_size'], help="batch size")
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
        opt.osim_dof_columns = OSIM_DOF_ALL
        opt.joints_3d = JOINTS_3D_ALL
        opt.data_path = opt.data_path_parent + '/b3d_with_arm/'
        opt.model_states_column_names = MODEL_STATES_COLUMN_NAMES_WITH_ARM
    else:
        opt.with_arm = False
        opt.osim_dof_columns = OSIM_DOF_ALL[:23]
        opt.joints_3d = {key_: value_ for key_, value_ in JOINTS_3D_ALL.items() if key_ in ['pelvis', 'hip_r', 'hip_l']}
        opt.data_path = opt.data_path_parent + '/b3d_no_arm/'
        opt.model_states_column_names = MODEL_STATES_COLUMN_NAMES_NO_ARM


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    opt = parser.parse_args()
    return opt
