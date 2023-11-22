import glob
import os
import pickle
import random
from functools import cmp_to_key
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from dataset.preprocess import Normalizer, vectorize_many
from dataset.quaternion import ax_to_6v


import glob
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm


def slice_and_process_motion(joint_orientations, root_translation, frame_rate, target_frame_rate, stride, length, out_dir, file_name):
    start_idx = 0
    window = int(length * frame_rate)
    stride_step = int(stride * frame_rate)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= joint_orientations.shape[0] - window:
        joint_orientations_slice, root_translation_slice = (
            joint_orientations[start_idx : start_idx + window],
            root_translation[start_idx : start_idx + window],
        )
        joint_orientations_slice = torch.tensor(joint_orientations_slice).reshape(joint_orientations_slice.shape[0],-1,3)[:,:22,:]
        joint_orientations_slice = ax_to_6v(joint_orientations_slice).numpy()
        if joint_orientations_slice.min() < -1.01 or joint_orientations_slice.max() > 1.01:
            print('joint_orientations_slice out of range:: ' + file_name)

        # downsample joint_orientations_slice and root_translation_slice from the frame_rate to 20Hz using linear interpolation without using numpy.interp
        # TODO: implement with SLERP instead of linear interpolation (should not matter all that much)
        joint_orientations_slice_resampled = np.zeros((int(length * target_frame_rate), joint_orientations_slice.shape[1], joint_orientations_slice.shape[2]))
        root_translation_slice_resampled = np.zeros((int(length * target_frame_rate), root_translation_slice.shape[1]))
        for i in range(joint_orientations_slice.shape[1]):
            for j in range(joint_orientations_slice.shape[2]):
                joint_orientations_slice_resampled[:,i,j] = np.interp(np.linspace(0, length, int(length * target_frame_rate)), np.linspace(0, length, window), joint_orientations_slice[:,i,j])
        for i in range(root_translation_slice.shape[1]):
            root_translation_slice_resampled[:,i] = np.interp(np.linspace(0, length, int(length * target_frame_rate)), np.linspace(0, length, window), root_translation_slice[:,i])

        joint_orientations_slice_resampled = torch.tensor(joint_orientations_slice_resampled.reshape(joint_orientations_slice_resampled.shape[0],-1))
        root_translation_slice_resampled = torch.tensor(root_translation_slice_resampled)


        pose = torch.concatenate((root_translation_slice_resampled, joint_orientations_slice_resampled), axis=1).float()
        if pose[...,3:].min() < -1.01 or pose[...,3:].max() > 1.01:
            print('joint_orientations_slice out of range:: ' + file_name)

        # shift root position to start in (x,y) = (0,0)
        pose[:,0] = pose[:,0].clone() - pose[0,0].clone()
        pose[:,1] = pose[:,1].clone() - pose[0,1].clone()

        torch.save(pose, f"{out_dir}/{file_name}_slice{slice_count}.pt")        # save train/test data
        start_idx += stride_step
        slice_count += 1
    return slice_count


def list_files(dir):
    r = []
    subdirs = [x[0] for x in os.walk(dir) if '.npz' in x[0]]
    for subdir in subdirs:
        files = os.walk(subdir).next()[2]
        if (len(files) > 0):
            for file in files:
                r.append(os.path.join(subdir, file))
    return r


def slice_AMASS(motion_dir, stride=1, length=3, target_frame_rate=20):
    """
    motion_dir: directory containing AMASS motion files
    stride: stride of the slicing window every time a slice is taken
    length: length of slicing window in seconds
    """
    print("Slicing train data")
    out_dir = motion_dir + "_sliced"
    os.makedirs(out_dir, exist_ok=True)
    motions = sorted(glob.glob(f"{motion_dir}/*.npz"))
    motions = [m.replace('\\', '/') for m in motions]
    motion_out = motion_dir + "_sliced"
    os.makedirs(motion_out, exist_ok=True)
    for motion in tqdm(motions):
        data = np.load(motion)
        # check if poses exists in data
        if 'poses' not in data.keys():
            print('No poses in data:: ' + motion)
        else:
            joint_orientations = data['poses']
            root_translation = data['trans']
            if 'mocap_framerate' in data.keys():
                frame_rate = data['mocap_framerate']
            else:
                frame_rate = data['mocap_frame_rate']
            number_of_slices = slice_and_process_motion(joint_orientations, root_translation, frame_rate, target_frame_rate, stride, length, out_dir, motion.split('/')[-1][:-4])


class MotionDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            backup_path: str,
            train: bool,
            normalizer: Any = None,
            data_len: int = -1,
            force_reload: bool = False,
    ):
        self.data_path = data_path
        self.raw_fps = 120
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"

        self.normalizer = normalizer
        self.data_len = data_len

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
        # load raw data
        if not force_reload and pickle_name in os.listdir(backup_path):
            print("Using cached dataset...")
            with open(os.path.join(backup_path, pickle_name), "rb") as f:
                data = pickle.load(f)
        else:
            print("Loading dataset...")
            self.data = self.load_AMASS()  # Call this last
            # with open(os.path.join(backup_path, pickle_name), "wb") as f:
            #     pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        self.length = self.data.shape[0]
        self.data[...,:2] = self.data[:,:,:2] - self.data[:,0:1,:2]
        global_pose_vec_input = self.data.float().detach()
        self.normalizer = Normalizer(global_pose_vec_input.clone())
        self.data = self.normalizer.normalize(global_pose_vec_input.clone())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx,...]

    def load_AMASS(self):
        split_data_path = self.data_path
        motion_path = split_data_path
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pt")))
        data_list = []
        for motion in motions:
            data = torch.load(motion)
            data_list.append(data)
        data = torch.stack(data_list, dim=0)
        return data


if __name__ == '__main__':
    path = 'D:/Local/Data/AMASS_Copy/'
    slice_AMASS(path)





















