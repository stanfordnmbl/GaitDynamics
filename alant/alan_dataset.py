import glob
import os
from typing import List
from dataset.quaternion import ax_to_6v
import numpy as np
import torch
from tqdm import tqdm
import nimblephysics as nimble


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


def slice_addb(motion_dir, length_sec=3, stride_sec=1, target_frame_rate=20):
    print("Slicing train data")
    out_dir = motion_dir + "_sliced"
    os.makedirs(out_dir, exist_ok=True)

    sample_stride = int(100 / target_frame_rate)
    length = int(length_sec * 100)
    window_stride = int(stride_sec * 100)

    subject_paths = []
    if os.path.isdir(motion_dir):
        for root, dirs, files in os.walk(motion_dir):
            for file in files:
                if file.endswith(".b3d") and "vander" not in file.lower():
                    subject_paths.append(os.path.join(root, file))

    windows = []
    subjects = {}
    for i, subject_path in enumerate(subject_paths):
        # Add the skeleton to the list of skeletons
        subject = nimble.biomechanics.SubjectOnDisk(subject_path)
        force_bodies = subject.getGroundForceBodies()
        if len(force_bodies) > 2:
            print(f'Subject {subject} has more than two force bodies, skipping')
            continue
        subject_name = subject_path.split('/')[-1].split('.')[0]
        subjects[subject_name] = subject

        for trial_index in range(subject.getNumTrials()):
            if np.abs(subject.getTrialTimestep(trial_index) - 0.01) > 0.001:
                raise ValueError('sampling rate not equal to 100 Hz')
            trial_length = subject.getTrialLength(trial_index)
            probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                            in subject.getMissingGRF(trial_index)]
            for window_start in range(0, max(trial_length - length - 1, 0), window_stride):
                if not any(probably_missing[window_start:window_start + length:sample_stride]):
                    assert window_start + length < trial_length
                    windows.append((subject_name, trial_index, window_start))

    for i_window, (subject_name, trial, window_start) in enumerate(tqdm(windows)):
        # Read the frames from disk
        subject = subjects[subject_name]
        frames: nimble.biomechanics.FrameList = subject.readFrames(trial,
                                                                   window_start,
                                                                   length // sample_stride,
                                                                   stride=sample_stride,
                                                                   includeSensorData=False,
                                                                   includeProcessingPasses=True)
        assert (len(frames) == length // sample_stride)
        first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]

        pose = [frame.pos for frame in first_passes]
        pose = torch.tensor(pose).float()
        pose[:, 3] = pose[:, 3].clone() - pose[0, 3].clone()
        pose[:, 4] = pose[:, 4].clone() - pose[0, 4].clone()
        pose[:, 5] = pose[:, 5].clone() - pose[0, 5].clone()

        trial_name = subject.getTrialName(trial)
        torch.save(pose, f"{out_dir}/{subject_name}_{trial_name}_slice{i_window}.pt")        # save train/test data

        # ax_to_6v


if __name__ == '__main__':
    path = '/mnt/e/MotionModelData/train/'
    slice_addb(path)





















