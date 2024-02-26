import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from scipy import signal

from dataset.quaternion import ax_to_6v

def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


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


def slice_AMASS(motion_dir, stride=1, length=3, target_frame_rate=20):
    """
    motion_dir: directory containing AMASS motion files
    stride: stride of the slicing window every time a slice is taken
    length: length of slicing window in seconds
    """
    out_dir = motion_dir + "_sliced"
    os.makedirs(out_dir, exist_ok=True)
    motions = sorted(glob.glob(f"{motion_dir}/*.npz"))
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
            frame_rate = data['mocap_framerate']
            number_of_slices = slice_and_process_motion(joint_orientations, root_translation, frame_rate, target_frame_rate, stride, length, out_dir, motion.split('/')[-1][:-4])



def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)
