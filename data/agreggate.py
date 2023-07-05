import glob
import shutil
import os
import pickle
from tqdm import tqdm


def agreggate(AMASS_path, AMASS_agreggated_path):
    subsets = glob.glob(AMASS_path + "/*/", recursive=True)
    for subset in subsets:
        subjects = glob.glob(subset + "/*/", recursive=True)
        for subject in tqdm(subjects):
            trials = glob.glob(f"{subject}/*.npz")
            for trial in trials:
                trial_name = trial.split('/')[-3] + '_' + trial.split('/')[-2] + '_' + trial.split('/')[-1]
                shutil.copy(trial, os.path.join(AMASS_agreggated_path, trial_name))
