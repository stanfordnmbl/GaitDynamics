import sys
import inspect
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import nimblephysics as nimble
from consts import WEIGHT_KG_OVERWRITE
import numpy as np


def get_heights_weights(data_path):
    subject_paths = []
    if os.path.isdir(data_path):
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".b3d"):
                    subject_paths.append(file_path)

    dset_subject_names, weights, heights, ages, sexes = [], [], [], [], []
    for i_sub, subject_path in enumerate(subject_paths):
        try:
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
        except RuntimeError:
            print(f'Failed loading subject {subject_path}, skipping')
            continue
        # print(f'Loading subject: {subject_path}')

        subject_name = subject_path.split('/')[-1].split('.')[0].split('_split')[0]
        dset_name = subject_path.split('/')[-3]
        if dset_name + '__' + subject_name in dset_subject_names:
            continue

        dset_subject_names.append(dset_name + '__' + subject_name)
        height_m = subject.getHeightM()
        if f'{dset_name}_{subject_name}' in WEIGHT_KG_OVERWRITE.keys():
            weight_kg = WEIGHT_KG_OVERWRITE[f'{dset_name}_{subject_name}']
            print(f'Overwriting {dset_name}_{subject_name}\'s weight to {weight_kg} kg')
        else:
            weight_kg = subject.getMassKg()
        heights.append(height_m)
        weights.append(weight_kg)
        age = subject.getAgeYears()
        ages.append(age)
        sex = subject.getBiologicalSex()
        sexes.append(sex)

    return heights, weights, dset_subject_names, ages, sexes


def get_text_from_log(log_path):
    with open(log_path, 'r', encoding='unicode_escape') as f:
        lines = f.readlines()
    lines_data_lens = []
    lines_removed_trial_num = []
    for line in lines:
        if line.startswith('In total, '):
            lines_data_lens.append(line)
        if line.startswith('Removed trials: '):
            lines_removed_trial_num.append(line)
    return lines_data_lens, lines_removed_trial_num


def get_total_num_of_hours(lines_of_log):
    total_hours = 0
    total_trial_num = 0
    for line in lines_of_log:
        hour_num = float(line.split(' trials, ')[1].split(' hours, ')[0])
        total_hours += hour_num
        trial_num = int(line.split('In total, ')[1].split(' trials, ')[0])
        total_trial_num += trial_num
    print('Total hours: {:.1f}'.format(total_hours))
    print('Total trials: {}'.format(total_trial_num))
    return total_trial_num


def get_total_num_of_removed_trials(lines_removed_trial_num_train, lines_removed_trial_num_test, total_trial_num):
    removed_trials = {'contact_body_num': 0, 'trial_length': 0, 'lumbar_rotation': 0, 'wrong_cop': 0,
                      'large_moving_direction_change': 0, 'jittery_sample': 0}
    for line in lines_removed_trial_num_train + lines_removed_trial_num_test:
        for key in removed_trials.keys():
            if key == 'jittery_sample':
                removed_trials[key] += int(line.split(key + '\': ')[1].split('}')[0])
            else:
                removed_trials[key] += int(line.split(key + '\': ')[1].split(', ')[0])
    total_trial_num += np.sum(list(removed_trials.values()))
    for key in removed_trials.keys():
        print(f'{key}: {removed_trials[key]} trials ({removed_trials[key] / total_trial_num * 100:.1f}%) ')


if __name__ == "__main__":
    heights_train, weights_train, subject_names_train, ages_train, sexes_train = get_heights_weights('/dataNAS/people/alanttan/mfm/data/b3d_no_arm/train_cleaned/')
    heights_test, weights_test, subject_names_test, ages_test, sexes_test = get_heights_weights('/dataNAS/people/alanttan/mfm/data/b3d_no_arm/test_cleaned/')

    print(ages_train+ages_test)
    print(sexes_train+sexes_test)

    print(np.sort(subject_names_train))
    print('Num subjects: {}'.format(len(heights_train)))
    print(np.sort(subject_names_test))
    print('Num subjects: {}'.format(len(heights_test)))
    print('Train')
    print('Heights: {:.2f} ± {:.2f}'.format(np.mean(heights_train), np.std(heights_train)))
    print('Weights: {:.1f} ± {:.1f}'.format(np.mean(weights_train), np.std(weights_train)))
    lines_data_lens_train, lines_removed_trial_num_train = get_text_from_log('/dataNAS/people/alanttan/mfm/code/slurm-14837.out')
    print('Test')
    print('Heights: {:.2f} ± {:.2f}'.format(np.mean(heights_test), np.std(heights_test)))
    print('Weights: {:.1f} ± {:.1f}'.format(np.mean(weights_test), np.std(weights_test)))
    lines_data_lens_test, lines_removed_trial_num_test = get_text_from_log('/dataNAS/people/alanttan/mfm/code/slurm-15056.out')

    total_trial_num_train = get_total_num_of_hours(lines_data_lens_train)
    print('Total train trial number: {}'.format(total_trial_num_train))
    total_trial_num_test = get_total_num_of_hours(lines_data_lens_test)
    print('Total test trial number: {}'.format(total_trial_num_test))
    get_total_num_of_removed_trials(lines_removed_trial_num_train, lines_removed_trial_num_test, total_trial_num_train+total_trial_num_test)













