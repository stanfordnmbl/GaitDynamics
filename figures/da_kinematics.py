import pickle
import numpy as np
import matplotlib.pyplot as plt
from consts import EXCLUDE_FROM_ASB


def get_results(mask_key):
    test_data_name = f'downstream_reconstruct_kinematics_{mask_key}'
    results_true, results_pred, _, _, columns, is_output_label, _, _ = \
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))

    params_of_interest = ['pelvis_tilt', 'pelvis_list', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    dset_list = list(results_true.keys())

    maes_all_dset = []
    for dset in dset_list:
        if dset in EXCLUDE_FROM_ASB:
            continue
        true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc]
        pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc]

        maes = np.rad2deg(np.mean(np.abs((true_ - pred_))))
        maes_all_dset.append(maes)

    return maes_all_dset


if __name__ == "__main__":
    name_pairs = {
        'ankle': 'Ankle Flexion Angle',
        'knee': 'Knee Flexion Angle',
        'knee_ankle': 'Ankle Flexion & Knee Flexion',
        'hip': '3-D Hip Angles',
        'knee_ankle_hip': 'Ankle Flexion & Knee Flexion & 3-D Hip'
    }
    [print(val_) for val_ in name_pairs.values()]
    for mask_key, _ in name_pairs.items():
        maes_all_dset = get_results(mask_key)
        print(f'{np.mean(maes_all_dset):.2f} ± {np.std(maes_all_dset):.2f}')