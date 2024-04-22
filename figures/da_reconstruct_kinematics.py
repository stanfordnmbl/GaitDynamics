import pickle
import numpy as np
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
import matplotlib.pyplot as plt


def get_results(results_true, results_pred, results_bl):
    params_of_interest = ['pelvis_tilt', 'pelvis_list', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    dset_list = list(results_true.keys())

    rmses_all_dset = []
    for dset in dset_list:
        if dset in EXCLUDE_FROM_ASB + ['Camargo2021_Formatted_No_Arm']:     # Camargo2021 has no lumbar
            continue
        bl_ = np.concatenate(list(results_bl[dset].values()))[:, params_of_interest_col_loc]
        in_gait_phase = np.all(bl_ != NOT_IN_GAIT_PHASE, axis=1)
        bl_ = bl_[in_gait_phase]

        true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc][in_gait_phase]
        pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc][in_gait_phase]

        verbose = False
        if verbose:
            plt.figure()
            plt.plot(true_[:, 2], label='True')
            plt.plot(pred_[:, 2], label='Predicted')
            plt.plot(bl_[:, 2], label='Baseline')
            plt.legend()
            plt.title('lumbar_extension')
            plt.show()

        rmses = np.rad2deg(np.sqrt(np.mean((true_ - pred_)**2, axis=0)))
        rmses_all_dset.append(rmses)
    return rmses_all_dset


if __name__ == "__main__":
    test_data_name = f'downstream_reconstruct_kinematics_ankle'
    results_true, results_pred, _, results_bl, columns, _, _, _ = \
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))

    name_pairs = {
        'ankle': 'Ankle Flexion Angle',
        'knee': 'Knee Flexion Angle',
        'knee_ankle': 'Ankle Flexion & Knee Flexion',
        'hip': '3-D Hip Angles',
        'knee_ankle_hip': 'Ankle Flexion & Knee Flexion & 3-D Hip'
    }

    for mask_key, _ in name_pairs.items():
        test_data_name = f'downstream_reconstruct_kinematics_{mask_key}'
        results_true, results_pred, _, results_bl, columns, _, _, _ = \
            pickle.load(open(f"results/{test_data_name}.pkl", "rb"))
        rmses_all_dset = get_results(results_true, results_pred, results_bl)
        print(f'{np.mean(rmses_all_dset):.2f} ± {np.std(rmses_all_dset):.2f}')

    rmses_all_dset_bl = get_results(results_true, results_bl, results_bl)
    print(f'{np.mean(rmses_all_dset_bl):.2f} ± {np.std(rmses_all_dset_bl):.2f}')


