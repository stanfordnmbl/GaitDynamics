import pickle
import numpy as np
import matplotlib.pyplot as plt
from consts import EXCLUDE_FROM_ASB
from fig_utils import get_scores, VanCriekMetaData, vancriek_bad_sub_and_trial_names


def get_results(mask_key):
    test_data_name = f'downstream_reconstruct_kinematics_{mask_key}'
    results_true, results_pred, _, _, columns, is_output_label, _, _ = \
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))
    results_true = {sub_trial_name: val for sub_trial_name, val in results_true.items() if sub_trial_name not in vancriek_bad_sub_and_trial_names}
    results_pred = {sub_trial_name: val for sub_trial_name, val in results_pred.items() if sub_trial_name not in vancriek_bad_sub_and_trial_names}

    params_of_interest = ['knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    trial_list = list(results_true.keys())

    true_all_subs, pred_all_subs = {}, {}
    for trial in trial_list:
        sub_name = trial.split('__')[0]
        if sub_name not in true_all_subs.keys():
            true_all_subs[sub_name], pred_all_subs[sub_name] = [], []
        true_ = results_true[trial][:, params_of_interest_col_loc]
        pred_ = results_pred[trial][:, params_of_interest_col_loc]

        true_all_subs[sub_name].append(np.rad2deg(true_))
        pred_all_subs[sub_name].append(np.rad2deg(pred_))

    fac_list, r2_list = [], []
    for sub_name in true_all_subs.keys():
        fac = int(meta_data.get_sub_meta(sub_name)['FAC'])
        scores = get_scores(np.concatenate(true_all_subs[sub_name], axis=0),
                            np.concatenate(pred_all_subs[sub_name], axis=0), params_of_interest, True)
        fac_list.append(fac)
        r2_list.append(scores[0]['rmse'])

    plt.figure()
    plt.scatter(fac_list, r2_list)
    plt.xlabel('FAC')
    plt.ylabel('R2')
    for i_sub, sub_name in enumerate(true_all_subs.keys()):
        plt.annotate(f"{sub_name}", (fac_list[i_sub], r2_list[i_sub]))
    plt.show()

    return true_all_subs


if __name__ == "__main__":
    name_pairs = {
        # 'ankle': 'Ankle Flexion Angle',
        # 'knee': 'Knee Flexion Angle',
        # 'knee_ankle': 'Ankle Flexion & Knee Flexion',
        # 'hip': '3-D Hip Angles',
        # 'knee_ankle_hip': 'Ankle Flexion & Knee Flexion & 3-D Hip',
        'pelvis_hip_old': '3-D Pelvis & 3-D Hip'
    }
    [print(val_) for val_ in name_pairs.values()]
    meta_data = VanCriekMetaData()
    for mask_key, _ in name_pairs.items():
        maes_all_dset = get_results(mask_key)
        # print(f'{np.mean(maes_all_dset):.2f} ± {np.std(maes_all_dset):.2f}')
