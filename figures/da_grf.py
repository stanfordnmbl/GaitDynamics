import pickle
from alant.alan_consts import NOT_IN_GAIT_PHASE
from fig_utils import get_scores, EXCLUDE_FROM_ASB
import numpy as np
import matplotlib.pyplot as plt


def get_results():
    results_true, results_pred, results_bl, columns, _, _, _ =\
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))
    dset_list = list(results_true.keys())

    params_of_interest = ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    rmse_ap, rmse_v, rmses_ml = [], [], []
    rmse_ap_bl, rmse_v_bl, rmses_ml_bl = [], [], []
    for dset in dset_list:
        if dset in EXCLUDE_FROM_ASB:
            continue

        bl_ = np.concatenate(list(results_bl[dset].values()))[:, params_of_interest_col_loc]
        in_gait_phase = np.all(bl_ != NOT_IN_GAIT_PHASE, axis=1)
        bl_ = bl_[in_gait_phase]

        true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc][in_gait_phase]
        pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc][in_gait_phase]

        # true_vals = [val_ * sub_heights[dset][key_] for key_, val_ in results_true[dset].items()]
        # pred_vals = [val_ * sub_heights[dset][key_] for key_, val_ in results_pred[dset].items()]
        # true_ = np.concatenate(true_vals)[:, params_of_interest_col_loc][in_gait_phase]
        # pred_ = np.concatenate(pred_vals)[:, params_of_interest_col_loc][in_gait_phase]

        if True:
            plt.subplots(3, 1, figsize=(10, 8))
            for i in range(3):
                plt.subplot(3, 1, i+1)
                plt.plot(true_[:, i], label='True')
                plt.plot(pred_[:, i], label='Predicted')
                plt.plot(bl_[:, i], label='Baseline')
                plt.grid()
            plt.legend()
            plt.suptitle(dset)

        scores_bl = get_scores(true_, bl_, params_of_interest, None)
        rmse_ap_bl.append(scores_bl[0]['rmse'])
        rmse_v_bl.append(scores_bl[1]['rmse'])
        rmses_ml_bl.append(scores_bl[2]['rmse'])

        scores = get_scores(true_, pred_, params_of_interest, None)
        rmse_ap.append(scores[0]['rmse'])
        rmse_v.append(scores[1]['rmse'])
        rmses_ml.append(scores[2]['rmse'])

    print(f'{np.mean(rmses_ml_bl):.2f} ± {np.std(rmses_ml_bl):.2f}')
    print(f'{np.mean(rmses_ml):.2f} ± {np.std(rmses_ml):.2f}')

    print(f'{np.mean(rmse_ap_bl):.2f} ± {np.std(rmse_ap_bl):.2f}')
    print(f'{np.mean(rmse_ap):.2f} ± {np.std(rmse_ap):.2f}')

    print(f'{np.mean(rmse_v_bl):.2f} ± {np.std(rmse_v_bl):.2f}')
    print(f'{np.mean(rmse_v):.2f} ± {np.std(rmse_v):.2f}')

    plt.show()


test_data_name = 'downstream_grf'
if __name__ == "__main__":
    get_results()



