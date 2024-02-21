import pickle
from fig_utils import get_scores, DSETS_FOR_ASB
import numpy as np


def get_results():
    results_true, results_pred, columns, _ =\
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))

    params_of_interest = ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    rmse_ap, rmse_v, rmses_ml = [], [], []
    for dset in DSETS_FOR_ASB:

        true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc]
        pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc]

        scores = get_scores(true_, pred_, params_of_interest, None)
        rmse_ap.append(scores[0]['rmse'])
        rmse_v.append(scores[1]['rmse'])
        rmses_ml.append(scores[2]['rmse'])

    print(rmse_ap)
    print(rmse_v)
    print(rmses_ml)

    print(f'{np.mean(rmses_ml):.2f} ± {np.std(rmses_ml):.2f}')
    print(f'{np.mean(rmse_ap):.2f} ± {np.std(rmse_ap):.2f}')
    print(f'{np.mean(rmse_v):.2f} ± {np.std(rmse_v):.2f}')


test_data_name = 'downstream_grf'
if __name__ == "__main__":
    get_results()



