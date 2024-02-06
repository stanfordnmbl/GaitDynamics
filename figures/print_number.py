import pickle
from fig_utils import get_scores
import numpy as np
import matplotlib.pyplot as plt


test_data_name = '0124'
results_true, results_pred, columns = pickle.load(open(f"results/results_true_pred_{test_data_name}.pkl", "rb"))

params_of_interest = ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']
params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

dset_list = list(results_true.keys())
to_print = ''

for dset in dset_list:

    true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc]
    pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc]

    scores = get_scores(true_, pred_, params_of_interest, None)

    to_print += f"{dset}\t"
    if 'Li' in dset:
        to_print += '\t'
    for score in scores:
        to_print += f'{score["r2"]:.2f}\t'

    diff = np.linalg.norm(true_ - pred_, axis=1)
    to_print += f'{np.mean(diff):.2f} \pm {np.std(diff):.2f}\t'

    to_print += '\n'

print(to_print)
plt.show()






