import pickle
import numpy as np
import matplotlib.pyplot as plt


test_data_name = '0124'
results_true, results_pred, columns, _, _, _ =\
    pickle.load(open(f"results/results_true_pred_{test_data_name}.pkl", "rb"))

params_of_interest = ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']
params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

dset_list = list(results_true.keys())

for dset in dset_list:
    true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc]
    pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc]

    for i_axis in range(3):
        plt.figure()
        plt.plot(true_[:, i_axis], label='Measured')
        plt.plot(pred_[:, i_axis], label='Predicted')
        plt.ylabel('Force (Body Weight)', fontdict={'fontsize': 14})
        plt.xlabel('Time (Frames)', fontdict={'fontsize': 14})
        plt.legend(fontsize=14)
    plt.show()








