import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import get_scores
import numpy as np
import matplotlib.pyplot as plt


def get_results():
    bl_true, bl_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_baseline.pkl", "rb"))
    ts_true, ts_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_trunk_sway.pkl", "rb"))
    dset_list = list(bl_true.keys())

    params_of_interest = ['knee_moment_r_x', 'knee_moment_r_z']
    params_of_interest_names = ['Adduction moment', 'Flexion moment']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    for dset in dset_list:

        # Plot every gait cycle
        # true_ = np.concatenate(bl_true[dset])[:, params_of_interest_col_loc]
        # pred_ = np.concatenate(bl_pred[dset])[:, params_of_interest_col_loc]
        # plt.figure()
        # plt.plot(true_[:, 0])
        # plt.plot(pred_[:, 0])
        # plt.title('Adduction moment')
        # plt.figure()
        # plt.plot(true_[:, 1])
        # plt.plot(pred_[:, 1])
        # plt.title('Flexion moment')
        # plt.show()

        # Average gait cycles
        true_averaged = np.mean(bl_true[dset], axis=0)[:, params_of_interest_col_loc]
        pred_averaged = np.mean(bl_pred[dset], axis=0)[:, params_of_interest_col_loc]
        ts_averaged = np.mean(ts_true[dset], axis=0)[:, params_of_interest_col_loc]
        true_std = np.std(bl_true[dset], axis=0)[:, params_of_interest_col_loc]
        pred_std = np.std(bl_pred[dset], axis=0)[:, params_of_interest_col_loc]
        ts_std = np.std(ts_true[dset], axis=0)[:, params_of_interest_col_loc]

        for i, plot_name in enumerate(params_of_interest_names):
            plt.figure()
            plt.plot(true_averaged[:, i], 'C0', label='Original')
            plt.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='C0', alpha=0.4)
            plt.plot(pred_averaged[:, i], 'C1', label='Large Trunk Sway - Synthetic')
            plt.fill_between(range(len(pred_averaged)), pred_averaged[:, i] - pred_std[:, i], pred_averaged[:, i] + pred_std[:, i], color='C1', alpha=0.4)
            plt.plot(ts_averaged[:, i], 'C2', label='Large Trunk Sway - Experimental')
            plt.fill_between(range(len(ts_averaged)), ts_averaged[:, i] - ts_std[:, i], ts_averaged[:, i] + ts_std[:, i], color='C2', alpha=0.4)
            plt.legend()
            plt.title(plot_name)
        plt.show()


if __name__ == "__main__":
    get_results()



