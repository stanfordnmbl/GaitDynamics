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
    sub_list = list(bl_true.keys())

    params_of_interest = ['knee_moment_l_x', 'knee_moment_l_z']
    # params_of_interest = ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']
    params_of_interest_names = ['Adduction moment', 'Flexion moment']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    for sub in sub_list:

        # Average gait cycles
        true_averaged = np.mean(bl_true[sub], axis=0)[:, params_of_interest_col_loc]
        pred_averaged = np.mean(bl_pred[sub], axis=0)[:, params_of_interest_col_loc]
        ts_averaged = np.mean(ts_true[sub], axis=0)[:, params_of_interest_col_loc]
        true_std = np.std(bl_true[sub], axis=0)[:, params_of_interest_col_loc]
        pred_std = np.std(bl_pred[sub], axis=0)[:, params_of_interest_col_loc]
        ts_std = np.std(ts_true[sub], axis=0)[:, params_of_interest_col_loc]

        font_size = 11
        for i, plot_name in enumerate(params_of_interest_names):
            plt.figure(figsize=(7, 3.5))
            plt.plot(true_averaged[:, i], 'gray', label='Baseline')
            plt.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='gray', alpha=0.3)
            plt.plot(pred_averaged[:, i], 'C0', label='Large Trunk Sway - Synthetic       ')
            plt.fill_between(range(len(pred_averaged)), pred_averaged[:, i] - pred_std[:, i], pred_averaged[:, i] + pred_std[:, i], color='C0', alpha=0.3)
            plt.plot(ts_averaged[:, i], 'C2', label='Large Trunk Sway - Experimental')
            plt.fill_between(range(len(ts_averaged)), ts_averaged[:, i] - ts_std[:, i], ts_averaged[:, i] + ts_std[:, i], color='C2', alpha=0.3)
            plt.xlabel('Gait Cycle (%)', fontsize=font_size)
            plt.ylabel(plot_name + ' (Nm)', fontsize=font_size)
            plt.legend(fontsize=font_size, frameon=False)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=font_size)
            plt.savefig(f'exports/da_guided_ts_{i}.png', bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    get_results()



