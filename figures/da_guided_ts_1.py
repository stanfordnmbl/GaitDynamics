import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import get_scores, FONT_SIZE_SMALL
import numpy as np
import matplotlib.pyplot as plt


def get_results():
    bl_true, bl_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_baseline.pkl", "rb"))
    ts_true, ts_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_trunk_sway.pkl", "rb"))
    condition_list = list(bl_true.keys())

    params_of_interest = ['knee_moment_l_x', 'knee_moment_l_z']
    params_of_interest_names = ['Adduction moment', 'Flexion moment']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    for i, plot_name in enumerate(params_of_interest_names):
        plt.figure(figsize=(7, 3.5))
        for i_condition, condition in enumerate(condition_list):
            condition_val = condition.split('_')[-1]

            # Average gait cycles
            true_averaged = np.mean(bl_true[condition], axis=0)[:, params_of_interest_col_loc]
            pred_averaged = np.mean(bl_pred[condition], axis=0)[:, params_of_interest_col_loc]
            ts_averaged = np.mean(ts_true[condition], axis=0)[:, params_of_interest_col_loc]
            true_std = np.std(bl_true[condition], axis=0)[:, params_of_interest_col_loc]
            pred_std = np.std(bl_pred[condition], axis=0)[:, params_of_interest_col_loc]
            ts_std = np.std(ts_true[condition], axis=0)[:, params_of_interest_col_loc]

            color = np.array([1] * 3) / float(condition_val)
            plt.plot(pred_averaged[:, i], color=color, label=f'{condition_val}x Trunk Sway - Synthetic         ')
            # plt.fill_between(range(len(pred_averaged)), pred_averaged[:, i] - pred_std[:, i], pred_averaged[:, i] + pred_std[:, i], color=color, alpha=0.25)
            if i_condition == len(condition_list) - 1:
                plt.plot(true_averaged[:, i], 'C0', label='Baseline')
                plt.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='C0', alpha=0.3)
                plt.plot(ts_averaged[:, i], 'C1', label='Large Trunk Sway - Experimental')
                plt.fill_between(range(len(ts_averaged)), ts_averaged[:, i] - ts_std[:, i], ts_averaged[:, i] + ts_std[:, i], color='C1', alpha=0.3)
                plt.xlabel('Gait Cycle (%)', fontsize=FONT_SIZE_SMALL)
                plt.ylabel(plot_name + ' (Nm)', fontsize=FONT_SIZE_SMALL)
                plt.legend(frameon=False)       # fontsize=font_size,
                ax = plt.gca()
                ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_SMALL)
            plt.savefig(f'exports/da_guided_ts_{i}.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    get_results()



