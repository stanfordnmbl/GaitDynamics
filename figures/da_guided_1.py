import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import get_scores
import numpy as np
import matplotlib.pyplot as plt
from args import parse_opt, set_with_arm_opt


def get_results():
    bl_true, bl_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_.pkl", "rb"))
    dset_list = list(bl_true.keys())

    # params_of_interest = ['calcn_r_force_vx', 'calcn_l_force_vx']
    # params_of_interest = ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']
    # params_of_interest = ['hip_flexion_r', 'hip_flexion_l']
    params_of_interest = ['pelvis_tx', 'calcn_r_force_normed_cop_x', 'hip_flexion_r', 'knee_angle_r', 'ankle_angle_r']
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
        true_std = np.std(bl_true[dset], axis=0)[:, params_of_interest_col_loc]
        pred_std = np.std(bl_pred[dset], axis=0)[:, params_of_interest_col_loc]

        for i, plot_name in enumerate(params_of_interest):
            plt.figure()
            plt.plot(true_averaged[:, i], 'C0', label='')
            plt.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='C0', alpha=0.4)
            plt.plot(pred_averaged[:, i], 'C1', label='')
            plt.fill_between(range(len(pred_averaged)), pred_averaged[:, i] - pred_std[:, i], pred_averaged[:, i] + pred_std[:, i], color='C1', alpha=0.4)
            plt.title(plot_name)
            plt.legend()
        plt.show()


if __name__ == "__main__":
    opt = parse_opt()
    get_results()



