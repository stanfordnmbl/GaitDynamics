import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec


def get_average_and_std(bl_true, bl_pred, ts_true, condition, params_of_interest_col_loc):
    # Average Stance phase
    true_averaged = np.mean(bl_true[condition], axis=0)[:, params_of_interest_col_loc]
    pred_averaged = np.mean(bl_pred[condition], axis=0)[:, params_of_interest_col_loc]
    ts_averaged = np.mean(ts_true[condition], axis=0)[:, params_of_interest_col_loc]
    true_std = np.std(bl_true[condition], axis=0)[:, params_of_interest_col_loc]
    pred_std = np.std(bl_pred[condition], axis=0)[:, params_of_interest_col_loc]
    ts_std = np.std(ts_true[condition], axis=0)[:, params_of_interest_col_loc]
    return true_averaged, pred_averaged, ts_averaged, true_std, pred_std, ts_std


def format_ticks(ax):
    ax.set_ylabel('RMSE of GRF Estimation (% BW)', fontdict=FONT_DICT_SMALL)
    # ax.set_xlabel('Stance Phase (%)', fontdict=FONT_DICT_SMALL)
    # ax.set_xlim(0, 100)
    ax.set_xticks([0.3, 1.3, 2.3])
    ax.set_xticklabels(['Vertical', 'Anterior-posterior', 'Medial-lateral'], fontdict=FONT_DICT_SMALL)
    # ax.set_ylim(-1, 3.5)
    # ax.set_yticks(range(-1, 4))
    # ax.set_yticklabels(range(-1, 4), fontdict=FONT_DICT_SMALL)


def draw_fig():
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    rmse_dict, rmse_std_dict = {}, {}
    for file_name in ['marker_based', 'opencap_based']:
        true_all, pred_all, pred_std_all, columns = pickle.load(open(f"results/{file_name}.pkl", "rb"))
        params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]
        rmse_dict[file_name] = []
        rmse_std_dict[file_name] = []
        for param_col_loc in params_of_interest_col_loc:
            rmse_dict[file_name].append(np.sqrt(np.mean((true_all[:, param_col_loc] - pred_all[:, param_col_loc]) ** 2)))
            rmse_std_dict[file_name].append(np.std(np.sqrt((true_all[:, param_col_loc] - pred_all[:, param_col_loc]) ** 2)))

    #         plt.figure()
    #         plt.plot(true_all[:, param_col_loc], label='True')
    #         plt.plot(pred_all[:, param_col_loc], label='Pred')
    # plt.show()

    rmse_dict['opencap_reported'] = [1.01, 0.28, 0.13]
    # rmse_dict['marker_based'] = [0.58, 0.2, 0.14]           # !!!!!!!
    # rmse_dict['opencap_based'] = [0.8, 0.22, 0.19]

    colors = [np.array(x) / 255 for x in [[123, 204, 196], [67, 162, 202], [8, 104, 172]]]

    rc('font', family='Arial')
    fig = plt.figure(figsize=(5, 3))
    for i_axis, axis in enumerate(['Vertical', 'Anterior-Posterior', 'Medial-Lateral']):
        bars = plt.bar([i_axis, i_axis + 0.3, i_axis + 0.6], [ele * 100 / 9.81 for ele in
                                                              [rmse_dict['opencap_reported'][i_axis],
                                                               rmse_dict['opencap_based'][i_axis],
                                                               rmse_dict['marker_based'][i_axis]]], color=colors, width=0.3)

    # ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_,
    #                                            capsize=0, ecolor='black', fmt='none', lolims=True,
    #                                            elinewidth=LINE_WIDTH)
    # format_errorbar_cap(caplines, 20)
    # plt.tight_layout(rect=[0., -0.01, 1, 1.01], w_pad=2, h_pad=3)
    # l2 = lines.Line2D([0.54, 0.54], [0.01, 0.96], linestyle='--', transform=fig.transFigure, color='gray')
    # fig.lines.extend([l2])

    # for i, (ax, scale) in enumerate(zip([ax_kam, ax_angles], [1, - 180/np.pi])):
    #     for i_condition, condition in enumerate(condition_list):
    #         condition_val = condition.split('_')[-1]
    #         true_averaged, pred_averaged, ts_averaged, true_std, pred_std, ts_std = [scale * ele for ele in get_average_and_std(
    #             bl_true, bl_pred, ts_true, condition, params_of_interest_col_loc)]
    #
    #         ax.plot(pred_averaged[:, i], color=colors[i_condition], linewidth=LINE_WIDTH_THICK, label=f'{condition_val} x Normal Trunk Sway - Synthetic         ')
    #         ax.grid(True, linewidth=1, alpha=0.5)
    #         if i_condition == len(condition_list) - 1:
    #             ax.plot(true_averaged[:, i], '--', color=[0.4, 0.4, 0.4], label='Normal Walking - Experimental')
    #             ax.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='gray', alpha=0.3)
    #             ax.plot(ts_averaged[:, i], '--', color='C3', label='Large Trunk Sway - Experimental')
    #             ax.fill_between(range(len(ts_averaged)), ts_averaged[:, i] - ts_std[:, i], ts_averaged[:, i] + ts_std[:, i], color='C3', alpha=0.3)
    format_axis(plt.gca())
    plt.legend(bars, ['OpenCap', 'Diffusion - Smartphone-based Kinematics', 'Diffusion - Marker-based Kinematics'], frameon=False, bbox_to_anchor=(0.25, 1.))       # fontsize=font_size,
    format_ticks(plt.gca())
    plt.savefig(f'exports/da_grf.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_fig()



