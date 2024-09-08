import pickle
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, format_axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


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
    ax.set_ylabel('Mean Absolute Error of GRF (% BW)', fontdict=FONT_DICT_SMALL)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10], fontdict=FONT_DICT_SMALL)
    ax.set_xticks([0.3, 1.3, 2.3])
    ax.set_xticklabels(['Vertical', 'Anterior-posterior', 'Medial-lateral'], fontdict=FONT_DICT_SMALL)


def draw_fig():
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    mae_dict, mae_std_dict = {}, {}
    for file_name in ['uhlrich_marker_based_none', 'opencap_based']:
        true_all, pred_all, pred_std_all, columns = pickle.load(open(f"results/{file_name}.pkl", "rb"))
        params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]
        mae_dict[file_name] = []
        mae_std_dict[file_name] = []
        for param_col_loc in params_of_interest_col_loc:
            mae_dict[file_name].append(np.mean(np.abs(true_all[:, param_col_loc] - pred_all[:, param_col_loc])))
            mae_std_dict[file_name].append(np.std(np.abs((true_all[:, param_col_loc] - pred_all[:, param_col_loc]))))

            plt.figure()
            plt.plot(true_all[:, param_col_loc], label='True')
            plt.plot(pred_all[:, param_col_loc], label='Pred')
    plt.show()

    mae_dict['opencap_reported'] = [0.82, 0.21, 0.11]

    colors = [np.array(x) / 255 for x in [[155, 155, 155], [0, 155, 155], [0, 111, 111]]]

    rc('font', family='Arial')
    fig = plt.figure(figsize=(5, 3.5))
    for i_axis, axis in enumerate(['Vertical', 'Anterior-Posterior', 'Medial-Lateral']):
        bars = plt.bar([i_axis, i_axis + 0.3, i_axis + 0.6], [ele * 100 / 9.81 for ele in
                                                              [mae_dict['opencap_reported'][i_axis],
                                                               mae_dict['opencap_based'][i_axis],
                                                               mae_dict['uhlrich_marker_based_none'][i_axis]]], color=colors, width=0.3)

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
    # plt.legend(bars, ['Physics-based Foot-Ground Contact Model', 'Diffusion - Smartphone-based Kinematics', 'Diffusion - Marker-based Kinematics'],
    plt.legend(bars, ['Physics-based Foot-Ground Contact Model', 'Proposed Model - Smartphone Input', 'Proposed Model - Marker Data Input'],
               frameon=False, fontsize=FONT_SIZE_SMALL, bbox_to_anchor=(0.2, 1.06))       # fontsize=font_size,
    format_ticks(plt.gca())
    plt.savefig(f'exports/da_grf.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_fig()



