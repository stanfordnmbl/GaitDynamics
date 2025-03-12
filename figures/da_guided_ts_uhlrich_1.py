import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis, FONT_DICT_LARGE, \
    FONT_DICT
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


def format_ticks(ax_kam, ax_angles):
    ax_kam.set_ylabel('Knee Adduction Moment (% BW·BH)', fontdict=FONT_DICT_SMALL)
    ax_angles.set_ylabel(r'Medial-Lateral Trunk Angle ($^\circ$)', fontdict=FONT_DICT_SMALL)    # \nTo Contralateral Leg        To Ipsilateral Leg
    for ax in [ax_kam, ax_angles]:
        ax.set_xlabel('Stance Phase (%)', fontdict=FONT_DICT_SMALL)
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xticklabels(range(0, 101, 20), fontdict=FONT_DICT_SMALL)
    ax_kam.set_ylim(-1, 4)
    ax_kam.set_yticks(range(-1, 5))
    ax_kam.set_yticklabels(range(-1, 5), fontdict=FONT_DICT_SMALL)
    ax_angles.set_ylim(-23, 35)
    ax_angles.set_yticks(range(-20, 41, 10))
    ax_angles.set_yticklabels(range(-20, 41, 10), fontdict=FONT_DICT_SMALL)
    ax_angles.set_title('Increased Trunk Sway', fontdict=FONT_DICT, pad=20)
    ax_kam.set_title('Reduced Knee Loading', fontdict=FONT_DICT, pad=20)
    ax_angles.annotate('', xy=(1.28, 0.6), xycoords='axes fraction', xytext=(1.08, 0.6), fontsize=20,
                       arrowprops=dict(facecolor='gray', edgecolor='gray', shrink=0.05),)


def draw_fig():
    bl_true, bl_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_baseline.pkl", "rb"))
    ts_true, ts_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_trunk_sway.pkl", "rb"))
    condition_list = list(bl_true.keys())
    condition_list = [condition for condition in condition_list if condition != '_1']

    params_of_interest = ['knee_moment_l_x', 'lumbar_bending']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    colors = [np.array(x) / 255 for x in [[110, 170, 220], [70, 130, 180], [30, 90, 140], [177, 124, 90]]]

    rc('font', family='Arial')
    fig = plt.figure(figsize=(9, 4))
    gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.5, width_ratios=[4, 4], height_ratios=[1, 4], left=0.07, right=0.98)
    ax_kam = fig.add_subplot(gs[1, 1])
    ax_angles = fig.add_subplot(gs[1, 0])
    for i, (ax, scale) in enumerate(zip([ax_kam, ax_angles], [1, - 180/np.pi])):
        for i_condition, condition in enumerate(condition_list):
            condition_val = float(condition.split('_')[-1])
            true_averaged, pred_averaged, ts_averaged, true_std, pred_std, ts_std = [scale * ele for ele in get_average_and_std(
                bl_true, bl_pred, ts_true, condition, params_of_interest_col_loc)]

            if i == 0:
                print('Pred 1st peak diff {:.1f}'.format(true_averaged[:, i].max() - pred_averaged[:, i].max()))
                print('Pred 2st peak diff {:.1f}'.format(true_averaged[51:, i].max() - pred_averaged[51:, i].max()))
            elif i == 1:
                print('Pred 1st peak value {:.1f}'.format(pred_averaged[:, i].max()))
            ax.plot(pred_averaged[:, i], '--', color=colors[i_condition], linewidth=LINE_WIDTH_THICK, label='{:.1f} x Normal Trunk Sway - Synthetic         '.format(condition_val))
            # ax.grid(True, linewidth=1, alpha=0.5)
            if i_condition == len(condition_list) - 1:
                ax.plot(true_averaged[:, i], '-', color=[0.4, 0.4, 0.4], label='Normal Walking - Experimental')
                ax.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='gray', alpha=0.3)
                ax.plot(ts_averaged[:, i], '-', color=colors[3], label='Large Trunk Sway - Experimental')
                ax.fill_between(range(len(ts_averaged)), ts_averaged[:, i] - ts_std[:, i], ts_averaged[:, i] + ts_std[:, i], color=colors[-1], alpha=0.3)
                if i == 0:
                    print('Exp 1st peak diff {:.1f}'.format(true_averaged[:, i].max() - ts_averaged[:, i].max()))
                    print('Exp 2st peak diff {:.1f}'.format(true_averaged[51:, i].max() - ts_averaged[51:, i].max()))
                elif i == 1:
                    print('Exp normal gait 1st peak value {:.1f}'.format(true_averaged[:, i].max()))
                    print('Exp large trunk sway 1st peak value {:.1f}'.format(ts_averaged[:, i].max()))
            format_axis(ax)

    plt.legend(frameon=False, bbox_to_anchor=(1.7, 1.6), ncol=2)       # fontsize=font_size,
    format_ticks(ax_kam, ax_angles)
    plt.savefig(f'exports/da_guided_ts.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_fig()



