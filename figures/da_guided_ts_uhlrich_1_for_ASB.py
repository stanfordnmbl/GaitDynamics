import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        p = mpatches.FancyArrow(0.5*width, 1.9*height, 0, -2.*height, length_includes_head=True, head_width=0.3*width, overhang=.6)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def get_average_and_std(bl_true, bl_pred, ts_true, condition, params_of_interest_col_loc):
    # Average Stance phase
    true_averaged = np.mean(bl_true[condition], axis=0)[:, params_of_interest_col_loc]
    pred_averaged = np.mean(bl_pred[condition], axis=0)[:, params_of_interest_col_loc]
    ts_averaged = np.mean(ts_true[condition], axis=0)[:, params_of_interest_col_loc]
    true_std = np.std(bl_true[condition], axis=0)[:, params_of_interest_col_loc]
    pred_std = np.std(bl_pred[condition], axis=0)[:, params_of_interest_col_loc]
    ts_std = np.std(ts_true[condition], axis=0)[:, params_of_interest_col_loc]
    return true_averaged, pred_averaged, ts_averaged, true_std, pred_std, ts_std


def format_ticks(ax_kam):
    ax_kam.set_ylabel('Knee Adduction Moment (% BW·BH)', fontdict=FONT_DICT_SMALL)
    ax_kam.yaxis.set_label_coords(-0.1, 0.43)
    ax_kam.set_xlabel('Stance Phase (%)', fontdict=FONT_DICT_SMALL)
    ax_kam.set_xlim(0, 100)
    ax_kam.set_xticks(range(0, 101, 20))
    ax_kam.set_xticklabels(range(0, 101, 20), fontdict=FONT_DICT_SMALL)
    ax_kam.set_ylim(-1, 3)
    ax_kam.set_yticks(range(-1, 4))
    ax_kam.set_yticklabels(range(-1, 4), fontdict=FONT_DICT_SMALL)


def curve_fun(x, y_start, scale):
    y = [y_start + scale * np.log(x_-x[0]+1) for x_ in x]
    return y


def draw_fig():
    bl_true, bl_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_baseline_stable.pkl", "rb"))
    ts_true, ts_pred, _, _, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_trunk_sway_stable.pkl", "rb"))
    condition_list = list(bl_true.keys())

    params_of_interest = ['knee_moment_l_x', 'lumbar_bending']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    color_bl = [.4, .4, .4]
    color_exp = [.8, .6, .3]
    color_pred = [.2, .6, .7]
    rc('font', family='Arial')
    fig = plt.figure(figsize=(3.5, 2.8))
    ax = fig.gca()
    i_param = 0
    for i_condition, condition in enumerate(condition_list[-1:]):
        condition_val = condition.split('_')[-1]
        true_averaged, pred_averaged, ts_averaged, true_std, pred_std, ts_std = [ele for ele in get_average_and_std(
            bl_true, bl_pred, ts_true, condition, params_of_interest_col_loc)]

        ax.plot(pred_averaged[:, i_param], color=color_pred, linewidth=LINE_WIDTH_THICK, label=f'Moment Reduction - Predicted')
        if i_condition == 0:
            ax.plot(true_averaged[:, i_param], linewidth=LINE_WIDTH_THICK, color=color_bl, label='Normal Walking - Experimental')
            ax.plot(ts_averaged[:, i_param], linewidth=LINE_WIDTH_THICK, color=color_exp, label='Moment Reduction - Experimental')
            for x_ in [60]:
                arrow1 = plt.arrow(x_, true_averaged[x_, i_param] - 0.05, 0, ts_averaged[x_, i_param] - true_averaged[x_, i_param] + 0.2,
                                   color=1-0.5*(1-np.array(color_exp)), linewidth=4, length_includes_head=True, head_width=5, head_length=0.2)
        for x_ in [50]:
            arrow2 = plt.arrow(x_, true_averaged[x_, i_param] - 0.05, 0, pred_averaged[x_, i_param] - true_averaged[x_, i_param] + 0.2,
                               color=1-0.5*(1-np.array(color_pred)), linewidth=4, length_includes_head=True, head_width=5, head_length=0.2, label='')
        # ax.grid(True, linewidth=1, alpha=0.5)
        format_axis(ax)

    format_ticks(ax)
    plt.tight_layout(rect=[-0.02, -0.02, 1.02, 1.02])
    ax.plot(true_averaged[:, i_param], linewidth=LINE_WIDTH_THICK, color=color_bl, label='Normal Walking - Experimental')

    s, e = 14, 36
    ax.plot(range(s, e), curve_fun(range(s, e), true_averaged[s, i_param], -0.14), '--', linewidth=LINE_WIDTH, color=color_bl, label='Normal Walking - Experimental')
    ax.text(e+1, curve_fun(range(s, e), true_averaged[s, i_param], -0.14)[-1]-0.1, 'In-Lab Baseline', color=color_bl, fontdict=FONT_DICT_SMALL)

    s, e = 14, 36
    ax.plot(range(s, e), curve_fun(range(s, e), ts_averaged[s, i_param], -0.1), '--', linewidth=LINE_WIDTH, color=color_exp)
    ax.text(e+1, curve_fun(range(s, e), ts_averaged[s, i_param], -0.1)[-1]-0.1, 'In-Lab Intervention', color=color_exp, fontdict=FONT_DICT_SMALL)

    s, e = 14, 36
    ax.plot(range(s, e), curve_fun(range(s, e), pred_averaged[s, i_param], -0.1), '--', linewidth=LINE_WIDTH, color=color_pred)
    ax.text(e+1, curve_fun(range(s, e), pred_averaged[s, i_param], -0.1)[-1]-0.1, 'Model Prediction', color=color_pred, fontdict=FONT_DICT_SMALL)

    plt.savefig(f'exports/da_guided_ts.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_fig()



