import pickle
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis, hide_axis_add_grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec


def draw_fig2():
    delta_exp_dict, delta_syn_dict = pickle.load(open(f"results/da_run_faster.pkl", "rb"))

    param_dict = {
        'knee_angle_l_max': (r'$\Delta$ Peak Knee Flexion', [delta_exp_dict['knee_angle_l_max'], delta_syn_dict['knee_angle_l_max']]),
        'hip_flexion_l_max': (r'$\Delta$ Peak Hip Flexion', [delta_exp_dict['hip_flexion_l_max'], delta_syn_dict['hip_flexion_l_max']]),
        'hip_flexion_l_min': (r'$\Delta$ Peak Hip Extension', [delta_exp_dict['hip_flexion_l_min'], delta_syn_dict['hip_flexion_l_min']]),
    }

    colors = [np.array(x) / 255 for x in [[204, 123, 155], [8, 104, 172]]]
    rc('font', family='Arial')
    fig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.4)        # , width_ratios=[5, 3, 3]

    ax_list = []
    for i_speed in range(1):
        for i_param, (param, (param_name_formal, param_list)) in enumerate(param_dict.items()):
            (delta_exp, delta_syn) = param_list
            delta_exp_sorted = sorted(delta_exp, reverse=True)
            delta_syn_sorted = [x for _, x in sorted(zip(delta_exp, delta_syn), reverse=True)]

            sign_ = [np.sign(exp_)*np.sign(syn_) for exp_, syn_ in zip(delta_exp_sorted, delta_syn_sorted)]

            ax = fig.add_subplot(gs[i_speed, i_param])
            hide_axis_add_grid()
            for i_sub in range(len(delta_exp)):
                ax.plot([i_sub], [delta_exp_sorted[i_sub]], 'o', color=colors[sign_[i_sub] > 0])

            ax.set_ylabel(param_name_formal, fontdict=FONT_DICT_SMALL)
            ax_list.append(ax)
    format_ticks(ax_list)

    plt.show()


def draw_fig3():
    delta_exp_dict, delta_syn_dict = pickle.load(open(f"results/da_run_faster.pkl", "rb"))
    param_dict = {
        'knee_angle_l_max': ('Peak Knee Flexion', [delta_exp_dict['knee_angle_l_max'], delta_syn_dict['knee_angle_l_max']]),
        'hip_flexion_l_max': ('Peak Hip Flexion', [delta_exp_dict['hip_flexion_l_max'], delta_syn_dict['hip_flexion_l_max']]),
        'hip_flexion_l_min': ('Peak Hip Extension', [delta_exp_dict['hip_flexion_l_min'], delta_syn_dict['hip_flexion_l_min']]),
    }

    colors = [np.array(x) / 255 for x in [[204, 123, 155], [8, 104, 172]]]
    rc('font', family='Arial')
    fig = plt.figure(figsize=(10, 3))
    ax = plt.gca()

    positive_plot, negative_plot = [], []
    for i_param, (param, (param_name_formal, param_list)) in enumerate(param_dict.items()):
        (delta_exp, delta_syn) = param_list
        if param == 'hip_flexion_l_min':
            delta_exp, delta_syn = -np.array(delta_exp), -np.array(delta_syn)
        delta_exp, delta_syn = np.rad2deg(delta_exp), np.rad2deg(delta_syn)
        delta_exp_sorted = sorted(delta_exp, reverse=True)
        delta_syn_sorted = [x for _, x in sorted(zip(delta_exp, delta_syn), reverse=True)]

        sign_ = [np.sign(exp_)*np.sign(syn_) for exp_, syn_ in zip(delta_exp_sorted, delta_syn_sorted)]

        # ax = fig.add_axes([0.1 + 0.3 * i_param, 0.1, 0.2, 0.8])
        hide_axis_add_grid()
        for i_sub in range(len(delta_exp)):
            plot_, = plt.plot([i_sub+i_param*11], [delta_exp_sorted[i_sub]], 'o', clip_on=False, color=colors[sign_[i_sub] > 0])
            if sign_ and not positive_plot:
                positive_plot = plot_
            if not sign_ and not negative_plot:
                negative_plot = plot_

        plt.text(2+i_param*11, -5, param_name_formal, fontdict=FONT_DICT_SMALL)

    ax.set_xlim(0, 31)
    ax.set_xticks([i for i in range(0, 10)] + [i for i in range(11, 21)] + [i for i in range(22, 32)])
    ax.set_xticklabels([], fontdict=FONT_DICT_SMALL)

    ax.set_ylim(-4.01, 12.01)
    ax.set_yticks(range(-4, 13, 4))
    ax.set_yticklabels(range(-4, 13, 4), fontdict=FONT_DICT_SMALL)

    ax.set_ylabel(r'$\Delta$ of Joint Angle from 4m/s to 5m/s', fontdict=FONT_DICT_SMALL, labelpad=30)
    # ax.set_xlabel(
    #     'Subject 1 - 10                                        Subject 1 - 10                                        Subject 1 - 10',
    #     fontdict=FONT_DICT_SMALL)

    # ax.annotate('', xy=(-2, 1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<->", color='b'))
    ax.arrow(-1.5, 1, 0, 3, width=0.2, clip_on=False, color=[0, 0, 0], edgecolor=None)
    ax.arrow(-1.5, -1, 0, -3, width=0.2, clip_on=False, color=[0, 0, 0], edgecolor=None)
    ax.text(-2.5, -4.9, 'Reduction', rotation=90, fontdict=FONT_DICT_SMALL)
    ax.text(-2.5, 1.1, 'Increase', rotation=90, fontdict=FONT_DICT_SMALL)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.grid(True)

    # plt.tight_layout()
    plt.savefig(f'exports/da_run_faster.png', dpi=300, bbox_inches='tight')
    plt.show()


def format_ticks(ax):
    # ax.set_xlim(-0.5, 9.5)
    # ax.set_xticks(range(0, 10, 1))
    # ax.set_xticklabels([], fontdict=FONT_DICT_SMALL)
    # ax.set_ylabel(r'$\Delta$ Angle', fontdict=FONT_DICT_SMALL)
    ax.set_ylim(-10, 10)
    ax.set_yticks(range(-10, 11, 2))
    ax.set_yticklabels(range(-10, 11, 2), fontdict=FONT_DICT_SMALL)


def draw_fig():
    delta_exp_dict, delta_syn_dict = pickle.load(open(f"results/da_run_faster.pkl", "rb"))
    param_dict = {
        'knee_angle_l_max': ('Max Knee Flexion', [delta_exp_dict['knee_angle_l_max'], delta_syn_dict['knee_angle_l_max']]),
        'hip_flexion_l_max': ('Max Hip Flexion', [delta_exp_dict['hip_flexion_l_max'], delta_syn_dict['hip_flexion_l_max']]),
        'hip_flexion_l_min': ('Min Hip Flexion', [delta_exp_dict['hip_flexion_l_min'], delta_syn_dict['hip_flexion_l_min']]),
        # 'ankle_angle_l_max': ('Max Hip Flexion', [delta_exp_dict['ankle_angle_l_max'], delta_syn_dict['ankle_angle_l_max']]),
        # 'ankle_angle_l_min': ('Min Hip Flexion', [delta_exp_dict['ankle_angle_l_min'], delta_syn_dict['ankle_angle_l_min']]),
    }

    colors = [np.array(x) / 255 for x in [[111, 111, 111], [20, 145, 145]]]
    rc('font', family='Arial')
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    bars = []
    for i_param, (param, (param_name_formal, param_list)) in enumerate(param_dict.items()):

        delta_exp = delta_exp_dict[param]
        delta_syn = delta_syn_dict[param]
        increased_idx = delta_exp > 0
        increased_num = np.sum(increased_idx)
        increased_num_syn = np.sum(delta_syn[increased_idx] > 0)
        bars.append(ax.bar(i_param, [increased_num, increased_num_syn], color=colors, label=['', ''], width=0.7))
        decrease_num_syn = np.sum(delta_syn[~increased_idx] < 0)
        bars.append(ax.bar(i_param, -np.array([delta_exp.shape[0] - increased_num, decrease_num_syn]), color=colors, width=0.7))
        # y_tick_list.append(param_name_formal)

    ax.set_ylabel(r'# of Subjects', labelpad=27)
    ax.text(-.95, -1.5, 'with Decreased Angles', rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='top')
    ax.text(-.95, 1.5, 'with Increased Angles', rotation=90, fontdict=FONT_DICT_SMALL)
    ax.arrow(-0.8, 1.5, 0, 9, width=0.02, head_length=1, clip_on=False, color=[0, 0, 0], edgecolor=None)
    ax.arrow(-0.8, -1.5, 0, -9, width=0.02, head_length=1, clip_on=False, color=[0, 0, 0], edgecolor=None)
    ax.set_yticks(np.arange(-10, 11, 2))
    ax.set_ylim([-10, 11])
    ax.set_xticks(range(len(param_dict)))
    ax.set_xticklabels([param_name for param_name, _ in param_dict.values()], fontdict=FONT_DICT_SMALL)
    ax.set_xlim([-0.5, 2.5])
    ax.xaxis.set_ticks_position('none')
    format_axis(ax)
    ax.spines['bottom'].set_visible(False)
    format_ticks(ax)
    plt.tight_layout(rect=[-0.02, -0.02, 1.02, 0.99])
    ax.legend([bars[-1][1], bars[-1][0]], ['Correct Prediction', 'Incorrect Prediction'], frameon=False, bbox_to_anchor=(0.1, 1.), ncol=2)
    ax.plot([-0.5, i_param+0.5], [0, 0], 'black', linewidth=2)
    plt.savefig(f'exports/da_run_faster.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_fig()

