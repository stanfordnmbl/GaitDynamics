import pickle
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib as mpl


# def draw_direction_fig():
#     def format_ticks(ax):
#         ax.set_ylim(-10, 10)
#         ax.set_yticks(range(-10, 11, 2))
#         ax.set_yticklabels(range(-10, 11, 2), fontdict=FONT_DICT_SMALL)
#     delta_exp_dict, delta_syn_dict = pickle.load(open(f"results/da_run_faster.pkl", "rb"))
#     param_dict = {
#         'knee_angle_l_max': ('Max Knee Flexion', [delta_exp_dict['knee_angle_l_max'], delta_syn_dict['knee_angle_l_max']]),
#         'hip_flexion_l_max': ('Max Hip Flexion', [delta_exp_dict['hip_flexion_l_max'], delta_syn_dict['hip_flexion_l_max']]),
#         'hip_flexion_l_min': ('Min Hip Flexion', [delta_exp_dict['hip_flexion_l_min'], delta_syn_dict['hip_flexion_l_min']]),
#         # 'ankle_angle_l_max': ('Max Hip Flexion', [delta_exp_dict['ankle_angle_l_max'], delta_syn_dict['ankle_angle_l_max']]),
#         # 'ankle_angle_l_min': ('Min Hip Flexion', [delta_exp_dict['ankle_angle_l_min'], delta_syn_dict['ankle_angle_l_min']]),
#     }
#
#     colors = [np.array(x) / 255 for x in [[111, 111, 111], [20, 145, 145]]]
#     rc('font', family='Arial')
#     plt.figure(figsize=(5, 4))
#     ax = plt.gca()
#     bars = []
#     for i_param, (param, (param_name_formal, param_list)) in enumerate(param_dict.items()):
#
#         delta_exp = delta_exp_dict[param]
#         delta_syn = delta_syn_dict[param]
#         increased_idx = delta_exp > 0
#         increased_num = np.sum(increased_idx)
#         increased_num_syn = np.sum(delta_syn[increased_idx] > 0)
#         bars.append(ax.bar(i_param, [increased_num, increased_num_syn], color=colors, label=['', ''], width=0.7))
#         decrease_num_syn = np.sum(delta_syn[~increased_idx] < 0)
#         bars.append(ax.bar(i_param, -np.array([delta_exp.shape[0] - increased_num, decrease_num_syn]), color=colors, width=0.7))
#
#     ax.set_ylabel(r'# of Subjects', labelpad=27)
#     ax.text(-.95, -1.5, 'with Decreased Angles', rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='top')
#     ax.text(-.95, 1.5, 'with Increased Angles', rotation=90, fontdict=FONT_DICT_SMALL)
#     ax.arrow(-0.8, 1.5, 0, 9, width=0.02, head_length=1, clip_on=False, color=[0, 0, 0], edgecolor=None)
#     ax.arrow(-0.8, -1.5, 0, -9, width=0.02, head_length=1, clip_on=False, color=[0, 0, 0], edgecolor=None)
#     ax.set_yticks(np.arange(-10, 11, 2))
#     ax.set_ylim([-10, 11])
#     ax.set_xticks(range(len(param_dict)))
#     ax.set_xticklabels([param_name for param_name, _ in param_dict.values()], fontdict=FONT_DICT_SMALL)
#     ax.set_xlim([-0.5, 2.5])
#     ax.xaxis.set_ticks_position('none')
#     format_axis(ax)
#     ax.spines['bottom'].set_visible(False)
#     format_ticks(ax)
#     plt.tight_layout(rect=[-0.02, -0.02, 1.02, 0.99])
#     ax.legend([bars[-1][1], bars[-1][0]], ['Correct Prediction', 'Incorrect Prediction'], frameon=False, bbox_to_anchor=(0.1, 1.), ncol=2)
#     ax.plot([-0.5, i_param+0.5], [0, 0], 'black', linewidth=2)
#     plt.savefig(f'exports/da_run_faster_0.png', dpi=300)


def format_errorbar_cap(caplines):
    for i_cap in range(2):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(10)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH*1.5)


def draw_speed_only_fig():
    colors = [np.array(x) / 255 for x in [[110, 170, 220], [70, 130, 180], [30, 90, 140], [177, 124, 90]]]
    [speed_param_syn, speed_param_exp] = pickle.load(open(f"results/da_run_faster_speeds.pkl", "rb"))
    fig = plt.figure(figsize=(11, 6))

    rc('text', usetex=True)
    plt.rc('font', family='Helvetica')

    gs = gridspec.GridSpec(nrows=2, ncols=4, wspace=0.35, hspace=0.35, width_ratios=[3, 4, 3, 4])        # , width_ratios=[8, 1, 8]
    param_y_range_dict = {
        'stride_length': ('Generated Stride Length (m)', [1.7, 2.0, 2.3, 2.6, 2.9, 3.2], [0, 0], 0),
        'peak_vgrf': (r'Generated Peak $f_v$ (\%Body Weight)', [210, 230, 250, 270], [-3, 7], -4),
        'knee_angle_l_max': ('Generated Peak Knee Flexion (deg)', [90, 100, 110, 120, 130], [0, 2], 1.5),
        'peak_apgrf': (r'Generated Peak $f_{ap}$ (\%Body Weight)', [20, 25, 30, 35, 40], [-1, 0], -1.5),
    }
    for i_param, (param_, (param_name_formal, y_range, ylim_adjustments, ylabel_adjustment)) in enumerate(param_y_range_dict.items()):
        # ax = fig.add_subplot(gs[i_param + int(np.floor((i_param+1)/2))])
        ax = fig.add_subplot(gs[i_param * 2 + 1])
        ax.grid(True, linewidth=LINE_WIDTH, alpha=0.5, axis='y')

        # Mean and std of syn
        param_syn_mean, param_syn_std = [], []
        for speed in speed_param_syn.keys():
            ratio = 1
            if 'Body Weight' in param_name_formal:
                ratio = 100
            param_syn_mean.append(np.mean(speed_param_syn[speed][param_]) * ratio)
            param_syn_std.append(np.std(speed_param_syn[speed][param_]) * ratio)
        ebar0, caplines, barlinecols = plt.errorbar(
            list(speed_param_syn.keys()), param_syn_mean, param_syn_std, capsize=6, linestyle='--',
            fmt='o', color=colors[1], alpha=0.8, linewidth=LINE_WIDTH, elinewidth=LINE_WIDTH, zorder=60)
        format_errorbar_cap(caplines)
        print(param_name_formal, param_syn_mean[-1] / param_syn_mean[0])

        # Mean and std of exp
        param_exp_mean, param_exp_std = [], []
        for speed in speed_param_exp.keys():
            param_exp_mean.append(np.mean(speed_param_exp[speed][param_]) * ratio)
            param_exp_std.append(np.std(speed_param_exp[speed][param_]) * ratio)
        ebar1, caplines, barlinecols = plt.errorbar(
            list(speed_param_exp.keys()), param_exp_mean, param_exp_std, capsize=5, linestyle='-',
            fmt='o', color=[0.4, 0.4, 0.4], alpha=1, elinewidth=LINE_WIDTH, zorder=50)
        format_errorbar_cap(caplines)

        format_axis(ax)
        ax.set_xlabel('Running Speed (m/s)', fontdict=FONT_DICT_SMALL)
        x_speed_labels = [round(val/10, 1) for val in list(speed_param_syn.keys())]
        ax.set_xticks(list(speed_param_syn.keys()))
        ax.set_xticklabels(x_speed_labels, fontdict=FONT_DICT_SMALL)
        ax.set_xlim(28.5, 51.5)

        ax.text(22.6, np.mean(ax.get_ylim())+np.mean(ylim_adjustments)+ylabel_adjustment, param_name_formal,
                rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='center')
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylim(y_range[0] + ylim_adjustments[0], y_range[-1] + ylim_adjustments[1])

        if i_param == 1:
            ax.legend([ebar0, ebar1], ['Parameter - Synthetic', 'Parameter - Experimental'], frameon=False,
                      bbox_to_anchor=(1., 1.25), ncol=2)
    plt.subplots_adjust(.08, .08, .98, .92)
    plt.savefig(f'exports/da_run_faster_1.png', dpi=300)


if __name__ == "__main__":
    # draw_direction_fig()
    draw_speed_only_fig()
    plt.show()
