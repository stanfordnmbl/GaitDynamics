import pickle
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis, hide_axis_add_grid, \
    format_errorbar_cap, FONT_DICT_LARGE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import matplotlib as mpl


# def draw_fig2():
#     def format_ticks(ax):
#         ax.set_ylim(-10, 10)
#         ax.set_yticks(range(-10, 11, 2))
#         ax.set_yticklabels(range(-10, 11, 2), fontdict=FONT_DICT_SMALL)
#     delta_exp_dict, delta_syn_dict = pickle.load(open(f"results/da_run_faster.pkl", "rb"))
#
#     param_dict = {
#         'knee_angle_l_max': (r'$\Delta$ Peak Knee Flexion', [delta_exp_dict['knee_angle_l_max'], delta_syn_dict['knee_angle_l_max']]),
#         'hip_flexion_l_max': (r'$\Delta$ Peak Hip Flexion', [delta_exp_dict['hip_flexion_l_max'], delta_syn_dict['hip_flexion_l_max']]),
#         'hip_flexion_l_min': (r'$\Delta$ Peak Hip Extension', [delta_exp_dict['hip_flexion_l_min'], delta_syn_dict['hip_flexion_l_min']]),
#     }
#
#     colors = [np.array(x) / 255 for x in [[204, 123, 155], [8, 104, 172]]]
#     rc('font', family='Arial')
#     fig = plt.figure(figsize=(10, 3))
#     gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.4)        # , width_ratios=[5, 3, 3]
#
#     ax_list = []
#     for i_speed in range(1):
#         for i_param, (param, (param_name_formal, param_list)) in enumerate(param_dict.items()):
#             (delta_exp, delta_syn) = param_list
#             delta_exp_sorted = sorted(delta_exp, reverse=True)
#             delta_syn_sorted = [x for _, x in sorted(zip(delta_exp, delta_syn), reverse=True)]
#
#             sign_ = [np.sign(exp_)*np.sign(syn_) for exp_, syn_ in zip(delta_exp_sorted, delta_syn_sorted)]
#
#             ax = fig.add_subplot(gs[i_speed, i_param])
#             hide_axis_add_grid()
#             for i_sub in range(len(delta_exp)):
#                 ax.plot([i_sub], [delta_exp_sorted[i_sub]], 'o', color=colors[sign_[i_sub] > 0])
#
#             ax.set_ylabel(param_name_formal, fontdict=FONT_DICT_SMALL)
#             ax_list.append(ax)
#     format_ticks(ax_list)
#
#     plt.show()


# def draw_fig3():
#     delta_exp_dict, delta_syn_dict = pickle.load(open(f"results/da_run_faster.pkl", "rb"))
#     param_dict = {
#         'knee_angle_l_max': ('Peak Knee Flexion', [delta_exp_dict['knee_angle_l_max'], delta_syn_dict['knee_angle_l_max']]),
#         'hip_flexion_l_max': ('Peak Hip Flexion', [delta_exp_dict['hip_flexion_l_max'], delta_syn_dict['hip_flexion_l_max']]),
#         'hip_flexion_l_min': ('Peak Hip Extension', [delta_exp_dict['hip_flexion_l_min'], delta_syn_dict['hip_flexion_l_min']]),
#     }
#
#     colors = [np.array(x) / 255 for x in [[204, 123, 155], [8, 104, 172]]]
#     rc('font', family='Arial')
#     fig = plt.figure(figsize=(10, 3))
#     ax = plt.gca()
#
#     positive_plot, negative_plot = [], []
#     for i_param, (param, (param_name_formal, param_list)) in enumerate(param_dict.items()):
#         (delta_exp, delta_syn) = param_list
#         if param == 'hip_flexion_l_min':
#             delta_exp, delta_syn = -np.array(delta_exp), -np.array(delta_syn)
#         delta_exp, delta_syn = np.rad2deg(delta_exp), np.rad2deg(delta_syn)
#         delta_exp_sorted = sorted(delta_exp, reverse=True)
#         delta_syn_sorted = [x for _, x in sorted(zip(delta_exp, delta_syn), reverse=True)]
#
#         sign_ = [np.sign(exp_)*np.sign(syn_) for exp_, syn_ in zip(delta_exp_sorted, delta_syn_sorted)]
#
#         # ax = fig.add_axes([0.1 + 0.3 * i_param, 0.1, 0.2, 0.8])
#         hide_axis_add_grid()
#         for i_sub in range(len(delta_exp)):
#             plot_, = plt.plot([i_sub+i_param*11], [delta_exp_sorted[i_sub]], 'o', clip_on=False, color=colors[sign_[i_sub] > 0])
#             if sign_ and not positive_plot:
#                 positive_plot = plot_
#             if not sign_ and not negative_plot:
#                 negative_plot = plot_
#
#         plt.text(2+i_param*11, -5, param_name_formal, fontdict=FONT_DICT_SMALL)
#
#     ax.set_xlim(0, 31)
#     ax.set_xticks([i for i in range(0, 10)] + [i for i in range(11, 21)] + [i for i in range(22, 32)])
#     ax.set_xticklabels([], fontdict=FONT_DICT_SMALL)
#
#     ax.set_ylim(-4.01, 12.01)
#     ax.set_yticks(range(-4, 13, 4))
#     ax.set_yticklabels(range(-4, 13, 4), fontdict=FONT_DICT_SMALL)
#
#     ax.set_ylabel(r'$\Delta$ of Joint Angle from 4m/s to 5m/s', fontdict=FONT_DICT_SMALL, labelpad=30)
#     # ax.set_xlabel(
#     #     'Subject 1 - 10                                        Subject 1 - 10                                        Subject 1 - 10',
#     #     fontdict=FONT_DICT_SMALL)
#
#     # ax.annotate('', xy=(-2, 1), xycoords='axes fraction', xytext=(1, -0.1), arrowprops=dict(arrowstyle="<->", color='b'))
#     ax.arrow(-1.5, 1, 0, 3, width=0.2, clip_on=False, color=[0, 0, 0], edgecolor=None)
#     ax.arrow(-1.5, -1, 0, -3, width=0.2, clip_on=False, color=[0, 0, 0], edgecolor=None)
#     ax.text(-2.5, -4.9, 'Reduction', rotation=90, fontdict=FONT_DICT_SMALL)
#     ax.text(-2.5, 1.1, 'Increase', rotation=90, fontdict=FONT_DICT_SMALL)
#     ax.xaxis.set_ticks_position('none')
#     ax.yaxis.set_ticks_position('none')
#     ax.grid(True)
#
#     # plt.tight_layout()
#     plt.savefig(f'exports/da_run_faster.png', dpi=300, bbox_inches='tight')
#     plt.show()


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


# def draw_speeds_fig():
#     delta_exp_dict, delta_syn_dict = pickle.load(open(f"results/da_run_faster.pkl", "rb"))
#     colors = [np.array(x) / 255 for x in [[20, 145, 145], [111, 111, 111]]]
#     bars = []
#     [speed_param_syn, speed_param_exp] = pickle.load(open(f"results/da_run_faster_speeds.pkl", "rb"))
#     fig = plt.figure(figsize=(8.5, 6))
#     rc('font', family='Arial')
#     gs = gridspec.GridSpec(nrows=3, ncols=4, wspace=0.2, width_ratios=[3, 6, 1, 7])        # , width_ratios=[8, 1, 8]
#     param_y_range_dict = {
#         'stride_length': ('Strike Length', [-20, -15, -10, -5]),
#         'knee_angle_l_max': ('Knee Flexion Max', [100, 105, 110, 115]),
#         'peak_vgrf': ('Peak vGRF', [45, 50, 55, 60]),
#     }
#     for i_param, (param_, (param_name_formal, y_range)) in enumerate(param_y_range_dict.items()):
#         ax = fig.add_subplot(gs[i_param, 1])
#
#         param_syn_mean, param_syn_std = [], []
#         for speed in speed_param_syn.keys():
#             param_syn_mean.append(np.mean(speed_param_syn[speed][param_]))
#             param_syn_std.append(np.std(speed_param_syn[speed][param_]))
#         ax.errorbar(list(speed_param_syn.keys()), param_syn_mean, param_syn_std, capsize=5, linestyle='-',
#                     fmt='o', color=colors[0], elinewidth=LINE_WIDTH)
#
#         param_exp_mean, param_exp_std = [], []
#         for speed in speed_param_exp.keys():
#             param_exp_mean.append(np.mean(speed_param_exp[speed][param_]))
#             param_exp_std.append(np.std(speed_param_exp[speed][param_]))
#         ax.errorbar(list(speed_param_exp.keys()), param_exp_mean, param_exp_std, capsize=5, linestyle='-',
#                     fmt='o', color=colors[1], elinewidth=LINE_WIDTH)
#
#         # ax.plot(list(speed_param_syn.keys()), param_syn_mean, 'o-', color=colors[0], linewidth=LINE_WIDTH)
#
#         # speeds = list(speed_param_syn.keys())
#         # for i_sub in range(len(speed_param_syn[speeds[0]][param_])):
#         #     ax.plot(speeds, [speed_param_syn[speed][param_][i_sub] for speed in speeds], 'o-', color=colors[0], linewidth=LINE_WIDTH)
#
#         format_axis(ax)
#         ax.grid(True, linewidth=1, alpha=0.5, axis='y')
#         kwargs = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=12,
#                       linestyle="none", color='k', mec='k', mew=3, clip_on=False)
#         if i_param != 2:
#             ax.spines.bottom.set_visible(False)
#             ax.set_xticks([])
#             ax.plot([0], [0], transform=ax.transAxes, **kwargs)
#
#         if i_param != 0:
#             ax.plot([0], [1.01], transform=ax.transAxes, **kwargs)
#
#         if i_param == 0:
#             # ax.set_ylim(y_range[0]-3.5, y_range[-1])
#             pass
#         elif i_param == 1:
#             # ax.set_ylim(y_range[0]-0.5, y_range[-1]+2)
#             pass
#         else:
#             # ax.set_ylim(y_range[0]-2.8, y_range[-1]+2.5)
#             ax.set_xlabel('Synthetic Running Speed (m/s)', fontdict=FONT_DICT_SMALL)
#             x_speed_labels = [round(val/10, 1) for val in list(speed_param_syn.keys())]
#             ax.set_xticks(list(speed_param_syn.keys()))
#             ax.set_xticklabels(x_speed_labels, fontdict=FONT_DICT_SMALL)
#         # ax.text(36.8, np.mean(ax.get_ylim()), r'Generated Angle ($\circ$)', rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='center')
#         ax.text(28, np.mean(ax.get_ylim()), param_name_formal, rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='center')
#         # ax.set_yticks(y_range)
#         # ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
#
#     ax = fig.add_subplot(gs[:, 3])
#     ax.grid(True, linewidth=1, alpha=0.5, zorder=0)
#     mpl.rcParams['hatch.linewidth'] = 4
#     mpl.rcParams['hatch.color'] = 'white'
#
#     for i_param, (param_, (_, _)) in enumerate(param_y_range_dict.items()):
#         delta_syn = delta_syn_dict[param_]
#         delta_exp = delta_exp_dict[param_]
#         y_loc = [2.6-i_param, 2.3-i_param]
#         bars.append(ax.barh(y_loc, [np.sum(delta_syn > 0), np.sum(delta_exp > 0)], color=colors, height=0.2, zorder=100, hatch=[None, '//']))
#         bars.append(ax.barh(y_loc, [-np.sum(delta_syn < 0), -np.sum(delta_exp < 0)], color=colors, height=0.2, zorder=100, hatch=[None, '//']))
#     ax.set_xticks(np.arange(-10, 11, 2))
#     ax.set_xticklabels(list(range(10, 0, -2)) + list(range(0, 11, 2)), fontdict=FONT_DICT_SMALL)
#     ax.set_xlim([-10, 10])
#     ax.set_ylim([0, 3])
#     ax.plot([0, 0], [0, 3], 'black', linewidth=2)
#     format_axis(ax)
#     ax.spines.left.set_visible(False)
#     ax.set_yticks([])
#
#     ax.set_xlabel(r'# of Subjects with', labelpad=30)
#     ax.text(-1, -0.37, 'Decreased Angles', fontdict=FONT_DICT_SMALL, horizontalalignment='right')
#     ax.text(1, -0.37, 'Increased Angles', fontdict=FONT_DICT_SMALL, horizontalalignment='left')
#     ax.arrow(1, -0.23, 9, 0, width=0.02, head_length=1, clip_on=False, color=[0, 0, 0], edgecolor=None)
#     ax.arrow(-1, -0.23, -9, 0, width=0.02, head_length=1, clip_on=False, color=[0, 0, 0], edgecolor=None)
#     ax.text(-40, 3.25, '(a)', fontdict=FONT_DICT_LARGE)
#     ax.text(-12, 3.25, '(b)', fontdict=FONT_DICT_LARGE)
#
#     # fig.patches.extend([plt.Rectangle((0.25,0.5),0.25,0.25, fill=True, color='g',zorder=-100, transform=fig.transFigure, figure=fig)])
#
#     plt.subplots_adjust(.12, .15, .97, .9, wspace=0., hspace=0.1)
#     ax.legend([bars[-1][0], bars[-1][1]], ['GaitDynamics Generation', 'Experimental Results'], frameon=False,
#               bbox_to_anchor=(0.9, 1.14), handlelength=2.6, handleheight=1.3, ncol=1)
#     plt.savefig(f'exports/da_run_faster_1.png', dpi=300)


def draw_speed_only_fig():
    colors = [np.array(x) / 255 for x in [[20, 145, 145], [111, 111, 111]]]
    [speed_param_syn, speed_param_exp] = pickle.load(open(f"results/da_run_faster_speeds.pkl", "rb"))
    fig = plt.figure(figsize=(3.5, 8))
    rc('font', family='Arial')
    gs = gridspec.GridSpec(nrows=3, ncols=1, wspace=0.35, hspace=0.35)        # , width_ratios=[8, 1, 8]
    param_y_range_dict = {
        'stride_length': ('Synthetic Stride Length (m)', [1.7, 2.0, 2.3, 2.6, 2.9, 3.2]),
        'knee_angle_l_max': (' Synthetic Peak Knee Flexion (deg)', [90, 95, 100, 105, 110]),
        'peak_vgrf': ('Synthetic Peak vGRF (Body Weight)', [2.3, 2.4, 2.5, 2.6, 2.7, 2.8]),
    }
    for i_param, (param_, (param_name_formal, y_range)) in enumerate(param_y_range_dict.items()):
        ax = fig.add_subplot(gs[i_param])
        ax.grid(True, linewidth=LINE_WIDTH, alpha=0.5, axis='y')

        # Mean and std of syn
        param_syn_mean, param_syn_std = [], []
        for speed in speed_param_syn.keys():
            param_syn_mean.append(np.mean(speed_param_syn[speed][param_]))
            param_syn_std.append(np.std(speed_param_syn[speed][param_]))
        ax.errorbar(list(speed_param_syn.keys()), param_syn_mean, param_syn_std, capsize=6, linestyle='-',
                    fmt='o', color=colors[0], linewidth=LINE_WIDTH, elinewidth=LINE_WIDTH)

        # # Mean and std of exp
        # param_exp_mean, param_exp_std = [], []
        # for speed in speed_param_exp.keys():
        #     param_exp_mean.append(np.mean(speed_param_exp[speed][param_]))
        #     param_exp_std.append(np.std(speed_param_exp[speed][param_]))
        # ax.errorbar(list(speed_param_exp.keys()), param_exp_mean, param_exp_std, capsize=5, linestyle='-',
        #             fmt='o', color=colors[1], elinewidth=LINE_WIDTH)

        # # lines for each single subject
        # speeds = list(speed_param_syn.keys())
        # for i_sub in range(len(speed_param_syn[speeds[0]][param_])):
        #     ax.plot(speeds, [speed_param_syn[speed][param_][i_sub] for speed in speeds], 'o-', color=colors[0], linewidth=LINE_WIDTH)

        format_axis(ax)
        ax.set_xlabel('Synthetic Running Speed (m/s)', fontdict=FONT_DICT_SMALL)
        x_speed_labels = [round(val/10, 1) for val in list(speed_param_syn.keys())]
        ax.set_xticks(list(speed_param_syn.keys()))
        ax.set_xticklabels(x_speed_labels, fontdict=FONT_DICT_SMALL)
        ax.set_xlim(28.5, 51.5)

        ax.text(24.1, np.mean(ax.get_ylim()), param_name_formal, rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='center')
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylim(y_range[0], y_range[-1])

    plt.subplots_adjust(.17, .07, .98, .99)
    plt.savefig(f'exports/da_run_faster_1.png', dpi=300)


if __name__ == "__main__":
    # draw_direction_fig()
    draw_speed_only_fig()
    plt.show()
