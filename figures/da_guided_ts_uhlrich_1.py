import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import pandas as pd
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis, FONT_DICT_LARGE, \
    FONT_DICT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec


def export_to_excel(bl_true, bl_pred, ts_true, condition_list, params_of_interest, columns, filename):
    param_names = {
        'knee_moment_l_x': 'Knee Adduction Moment (% BW·BH)',
        'lumbar_bending': 'Medial-Lateral Trunk Angle (°)'
    }
    
    data = []
    params_col_loc = [columns.index(col) for col in params_of_interest]
    
    # Get normal walking and large trunk sway data (use first condition for reference)
    first_condition = condition_list[0]
    true_avg, _, ts_avg, true_std, _, ts_std = get_average_and_std(bl_true, bl_pred, ts_true, first_condition, params_col_loc)
    
    # Add normal walking data
    for i, param in enumerate(params_of_interest):
        scale = -180/np.pi if param == 'lumbar_bending' else 1
        for phase in range(len(true_avg)):
            data.append({
                'Condition': 'Normal Walking - Experimental',
                'Parameter': param_names[param],
                'Stance Phase (%)': phase,
                'Mean': true_avg[phase, i] * scale,
                'One Standard Deviation': np.abs(true_std[phase, i] * scale)
            })
    
    # Add large trunk sway data
    for i, param in enumerate(params_of_interest):
        scale = -180/np.pi if param == 'lumbar_bending' else 1
        for phase in range(len(ts_avg)):
            data.append({
                'Condition': 'Large Trunk Sway - Experimental',
                'Parameter': param_names[param],
                'Stance Phase (%)': phase,
                'Mean': ts_avg[phase, i] * scale,
                'One Standard Deviation': np.abs(ts_std[phase, i] * scale)
            })
    
    # Add synthetic conditions
    for condition in condition_list:
        condition_val = float(condition.split('_')[-1])
        _, pred_avg, _, _, pred_std, _ = get_average_and_std(bl_true, bl_pred, ts_true, condition, params_col_loc)
        
        for i, param in enumerate(params_of_interest):
            scale = -180/np.pi if param == 'lumbar_bending' else 1
            for phase in range(len(pred_avg)):
                data.append({
                    'Condition': f'{condition_val:.1f}x Normal Trunk Sway - Synthetic',
                    'Parameter': param_names[param],
                    'Stance Phase (%)': phase,
                    'Mean': pred_avg[phase, i] * scale,
                    'One Standard Deviation': np.abs(pred_std[phase, i] * scale)
                })
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(data).to_excel(filename, index=False)
    print(f"Exported to: {filename}")

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bl_true, bl_pred, _, _, columns, _, _, _, = \
        pickle.load(open(os.path.join(script_dir, "results", "da_guided_baseline.pkl"), "rb"))
    ts_true, ts_pred, _, _, columns, _, _, _, = \
        pickle.load(open(os.path.join(script_dir, "results", "da_guided_trunk_sway.pkl"), "rb"))
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
                print('Pred 1st peak diff {:.1f}'.format(true_averaged[:50, i].max() - pred_averaged[:50, i].max()))
                print('Pred 2st peak diff {:.1f}'.format(true_averaged[51:, i].max() - pred_averaged[51:, i].max()))
            elif i == 1:
                print('Pred 1st peak value {:.1f}'.format(pred_averaged[:50, i].max()))
            ax.plot(pred_averaged[:, i], '--', color=colors[i_condition], linewidth=LINE_WIDTH_THICK, label='{:.1f} x Normal Trunk Sway - Synthetic         '.format(condition_val))
            if i_condition == len(condition_list) - 1:
                ax.plot(true_averaged[:, i], '-', color=[0.4, 0.4, 0.4], label='Normal Walking - Experimental')
                ax.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='gray', alpha=0.3)
                ax.plot(ts_averaged[:, i], '-', color=colors[-1], label='Large Trunk Sway - Experimental')
                ax.fill_between(range(len(ts_averaged)), ts_averaged[:, i] - ts_std[:, i], ts_averaged[:, i] + ts_std[:, i], color=colors[-1], alpha=0.3)
                if i == 0:
                    print('Exp 1st peak diff {:.1f}'.format(true_averaged[:, i].max() - ts_averaged[:, i].max()))
                    print('Exp 2st peak diff {:.1f}'.format(true_averaged[51:, i].max() - ts_averaged[51:, i].max()))
                elif i == 1:
                    print('Exp normal gait 1st peak value {:.1f}'.format(true_averaged[:, i].max()))
                    print('Exp large trunk sway 1st peak value {:.1f}'.format(ts_averaged[:, i].max()))
            format_axis(ax)

    # Export data
    filename = os.path.join(script_dir, 'exports', 'Figure 3 Source Data.xlsx')
    export_to_excel(bl_true, bl_pred, ts_true, condition_list, params_of_interest, columns, filename)

    plt.legend(frameon=False, bbox_to_anchor=(1.7, 1.6), ncol=2)       # fontsize=font_size,
    format_ticks(ax_kam, ax_angles)
    plt.savefig(os.path.join(script_dir, 'exports', 'da_guided_ts.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


def draw_supplementary_fig():
    def format_ticks_supplementary(ax_kam, ax_angles):
        for ax in [ax_kam, ax_angles]:
            ax.set_xlabel('Stance Phase (%)', fontdict=FONT_DICT_SMALL)
            ax.set_xlim(0, 100)
            ax.set_xticks(range(0, 101, 20))
            ax.set_xticklabels(range(0, 101, 20), fontdict=FONT_DICT_SMALL)
        ax_kam.set_ylim(-1, 4)
        ax_kam.set_yticks(range(-1, 5))
        ax_kam.set_yticklabels(range(-1, 5), fontdict=FONT_DICT_SMALL)
        ax_angles.set_ylim(-23, 35)
        ax_angles.set_yticks(range(-20, 51, 10))
        ax_kam.set_ylabel('Knee Adduction Moment (% BW·BH)', fontdict=FONT_DICT_SMALL)
        ax_angles.set_ylabel(r'Medial-Lateral Trunk Angle ($^\circ$)', fontdict=FONT_DICT_SMALL)    # \nTo Contralateral Leg        To Ipsilateral Leg
        ax_angles.set_yticklabels(range(-20, 51, 10), fontdict=FONT_DICT_SMALL)
        ax_angles.annotate('', xy=(1.28, 0.6), xycoords='axes fraction', xytext=(1.08, 0.6), fontsize=20,
                        arrowprops=dict(facecolor='gray', edgecolor='gray', shrink=0.05),)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bl_true, bl_pred, _, _, columns, _, _, _, = \
        pickle.load(open(os.path.join(script_dir, "results", "da_guided_baseline.pkl"), "rb"))
    ts_true, ts_pred, _, _, columns, _, _, _, = \
        pickle.load(open(os.path.join(script_dir, "results", "da_guided_trunk_sway.pkl"), "rb"))
    condition_list = list(bl_true.keys())
    # condition_list = [condition for condition in condition_list if condition != '_1']

    params_of_interest = ['knee_moment_l_x', 'lumbar_bending']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    # colors = [np.array(x) / 255 for x in [[110, 170, 220], [80, 140, 190], [50, 110, 160], [20, 80, 130], [177, 124, 90]]]
    colors = [np.array(x) / 255 for x in [[110, 170, 240], [110, 170, 220], [70, 130, 180], [30, 90, 140], [177, 124, 90]]]

    rc('font', family='Arial')
    fig = plt.figure(figsize=(9, 11))
    gs = gridspec.GridSpec(nrows=4, ncols=2, wspace=0.5, hspace=0.6, top=0.92, bottom=0.01, left=0.07, right=0.98)
    for i_condition, condition in enumerate(condition_list):
        ax_kam = fig.add_subplot(gs[i_condition, 1])
        ax_angles = fig.add_subplot(gs[i_condition, 0])
        for i, (ax, scale) in enumerate(zip([ax_kam, ax_angles], [1, - 180/np.pi])):
            condition_val = float(condition.split('_')[-1])
            true_averaged, pred_averaged, ts_averaged, true_std, pred_std, ts_std = [scale * ele for ele in get_average_and_std(
                bl_true, bl_pred, ts_true, condition, params_of_interest_col_loc)]

            if i == 0:
                print('Pred 1st peak diff {:.1f}'.format(true_averaged[:, i].max() - pred_averaged[:, i].max()))
                print('Pred 2st peak diff {:.1f}'.format(true_averaged[51:, i].max() - pred_averaged[51:, i].max()))
            elif i == 1:
                print('Pred 1st peak value {:.1f}'.format(pred_averaged[:, i].max()))
            syn_plot, = ax.plot(pred_averaged[:, i], '--', color=colors[i_condition],
                    linewidth=LINE_WIDTH_THICK, label='{:.1f} x Normal Trunk Sway - Synthetic         '.format(condition_val), zorder=20)
            ax.fill_between(range(len(pred_averaged)), pred_averaged[:, i] - pred_std[:, i], pred_averaged[:, i] + pred_std[:, i], color=colors[1], alpha=0.3, zorder=18)

            bl_plot, = ax.plot(true_averaged[:, i], '-', color=[0.4, 0.4, 0.4], label='Normal Walking - Experimental', zorder=10)
            ax.fill_between(range(len(true_averaged)), true_averaged[:, i] - true_std[:, i], true_averaged[:, i] + true_std[:, i], color='gray', alpha=0.3)
            ts_plot, = ax.plot(ts_averaged[:, i], '-', color=colors[-1], label='Large Trunk Sway - Experimental')
            ax.fill_between(range(len(ts_averaged)), ts_averaged[:, i] - ts_std[:, i], ts_averaged[:, i] + ts_std[:, i], color=colors[-1], alpha=0.3)
            format_axis(ax)
            if i == 0:
                ax.legend(frameon=False, bbox_to_anchor=(-0.4, 1.15), handles=[syn_plot])
        format_ticks_supplementary(ax_kam, ax_angles)
    
    plt.legend(frameon=False, bbox_to_anchor=(0.935, 6.25), ncol=1, handles=[bl_plot, ts_plot])

    plt.savefig(os.path.join(script_dir, 'exports', 'da_guided_ts_supplementary.jpg'), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_fig()
    draw_supplementary_fig()



