import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import pickle
import pandas as pd
from fig_utils import FONT_DICT_SMALL, LINE_WIDTH, format_axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
from model.utils import data_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_omnicontrol(speeds=[3, 3.4, 3.8, 4.2, 4.6, 5]):
    JOINT_POS_MAP = {
        'MidHip': 0,
        'LHip': 1, 'LKnee': 4, 'LAnkle': 7, 'LFoot': 10,
        'RHip': 2, 'RKnee': 5, 'RAnkle': 8, 'RFoot': 11,
        'LShoulder': 16, 'LElbow': 18, 'LWrist': 20,
        'RShoulder': 17, 'RElbow': 19, 'RWrist': 21,
        'spine1': 3, 'spine2': 6, 'spine3': 9, 'Neck': 12, 'Head': 15,
        'LCollar': 13, 'Rcollar': 14,
    }

    def get_right_knee_angle_from_joint_center(positions, right_stance_starts):
        r_hip = positions[JOINT_POS_MAP['RHip']]
        r_knee = positions[JOINT_POS_MAP['RKnee']]
        r_ankle = positions[JOINT_POS_MAP['RAnkle']]
        vector_0 = (r_hip - r_knee).swapaxes(0, 1)
        vector_1 = (r_ankle - r_knee).swapaxes(0, 1)
        angle_profile = np.arccos(np.sum(np.multiply(vector_0, vector_1), axis=-1) / (np.linalg.norm(vector_0, axis=-1) * np.linalg.norm(vector_1, axis=-1)))
        max_flexion_angle = []
        # plt.figure()
        for i_stance in range(len(right_stance_starts)-1):
            max_flexion_angle.append(np.min(angle_profile[right_stance_starts[i_stance]:right_stance_starts[i_stance+1]]))
            # plt.plot(angle_profile[stance_starts[i_stance]:stance_starts[i_stance+1]])
        # plt.show()
        max_flexion_angle = [180 - np.rad2deg(x) for x in max_flexion_angle]
        return max_flexion_angle

    def get_right_stride_length(positions, right_stance_starts):
        right_foot = positions[JOINT_POS_MAP['RFoot']]
        stride_length = []
        for i_start in range(len(right_stance_starts)-1):
            stride_vec = right_foot[:, right_stance_starts[i_start+1]] - right_foot[:, right_stance_starts[i_start]]
            stride_length.append(np.linalg.norm((stride_vec[0], stride_vec[2])))
        return stride_length
    
    def get_cadence(positions, right_stance_starts, speed):
        stride_length = get_right_stride_length(positions, right_stance_starts)
        if len(right_stance_starts) > 1:
            cadence_from_duration = 20 * (len(right_stance_starts) - 1) / (right_stance_starts[-1] - right_stance_starts[0]) * 60 * 2
            cadence_from_speed = np.nanmean([speed * 60 / item * 2 for item in stride_length])
        else:
            cadence_from_duration = np.nan
            cadence_from_speed = np.nan
        return cadence_from_duration

    def get_stance_starts(positions, segment='RFoot'):
        right_foot = positions[JOINT_POS_MAP[segment]]
        right_foot_planar = np.linalg.norm(np.column_stack([right_foot[0], right_foot[2]]), axis=1)
        right_foot_planar_speed = np.diff(data_filter(right_foot_planar, 3, 20))
        stance_phase = right_foot_planar_speed < 0.1
        stance_starts = np.where(stance_phase[:-1] & ~stance_phase[1:])[0]
        return stance_starts
    
    num_of_speeds = len(speeds)
    npy_path = '/mnt/d/Local/MotionPrior/OmniControl/save/omnicontrol_ckpt/samples_omnicontrol/results.npy'
    results = np.load(npy_path, allow_pickle=True).item()
    num_of_repetitions = int(len(results['text']) / num_of_speeds)

    stride_length_all = []
    angle_all = []
    cadence_all = []
    for i_speed in range(num_of_speeds):
        print('\n', speeds[i_speed], 'm/s, num of steps: ', end=' ')
        stride_length = []
        angle = []
        cadence = []
        for i_rep in range(num_of_repetitions):
            i_hint = i_rep * num_of_speeds + i_speed
            stance_starts = get_stance_starts(results['motion'][i_hint])
            stride_length.append(np.mean(get_right_stride_length(results['motion'][i_hint], stance_starts)))
            angle.append(np.mean(180 - np.array(get_right_knee_angle_from_joint_center(results['motion'][i_hint], stance_starts))))
            cadence.append(get_cadence(results['motion'][i_hint], stance_starts, speeds[i_speed]))
            print(len(get_right_stride_length(results['motion'][i_hint], stance_starts)), end='  ')
        stride_length_all.append(stride_length)
        angle_all.append(angle)
        cadence_all.append(cadence)

    return stride_length_all, angle_all, cadence_all


def export_to_excel(speed_param_syn, speed_param_exp, filename):
    param_names = {
        'stride_length': 'Stride Length (m)',
        'peak_vgrf': 'Peak Vertical Force (% BW)',
        'knee_angle_r_max': 'Peak Knee Flexion (°)',
        'peak_apgrf': 'Peak Anterior-Posterior Force (% BW)'
    }
    
    data = []
    all_speeds = set(speed_param_syn.keys()) | set(speed_param_exp.keys())
    
    for param, param_name in param_names.items():
        ratio = 100 if 'BW' in param_name else 1
        
        for condition, speed_param_data in [('GaitDynamics Generation', speed_param_syn), ('Experimental Measurement', speed_param_exp)]:
            for speed in sorted(all_speeds):
                if speed in speed_param_data and param in speed_param_data[speed]:
                    values = speed_param_data[speed][param]
                    data.append({
                        'Parameter': param_name,
                        'Condition': condition,
                        'Speed (m/s)': float(speed) / 10,
                        'Mean': np.mean(values) * ratio,
                        'One Standard Deviation': np.abs(np.std(values) * ratio)
                    })
        
        # Add OmniControl data for relevant parameters
        if param in ['stride_length', 'knee_angle_r_max']:
            stride_length_all, angle_all, cadence_all = load_omnicontrol()
            if param == 'stride_length':
                omni_data = stride_length_all
            else:
                omni_data = angle_all
            
            for i, speed in enumerate(sorted(all_speeds)):
                if i < len(omni_data):
                    data.append({
                        'Parameter': param_name,
                        'Condition': 'OmniControl Generation',
                        'Speed (m/s)': float(speed) / 10,
                        'Mean': np.nanmean(omni_data[i]) * ratio,
                        'One Standard Deviation': np.abs(np.nanstd(omni_data[i]) * ratio)
                    })
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    pd.DataFrame(data).to_excel(filename, index=False)
    print(f"Exported to: {filename}")

def format_errorbar_cap(caplines):
    for i_cap in range(2):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(10)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH*1.5)
        
        
def draw_speed_only_fig():
    colors = [np.array(x) / 255 for x in [[110, 170, 220], [70, 130, 180], [30, 90, 140], [177, 124, 90]]]
    [speed_param_syn, speed_param_exp] = pickle.load(open(os.path.join(
        SCRIPT_DIR, "results", "da_run_faster_speed_base_40_speed_lower_30_increment_4.pkl"), "rb"))
    fig = plt.figure(figsize=(10, 6))

    rc('text', usetex=True)
    plt.rc('font', family='Helvetica')

    gs = gridspec.GridSpec(nrows=2, ncols=4, wspace=0.35, hspace=0.35, width_ratios=[4, 5, 3, 5])        # , width_ratios=[8, 1, 8]
    param_y_range_dict = {
        'stride_length': ('Stride Length (m)', [1.2, 1.6, 2.0, 2.4, 2.8, 3.2], [0, 0], 0),
        'peak_vgrf': (r'Peak Vertical Force (\% BW)', [210, 230, 250, 270], [-3, 7], -4),
        'knee_angle_r_max': (r'Peak Knee Flexion ($^\circ$)', [70, 85, 100, 115, 130], [-2, 3], 1.5),
        'peak_apgrf': (r'Peak Anterior-Posterior Force (\% BW)', [20, 25, 30, 35, 40], [-2, 2], -1.5),
    }
    for i_param, (param_, (param_name_formal, y_range, ylim_adjustments, ylabel_adjustment)) in enumerate(param_y_range_dict.items()):
        ax = fig.add_subplot(gs[i_param * 2 + 1])
        # Mean and std of syn
        param_syn_mean, param_syn_std = [], []
        for speed in speed_param_syn.keys():
            ratio = 1
            if 'BW' in param_name_formal:
                ratio = 100
            param_syn_mean.append(np.mean(speed_param_syn[speed][param_]) * ratio)
            param_syn_std.append(np.std(speed_param_syn[speed][param_]) * ratio)
        ebar0, caplines, barlinecols = plt.errorbar(
            list(speed_param_syn.keys()), param_syn_mean, param_syn_std, capsize=6, linestyle='--',
            fmt='o', color=colors[1], alpha=0.8, linewidth=LINE_WIDTH, elinewidth=LINE_WIDTH, zorder=60)
        format_errorbar_cap(caplines)
        print(param_name_formal, '3 m/s: ', param_syn_mean[0], '5 m/s: ', param_syn_mean[-1], 'ratio: ', param_syn_mean[-1] / param_syn_mean[0])

        # Mean and std of exp
        param_exp_mean, param_exp_std = [], []
        for speed in speed_param_exp.keys():
            param_exp_mean.append(np.mean(speed_param_exp[speed][param_]) * ratio)
            param_exp_std.append(np.std(speed_param_exp[speed][param_]) * ratio)
        ebar1, caplines, barlinecols = plt.errorbar(
            list(speed_param_exp.keys()), param_exp_mean, param_exp_std, capsize=5, linestyle='-',
            fmt='o', color=[0.4, 0.4, 0.4], alpha=1, elinewidth=LINE_WIDTH, zorder=50)
        format_errorbar_cap(caplines)
        print(param_name_formal.replace('Generated', 'Measured'), '3 m/s: ', param_exp_mean[0], '5 m/s: ', param_exp_mean[-1], 'ratio: ', param_exp_mean[-1] / param_exp_mean[0])

        # Mean and std of OmniControl
        if param_ in ['stride_length', 'knee_angle_r_max']:
            stride_length_all, angle_all, cadence_all = load_omnicontrol()
            if param_ == 'stride_length':
                data_ = stride_length_all
            else:
                data_ = angle_all
            ebar2, caplines, barlinecols = plt.errorbar(
                list(speed_param_syn.keys()), [np.nanmean(values) for values in data_],
                [np.nanstd(values) for values in data_], capsize=5, linestyle='--',
                fmt='o', color=colors[3], alpha=0.6, elinewidth=LINE_WIDTH, zorder=30)
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
            ax.legend([ebar1, ebar0, ebar2],
                      ['Experimental Measurement', 'GaitDynamics Generation', 'OmniControl Generation'],
                      frameon=False, bbox_to_anchor=(0.8, 1.25), ncol=3, handlelength=4)
    plt.subplots_adjust(.08, .08, .98, .92)
    plt.savefig(os.path.join(SCRIPT_DIR, "exports", "da_run_faster_1.svg"), dpi=300)

        
def draw_extreme_speed_fig():
    colors = [np.array(x) / 255 for x in [[110, 170, 220], [30, 90, 140]]]
    fig = plt.figure(figsize=(10, 6))

    rc('text', usetex=True)
    plt.rc('font', family='Helvetica')

    gs = gridspec.GridSpec(nrows=2, ncols=4, wspace=0.35, hspace=0.35, width_ratios=[3, 5, 3, 5])
    param_y_range_dict = {
        'stride_length': ('Stride Length (m)', [1, 2, 3, 4], [-0.1, 0.4], 0),
        'peak_vgrf': (r'Peak Vertical Force (\% BW)', [160, 200, 240, 280], [-2, 0], -4),
        'knee_angle_r_max': (r'Peak Knee Flexion ($^\circ$)', [80, 100, 120, 140], [-2, 0], 1.5),
        'peak_apgrf': (r'Peak Anterior-Posterior Force (\% BW)', [10, 20, 30, 40], [-1, 2], -1.5),
    }
    [_, speed_param_exp] = pickle.load(open(os.path.join(
        SCRIPT_DIR, "results", f"da_run_faster_speed_base_20_speed_lower_20_increment_4.pkl"), "rb"))
    for i_param, (param_, (param_name_formal, y_range, ylim_adjustments, ylabel_adjustment)) in enumerate(param_y_range_dict.items()):
        ax = fig.add_subplot(gs[i_param * 2 + 1])
        ratio = 1 if 'BW' not in param_name_formal else 100
        param_exp_mean, param_exp_std = [], []
        for speed in speed_param_exp.keys():
            param_exp_mean.append(np.mean(speed_param_exp[speed][param_]) * ratio)
            param_exp_std.append(np.std(speed_param_exp[speed][param_]) * ratio)
        ebar0, caplines, barlinecols = plt.errorbar(
            list(speed_param_exp.keys()), param_exp_mean, param_exp_std, capsize=5, linestyle='-',
            fmt='o', color=[0.4, 0.4, 0.4], alpha=1, elinewidth=LINE_WIDTH, zorder=80)
        format_errorbar_cap(caplines)
        
        ebars_syn = []
        for i_speed_base, speed_base in enumerate([20, 50]):
            [speed_param_syn, speed_param_exp] = pickle.load(open(os.path.join(
                SCRIPT_DIR, "results", f"da_run_faster_speed_base_{speed_base}_speed_lower_20_increment_4.pkl"), "rb"))

            param_syn_mean, param_syn_std = [], []
            for speed in speed_param_syn.keys():
                param_syn_mean.append(np.mean(speed_param_syn[speed][param_]) * ratio)
                param_syn_std.append(np.std(speed_param_syn[speed][param_]) * ratio)

            ebar1, caplines, barlinecols = plt.errorbar(
                list(speed_param_syn.keys()), param_syn_mean, param_syn_std, capsize=6, linestyle='--',
                fmt='o', color=colors[i_speed_base], alpha=0.8, linewidth=LINE_WIDTH, elinewidth=LINE_WIDTH, zorder=60)
            format_errorbar_cap(caplines)
            ebars_syn.append(ebar1)

        format_axis(ax)
        ax.set_xlabel('Running Speed (m/s)', fontdict=FONT_DICT_SMALL)
        ax.set_xticks([20, 30, 40, 50, 60])
        ax.set_xticklabels([2, 3, 4, 5, 6], fontdict=FONT_DICT_SMALL)
        ax.set_xlim(17, 63)
        ax.set_ylabel(param_name_formal, fontdict=FONT_DICT_SMALL)
        ax.set_yticks(y_range)
        ax.set_yticklabels(y_range, fontdict=FONT_DICT_SMALL)
        ax.set_ylim(y_range[0] + ylim_adjustments[0], y_range[-1] + ylim_adjustments[1])

        if i_param == 1:
            ax.legend([ebar0, ebars_syn[0], ebars_syn[1]],
                      ['Experimental Measurement', 'GaitDynamics Generation Guided by 2 m/s Data', 'GaitDynamics Generation Guided by 5 m/s Data'],
                      frameon=False, bbox_to_anchor=(1.1, 1.25), ncol=3, handlelength=3)
    plt.subplots_adjust(.08, .08, .98, .92)
    plt.savefig(os.path.join(SCRIPT_DIR, "exports", "da_run_faster_1_extreme.jpg"), dpi=600)


if __name__ == "__main__":
    # draw_speed_only_fig()
    draw_extreme_speed_fig()
    
    # Export source data
    [speed_param_syn, speed_param_exp] = pickle.load(open(os.path.join(
        SCRIPT_DIR, "results", "da_run_faster_speed_base_40_speed_lower_30_increment_4.pkl"), "rb"))
    filename = os.path.join(SCRIPT_DIR, 'exports', 'Figure 4 Source Data.xlsx')
    export_to_excel(speed_param_syn, speed_param_exp, filename)

    plt.show()
