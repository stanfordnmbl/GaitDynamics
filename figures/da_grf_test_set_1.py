import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add base path for file operations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from args import parse_opt
from consts import NOT_IN_GAIT_PHASE, RUNNING_DSET_SHORT_NAMES, OVERGROUND_DSETS
from da_grf_test_set_0 import cols_to_unmask_main, dset_to_skip, drop_frame_num_range, cols_to_unmask_big_table, segment_to_osim_param
from data.addb_dataset import MotionDataset
from matplotlib import rc, lines
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, format_axis, LINE_WIDTH, FONT_DICT_LARGE, FONT_DICT
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
import random
from model.utils import fix_seed
from matplotlib.lines import Line2D


def format_errorbar_cap(caplines, size=15):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


def print_table_1(fast_run=False):
    """ Iterate through mask conditions, for Supplementary Table 2 """
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    metric_tf_dict, metric_diffusion_dict, masked_segment_col_loc = {}, {}, {}

    folder = 'fast' if fast_run else 'full'
    for i_test, test_name in enumerate(list(cols_to_unmask_main.keys())[1:]):
        metric_diffusion_inpainting = get_all_the_metrics(model_key=f'/{folder}/diffusion_{test_name}_diffusion_filling')
        metric_tf_inpainting = get_all_the_metrics(model_key=f'/{folder}/tf_{test_name}_diffusion_filling')
        metric_tf_dict[test_name] = metric_tf_inpainting
        metric_diffusion_dict[test_name] = metric_diffusion_inpainting

    for test_name in list(cols_to_unmask_main.keys())[1:]:
        string_ = ''
        for i_segment, (segment, params) in enumerate(list(segment_to_osim_param.items())[1:]):
            if segment not in test_name:
                string_ += '✓\t'
            else:
                # if segment == 'velocity':
                #     unit, scale = ' m/s', 1
                # else:
                #     unit, scale = ' deg', 180 / np.pi
                # param_metric = []
                # for param in params:
                #     param_metric.extend(metric_tf_dict[test_name][param])
                # string_ += f'{np.mean(param_metric)*scale:.1f} ± {np.std(param_metric)*scale:.1f}{unit}\t'
                string_ += '✕\t'

        print(string_, end='\t')
        for i_param, param in enumerate(params_of_interest):
            print(f'{np.mean(metric_diffusion_dict[test_name][param]):.1f} ± {np.std(metric_diffusion_dict[test_name][param]):.1f}', end='\t')
        print('\t', end='')
        for i_param, param in enumerate(params_of_interest):
            print(f'{np.mean(metric_tf_dict[test_name][param]):.1f} ± {np.std(metric_tf_dict[test_name][param]):.1f}', end='\t')
        print('')
    return metric_tf_dict


def combine_splits(results_):
    true_, pred_, pred_std_, columns = results_
    true_all, pred_all = {}, {}
    for dset in true_.keys():
        dset_short = dset.split('_Formatted_')[0]
        if dset_short not in true_all.keys():
            true_all[dset_short] = true_[dset]
            pred_all[dset_short] = pred_[dset]
        else:
            true_all[dset_short].extend(true_[dset])
            pred_all[dset_short].extend(pred_[dset])
    return (true_all, pred_all, pred_std_, columns)


def print_table_2():
    """ Accuracies of each dataset """
    dset_order = ['Camargo2021', 'Moore2015', 'Tan2022', 'vanderZee2022', 'Wang2023', 'Carter2023', 'Tan2021',
                  'Falisse2017', 'Han2023', 'Tiziana2019', 'Uhlrich2023', 'Hamner2013', 'Fregly2012', 'Li2021']
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']

    metric_all_dsets = get_all_the_metrics(model_key='full/tf_none_diffusion_filling')

    results_array = [[] for _ in range(len(dset_order))]
    for i_dset, dset_short in enumerate(dset_order):
        dset_index = metric_all_dsets['dset_short'].index(dset_short)

        for i_param, param_col in enumerate(params_of_interest):
            print(f'{metric_all_dsets[param_col][dset_index]:.1f}', end='\t')
            results_array[i_dset].append(metric_all_dsets[param_col][dset_index])
        print()
    results_average = np.mean(np.array(results_array), axis=0)
    results_std = np.std(np.array(results_array), axis=0)
    [print(f'{mean_:.1f} ± {std_:.1f}', end='\t') for mean_, std_ in zip(results_average, results_std)]


def dset_data_profile_to_peak(true_, pred_, columns, dset_short):
    param_true_dict = {
        'calcn_l_force_vy_max': [],
        'calcn_l_force_vx_max': [],
        'calcn_l_force_vx_min': [],
    }
    if 'knee_moment_l_x' in columns:
        param_true_dict['knee_moment_l_x_max'] = []
        param_true_dict['knee_moment_l_z_max'] = []

    param_pred_dict = copy.deepcopy(param_true_dict)
    gait_phase_label = []
    for i_trial in range(len(true_)):
        if len(true_[i_trial]) == 0:
            continue
        if dset_short == 'Li2021':
            stance_len_thds, cycle_len_thds = [10, 300], [20, 400]
        else:
            stance_len_thds, cycle_len_thds = None, None
        trial_gait_phase_label, stance_start_valid, stance_end_valid = MotionDataset.grf_to_trial_gait_phase_label(
            true_[i_trial][:, columns.index('calcn_l_force_vy')], opt.window_len, opt.target_sampling_rate, stance_len_thds, cycle_len_thds)

        if len(stance_start_valid) == 0:        # for one gait cycle trials, use the whole trial
            gait_phase_label.append(np.linspace(0, 1000, true_[i_trial].shape[0]))
            if dset_short.split('20')[0] in OVERGROUND_DSETS:
                stance_start_valid, stance_end_valid = [0], [true_[i_trial].shape[0]]
            else:
                continue
        else:
            gait_phase_label.append(trial_gait_phase_label)
        for start_, end_ in zip(stance_start_valid, stance_end_valid):
            # for param_dict, data_dict in zip([param_true_dict, param_pred_dict], [true_, pred_]):
            max_vy = np.max(true_[i_trial][start_:end_, columns.index('calcn_l_force_vy')])
            if dset_short not in RUNNING_DSET_SHORT_NAMES and (max_vy > 13 or max_vy < 8):
                continue        # skip abnormal max vGRF
            for param_dict, data_dict in zip([param_true_dict, param_pred_dict], [true_, pred_]):
                param_dict['calcn_l_force_vy_max'].append(np.max(data_dict[i_trial][start_:end_, columns.index('calcn_l_force_vy')]))
                param_dict['calcn_l_force_vx_max'].append(np.max(data_dict[i_trial][start_:end_, columns.index('calcn_l_force_vx')]))
                param_dict['calcn_l_force_vx_min'].append(np.min(data_dict[i_trial][start_:end_, columns.index('calcn_l_force_vx')]))
                if 'knee_moment_l_x' in columns:
                    param_dict['knee_moment_l_x_max'].append(np.max(data_dict[i_trial][start_:end_, columns.index('knee_moment_l_x')]))
                    param_dict['knee_moment_l_z_max'].append(np.max(data_dict[i_trial][start_:end_, columns.index('knee_moment_l_z')]))
    return param_true_dict, param_pred_dict, gait_phase_label


def get_all_the_metrics(model_key):
    results_ = pickle.load(open(os.path.join(SCRIPT_DIR, f"results/{model_key}.pkl"), "rb"))
    results_ = combine_splits(results_)
    true_, pred_, _, columns = results_
    dset_list = list(true_.keys())
    param_pattern_and_ratio = {'calcn_l_force_v': 100 / 9.81, 'calcn_l_force_normed_cop': 100, 'moment': 1}

    metric_all_dsets = {'dset_short': []}
    for i_dset, dset_short in enumerate(dset_list):
        metric_dset = {}
        dset = dset_short + '_Formatted_No_Arm'
        if dset in dset_to_skip:
            continue
        metric_all_dsets['dset_short'].append(dset_short)
        param_true_dict, param_pred_dict, gait_phase_label = dset_data_profile_to_peak(true_[dset_short], pred_[dset_short], columns, dset_short)

        for param_col in param_true_dict.keys():
            ratio = [v for k, v in param_pattern_and_ratio.items() if k in param_col]
            assert len(ratio) == 1
            # metric_mean = np.sqrt(np.mean((np.array(param_true_dict[param_col]) - np.array(param_pred_dict[param_col]))**2)) * ratio[0]
            metric_mean = np.mean(np.abs(np.array(param_true_dict[param_col]) - np.array(param_pred_dict[param_col]))) * ratio[0]
            metric_dset[param_col] = metric_mean

        true_concat = np.concatenate(true_[dset_short], axis=0)
        pred_concat = np.concatenate(pred_[dset_short], axis=0)
        gait_phase_label_concat = np.concatenate(gait_phase_label, axis=0)
        for i_param, param_col in enumerate(columns):
            param_col_loc = i_param
            ratio = [v for k, v in param_pattern_and_ratio.items() if k in param_col]
            if len(ratio) == 0:
                ratio = [1]
            elif len(ratio) > 1:
                raise ValueError('2 matching ratios')
            within_gait_cycle = (gait_phase_label_concat != NOT_IN_GAIT_PHASE)
            diff_abs = np.abs(true_concat[within_gait_cycle, param_col_loc] - pred_concat[within_gait_cycle, param_col_loc])
            metric_mean = np.mean(diff_abs) * ratio[0]

            if 'pelvis_t' not in param_col and 'force' not in param_col and 'moment' not in param_col and diff_abs.max() > 0.9 * np.pi:
                print(dset_short + ' has large angular error in ' + param_col)
                # plt.figure()
                # plt.plot(true_concat[within_gait_cycle, param_col_loc])
                # plt.plot(pred_concat[within_gait_cycle, param_col_loc])
                # plt.show()

            if 'normed_cop' in param_col or 'moment' in param_col:
                stance_phase = (np.abs(true_concat[:, columns.index('calcn_l_force_normed_cop_x')]) > 1e-10) & (
                        np.abs(pred_concat[:, columns.index('calcn_l_force_normed_cop_x')]) > 1e-10)
                metric_mean = np.mean(np.abs(true_concat[stance_phase & within_gait_cycle, param_col_loc] -
                                             pred_concat[stance_phase & within_gait_cycle, param_col_loc])) * ratio[0]

            metric_dset[param_col] = metric_mean
        # print(dset_short)
        for param_col, metric_mean in metric_dset.items():
            # print(f'{param_col}, {metric_mean:.2f}')
            if param_col not in metric_all_dsets.keys():
                metric_all_dsets[param_col] = [metric_mean]
            else:
                metric_all_dsets[param_col].append(metric_mean)

        # if 'Li' in dset_short:
        #     plt.figure()
        #     param = 'calcn_l_force_vy'
        #     param_col_loc = columns.index(param)
        #     within_gait_cycle = (gait_phase_label_concat != NOT_IN_GAIT_PHASE)
        #     plt.plot(true_concat[within_gait_cycle, param_col_loc])
        #     plt.plot(pred_concat[within_gait_cycle, param_col_loc])
        #     plt.title(dset_short + ' ' + str(metric_dset[param]))
        # plt.show()

    # for param_col, metric_list in metric_all_dsets.items():
    #     if param_col == 'dset_short':
    #         continue
    #     print(f'{param_col}, {np.mean(metric_list):.2f}')
    return metric_all_dsets


def get_metrics_linear_velocity(model_key):
    results_ = pickle.load(open(os.path.join(SCRIPT_DIR, f"results/{model_key}.pkl"), "rb"))
    results_ = combine_splits(results_)
    true_, pred_, _, columns = results_
    dset_list = list(true_.keys())
    # param_pattern_and_ratio = {'calcn_l_force_v': 100 / 9.81, 'calcn_l_force_normed_cop': 100, 'moment': 1}

    metric_all_dsets = {'dset_short': []}
    for i_dset, dset_short in enumerate(dset_list):
        true_concat = np.concatenate(true_[dset_short], axis=0)
        pred_concat = np.concatenate(pred_[dset_short], axis=0)
        param_true_dict, param_pred_dict, gait_phase_label = dset_data_profile_to_peak(true_[dset_short], pred_[dset_short], columns, dset_short)
        gait_phase_label_concat = np.concatenate(gait_phase_label, axis=0)
        within_gait_cycle = (gait_phase_label_concat != NOT_IN_GAIT_PHASE)
        connecting_point_index = np.ones(within_gait_cycle.shape, dtype=bool)
        i_index = 0
        for i_trial, gait_phase_label_trial in enumerate(gait_phase_label):
            i_index += (gait_phase_label_trial.shape[0] - 1)
            connecting_point_index[i_index] = False
        ap_speed_true = np.diff(true_concat[:, columns.index('pelvis_tx')], append=true_concat[-1, columns.index('pelvis_tx')]) * opt.target_sampling_rate
        invalid_speed = np.ones(within_gait_cycle.shape, dtype=bool)
        invalid_speed[np.abs(ap_speed_true) > 5] = 0
        index_for_valid_speed = within_gait_cycle & connecting_point_index & invalid_speed
        
        metric_dset = {}
        for i_param, param_col in enumerate(columns):
            if param_col not in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']:
                continue
            ratio = [1]
            param_col_loc = i_param
            
            true_speed = np.diff(true_concat[:, param_col_loc], append=true_concat[-1, param_col_loc]) * opt.target_sampling_rate
            pred_speed = np.diff(pred_concat[:, param_col_loc], append=pred_concat[-1, param_col_loc]) * opt.target_sampling_rate
            
            diff_abs = np.abs(true_speed[index_for_valid_speed] - pred_speed[index_for_valid_speed])
            metric_mean = np.mean(diff_abs) * ratio[0]
                
            metric_dset[param_col] = metric_mean
        for param_col, metric_mean in metric_dset.items():
            if param_col not in metric_all_dsets.keys():
                metric_all_dsets[param_col] = [metric_mean]
            else:
                metric_all_dsets[param_col].append(metric_mean)

    return metric_all_dsets


def sigifi_sign(mean_, std_, bar_locs, draw_23=True, draw_12=True, draw_13=True):
    i = 0
    lo = (0.03, 0.03, 0.03)
    
    bar_to_line_distance = 0.4
    y_top = max([a + b for a, b in zip(mean_[i:i+3], std_[i:i+3])])
    top_line_y0, top_line_y1 = y_top + 0.8, y_top + 2.3
    if not draw_12 and not draw_23:
        top_line_y1 = top_line_y0
        
    if draw_12:
        diff_line_0x = [bar_locs[i]+lo[0], bar_locs[i]+lo[0], bar_locs[i+1]-lo[1], bar_locs[i+1]-lo[1]]
        diff_line_0y = [mean_[i] + std_[i] + bar_to_line_distance, top_line_y0, top_line_y0, mean_[i+1] + std_[i+1] + bar_to_line_distance]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.5 + bar_locs[i+1]*0.5, top_line_y0-0.28, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=20, ha='center')

    if draw_23:
        diff_line_0x = [bar_locs[i+1]+lo[1], bar_locs[i+1]+lo[1], bar_locs[i+2]-lo[2], bar_locs[i+2]-lo[2]]
        diff_line_0y = [mean_[i+1] + std_[i+1] + bar_to_line_distance, top_line_y0, top_line_y0, mean_[i+2] + std_[i+2] + bar_to_line_distance]
        plt.plot(diff_line_0x, diff_line_0y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i+1]*0.5 + bar_locs[i+2]*0.5, top_line_y0-0.28, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=20, ha='center')

    if draw_13:
        diff_line_1x = [bar_locs[i]-lo[0], bar_locs[i]-lo[0], bar_locs[i+2]+lo[2], bar_locs[i+2]+lo[2]]
        diff_line_1y = [mean_[i] + std_[i] + bar_to_line_distance, top_line_y1, top_line_y1, mean_[i+2] + std_[i+2] + bar_to_line_distance]
        plt.plot(diff_line_1x, diff_line_1y, 'black', linewidth=LINE_WIDTH)
        plt.text(bar_locs[i]*0.5 + bar_locs[i+2]*0.5, top_line_y1-0.28, '*', fontdict={'fontname': 'Times New Roman'}, color='black', size=20, ha='center')


def export_metrics_to_excel(metrics_dict, params_dict, filename, method_names=None, p_values=None):
    """Generic function to export metrics to Excel"""
    if method_names is None:
        method_names = list(metrics_dict.keys())
    
    datasets = list(metrics_dict.values())[0]['dset_short']
    data = []
    
    for param, param_name in params_dict.items():
        for i, dataset in enumerate(datasets):
            for method_key, method_name in zip(metrics_dict.keys(), method_names):
                data.append({
                    'Dataset': dataset, 
                    'Parameter': param_name, 
                    'Model Name': method_name, 
                    'Mean Absolute Error (% BW)': metrics_dict[method_key][param][i]
                })
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        pd.DataFrame(data).to_excel(writer, sheet_name='Data Points', index=False)
        if p_values:
            pd.DataFrame(p_values).to_excel(writer, sheet_name='P_Values', index=False)
    
    print(f"Exported to: {filename}")
    return filename

def draw_fig_1(fast_run=False):
    def format_ticks(ax):
        ax.text(-0.7, 5, 'Mean Absolute Error (\% BW)', rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='center')
        ax.text(-0.8, 12.5, 'Better', rotation=90, fontdict=FONT_DICT_SMALL, color='green', verticalalignment='center')
        ax.annotate('', xy=(-0.08, 0.78), xycoords='axes fraction', xytext=(-0.08, 0.98),
                    arrowprops=dict(arrowstyle="->", color='green'))

        ax.set_yticks(range(0, 15, 2))
        ax.set_yticklabels(range(0, 15, 2), fontdict=FONT_DICT_SMALL)
        ax.set_xlim([-0.3, 3.8])

        ax.set_xticks([])
        for i in range(4):
            ax.text(i+0.25, -1.6, list(params_name_formal_name_pairs.values())[i], fontdict=FONT_DICT_SMALL, ha='center')

    colors = [np.array(x) / 255 for x in [[70, 130, 180], [207, 154, 130], [177, 124, 90]]]        #  [207, 154, 130], [100, 155, 227]
    folder = 'fast' if fast_run else 'full'
    metric_tf = get_all_the_metrics(model_key=f'/{folder}/tf_none_diffusion_filling')
    metric_groundlink = get_all_the_metrics(model_key=f'/{folder}/groundlink_none_diffusion_filling')
    metric_sugainet = get_all_the_metrics(model_key=f'/{folder}/sugainet_none_diffusion_filling')

    # Export source data
    metrics_dict = {'tf': metric_tf, 'groundlink': metric_groundlink, 'sugainet': metric_sugainet}
    params_name_excel_pairs = {
        'calcn_l_force_vy': 'Vertical Force Profile',
        'calcn_l_force_vx': 'Anterior-Posterior Force Profile', 
        'calcn_l_force_vz': 'Medial-Lateral Force Profile',
        'calcn_l_force_vy_max': 'Vertical Force Peak'
    }

    params_name_formal_name_pairs = {
        'calcn_l_force_vy': 'Vertical\nForce—Profile',
        'calcn_l_force_vx': 'Anterior-Posterior\nForce—Profile',
        'calcn_l_force_vz': 'Medial-Lateral\nForce\u2014Profile',
        'calcn_l_force_vy_max': 'Vertical\nForce\u2014Peak'}
    params_of_interest = list(params_name_formal_name_pairs.keys())

    rc('text', usetex=True)
    plt.rc('font', family='Helvetica')

    fig = plt.figure(figsize=(7.7, 4.5))
    p_values_data = []
    print('Parameter\tAll\t1-2\t1-3\t2-3')
    for i_axis, param in enumerate(params_of_interest):
        # print(np.mean(metric_tf[param]))
        bar_locs = [i_axis, i_axis + 0.25, i_axis + 0.5]
        mean_ = [np.mean(ele) for ele in [metric_tf[param], metric_groundlink[param], metric_sugainet[param]]]
        std_ = [np.std(ele) for ele in [metric_tf[param], metric_groundlink[param], metric_sugainet[param]]]
        bars = plt.bar(bar_locs, mean_, color=colors, width=0.23)
        ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_, capsize=0, ecolor='black', fmt='none', lolims=True, elinewidth=LINE_WIDTH)
        format_errorbar_cap(caplines, 8)

        p_friedmanchisquare = friedmanchisquare(metric_tf[param], metric_groundlink[param], metric_sugainet[param]).pvalue
        print('{0: <15}'.format(param[8:]), end='\t')
        print(round(p_friedmanchisquare, 3), end='\t')
        p_wilcoxon_tf_groundlink = wilcoxon(metric_tf[param], metric_groundlink[param], alternative='two-sided').pvalue
        p_wilcoxon_tf_sugainet = wilcoxon(metric_tf[param], metric_sugainet[param], alternative='two-sided').pvalue
        p_wilcoxon_groundlink_sugainet = wilcoxon(metric_groundlink[param], metric_sugainet[param], alternative='two-sided').pvalue
        [print(round(p_val, 5), end='\t') for p_val in [p_wilcoxon_tf_groundlink, p_wilcoxon_tf_sugainet, p_wilcoxon_groundlink_sugainet]]
        print()
        
        p_values_data.append({
            'Parameter': params_name_excel_pairs[param],
            'Friedman': round(p_friedmanchisquare, 2),
            'GaitDynamics vs CNN': round(p_wilcoxon_tf_groundlink, 2),
            'GaitDynamics vs RNN': round(p_wilcoxon_tf_sugainet, 2),
            'CNN vs RNN': round(p_wilcoxon_groundlink_sugainet, 2)
        })
        
        if i_axis == 0:
            sigifi_sign(mean_, std_, bar_locs)
        elif i_axis == 2:
            sigifi_sign(mean_, std_, bar_locs, draw_12=True, draw_23=False, draw_13=True)
        elif i_axis == 3:
            sigifi_sign(mean_, std_, bar_locs, draw_12=False, draw_23=True, draw_13=True)

    # From "Comparison of different machine learning models to enhance sacral acceleration-based estimations of running stride temporal variables and peak vertical ground reaction force"
    line0, = plt.plot([2.8, 3.7], [13, 13], '--', linewidth=2, color=[0.0, 0.0, 0.0], alpha=0.5)
    plt.text(4.1, 14, 'Running MDC - Healthy$^{\ 31}$', fontdict=FONT_DICT_SMALL, color=[0.0, 0.0, 0.0], va='center')
    plt.annotate('', xytext=(4.05, 14), xycoords='data', xy=(3.75, 13), arrowprops=dict(arrowstyle="->"))

    # From "Intra-rater repeatability of gait parameters in healthy adults during self-paced treadmill-based virtual reality walking"
    line1, = plt.plot([2.8, 3.7], [10.18, 10.18], '--', linewidth=2, color=[0.0, 0.0, 0.0], alpha=0.5)
    plt.text(4.1, 11.18, 'Walking MDC - Healthy$^{\ 32}$', fontdict=FONT_DICT_SMALL, color=[0.0, 0.0, 0.0], va='center')
    plt.annotate('', xytext=(4.05, 11.18), xycoords='data', xy=(3.75, 10.18), arrowprops=dict(arrowstyle="->"))

    # and "Minimal detectable change for gait variables collected during treadmill walking in individuals post-stroke"
    line2, = plt.plot([2.8, 3.7], [4.65, 4.65], '--', linewidth=2, color=[0.0, 0.0, 0.0], alpha=0.5)
    plt.text(4.1, 5.65, 'Walking MDC - Stroke$^{\ 33}$', fontdict=FONT_DICT_SMALL, color=[0.0, 0.0, 0.0], va='center')
    plt.annotate('', xytext=(4.05, 5.65), xycoords='data', xy=(3.75, 4.65), arrowprops=dict(arrowstyle="->"))

    format_axis(plt.gca())
    format_ticks(plt.gca())
    plt.tight_layout(rect=[-0.03, 0., 1.03, 0.88])
    plt.legend(list(bars), ['GaitDynamics', 'Convolutional Neural Network$^{\ 21}$', 'Recurrent Neural Network$^{\ 22}$'],
               frameon=False, fontsize=FONT_SIZE_SMALL, bbox_to_anchor=(0.7, 1.2), ncols=1)
    plt.savefig(os.path.join(SCRIPT_DIR, f'exports/da_grf.pdf'), dpi=300, bbox_inches='tight')
    plt.show()
    
    filename = os.path.join(SCRIPT_DIR, f'exports/Figure 1 Source Data.xlsx')
    export_metrics_to_excel(metrics_dict, params_name_excel_pairs, filename, ['GaitDynamics', 'Convolutional Neural Network', 'Recurrent Neural Network'], p_values_data)


def draw_fig_2(fast_run=False):
    def format_ticks(ax_plt):
        ax_plt.text(-0.7, 33, 'Mean Absolute Error of Peak Vertical Force Estimation (\% BW)', rotation=90, fontdict=FONT_DICT_SMALL, verticalalignment='center')
        ax_plt.text(-1., 73, 'Better', rotation=90, fontdict=FONT_DICT_SMALL, color='green', verticalalignment='center')
        ax_plt.annotate('', xy=(-0.11, 0.8), xycoords='axes fraction', xytext=(-0.11, 1.),
                        arrowprops=dict(arrowstyle="->", color='green'))
        ax_plt.set_yticks([0, 20, 40, 60, 80])
        ax_plt.set_yticklabels([0, 20, 40, 60, 80], fontdict=FONT_DICT_SMALL)
        ax_plt.set_ylim([0, 80])
        ax_plt.set_xlim([-0.3, 4.6])
        ax_plt.set_xticks([])

        ax_text = fig.add_axes([0.1, 0., 0.87, 0.3])
        plt.axis('off')
        ax_text.set_xlim(ax_plt.get_xlim())
        ax_text.set_ylim([3, 10])
        segment_list = ['trunk', 'pelvis', 'hips', 'knees', 'ankles']
        for i_test, test_name in enumerate(list(cols_to_unmask_main.keys())[1:]):
            masked_segments = test_name.split('_')
            for i_segment, segment in enumerate(segment_list):
                if segment in masked_segments or segment[:-1] in masked_segments:
                    ax_text.text(i_test*0.96+0.35, 7.9 - i_segment*1.1, segment, fontdict=FONT_DICT_SMALL, color=[0.8, 0.8, 0.8], ha='center')
                else:
                    ax_text.text(i_test*0.96+0.35, 7.9 - i_segment*1.1, segment, fontdict=FONT_DICT_SMALL, ha='center')

    colors = [np.array(x) / 255 for x in [[70, 130, 180], [207, 154, 130]]]        #  [207, 154, 130], [100, 155, 227]
    folder = 'fast' if fast_run else 'full'
    param_of_interest = 'calcn_l_force_vy_max'
    fig = plt.figure(figsize=(7.7, 4.8))
    rc('text', usetex=True)
    plt.rc('font', family='Helvetica')
    ax_plt = fig.add_axes([0.14, 0.25, 0.83, 0.72])

    full_input = get_all_the_metrics(model_key=f'/{folder}/tf_none_diffusion_filling')[param_of_interest]
    print(np.mean(full_input))
    line_1, = plt.plot([-0.3, 7.6], [np.mean(full_input), np.mean(full_input)], color=np.array([70, 130, 180])/255, linewidth=LINE_WIDTH, linestyle='--')

    # Collect metrics for export
    metrics_dict = {}
    full_input_metric = get_all_the_metrics(model_key=f'/{folder}/tf_none_diffusion_filling')
    metrics_dict['full_input'] = full_input_metric
    
    for i_test, test_name in enumerate(list(cols_to_unmask_main.keys())[1:]):
        metric_tf_inpainting = get_all_the_metrics(model_key=f'/{folder}/tf_{test_name}_diffusion_filling')
        metric_tf_medianfilling = get_all_the_metrics(model_key=f'/{folder}/tf_{test_name}_median_filling')
        metrics_dict[f'inpainting_{test_name}'] = metric_tf_inpainting
        metrics_dict[f'median_{test_name}'] = metric_tf_medianfilling
        bar_locs = [i_test, i_test + 0.3]
        mean_ = [np.mean(ele) for ele in [metric_tf_inpainting[param_of_interest], metric_tf_medianfilling[param_of_interest]]]
        std_ = [np.std(ele) for ele in [metric_tf_inpainting[param_of_interest], metric_tf_medianfilling[param_of_interest]]]
        bars = plt.bar(bar_locs, mean_, color=colors[:2], width=0.27)
        ebar, caplines, barlinecols = plt.errorbar(bar_locs, mean_, std_, capsize=0, ecolor='black', fmt='none', lolims=True, elinewidth=LINE_WIDTH)
        format_errorbar_cap(caplines, 8)

        print(test_name, mean_ - np.mean(full_input))

    # Export source data
    params_dict = {param_of_interest: 'Vertical Force Peak'}
    filename = os.path.join(SCRIPT_DIR, f'exports/Figure 2 Source Data.xlsx')
    method_names = ['Full-Body Kinematics (GaitDynamics)'] + [f'{method}_{test}' for test in list(cols_to_unmask_main.keys())[1:] for method in ['Partial-Body Kinematics with Inpainting (GaitDynamics)', 'Partial-Body Kinematics with Median Filling']]
    export_metrics_to_excel(metrics_dict, params_dict, filename, method_names)
    
    format_axis(plt.gca())
    format_ticks(ax_plt)
    ax_plt.legend(list(bars) + [line_1], [
        'Partial-Body Kinematics with Inpainting (GaitDynamics)', 'Partial-Body Kinematics with Median Filling', 'Full-Body Kinematics (GaitDynamics)'],
                  frameon=False, fontsize=FONT_SIZE_SMALL, bbox_to_anchor=(0.36, 0.8), loc='lower left')
    plt.savefig(os.path.join(SCRIPT_DIR, f'exports/da_segment_filling.pdf'), dpi=300, bbox_inches='tight')
    plt.show()


def draw_fig_4(fast_run=False):
    def format_ticks(ax_plt):
        ax_plt.set_ylabel(r'MAE of $f_v$ Peak (% Body Weight)', fontdict=FONT_DICT_SMALL)
        ax_plt.set_yticks([0, 2, 4, 6, 8, 10])
        ax_plt.set_yticklabels([0, 2, 4, 6, 8, 10], fontdict=FONT_DICT_SMALL)
        ax_plt.set_ylim([0, 10])
        ax_plt.set_xlim([0, 550])
        ax_plt.set_xlabel('Duration of Package Drop (ms)', fontdict=FONT_DICT_SMALL)

    colors = [np.array(x) / 255 for x in [[20, 145, 145], [191, 166, 203], [174, 118, 173]]]        #  [207, 154, 130], [100, 155, 227]
    folder = 'fast' if fast_run else 'full'
    param_of_interest = 'calcn_l_force_vy_max'
    fig = plt.figure(figsize=(5, 4))
    ax_plt = fig.add_axes([0.13, 0.15, 0.82, 0.6])
    lines = []
    for i_method, filling_method in enumerate(['diffusion', 'interpo', 'median']):       # 'diffusion', 'interpo', 'median'
        results_mean = []
        for drop_frame_num in drop_frame_num_range:
            results_current_drop_num = get_all_the_metrics(model_key=f'/{folder}/tf_{drop_frame_num}_{filling_method}_filling')[param_of_interest]
            results_mean.append(np.nanmean(results_current_drop_num))

        line_, = plt.plot([ele*10 for ele in drop_frame_num_range], results_mean, marker='o', color=colors[i_method], label=filling_method, linewidth=LINE_WIDTH)
        lines.append(line_)

    no_drop = get_all_the_metrics(model_key=f'/{folder}/tf_none_diffusion_filling')[param_of_interest]
    line_1, = plt.plot([50, 500], [np.mean(no_drop), np.mean(no_drop)], color=[0., 0., 0.1], alpha=0.3, linewidth=LINE_WIDTH, linestyle='--')
    format_axis(plt.gca())
    format_ticks(ax_plt)
    ax_plt.legend(lines + [line_1], [
        'GaitDynamics - Diffusion Filling', 'GaitForce - Interpolation', 'GaitForce - Median Filling', 'GaitForce - No Package Drop'],
                  frameon=False, fontsize=FONT_SIZE_SMALL, bbox_to_anchor=(1, 1.4))
    plt.savefig(os.path.join(SCRIPT_DIR, f'exports/da_temporal_filling.png'), dpi=300, bbox_inches='tight')
    plt.show()

def p_val_with_without_data_filter(fast_run=False):
    folder = 'fast' if fast_run else 'full'
    metric_tf_filtered = get_all_the_metrics(model_key=f'/{folder}/tf_none_diffusion_filling')
    metric_tf_unfiltered = get_all_the_metrics(model_key=f'/{folder}/tf_no_data_filter_none_diffusion_filling')

    params_name_formal_name_pairs = {
        'calcn_l_force_vy': 'Vertical\nForce - Profile',
        'calcn_l_force_vx': 'Anterior-Posterior\nForce - Profile',
        'calcn_l_force_vz': 'Medial-Lateral\nForce - Profile',
        'calcn_l_force_vy_max': 'Vertical\nForce - Peak'}
    params_of_interest = list(params_name_formal_name_pairs.keys())
    # print('Parameter\t\tAll\t\t1-2\t\t1-3\t\t2-3')
    for i_axis, param in enumerate(params_of_interest):
        print(param, end='\t')
        p_val = wilcoxon(metric_tf_filtered[param], metric_tf_unfiltered[param]).pvalue
        print(round(p_val, 3))
        print('{:.2f} ± {:.2f}'.format(np.mean(metric_tf_filtered[param]), np.std(metric_tf_filtered[param])))
        print('{:.2f} ± {:.2f}'.format(np.mean(metric_tf_unfiltered[param]), np.std(metric_tf_unfiltered[param])))


def draw_fig_for_meeting(fast_run=False):
    """ Iterate through mask conditions """
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    metric_tf_dict, metric_diffusion_dict, masked_segment_col_loc = {}, {}, {}

    folder = 'fast' if fast_run else 'full'
    for i_test, test_name in enumerate(list(cols_to_unmask_main.keys())[0:1]):
        metric_tf_inpainting = get_all_the_metrics(model_key=f'/{folder}/tf_{test_name}_diffusion_filling')
        metric_diffusion_inpainting = get_all_the_metrics(model_key=f'/{folder}/diffusion_{test_name}_diffusion_filling')
        metric_tf_dict[test_name] = metric_tf_inpainting
        metric_diffusion_dict[test_name] = metric_diffusion_inpainting

    plt.figure(figsize=(6, 3.5))
    labels = ['end2end model', 'diffusion model (inpainting)']
    # labels = ['GRF Refinement Model', 'Diffusion Model (Inpainting)']
    bars = plt.bar([0, 3, 6], [np.mean(metric_tf_dict[test_name][param]) for param in params_of_interest], color='C0', width=1, label=labels[0])
    bars = plt.bar([1, 4, 7], [np.mean(metric_diffusion_dict[test_name][param]) for param in params_of_interest], color='C1', width=1, label=labels[1])
    ax = plt.gca()
    plt.xticks([0.5, 3.5, 6.5])
    ax.set_xticklabels(['Vertical', 'Anterior-posterior', 'Medial-lateral'], fontdict=FONT_DICT_SMALL)
    plt.yticks(range(0, 11, 2))
    ax.set_yticklabels(range(0, 11, 2), fontdict=FONT_DICT_SMALL)
    plt.ylabel('Mean Absolute Error (% Body Weight)', fontdict=FONT_DICT_SMALL)
    ebar, caplines, barlinecols = plt.errorbar([0, 3, 6], [np.mean(metric_tf_dict[test_name][param]) for param in params_of_interest],
                                               [np.std(metric_tf_dict[test_name][param]) for param in params_of_interest],
                                               capsize=0, ecolor='black', fmt='none', lolims=True, elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 8)
    ebar, caplines, barlinecols = plt.errorbar([1, 4, 7], [np.mean(metric_diffusion_dict[test_name][param]) for param in params_of_interest],
                                               [np.std(metric_diffusion_dict[test_name][param]) for param in params_of_interest],
                                               capsize=0, ecolor='black', fmt='none', lolims=True, elinewidth=LINE_WIDTH)
    format_errorbar_cap(caplines, 8)
    plt.legend(fontsize=FONT_SIZE_SMALL)
    plt.savefig(os.path.join(SCRIPT_DIR, 'exports/for_meeting.png'))
    plt.show()


def find_first_consecutive_trues_numpy(gait_phase_label, n):
    # Convert to numpy array if it's not already
    arr = np.asarray(gait_phase_label)

    # Find runs of consecutive True values
    runs = np.split(np.where(arr)[0], np.where(np.diff(np.where(arr)[0]) != 1)[0] + 1)

    # Find the first run with length >= n
    matching_runs = [r for r in runs if len(r) >= n]
    if len(matching_runs) == 0:
        return NOT_IN_GAIT_PHASE
    else:
        random_idx = np.random.randint(0, len(matching_runs))
        return matching_runs[random_idx][0]


def get_random_non_negative_one_numpy(first_true):
    # Convert to numpy array
    arr = np.array(first_true)

    # Find indices where values are not -1
    indices = np.where(arr != NOT_IN_GAIT_PHASE)[0]

    # Check if there are any non-(-1) values
    if indices.size > 0:
        # Select a random index and return the corresponding value
        random_idx = np.random.choice(indices)
        return random_idx
    else:
        return None


def draw_fig_supplementary_1(fast_run=False, data_len=300):
    rc('text', usetex=True)
    fix_seed()
    params_of_interest = {'calcn_l_force_vy': ('f_v', (0, 60, 120)),
                          'calcn_l_force_vx': ('f_{ap}', (-20, 0, 20)),
                          'calcn_l_force_vz': ('f_{ml}', (0, 7.5, 15))}     # 'hip_flexion_r', 'knee_angle_r', 'ankle_angle_r'
    folder = 'fast' if fast_run else 'full'
    colors = [np.array(x) / 255 for x in [[70, 130, 180], [207, 154, 130], [177, 124, 90]]]        #  [207, 154, 130], [100, 155, 227]

    results_tf = pickle.load(open(os.path.join(SCRIPT_DIR, f"results/{folder}/tf_none_diffusion_filling.pkl"), "rb"))
    results_tf = combine_splits(results_tf)
    true_tf, pred_tf, _, columns = results_tf
    
    results_tf_no_overlapping = pickle.load(open(os.path.join(SCRIPT_DIR, f"results/{folder}/tf_none_diffusion_filling_no_overlapping.pkl"), "rb"))
    results_tf_no_overlapping = combine_splits(results_tf_no_overlapping)
    _, pred_tf_no_overlapping, _, _ = results_tf_no_overlapping

    # Use GridSpec for more control over subplot spacing
    from matplotlib.gridspec import GridSpec
    
    random_dset_trial_id_and_start_sample_dict = {}
    for i_dset, dset in enumerate(list(true_tf.keys())):
        _, _, gait_phase_label = dset_data_profile_to_peak(true_tf[dset], pred_tf[dset], columns, dset)
        start_samples = [find_first_consecutive_trues_numpy(gait_phase_label[i_trial] != NOT_IN_GAIT_PHASE, data_len) for i_trial in range(len(gait_phase_label))]
        random_trial_id = get_random_non_negative_one_numpy(start_samples)
        if random_trial_id is not None:
            dset_name_print = dset.split('_')[0]
            dset_name_print = 'Data from {} et al., {}'.format(dset_name_print[:-4], dset_name_print[-4:])
            random_dset_trial_id_and_start_sample_dict[dset] = (random_trial_id, start_samples[random_trial_id], dset_name_print)

    for i_fig, (pred_tf_to_plot, trial_id_offset, generation_label, color_pred) in enumerate([
        (pred_tf, 0, 'GaitDynamics Generation', colors[0]), 
        (pred_tf_no_overlapping, 8, 'GaitDynamics Generation (None Overlapping Windows)', 'C2')]):

        fig = plt.figure(figsize=(9, 12.5))
        gs = GridSpec(11, 1, figure=fig, height_ratios=[1, 1, 1, 0.1, 1, 1, 1, 0.1, 1, 1, 1], hspace=0.4, left=0.11, right=0.98, bottom=0.04, top=0.96)
        fix_seed()
        random_dset_list = random.sample(list(random_dset_trial_id_and_start_sample_dict.keys()), 3)
        for i_dset, dset in enumerate(random_dset_list):
            random_trial_id, start_sample, dset_name_print = random_dset_trial_id_and_start_sample_dict[dset]
            if i_dset == 2:
                random_trial_id_pred = random_trial_id + trial_id_offset
            else:
                random_trial_id_pred = random_trial_id
            for i_param, param_col in enumerate(params_of_interest.keys()):
                ax = fig.add_subplot(gs[4 * i_dset + i_param])
                plt.plot(true_tf[dset][random_trial_id][start_sample:start_sample+data_len, columns.index(param_col)] * 100 / 9.81,
                         color=[0.4, 0.4, 0.4], label='Experimental Measurement')
                plt.plot(pred_tf_to_plot[dset][random_trial_id_pred][start_sample:start_sample+data_len, columns.index(param_col)] * 100 / 9.81,
                         color=color_pred, label=generation_label)
                
                if i_fig == 1:
                    if i_dset == 0 and i_param == 1:
                        rect = plt.Rectangle((43, 5), 20, 15, linewidth=2, edgecolor=[0.6, 0.6, 0.6], facecolor='none', label='Subtle Discontinuity', linestyle='--')
                        ax.add_patch(rect)
                    if i_dset == 2 and i_param == 0:
                        rect = plt.Rectangle((188, 80), 20, 40, linewidth=2, edgecolor=[0.6, 0.6, 0.6], facecolor='none', label='Subtle Discontinuity', linestyle='--')
                        ax.add_patch(rect)
                
                format_axis(ax)
                ax.set_xlim([0, data_len])
                ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
                ax.set_xticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3], fontdict=FONT_DICT_SMALL)
                yticks = params_of_interest[param_col][1]
                ax.set_yticks(yticks)
                ax.set_yticklabels([str(ele) for ele in yticks], fontdict=FONT_DICT_SMALL)
                ax.set_ylabel(r'${}$ (\% BW)'.format(params_of_interest[param_col][0]), fontdict=FONT_DICT_SMALL)
                if i_param == 1:
                    ax.text(-0.1, 1.8, dset_name_print, fontdict=FONT_DICT, transform=ax.transAxes, ha='center', rotation=90, va='top')
                if i_param == 2:
                    ax.set_xlabel('Time (s)', fontdict=FONT_DICT_SMALL)
                if i_param == 1 and i_dset == 0:
                    plt.legend(fontsize=FONT_SIZE_SMALL, frameon=False, bbox_to_anchor=(1.02, 2.45), loc='lower right', ncols=3)

        for y_pos in [0.65, 0.325]:
            line = Line2D([0, 1], [y_pos, y_pos], color='gray', linestyle='-', linewidth=0.5, alpha=0.5, transform=fig.transFigure)
            fig.add_artist(line)
        plt.savefig(os.path.join(SCRIPT_DIR, f'exports/da_grf_supplement{trial_id_offset}.jpg'), dpi=600)
    plt.show()


def print_table_4(fast_run=False):
    """ Big table """
    folder = 'fast' if fast_run else 'full'
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    segments_full = ['trunk', 'pelvis angles', 'pelvis linear position', 'right hip', 'left hip', 'right knee', 'left knee', 'right ankle', 'left ankle']

    detailed_segment_to_osim_param = {
        'trunk': ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation'],
        'pelvis angles': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
        'pelvis linear position': ['pelvis_tx', 'pelvis_ty', 'pelvis_tz'],
        'right hip': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r'],
        'left hip': ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'],
        'right knee': ['knee_angle_r'],
        'left knee': ['knee_angle_l'],
        'right ankle': ['ankle_angle_r', 'subtalar_angle_r'],
        'left ankle': ['ankle_angle_l', 'subtalar_angle_l'],
    }

    csv_data = []

    header_row = segments_full + params_of_interest + ['test_name']
    csv_data.append(header_row)
    
    for i_test, test_name in enumerate(list(cols_to_unmask_main.keys()) + list(cols_to_unmask_big_table.keys())):
        if test_name == 'only_left_leg':
            segments_masked = ['trunk', 'pelvis angles', 'pelvis linear position'] + ['right ' + segment for segment in ['hip', 'knee', 'ankle']]
        elif test_name == 'only_right_leg':
            segments_masked = ['trunk', 'pelvis angles', 'pelvis linear position'] + ['left ' + segment for segment in ['hip', 'knee', 'ankle']]
        else:
            segments_masked = test_name.split('_')
            segments_masked += ['left ' + segment for segment in ['hip', 'knee', 'ankle'] if segment in segments_masked]
            segments_masked += ['right ' + segment for segment in ['hip', 'knee', 'ankle'] if segment in segments_masked]
            segments_masked += ['pelvis angles', 'pelvis linear position'] if 'pelvis' in segments_masked else []
        
        # Create row data
        row_data = []
        for segment in segments_full:
            if segment in segments_masked:
                if segment == 'pelvis linear position':
                    unit, scale, precision = ' m/s', 1, 2
                    all_the_metrics = get_metrics_linear_velocity(model_key=f'/{folder}/tf_{test_name}_diffusion_filling')
                else:
                    unit, scale, precision = ' deg', 180 / np.pi, 1
                    all_the_metrics = get_all_the_metrics(model_key=f'/{folder}/tf_{test_name}_diffusion_filling')
                param_metric = []
                for param in detailed_segment_to_osim_param[segment]:
                    param_metric.append(all_the_metrics[param])
                param_metric = np.mean(param_metric, axis=0)
                row_data.append(f'{np.mean(param_metric)*scale:.{precision}f}±{np.std(param_metric)*scale:.{precision}f}')
            else:
                row_data.append('✓')
                
        # Add metric values
        for param in params_of_interest:
            metrics = get_all_the_metrics(model_key=f'/{folder}/tf_{test_name}_diffusion_filling')
            row_data.append(f'{np.mean(metrics[param]):.1f}±{np.std(metrics[param]):.1f}')
        
        csv_data.append(row_data)
    
    csv_filename = os.path.join(SCRIPT_DIR, f"exports/table_4_{folder}.csv")
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_data)
    
    print(f"Table 4 saved to: {csv_filename}")
    
    print('\t'.join(segments_full))
    for row in csv_data[1:]:  # Skip header row
        print('\t'.join(row))

opt = parse_opt()
if __name__ == "__main__":
    # get_all_the_metrics(model_key=f'/full/tf_none_diffusion_filling')
    # print_table_1()
    # print_table_2()
    # draw_fig_1()
    draw_fig_2()
    # draw_fig_4()
    # draw_fig_supplementary_1()
    # print_table_4()
    # p_val_with_without_data_filter()
    # draw_fig_for_meeting()











