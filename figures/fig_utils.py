import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import time
import pandas as pd

FONT_SIZE_LARGE = 15
FONT_SIZE = 13
FONT_SIZE_SMALL = 11
FONT_DICT = {'fontsize': FONT_SIZE}
FONT_DICT_LARGE = {'fontsize': FONT_SIZE_LARGE}
FONT_DICT_SMALL = {'fontsize': FONT_SIZE_SMALL}
FONT_DICT_X_SMALL = {'fontsize': 15}
LINE_WIDTH = 1.5
LINE_WIDTH_THICK = 2


def naive_stance_phase_extractor(v_grf):
    """ Can only be used for clean data such as Hamner2013. """
    stance_vgrf_thd = 5    # 100% of body mass. Needs to be large because some datasets are noisy.
    stance_start_valid, stance_end_valid = [], []
    stance_flag = np.abs(v_grf) > stance_vgrf_thd
    stance_flag = stance_flag.astype(int)
    start_end_indicator = np.diff(stance_flag)
    stance_start = np.where(start_end_indicator == 1)[0]
    stance_end = np.where(start_end_indicator == -1)[0]
    for i_start in range(0, len(stance_start)):
        try:
            end_ = stance_end[(stance_start[i_start] < stance_end)][0]
        except IndexError:
            continue
        stance_start_valid.append(stance_start[i_start])
        stance_end_valid.append(end_)
    return stance_start_valid, stance_end_valid


def extract_gait_parameters_from_osim_states_and_append(poses, skel, opt, param_dict, max_num_to_append=1):
    param_dict_new = extract_gait_parameters_from_osim_states(poses, skel, opt)
    for key in param_dict_new.keys():
        if key not in param_dict:
            param_dict[key] = param_dict_new[key]
        else:
            param_dict[key].extend(param_dict_new[key][:max_num_to_append])
    return param_dict


def extract_gait_parameters_from_osim_states(poses, skel, opt):
    v_grf = poses[:, opt.osim_dof_columns.index('calcn_r_force_vy')]
    stance_start_valid, stance_end_valid = naive_stance_phase_extractor(v_grf)
    param_dict = {'hip_flexion_r': [], 'knee_angle_r': [], 'ankle_angle_r': [],
                  'hip_flexion_r_max': [], 'knee_angle_r_max': [], 'ankle_angle_r_max': [],
                  'hip_flexion_r_min': [], 'knee_angle_r_min': [], 'ankle_angle_r_min': [],
                  'stride_length_r': [], 'stride_time_r': [], 'stance_time_r': []}
    for i_step in range(len(stance_start_valid) - 1):
        start_, end_ = stance_start_valid[i_step], stance_start_valid[i_step+1]
        for key in ('hip_flexion_r', 'knee_angle_r', 'ankle_angle_r'):
            param_dict[key].append(poses[start_:end_, opt.osim_dof_columns.index(key)])
            param_dict[key + '_max'].append(np.max(poses[start_:end_, opt.osim_dof_columns.index(key)]))
            param_dict[key + '_min'].append(np.min(poses[start_:end_, opt.osim_dof_columns.index(key)]))

        stance_time = (stance_end_valid[i_step] - start_) / opt.target_sampling_rate
        stride_time = (end_ - start_) / opt.target_sampling_rate
        skel.setPositions(poses[start_, opt.kinematic_osim_col_loc])
        foot_loc_r_start = skel.getBodyNode('calcn_r').getWorldTransform().translation()
        skel.setPositions(poses[end_, opt.kinematic_osim_col_loc])
        foot_loc_r_end = skel.getBodyNode('calcn_r').getWorldTransform().translation()
        param_dict['stance_time_r'].append(stance_time)
        param_dict['stride_time_r'].append(stride_time)
        param_dict['stride_length_r'].append((foot_loc_r_end - foot_loc_r_start)[0])
    return param_dict


def set_up_gui():
    world = nimble.simulation.World()
    world.setGravity([0, -9.81, 0])
    gui = NimbleGUI(world)
    gui_inited = False
    port = 8090
    while not gui_inited:
        try:
            gui.serve(port)
            gui_inited = True
        except Exception:
            port += 1
    return gui


def show_skeletons(opt, name_states_dict, gui, skel):
    num_frames = list(name_states_dict.values())[0].shape[0]
    for i_frame in range(num_frames):
        for i_skel, states in enumerate(list(name_states_dict.values())):
            states[i_frame, 5] += 0.5 * i_skel
            states[i_frame, opt.cop_osim_col_loc[2]] += 0.5 * i_skel
            states[i_frame, opt.cop_osim_col_loc[5]] += 0.5 * i_skel

            poses = states[i_frame, opt.kinematic_osim_col_loc]
            skel.setPositions(poses)
            gui.nativeAPI().renderSkeleton(skel, prefix='skel' + str(i_skel))
            for i_force, contact_body in enumerate(['calcn_r', 'calcn_l']):
                forces = states[i_frame, opt.grf_osim_col_loc[3 * i_force:3 * (i_force + 1)]]
                cop = states[i_frame, opt.cop_osim_col_loc[3 * i_force:3 * (i_force + 1)]]
                gui.nativeAPI().createLine(f'line_{i_skel}_{i_force}', [cop, cop + 0.1 * forces], color=[1, 0., 0., 1])
        time.sleep(0.2)


def get_scores(y_true, y_pred, y_fields, exclude_swing_phase=False):
    scores = []
    for col, field in enumerate(y_fields):
        if exclude_swing_phase:
            non_zero_idx = np.where(y_true[:, col] != 0)[0]
            true_, pred_ = y_true[non_zero_idx, col], y_pred[non_zero_idx, col]
        else:
            true_, pred_ = y_true[:, col], y_pred[:, col]

        r2 = r2_score(true_, pred_)
        rmse = np.sqrt(mse(true_, pred_))
        mae = np.mean(np.abs(true_ - pred_))
        cor_value = pearsonr(true_, pred_)[0]
        score_one_field = {'field': field, 'r2': r2, 'rmse': rmse, 'cor_value': cor_value, 'mae': mae}
        scores.append(score_one_field)
    return scores


class VanCriekMetaData:
    def __init__(self):
        self.meta_table = pd.read_csv('/mnt/g/Shared drives/NMBL Shared Data/datasets/VanCriekinge2023/meta_data.csv')
        self.meta_table.columns = [col.split('\n')[0] for col in self.meta_table.columns]

    def get_sub_meta(self, sub):
        # meta_table = pd.read_csv('/mnt/g/Shared drives/NMBL Shared Data/datasets/VanCriekinge2023/meta_data.csv')
        # meta_table.columns = [col.split('\n')[0] for col in meta_table.columns]
        sub_meta = self.meta_table[self.meta_table['ID'] == sub]
        assert len(sub_meta) == 1
        sub_meta_dict = sub_meta.iloc[0].to_dict()
        if np.isnan(sub_meta_dict['FAC']):
            sub_meta_dict['FAC'] = 0
        return sub_meta_dict


vancriek_bad_sub_and_trial_names = ['TVC23__BWA5_segment_0'] + \
                                   []


def find_peak_max(data_clip, height, width=None, prominence=None):
    """
    find the maximum peak
    :return:
    """
    peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
    if len(peaks) == 0:
        return None
    peak_heights = properties['peak_heights']
    return np.max(peak_heights)


def save_fig(name, dpi=300):
    plt.savefig('exports/' + name + '.png', dpi=dpi)


def format_axis(ax=None, line_width=LINE_WIDTH):
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(width=line_width)
    ax.yaxis.set_tick_params(width=line_width)
    ax.spines['left'].set_linewidth(line_width)
    ax.spines['bottom'].set_linewidth(line_width)


def hide_axis_add_grid():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(color=[0.9, 0.9, 0.9], linewidth=1)
    ax.tick_params(color=[0.9, 0.9, 0.9], width=1)







