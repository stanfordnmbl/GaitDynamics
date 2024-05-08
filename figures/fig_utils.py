import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error as mse
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import time


FONT_SIZE_LARGE = 15
FONT_SIZE = 13
FONT_SIZE_SMALL = 11
FONT_DICT = {'fontsize': FONT_SIZE}
FONT_DICT_LARGE = {'fontsize': FONT_SIZE_LARGE}
FONT_DICT_SMALL = {'fontsize': FONT_SIZE_SMALL}
FONT_DICT_X_SMALL = {'fontsize': 15}
LINE_WIDTH = 1.5
LINE_WIDTH_THICK = 2


def set_up_gui():
    world = nimble.simulation.World()
    world.setGravity([0, -9.81, 0])
    gui = NimbleGUI(world)
    gui.serve(8090)
    return gui


def show_skeletons(opt, name_states_dict, gui, skels, trial):
    num_frames = list(name_states_dict.values())[0].shape[0]
    for i_frame in range(num_frames):
        for i_skel, states in enumerate(list(name_states_dict.values())):
            states[i_frame, 5] += 0.5 * i_skel
            states[i_frame, opt.cop_osim_col_loc[2]] += 0.5 * i_skel
            states[i_frame, opt.cop_osim_col_loc[5]] += 0.5 * i_skel

            poses = states[i_frame, opt.kinematic_osim_col_loc]
            skels[i_skel].setPositions(poses)
            gui.nativeAPI().renderSkeleton(skels[i_skel], prefix='skel' + str(i_skel))
            for i_force, contact_body in enumerate(['calcn_r', 'calcn_l']):
                forces = states[i_frame, opt.grf_osim_col_loc[3 * i_force:3 * (i_force + 1)]]
                cop = states[i_frame, opt.cop_osim_col_loc[3 * i_force:3 * (i_force + 1)]]
                gui.nativeAPI().createLine(f'line_{i_skel}_{i_force}', [cop, cop + 0.1 * forces], color=[1, 0., 0., 1])
        time.sleep(0.05)


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


def format_errorbar_cap(caplines, size=15):
    for i_cap in range(1):
        caplines[i_cap].set_marker('_')
        caplines[i_cap].set_markersize(size)
        caplines[i_cap].set_markeredgewidth(LINE_WIDTH)


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
    ax.grid(color='lightgray', linewidth=1.5)
    ax.tick_params(color='lightgray', width=1.5)







