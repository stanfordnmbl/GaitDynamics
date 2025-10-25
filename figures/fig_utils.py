import matplotlib.pyplot as plt
import nimblephysics as nimble
from nimblephysics import NimbleGUI
import time


FONT_SIZE_LARGE = 15
FONT_SIZE = 13
FONT_SIZE_SMALL = 11
FONT_DICT = {'fontsize': FONT_SIZE}
FONT_DICT_LARGE = {'fontsize': FONT_SIZE_LARGE}
FONT_DICT_SMALL = {'fontsize': FONT_SIZE_SMALL}
FONT_DICT_X_SMALL = {'fontsize': FONT_SIZE_SMALL - 2}
LINE_WIDTH = 1.5
LINE_WIDTH_THICK = 2


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







