import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from args import parse_opt
from data.addb_dataset import MotionDataset
import numpy as np
import matplotlib.pyplot as plt
from da_grf_test_set_0 import load_model, loop_all
from model.utils import fix_seed
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, format_axis, LINE_WIDTH, FONT_DICT_LARGE, FONT_DICT
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


def one_trial_for_quantifying_randomness(num_of_tests):
    model, model_key = load_model(model_to_test=0)

    cols_to_unmask_conditions = {
        'Inputs: All Kinematics except Knee Kinematics': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('knee' not in col)],
        'Inputs: Hips, Knees and Ankles Kinematics': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('hip' in col or 'knee' in col or 'ankle' in col or 'subtalar' in col)],
        'Inputs: Hips Kinematics Only': [i_col for i_col, col in enumerate(opt.model_states_column_names) if ('hip' in col)],
    }
    
    test_dataset = MotionDataset(
        data_path='/mnt/d/Local/Data/MotionPriorData/b3d_no_arm/Camargo2021_Formatted_No_Arm/',
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        include_trials_shorter_than_window_len=True,
        specific_trial='treadmill_01_01_segment_5'      # segment_5 has a normal walking speed
    )
    color_true = np.array([100, 100, 100]) / 255
    color_pred = np.array([70, 130, 180]) / 255

    pred_param, true_param = {}, {}
    params_of_interest = ['calcn_l_force_vy', 'knee_angle_l']
    params_of_interest_ratios = [1, 180 / np.pi]
    params_of_interest_names_formal = ['Vertical Force (% BW)', 'Knee Flexion Angle (°)']
    fig = plt.figure(figsize=(9, 9))
    gs = GridSpec(3, 2, figure=fig, hspace=0.6, wspace=0.3, left=0.09, right=0.98, bottom=0.06, top=0.91)

    for i_condition, (condition_name, cols_to_unmask) in enumerate(cols_to_unmask_conditions.items()):
        pred_param[condition_name] = {}
        true_param[condition_name] = {}
        windows, s_list, e_list = test_dataset.get_overlapping_wins(cols_to_unmask, 150)
        for i_test in range(num_of_tests):
            fix_seed(i_test)
            true_sub, pred_sub, pred_std_sub, column_names = loop_all(
                model, opt, test_dataset.skels, test_dataset.trials, windows, windows,
                s_list, e_list, exclude_probably_missing=False)
            
            col_loc_of_interest = [column_names.index(param) for param in params_of_interest]
            pred_param[condition_name][i_test] = pred_sub[0][:150, col_loc_of_interest]
            true_param[condition_name][i_test] = true_sub[0][:150, col_loc_of_interest]
        
        for i_param in range(len(params_of_interest)):
            preds = np.stack([pred_param[condition_name][i_test][:, i_param] * params_of_interest_ratios[i_param] for i_test in range(num_of_tests)], axis=0)
            mean_pred = np.mean(preds, axis=0)
            std_pred = np.std(preds, axis=0)
            ax = fig.add_subplot(gs[i_condition * 2 + i_param])

            plt.plot(true_param[condition_name][0][:, i_param] * params_of_interest_ratios[i_param],
                     label='Experimental Measurement', color=color_true, linewidth=2, zorder=100)
            plt.fill_between(np.arange(mean_pred.shape[0]), mean_pred - std_pred, mean_pred + std_pred, color=color_pred, alpha=0.3)
            
            plt.plot(pred_param[condition_name][0][:, i_param] * params_of_interest_ratios[i_param], '--', color=color_pred, label='GaitDynamics Generation')
            for i_test in range(1, num_of_tests):
                plt.plot(pred_param[condition_name][i_test][:, i_param] * params_of_interest_ratios[i_param], '--', color=color_pred)
            plt.xlabel('Time (s)', fontsize=FONT_SIZE_SMALL)
            plt.ylabel(params_of_interest_names_formal[i_param], fontsize=FONT_SIZE_SMALL)
            format_axis()
            # ax = plt.gca()
            ax.set_xlim([0, 150])
            ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
            ax.set_xticklabels([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5], fontdict=FONT_DICT_SMALL)
            if i_param == 0:
                if i_condition == 0:
                    ax.set_ylim([-0.5, 15])
                    ax.set_yticks([0, 3, 6, 9, 12, 15])
                    ax.set_yticklabels([0, 3, 6, 9, 12, 15], fontdict=FONT_DICT_SMALL)
                elif i_condition == 1:
                    ax.set_ylim([-0.5, 15])
                    ax.set_yticks([0, 3, 6, 9, 12, 15])
                    ax.set_yticklabels([0, 3, 6, 9, 12, 15], fontdict=FONT_DICT_SMALL)
                elif i_condition == 2:
                    ax.set_ylim([-3, 25])
                    ax.set_yticks([0, 5, 10, 15, 20, 25])
                    ax.set_yticklabels([0, 5, 10, 15, 20, 25], fontdict=FONT_DICT_SMALL)
            else:
                ax.set_ylim([0, 90])
                ax.set_yticks([0, 20, 40, 60, 80])
                ax.set_yticklabels([0, 20, 40, 60, 80], fontdict=FONT_DICT_SMALL)
                
            if i_param == 0:
                ax.text(-0.15, 1.15, condition_name, fontdict=FONT_DICT, transform=ax.transAxes, ha='left', va='center')
            if i_param == 1 and i_condition == 0:
                plt.legend(fontsize=FONT_SIZE_SMALL, frameon=False, bbox_to_anchor=(0.9, 1.4), ncols=1)
    for y_pos in [0.64, 0.32]:
        line = Line2D([0, 1], [y_pos, y_pos], color='gray', linestyle='-', linewidth=0.5, alpha=0.5, transform=fig.transFigure)
        fig.add_artist(line)
    plt.savefig(os.path.join(SCRIPT_DIR, f'exports/da_randomness.png'), dpi=300)
    plt.show()
            

opt = parse_opt()
if __name__ == "__main__":
    one_trial_for_quantifying_randomness(num_of_tests=10)






