import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from data.addb_dataset import MotionDataset
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from da_grf_test_set_0 import cols_to_unmask, dset_to_skip
# cols_to_unmask = {key: cols_to_unmask[key] for key in ['trunk_pelvis_knee_ankle', 'velocity_hip']}  # !!!


def draw_figure():
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    metric_dict, metric_std_dict = {}, {}
    f_name = f'addb_marker_based_{model_key}'
    results_dict = pickle.load(open(f"results/{f_name}.pkl", "rb"))

    for test_name in results_dict.keys():
        true_, pred_, pred_std_, columns = results_dict[test_name]
        params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

        true_all, pred_all = {}, {}
        for dset in true_.keys():
            gait_phase_label, stance_start_valid, stance_end_valid = MotionDataset.grf_to_trial_gait_phase_label(
                true_[dset][:, columns.index('calcn_l_force_vy')], 150, 100)

            true_to_keep, pred_to_keep = [], []
            for i in range(0, true_[dset].shape[0], 150):
                if (gait_phase_label[i-150:i+300] != NOT_IN_GAIT_PHASE).any() and (true_[dset][i-150:i+300, columns.index('calcn_l_force_vy')] > -1).all():
                    true_to_keep.append(true_[dset][i:i+150])
                    pred_to_keep.append(pred_[dset][i:i+150])
            if len(true_to_keep) == 0:
                continue
            true_[dset] = np.concatenate(true_to_keep, axis=0)
            pred_[dset] = np.concatenate(pred_to_keep, axis=0)

            dset_short = dset.split('_Formatted_')[0]
            if dset_short not in true_all.keys():
                true_all[dset_short] = true_[dset]
                pred_all[dset_short] = pred_[dset]
            else:
                true_all[dset_short] = np.concatenate((true_all[dset_short], true_[dset]), axis=0)
                pred_all[dset_short] = np.concatenate((pred_all[dset_short], pred_[dset]), axis=0)

        metric_dict[test_name] = [[] for _ in range(len(params_of_interest))]
        true_all = {k: v for k, v in true_all.items() if k + '_Formatted_No_Arm' not in dset_to_skip}
        pred_all = {k: v for k, v in pred_all.items() if k + '_Formatted_No_Arm' not in dset_to_skip}
        for i_dset, dset in enumerate(true_all.keys()):
            for i_param, param_col_loc in enumerate(params_of_interest_col_loc):
                metric_dict[test_name][i_param].append(np.mean(np.abs(true_all[dset][:, param_col_loc] - pred_all[dset][:, param_col_loc])))
                # metric_dict[test_name][i_param].append(np.sqrt(np.mean((true_all[dset][:, param_col_loc] - pred_all[dset][:, param_col_loc])**2)))

            print('{}, {:.2f}'.format(dset, metric_dict[test_name][0][i_dset]))

    for test_name in results_dict.keys():
        # test_name = f'addb_marker_based_{name}'
        print(f'{string_map[test_name]}', end=' ')
        for i_param, param_col_loc in enumerate(params_of_interest_col_loc):
            print(f'& {np.mean(metric_dict[test_name][i_param]):.2f} $\pm$ {np.std(metric_dict[test_name][i_param]):.2f}', end='\t')
        print('\\\\')


if __name__ == "__main__":
    # model_key = 'baseline_tf'
    model_key = 'diffusion'
    print_table()



