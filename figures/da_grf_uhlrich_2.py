import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from data.addb_dataset import MotionDataset
from fig_utils import FONT_DICT_SMALL, FONT_SIZE_SMALL, LINE_WIDTH, LINE_WIDTH_THICK, format_axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from da_grf_test_set_0 import cols_to_unmask, dset_to_skip
# cols_to_unmask = {key: cols_to_unmask[key] for key in ['trunk_pelvis_knee_ankle', 'velocity_hip']}  # !!!


def print_table():
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    metric_dict, metric_std_dict = {}, {}
    for name in cols_to_unmask.keys():
        file_name = f'addb_marker_based_{name}'

        true_, pred_, pred_std_, columns = pickle.load(open(f"results/{file_name}.pkl", "rb"))
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

        metric_dict[file_name] = [[] for _ in range(len(params_of_interest))]
        true_all = {k: v for k, v in true_all.items() if k + '_Formatted_No_Arm' not in dset_to_skip}
        pred_all = {k: v for k, v in pred_all.items() if k + '_Formatted_No_Arm' not in dset_to_skip}
        for i_dset, dset in enumerate(true_all.keys()):
            for i_param, param_col_loc in enumerate(params_of_interest_col_loc):
                metric_dict[file_name][i_param].append(np.mean(np.abs(true_all[dset][:, param_col_loc] - pred_all[dset][:, param_col_loc])))
                # metric_dict[file_name][i_param].append(np.sqrt(np.mean((true_all[dset][:, param_col_loc] - pred_all[dset][:, param_col_loc])**2)))

            print('{}, {:.2f}'.format(dset, metric_dict[file_name][0][i_dset]))

    #         plt.figure()
    #         plt.plot(true_all[dset][:, params_of_interest_col_loc[0]], label='True')
    #         plt.plot(pred_all[dset][:, params_of_interest_col_loc[0]], label='Pred')
    #         plt.title('{}, {:.2f}'.format(dset, metric_dict[file_name][0][i_dset]))
    # plt.show()

    for name in cols_to_unmask.keys():
        file_name = f'addb_marker_based_{name}'
        print(f'{string_map[name]}', end=' ')
        for i_param, param_col_loc in enumerate(params_of_interest_col_loc):
            print(f'& {np.mean(metric_dict[file_name][i_param]):.2f} $\pm$ {np.std(metric_dict[file_name][i_param]):.2f}', end='\t')
        print('\\\\')


string_map = {
    'none': '\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&',
    'velocity': '\t\t\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&',
    'trunk': '\ding{51}\t&\t\t\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&',
    'pelvis': '\ding{51}\t&\ding{51}\t&\t\t\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&',
    'hip': '\ding{51}\t&\ding{51}\t&\ding{51}\t&\t\t\t&\ding{51}\t&\ding{51}\t&',
    'knee': '\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\t\t\t&\ding{51}\t&',
    'ankle': '\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&\t\t\t&',
    # 'trunk_pelvis': '\ding{51}\t&\t\t\t&\t\t\t&\ding{51}\t&\ding{51}\t&\ding{51}\t&',
    # 'trunk_hip': '\ding{51}\t&\t\t\t&\ding{51}\t&\t\t\t&\ding{51}\t&\ding{51}\t&',
    # 'pelvis_hip': '\ding{51}\t&\ding{51}\t&\t\t\t&\t\t\t&\ding{51}\t&\ding{51}\t&',
    # 'hip_knee_ankle': '\ding{51}\t&\ding{51}\t&\ding{51}\t&\t\t\t&\t\t\t&\t\t\t&',
    'trunk_pelvis_knee_ankle': '\ding{51}\t&\t\t\t&\t\t\t&\ding{51}\t&\t\t\t&\t\t\t&',
    'velocity_hip': '\t\t\t&\ding{51}\t&\ding{51}\t&\t\t\t&\ding{51}\t&\ding{51}\t&',
}


if __name__ == "__main__":
    print_table()



