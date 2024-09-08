import pickle
import numpy as np
import matplotlib.pyplot as plt
from da_grf_test_set_0 import cols_to_unmask, dset_to_skip


def print_table_1(model_key):
    print(model_key)
    segment_to_param = {
        'velocity': ['pelvis_tx', 'pelvis_ty', 'pelvis_tz'],
        'trunk': ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation'],
        'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
        'hip': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'],
        'knee': ['knee_angle_r', 'knee_angle_l'],
        'ankle': ['ankle_angle_r', 'subtalar_angle_r', 'ankle_angle_l', 'subtalar_angle_l'],
    }
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    segment_list = ['velocity', 'trunk', 'pelvis', 'hip', 'knee', 'ankle']
    metric_dict, masked_segment_col_loc = {}, {}
    f_name = f'addb_marker_based_{model_key}'
    results_dict = pickle.load(open(f"results/{f_name}.pkl", "rb"))

    for test_name in results_dict.keys():
        # true_, pred_, pred_std_, params_of_interest, masked_osim_dofs = results_dict[test_name]
        true_, pred_, pred_std_, columns = results_dict[test_name]
        # masked_osim_dofs = [col for col in columns if col not in params_of_interest]
        # params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]
        # masked_osim_dofs_col_loc = [columns.index(col) for col in masked_osim_dofs]

        masked_segment_col_loc[test_name] = [segment_list.index(segment) for segment in test_name.split('_')] if test_name != 'none' else []

        true_all, pred_all = {}, {}
        for dset in true_.keys():
            # gait_phase_label, stance_start_valid, stance_end_valid = MotionDataset.grf_to_trial_gait_phase_label(
            #     true_[dset][:, columns.index('calcn_l_force_vy')], 150, 100)
            #
            # true_to_keep, pred_to_keep = [], []
            # for i in range(0, true_[dset].shape[0], 150):
            #     if (gait_phase_label[i-150:i+300] != NOT_IN_GAIT_PHASE).any() and (true_[dset][i-150:i+300, columns.index('calcn_l_force_vy')] > -1).all():
            #         true_to_keep.append(true_[dset][i:i+150])
            #         pred_to_keep.append(pred_[dset][i:i+150])
            # if len(true_to_keep) == 0:
            #     continue
            # true_[dset] = np.concatenate(true_to_keep, axis=0)
            # pred_[dset] = np.concatenate(pred_to_keep, axis=0)

            dset_short = dset.split('_Formatted_')[0]
            if dset_short not in true_all.keys():
                true_all[dset_short] = true_[dset]
                pred_all[dset_short] = pred_[dset]
            else:
                true_all[dset_short] = np.concatenate((true_all[dset_short], true_[dset]), axis=0)
                pred_all[dset_short] = np.concatenate((pred_all[dset_short], pred_[dset]), axis=0)

        metric_dict[test_name] = {param: [] for param in columns}
        true_all = {k: v for k, v in true_all.items() if k + '_Formatted_No_Arm' not in dset_to_skip}
        pred_all = {k: v for k, v in pred_all.items() if k + '_Formatted_No_Arm' not in dset_to_skip}
        for i_dset, dset in enumerate(true_all.keys()):
            for i_param, param in enumerate(columns):
                param_col_loc = columns.index(param)
                metric_dict[test_name][columns[i_param]].append(np.mean(np.abs(true_all[dset][:, param_col_loc] - pred_all[dset][:, param_col_loc])))
                # metric_dict[test_name][i_param].append(np.sqrt(np.mean((true_all[dset][:, param_col_loc] - pred_all[dset][:, param_col_loc])**2)))
                if 'force_v' in param:
                    metric_dict[test_name][columns[i_param]][-1] = metric_dict[test_name][columns[i_param]][-1] * 100 / 9.81


            print('{}, {:.1f}'.format(dset, metric_dict[test_name]['calcn_l_force_vy'][i_dset]))

            plt.figure()
            plt.plot(true_all[dset][:, columns.index('calcn_l_force_vy')], label='True')
            plt.plot(pred_all[dset][:, columns.index('calcn_l_force_vy')], label='Pred')
            plt.title('{}, {:.2f}'.format(dset, metric_dict[test_name]['calcn_l_force_vy'][i_dset]))
        plt.show()

        #     dof = 'knee_angle_l'
        #     if dof in columns:
        #         plt.figure()
        #         plt.plot(true_all[dset][:, columns.index(dof)], label='True')
        #         plt.plot(pred_all[dset][:, columns.index(dof)], label='Pred')
        #         plt.title('{}, {}, {:.1f}'.format(test_name, dset, metric_dict[test_name][dof][i_dset]))
        # plt.show()

    for test_name in results_dict.keys():
        string_ = ''
        for i_segment, (segment, params) in enumerate(segment_to_param.items()):
            if segment not in test_name:
                string_ += '\ding{51}\t&'
            else:
                if segment == 'velocity':
                    unit, scale = ' m/s', 1
                else:
                    unit, scale = ' deg', 180 / np.pi
                param_metric = []
                for param in params:
                    param_metric.extend(metric_dict[test_name][param])
                string_ += f'{np.mean(param_metric)*scale:.1f} $\pm$ {np.std(param_metric)*scale:.1f}{unit}\t&'

        print(string_, end=' ')
        for i_param, param in enumerate(params_of_interest):
            print(f'& {np.mean(metric_dict[test_name][param]):.1f} $\pm$ {np.std(metric_dict[test_name][param]):.1f}', end='\t')
        print('\\\\')


def combine_splits(results_dict):
    results_dict_combined = {}
    for test_name in results_dict.keys():
        true_, pred_, pred_std_, columns = results_dict[test_name]
        true_all, pred_all = {}, {}
        for dset in true_.keys():
            dset_short = dset.split('_Formatted_')[0]
            if dset_short not in true_all.keys():
                true_all[dset_short] = true_[dset]
                pred_all[dset_short] = pred_[dset]
            else:
                true_all[dset_short] = np.concatenate((true_all[dset_short], true_[dset]), axis=0)
                pred_all[dset_short] = np.concatenate((pred_all[dset_short], pred_[dset]), axis=0)
        results_dict_combined[test_name] = (true_all, pred_all, pred_std_, columns)
    return results_dict_combined


def print_table_2():
    """ Accuracies of each dataset """
    dset_order = ['Camargo2021', 'Carter2023', 'Hamner2013', 'Tan2021', 'Moore2015', 'Tan2022', 'vanderZee2022',
                  'Wang2023', 'Fregly2012', 'Falisse2017', 'Han2023', 'Li2021', 'Tiziana2019', 'Uhlrich2023']
    params_of_interest = ['calcn_l_force_vy', 'calcn_l_force_vx', 'calcn_l_force_vz']
    results_tf_dict = pickle.load(open(f"results/addb_marker_based_baseline_sugainet.pkl", "rb"))
    results_tf_dict = combine_splits(results_tf_dict)
    results_diffusion_dict = pickle.load(open(f"results/addb_marker_based_diffusion.pkl", "rb"))
    results_diffusion_dict = combine_splits(results_diffusion_dict)

    test_name = 'none'
    results_array = [[] for _ in range(len(dset_order))]
    for i_dset, dset_short in enumerate(dset_order):
        dset = dset_short + '_Formatted_No_Arm'
        if dset in dset_to_skip:        #  or dset_short not in results_tf_dict[test_name][0].keys()
            continue
        for results_dict in [results_tf_dict, results_diffusion_dict]:
            true_, pred_, pred_std_, columns = results_dict[test_name]
            params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]
            for i_param, param_col_loc in enumerate(params_of_interest_col_loc):
                metric_mean = np.mean(np.abs(true_[dset_short][:, param_col_loc] - pred_[dset_short][:, param_col_loc])) * 100 / 9.81
                print(f'{metric_mean:.1f}', end='\t')
                results_array[i_dset].append(metric_mean)
        print()
    results_average = np.mean(np.array(results_array), axis=0)
    [print(round(element, 1), end='\t') for element in results_average]


def print_table_3():
    """ Accuracies of joint moments """
    results_tf_dict = pickle.load(open(f"results/addb_marker_based_baseline_tf.pkl", "rb"))
    results_tf_dict = combine_splits(results_tf_dict)
    params_of_interest = ['knee_moment_r_x', 'knee_moment_r_z', 'hip_moment_r_z', 'ankle_moment_r_z']
    test_name = 'none'
    dset_list = list(results_tf_dict[test_name][0].keys())

    print('\t\t', end='')
    for i_param, param_col in enumerate(params_of_interest):
        print(param_col, end='\t')
    print()

    results_array = [[] for _ in range(len(dset_list))]
    for i_dset, dset_short in enumerate(dset_list):
        dset = dset_short + '_Formatted_No_Arm'
        if dset in dset_to_skip:        #  or dset_short not in results_tf_dict[test_name][0].keys()
            continue
        print(f'{dset_short[:7]}\t', end='')
        true_, pred_, pred_std_, columns = results_tf_dict[test_name]
        for i_param, param_col in enumerate(params_of_interest):
            param_col_loc = columns.index(param_col)
            metric_mean = np.mean(np.abs(true_[dset_short][:, param_col_loc] - pred_[dset_short][:, param_col_loc]))
            print(f'{metric_mean:.2f}', end='\t\t\t')
            results_array[i_dset].append(metric_mean)
        print()
    results_average = np.mean(np.array(results_array), axis=0)
    print('Average\t', end='')
    [print(round(element, 1), end='\t') for element in results_average]


if __name__ == "__main__":
    print_table_1(model_key='diffusion')        # 'diffusion', 'baseline_tf', 'baseline_groundlink', 'baseline_sugainet'

    # print_table_2()

    # print_table_3()

