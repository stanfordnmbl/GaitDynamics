import pickle
from consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import get_scores
import numpy as np
import matplotlib.pyplot as plt
from fig_utils import set_up_gui
from args import parse_opt
import nimblephysics as nimble
import time


def get_results():
    results_true, results_pred, results_pred_std, results_bl, columns, _, _, _, =\
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))
    dset_list = list(results_true.keys())

    params_of_interest = ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']
    params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

    rmse_ap, rmse_v, rmses_ml = [], [], []
    rmse_ap_bl, rmse_v_bl, rmses_ml_bl = [], [], []
    if visualize_as_video:
        gui = set_up_gui()
        opt = parse_opt()
        customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(
            f'/mnt/d/Local/Data/MotionPriorData/model_and_geometry/unscaled_generic_no_arm.osim')
        skel = customOsim.skeleton
        colors = [[1, 0., 0., 1], [0., 1, 0., 1]]

    for dset in dset_list:
        if dset in EXCLUDE_FROM_ASB:
            continue

        bl_ = np.concatenate(list(results_bl[dset].values()))[:, params_of_interest_col_loc]
        in_gait_phase = np.all(bl_ != NOT_IN_GAIT_PHASE, axis=1)
        bl_ = bl_[in_gait_phase]

        true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc][in_gait_phase]
        pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc][in_gait_phase]
        pred_std = np.concatenate(list(results_pred_std[dset].values()))[:, params_of_interest_col_loc][in_gait_phase]

        if visualize_as_video:
            true_pose = np.concatenate(list(results_true[dset].values()))
            pred_pose = np.concatenate(list(results_pred[dset].values()))
            start_frame = 300
            end_frame = start_frame + 150
            # gui.nativeAPI().createText('Dataset', 'Dataset ' + dset.split('_')[0], [1000, 50], [500, 200])
            x_list, y_list = {0: {0: [], 1: []}, 1: {0: [], 1: []}}, {0: {0: [], 1: []}, 1: {0: [], 1: []}}
            for i_frame in range(start_frame, end_frame):
                poses = true_pose[i_frame, opt.kinematic_osim_col_loc]
                skel.setPositions(poses)
                gui.nativeAPI().renderSkeleton(skel, prefix='skel')
                gui.nativeAPI().createRichPlot(f'plot_{0}', [400, 200], [500, 350], 0, 150, 0, 30*75, 'Right Foot 3-D Force Prediction', 'Time', 'Force (N)')
                for i_source, grf_source in enumerate([true_pose, pred_pose]):
                    for i_force, contact_body in enumerate(['calcn_r', 'calcn_l']):
                        forces = grf_source[i_frame, opt.grf_osim_col_loc[3 * i_force:3 * (i_force + 1)]]
                        cop = skel.getBodyNode(contact_body).getWorldTransform().translation()
                        x_list[i_source][i_force].append(i_frame - start_frame)
                        y_list[i_source][i_force].append(np.linalg.norm(forces) * 75)       # 75 kg
                        gui.nativeAPI().createLine(f'line_{i_source}_{i_force}', [cop, cop + 0.06 * forces], color=colors[i_source])

                        if i_force == 0:
                            gui.nativeAPI().setRichPlotData(f'plot_{0}', 'Measured', 'red', 'line', x_list[0][i_force], y_list[0][i_force])
                            gui.nativeAPI().setRichPlotData(f'plot_{0}', 'Predicted', 'green', 'line', x_list[1][i_force], y_list[1][i_force])
                time.sleep(0.01)

        if not visualize_as_video:
            plt.subplots(3, 1, figsize=(10, 8))
            for i in range(3):
                plt.subplot(3, 1, i+1)
                plt.plot(true_[:, i], label='True')
                plt.plot(pred_[:, i], label='Predicted')
                plt.fill_between(range(len(pred_)), pred_[:, i] - pred_std[:, i], pred_[:, i] + pred_std[:, i], color='C1', alpha=0.5)
                plt.grid()
            plt.legend()
            plt.suptitle(dset)

        scores_bl = get_scores(true_, bl_, params_of_interest, None)
        rmse_ap_bl.append(scores_bl[0]['rmse'])
        rmse_v_bl.append(scores_bl[1]['rmse'])
        rmses_ml_bl.append(scores_bl[2]['rmse'])

        scores = get_scores(true_, pred_, params_of_interest, None)
        rmse_ap.append(scores[0]['rmse'])
        rmse_v.append(scores[1]['rmse'])
        rmses_ml.append(scores[2]['rmse'])

        print('{}, {:.2f}'.format(dset, scores[1]['rmse']))

    print(f'{np.mean(rmses_ml_bl):.2f} ± {np.std(rmses_ml_bl):.2f}')
    print(f'{np.mean(rmses_ml):.2f} ± {np.std(rmses_ml):.2f}')

    print(f'{np.mean(rmse_ap_bl):.2f} ± {np.std(rmse_ap_bl):.2f}')
    print(f'{np.mean(rmse_ap):.2f} ± {np.std(rmse_ap):.2f}')

    print(f'{np.mean(rmse_v_bl):.2f} ± {np.std(rmse_v_bl):.2f}')
    print(f'{np.mean(rmse_v):.2f} ± {np.std(rmse_v):.2f}')

    plt.show()


visualize_as_video = False
test_data_name = 'downstream_grf'
if __name__ == "__main__":
    get_results()



