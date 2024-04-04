import pickle
from matplotlib.legend_handler import HandlerBase
from alant.alan_consts import NOT_IN_GAIT_PHASE, EXCLUDE_FROM_ASB
from fig_utils import get_scores, format_axis, FONT_DICT, LINE_WIDTH_THICK, save_fig
import numpy as np
import matplotlib.pyplot as plt


params_of_interest_sets = {'Ankle': ['ankle_angle_r'],
                           'Knee': ['knee_angle_r'],
                           'Hip': ['hip_flexion_r']}
average_all_end_loc = {key_: [] for key_ in params_of_interest_sets.keys()}
bl_val = {}
start_loc = 130
end_loc = 150
step = 2

for end_of_known in range(start_loc, end_loc, step):
    test_data_name = f'downstream_future_motion_{end_of_known}'
    results_true, results_pred, results_bl, columns, is_output_label, _, _ =\
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))

    window_len = is_output_label.shape[0]
    output_start = np.where(is_output_label)[0][0]
    time_step_label = np.zeros((is_output_label.shape[0]))
    time_step_label[output_start:] = range(1, 1 + is_output_label.shape[0] - output_start)

    for param_name, params_of_interest in params_of_interest_sets.items():
        params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

        dset_list = list(results_true.keys())
        average_dset_all, average_dset_bl_all = [], []

        for dset in dset_list[:]:
            if dset in EXCLUDE_FROM_ASB:
                continue

            rmse_time_steps = {param: [] for param in params_of_interest}
            rmse_time_steps_bl = {param: [] for param in params_of_interest}

            true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc]
            pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc]
            bl_ = np.concatenate(list(results_bl[dset].values()))[:, params_of_interest_col_loc]

            output_time_step = np.zeros((bl_.shape[0], len(params_of_interest)))
            for i in range(0, bl_.shape[0] - window_len + 1, window_len):
                output_time_step[i:i + window_len, :] = time_step_label.repeat(len(params_of_interest)).reshape([-1, len(params_of_interest)])

            in_gait_phase = np.all(bl_ != NOT_IN_GAIT_PHASE, axis=1).reshape([-1, len(params_of_interest)])
            in_time_loc = output_time_step == output_time_step.max()

            selected_loc = in_gait_phase & in_time_loc

            if len(true_[selected_loc]) == len(params_of_interest):
                continue

            scores = get_scores(true_[selected_loc].reshape([-1, len(params_of_interest)]),
                                pred_[selected_loc].reshape([-1, len(params_of_interest)]), params_of_interest, None)
            scores_bl = get_scores(true_[selected_loc].reshape([-1, len(params_of_interest)]),
                                   bl_[selected_loc].reshape([-1, len(params_of_interest)]), params_of_interest, None)

            for score, score_bl, param in zip(scores, scores_bl, params_of_interest):
                rmse_time_steps[param].append(np.rad2deg(score['rmse']))
                rmse_time_steps_bl[param].append(np.rad2deg(score_bl['rmse']))

            average_dset = np.mean(np.array(list(rmse_time_steps.values())))
            average_dset_bl = np.mean(np.array(list(rmse_time_steps_bl.values())))
            average_dset_all.append(average_dset)
            average_dset_bl_all.append(average_dset_bl)
            print(f'{dset}, {average_dset:.2f}, {average_dset_bl:.2f}')

            verbose = False
            if verbose:
                plt.figure()
                plt.plot(true_[:, 0][in_gait_phase.ravel()], label='True')
                plt.plot(pred_[:, 0][in_gait_phase.ravel()], label='Predicted')
                plt.plot(bl_[:, 0][in_gait_phase.ravel()], label='Baseline')
                plt.show()

        average_all_end_loc[param_name].append(np.mean(average_dset_all))
        bl_val[param_name] = np.mean(average_dset_bl_all)


class AnyObjectHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [1*height, 1*height], linestyle='--', color='C0')
        l2 = plt.Line2D([x0,y0+width], [0.5*height, 0.5*height], linestyle='--', color='C1')
        l3 = plt.Line2D([x0,y0+width], [0.*height, 0.*height], linestyle='--', color='C2')
        return [l1, l2, l3]


plt.figure(figsize=(3.7, 3.3))
figs = []
for i_param, param_name in enumerate(params_of_interest_sets.keys()):
    fig, = plt.plot(range(end_loc, start_loc, -step), average_all_end_loc[param_name], marker="o",
                    color=f"C{i_param}", markersize=6, linewidth=LINE_WIDTH_THICK)
    plt.plot([end_loc, start_loc], [bl_val[param_name], bl_val[param_name]], '--', color=f"C{i_param}")
    figs.append(fig)

ax = plt.gca()

ax.set_xlabel('Time (ms)', fontdict=FONT_DICT)
ax.set_xticks(np.arange(end_loc, start_loc - 1, -5))
ax.set_xlim(start_loc, end_loc+1)
ax.set_xticklabels(['200', '150', '100', '50', '0'], fontdict=FONT_DICT)

ax.set_ylabel('RMSE (deg)', fontdict=FONT_DICT)
ax.set_yticks([0, 4, 8, 12])
ax.set_yticklabels([0, 4, 8, 12], fontdict=FONT_DICT)
ax.set_ylim(0, 12)
plt.grid(True, linestyle='-', alpha=0.7)

plt.tight_layout(rect=[-0.02, -0.03, 1.02, 0.86])
plt.legend(figs + [object], list(params_of_interest_sets.keys()) + ['Baseline'],
           handler_map={object: AnyObjectHandler()}, frameon=False, bbox_to_anchor=(0.04, 0.97), ncol=2,
           fontsize=FONT_DICT['fontsize'])
format_axis()
save_fig('da_future_motion')
plt.show()






