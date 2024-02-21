import pickle
from fig_utils import get_scores, DSETS_FOR_ASB, format_axis, FONT_DICT, LINE_WIDTH_THICK, save_fig
import numpy as np
import matplotlib.pyplot as plt


params_of_interest_sets = {'Ankle': ['ankle_angle_r', 'ankle_angle_l'],
                           'Knee': ['knee_angle_r', 'knee_angle_l'],
                           'Hip': ['hip_flexion_r', 'hip_flexion_l']}
average_all_end_loc = {key_: [] for key_ in params_of_interest_sets.keys()}
start_loc = 70
end_loc = 90
step = 2

for end_of_known in range(start_loc, end_loc, step):
    test_data_name = f'downstream_future_motion_{end_of_known}'
    results_true, results_pred, columns, is_output_label =\
        pickle.load(open(f"results/{test_data_name}.pkl", "rb"))

    output_start = np.where(is_output_label)[0][0]
    time_step_label = np.zeros((is_output_label.shape[0]))
    time_step_label[output_start:] = range(1, 1 + is_output_label.shape[0] - output_start)

    for param_name, params_of_interest in params_of_interest_sets.items():
        params_of_interest_col_loc = [columns.index(col) for col in params_of_interest]

        dset_list = list(results_true.keys())
        average_dset_all = []

        for dset in dset_list:
            if dset not in DSETS_FOR_ASB:
                continue

            rmse_time_steps = {param: [] for param in params_of_interest}

            print(dset)
            true_ = np.concatenate(list(results_true[dset].values()))[:, params_of_interest_col_loc]
            pred_ = np.concatenate(list(results_pred[dset].values()))[:, params_of_interest_col_loc]

            window_len = 90
            trial_len = true_.shape[0]

            output_time_step = np.zeros((trial_len, len(params_of_interest)))
            for i in range(0, trial_len - window_len + 1, window_len):
                output_time_step[i:i + window_len, :] = time_step_label.repeat(len(params_of_interest)).reshape([-1, len(params_of_interest)])
            output_time_step[-min(window_len, trial_len - window_len - i + 1):, :] =\
                time_step_label[-min(window_len, trial_len - window_len - i + 1):].repeat(len(params_of_interest)).reshape([-1, len(params_of_interest)])

            time_loc = np.where(output_time_step == output_time_step.max())

            scores = get_scores(true_[time_loc].reshape([-1, len(params_of_interest)]),
                                pred_[time_loc].reshape([-1, len(params_of_interest)]), params_of_interest, None)
            for score, param in zip(scores, params_of_interest):
                rmse_time_steps[param].append(np.rad2deg(score['rmse']))

            average_dset = np.mean(np.array(list(rmse_time_steps.values())))
            average_dset_all.append(average_dset)
        average_all_end_loc[param_name].append(np.mean(average_dset_all))


plt.figure(figsize=(3.7, 3))
for param_name in params_of_interest_sets.keys():
    plt.plot(range(end_loc, start_loc, -step), average_all_end_loc[param_name], marker="o",
             markersize=6, linewidth=LINE_WIDTH_THICK)

ax = plt.gca()

ax.set_xlabel('Time (ms)', fontdict=FONT_DICT)
ax.set_xticks(np.arange(end_loc, start_loc - 1, -5))
ax.set_xlim(start_loc, end_loc+1)
ax.set_xticklabels(['200', '150', '100', '50', '0'], fontdict=FONT_DICT)

ax.set_ylabel('RMSE (deg)', fontdict=FONT_DICT)
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_yticklabels([0, 2, 4, 6, 8], fontdict=FONT_DICT)
ax.set_ylim(0, 8.5)
plt.grid(True, linestyle='-', alpha=0.7)

plt.tight_layout(rect=[-0.01, -0.01, 1.01, 0.99])
plt.legend(params_of_interest_sets.keys(), frameon=False, bbox_to_anchor=(0.5, 0.62), fontsize=FONT_DICT['fontsize'])
format_axis()
save_fig('da_future_motion')
plt.show()






