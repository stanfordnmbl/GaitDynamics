import pickle
import pandas as pd
import numpy as np
from args import parse_opt
from fig_utils import format_axis
from example_usage.gait_dynamics import convertDfToGRFMot
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter


def convertDfToMotionMot(df, out_path, dt, dof_to_include, max_time=None):
    numFrames = df.shape[0]
    for key in df.keys():
        if key == 'TimeStamp':
            continue

    out_file = open(out_path, 'w')
    out_file.write('Coordinates\n')
    out_file.write('version=1\n')
    out_file.write(f'nRows={numFrames}\n')
    out_file.write(f'nColumns={len(dof_to_include)+1}\n')
    out_file.write('inDegrees=no\n')
    out_file.write('\n')
    out_file.write('If the header above contains a line with \'inDegrees\', this indicates whether rotational values are in degrees (yes) or radians (no).\n')
    out_file.write('\n')
    out_file.write('\n')
    out_file.write('endheader\n')
    out_file.write('time')

    for dof in dof_to_include:
        out_file.write('\t' + dof)

    out_file.write('\n')
    for i in range(numFrames):
        out_file.write(str(round(dt * i, 5)))
        for dof in dof_to_include:
            out_file.write('\t' + str(df[dof][i]))
        out_file.write('\n')
    out_file.close()


def export_grf():
    trial_to_select_of_each_dset = {
        'Camargo2021_Formatted_No_Arm': 81,
        'Moore2015_Formatted_No_Arm': 3,
        # 'vanderZee2022_Formatted_No_Arm': 1,
        'Wang2023_Formatted_No_Arm': 13,
    }
    results_ = pickle.load(open(f"results/full/tf_for_video.pkl", "rb"))

    max_time = 300      # ms
    pause_time = 40     # ms
    true_df_list, pred_df_list = [], []
    current_ms = 0
    for dset, (true_sub, pred_sub, column_names, height_m) in results_.items():
        if dset not in trial_to_select_of_each_dset.keys():
            trial_to_select_of_each_dset[dset] = 0
        # if dset in ['Carter2023_Formatted_No_Arm']:
        #     continue
        true_sub = true_sub[trial_to_select_of_each_dset[dset]][:max_time]
        pred_sub = pred_sub[trial_to_select_of_each_dset[dset]][:max_time]

        current_ms += (min(true_sub.shape[0], max_time)+pause_time)
        print('{}:{}'.format(int(np.floor(current_ms/100)), str(round((current_ms%100)/100*24)).zfill(2)))
        print('Data from ' + dset.split('20')[0] + ' et al., 20' + dset.split('20')[1][:2])

        force_magnitude_index = [i for i, col in enumerate(column_names) if 'force_v' in col]
        true_sub[:, force_magnitude_index] = true_sub[:, force_magnitude_index] * weight_kg
        pred_sub[:, force_magnitude_index] = pred_sub[:, force_magnitude_index] * weight_kg

        for i_plate in range(2):
            true_v_force_sum = np.abs(true_sub[:, force_magnitude_index[i_plate*3+1]])
            no_force_index = true_v_force_sum < 1e-3
            for j in range(3):
                pred_sub[no_force_index, force_magnitude_index[i_plate*3+j]] = 0

        pelvis_t_index = [i for i, col in enumerate(column_names) if col in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]
        r_cop_index = [i for i, col in enumerate(column_names) if 'calcn_r_force_normed_cop' in col]
        l_cop_index = [i for i, col in enumerate(column_names) if 'calcn_l_force_normed_cop' in col]

        true_sub[:, pelvis_t_index] = true_sub[:, pelvis_t_index] / height_m * 1.67
        true_sub[:, r_cop_index] = true_sub[:, r_cop_index] / height_m * 1.67
        true_sub[:, l_cop_index] = true_sub[:, l_cop_index] / height_m * 1.67
        pred_sub[:, r_cop_index] = pred_sub[:, r_cop_index] / height_m * 1.67
        pred_sub[:, l_cop_index] = pred_sub[:, l_cop_index] / height_m * 1.67

        true_df = pd.DataFrame(true_sub, columns=column_names)
        true_df = pd.concat([true_df, *[true_df.tail(1)]*pause_time])
        true_df_list.append(true_df)

        pred_df = pd.DataFrame(pred_sub, columns=column_names)
        pred_df = pd.concat([pred_df, *[pred_df.tail(1)]*pause_time])
        pred_df_list.append(pred_df)
    true_df = pd.concat(true_df_list, ignore_index=True)
    pred_df = pd.concat(pred_df_list, ignore_index=True)
    convertDfToMotionMot(true_df, f'exports/mot_grf_q.mot', 0.01, column_names[:23])
    convertDfToGRFMot(true_df, f'exports/mot_grf_f.mot', 0.01, max_time=None)
    convertDfToGRFMot(pred_df, f'exports/mot_grf_f_pred.mot', 0.01, max_time=None)

    true_df.iloc[:, r_cop_index] -= true_df.iloc[:, pelvis_t_index].values
    true_df.iloc[:, l_cop_index] -= true_df.iloc[:, pelvis_t_index].values
    pred_df.iloc[:, r_cop_index] -= true_df.iloc[:, pelvis_t_index].values
    pred_df.iloc[:, l_cop_index] -= true_df.iloc[:, pelvis_t_index].values

    true_df.iloc[:, pelvis_t_index] = 0
    convertDfToMotionMot(true_df, f'exports/mot_grf_no_trans_q.mot', 0.01, column_names[:23])
    convertDfToGRFMot(true_df, f'exports/mot_grf_no_trans_f.mot', 0.01, max_time=None)
    convertDfToGRFMot(pred_df, f'exports/mot_grf_no_trans_f_pred.mot', 0.01, max_time=None)


def export_walk_ts():
    _a, _b, bl_true_ori, bl_pred_ori, columns, _c, _d, _e, = \
        pickle.load(open(f"results/da_guided_baseline.pkl", "rb"))
    _, _, ts_true_ori, ts_pred_ori, columns, _, _, _, = \
        pickle.load(open(f"results/da_guided_trunk_sway.pkl", "rb"))

    trial_index = 8
    to_export = {
        'bl_exp': (bl_true_ori['_1'][trial_index],),
        'ts_exp': (ts_true_ori['_1'][trial_index],),
        'ts_syn': (bl_pred_ori['_3'][trial_index],)
    }

    for trial_name, data_ in to_export.items():
        force_magnitude_index = [i for i, col in enumerate(columns) if 'force_v' in col]
        data_[0][:, force_magnitude_index] = data_[0][:, force_magnitude_index] * weight_kg
        generation_1ms_df = pd.DataFrame(data_[0], columns=columns)
        convertDfToMotionMot(generation_1ms_df, f'exports/mot_walk_{trial_name}_q.mot', 0.01, columns[:23])
        convertDfToGRFMot(generation_1ms_df, f'exports/mot_walk_{trial_name}_f.mot', 0.01, max_time=None)
        data_[0][:, force_magnitude_index] = data_[0][:, force_magnitude_index] / weight_kg


def export_running_speeds(angle_to_plot='knee_angle_r'):
    win_exp_list, win_syn_list = pickle.load(open(f"results/da_run_faster_win.pkl", "rb"))

    to_export = {
        'exp_400': [win_exp_list[6][1],],
        'syn_300': [win_syn_list[6][0],],
        'syn_500': [win_syn_list[6][-1],],
    }

    for trial_name, data_ in to_export.items():
        force_magnitude_index = [i for i, col in enumerate(opt.osim_dof_columns) if 'force_v' in col]
        data_[0][:, force_magnitude_index] = data_[0][:, force_magnitude_index] * weight_kg
        data_[0][:, 5] = data_[0][:, 5]
        generation_1ms_df = pd.DataFrame(data_[0], columns=opt.osim_dof_columns)
        convertDfToMotionMot(generation_1ms_df, f'exports/mot_run_{trial_name}_q.mot', 0.01, opt.osim_dof_columns[:23])
        convertDfToGRFMot(generation_1ms_df, f'exports/mot_run_{trial_name}_f.mot', 0.01, max_time=None)
        data_[0][:, force_magnitude_index] = data_[0][:, force_magnitude_index] / weight_kg

        to_export[trial_name].append(generation_1ms_df[angle_to_plot] * 180 / np.pi)

    matplotlib.rc('font', size=14)
    colors = [np.array(x) / 255 for x in [[110, 170, 220], [30, 90, 140], [177, 124, 90]]]
    fig, ax = plt.subplots(figsize=(6, 6))
    line2 = ax.plot(to_export['syn_500'][1][:1], '--', linewidth=2, label=f'5.0 m/s - synthetic', color=colors[0])[0]
    line1 = ax.plot(to_export['exp_400'][1][:1], linewidth=2, label=f'4.0 m/s - experimental', color=colors[2])[0]
    line0 = ax.plot(to_export['syn_300'][1][:1], '--', linewidth=2, label=f'3.0 m/s - synthetic', color=colors[1])[0]
    ax.set(xlim=[0, 150], ylim=[0, 170],
           xticks=range(0, 151, 30), yticks=range(0, 161, 40),
           xlabel='Time (s)', ylabel='Right Knee Flexion (deg)')
    ax.legend()
    ax.grid(True, linewidth=1, alpha=0.5)
    format_axis(ax)
    plt.tight_layout()

    def update(frame):
        line2.set_data(range(frame), to_export['syn_500'][1][:frame])
        line1.set_data(range(frame), to_export['exp_400'][1][:frame])
        line0.set_data(range(frame), to_export['syn_300'][1][:frame])
        return (line0, line1, line2)

    update(150)
    plt.show()

    ani = FuncAnimation(fig=fig, func=update, frames=150, interval=50)
    ani.save("exports/da3.gif", dpi=300, writer=PillowWriter(fps=20))
    plt.show()


if __name__ == "__main__":
    weight_kg = 70
    opt = parse_opt()
    export_grf()
    export_walk_ts()
    export_running_speeds()



