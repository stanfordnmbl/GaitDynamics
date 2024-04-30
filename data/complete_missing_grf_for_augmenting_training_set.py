from args import parse_opt, set_with_arm_opt
import torch
from tqdm import tqdm
import more_itertools as mit
import os
from consts import DATASETS_NO_ARM, NOT_IN_GAIT_PHASE
from model.model import MotionModel
from data.addb_dataset import MotionDataset
import numpy as np
from data.osim_fk import get_model_offsets, get_knee_rotation_coefficients, forward_kinematics
import matplotlib.pyplot as plt
import nimblephysics as nimble
from model.utils import extract, make_beta_schedule, linear_resample_data, update_d_dd, fix_seed, nan_helper, \
    from_foot_loc_to_foot_vel, moving_average_filtering, convert_addb_state_to_model_input, identity, maybe_wrap, \
    inverse_convert_addb_state_to_model_input, align_moving_direction, inverse_align_moving_direction


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)

    model = MotionModel(opt, repr_dim)
    dset_list = DATASETS_NO_ARM
    results_true, results_pred, height_m_all, weights_kg_all = {}, {}, {}, {}
    is_output_label_array = torch.zeros([150, 35])

    for dset in dset_list:
        print(dset)
        dataset_to_process = MotionDataset(
            data_path=opt.data_path_train,
            train=False,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            specific_dset=dset,
            max_trial_num=2,
            # trial_start_num=6,
        )

        minimal_has_grf_num = 0.5 * opt.window_len
        check_sample_len = int(opt.window_len / 5)

        subject_paths = []
        if os.path.isdir(dataset_to_process.data_path):
            for root, dirs, files in os.walk(dataset_to_process.data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(".b3d"):
                        subject_paths.append(file_path)

        # trial_protos = {}
        # i_loaded_trial = 0
        subject_loaded = []
        for i_sub, subject_path in enumerate(subject_paths):
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)
            subject_name = subject_path.split('/')[-1].split('.')[0]
            assert subject_name not in subject_loaded
            subject_loaded.append(subject_name)
            trial_protos = subject.getHeaderProto().getTrials()
            trial_forceplate_dict = {}

            # if dataset_to_process.trials[i_loaded_trial].sub_and_trial_name.split('_')[0] == subject_name:

            windows_to_complete = []
            windows_probably_missing = []
            windows_start_time_sample = []
            windows_trial_id = []
            for trial_loaded in dataset_to_process.trials:
                if not trial_loaded.sub_and_trial_name.split('__')[0] == subject_name:
                    continue

                poses = trial_loaded.converted_pose
                probably_missing = trial_loaded.probably_missing
                for i in range(0, poses.shape[0] - opt.window_len, check_sample_len):

                    # debug
                    print(sum(probably_missing[i:i+opt.window_len]))
                    print(sum(probably_missing[i:i+opt.window_len]) <= minimal_has_grf_num)
                    if sum(probably_missing[i:i+opt.window_len]) and sum(probably_missing[i:i+opt.window_len]) <= minimal_has_grf_num:
                        windows_to_complete.append(trial_loaded.converted_pose[i:i+opt.window_len, ...])
                        windows_probably_missing.append(probably_missing[i:i+opt.window_len])
                        windows_start_time_sample.append(i)
                        windows_trial_id.append(trial_loaded.trial_id)

                trial_forceplate_dict[trial_loaded.trial_id] = trial_protos[trial_loaded.trial_id].getForcePlates()

            if len(windows_to_complete) == 0:
                continue

            # state_pred_list = [[] for _ in range(skel_num-1)]
            for i_win_chunk in range(0, len(windows_to_complete), opt.batch_size_inference):

                state_true = windows_to_complete[i_win_chunk:i_win_chunk+opt.batch_size_inference]
                state_true = torch.stack(state_true)

                # For GRF estimation
                masks = torch.ones_like(state_true)      # 0 for masking, 1 for unmasking

                for i_win in range(i_win_chunk, min(i_win_chunk+opt.batch_size_inference, len(windows_to_complete))):
                    win_grf_mask = masks[i_win, :, opt.grf_moment_cop_model_col_loc]
                    win_grf_mask[windows_probably_missing[i_win]] = 0
                    masks[i_win, :, opt.grf_moment_cop_model_col_loc] = win_grf_mask

                state_pred_list_batch = model.eval_loop(opt, state_true, masks, num_of_generation_per_window=skel_num-1)
                state_pred_list_batch = inverse_convert_addb_state_to_model_input(
                    state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0])

                pos_vec = test_dataset.trials[windows[i_win][2]].pos_vec_for_pos_alignment
                state_pred_list_batch = inverse_convert_addb_state_to_model_input(
                    state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, pos_vec)

                # for i_skel in range(skel_num-1):
                #     state_pred_list[i_skel] += state_pred_list_batch[i_skel]

                averaged = torch.zeros_like(state_pred_list_batch[0])
                for i_skel in range(skel_num-1):
                    averaged += state_pred_list_batch[i_skel]
                averaged /= (skel_num-1)
                averaged_grf = averaged[:, :, opt.grf_model_col_loc].cpu().numpy()
                averaged_moment = averaged[:, :, opt.grf_moment_cop_model_col_loc].cpu().numpy()
                cop_x = averaged_moment[:, :, 2] / averaged_grf[:, :, 1]
                cop_z = - averaged_moment[:, :, 0] / averaged_grf[:, :, 1]
                # averaged_cop = np.concatenate([cop_x, cop_z], axis=1)

                for i_win in range(i_win_chunk, min(i_win_chunk+opt.batch_size_inference, len(windows_to_complete))):
                    trial_id = windows_trial_id[i_win]
                    start_sample = windows_start_time_sample[i_win]
                    for i_sample in range(start_sample, start_sample+opt.window_len):
                        if windows_probably_missing[i_win][i_sample-start_sample]:
                            idx_0 = i_win%opt.batch_size_inference
                            idx_1 = i_sample-start_sample

                            # [debug] from here, wait until no force issue fixed
                            trial_forceplate_dict[trial_id][i_sample].forces = averaged_grf[idx_0, idx_1, :]
                            trial_forceplate_dict[trial_id][i_sample].centersOfPressure = np.array(
                                [cop_x[idx_0, idx_1], trial_forceplate_dict[trial_id][i_sample].centersOfPressure[1], cop_z[idx_0, idx_1]])

            for trial_loaded in dataset_to_process.trials:
                trial_id = trial_loaded.trial_id
                trial_protos[trial_id].setForcePlates(trial_forceplate_dict[trial_id])

                # raw_force_plates = trial_protos[trial].getForcePlates()
                # for f in range(len(raw_force_plates)):
                #     raw_force_plates[f].centersOfPressure = cops[f]
                #     raw_force_plates[f].forces = forces[f]
                # trial_protos[0].setForcePlates()

            subject_path_new = subject_path.replace('train_cleaned', 'train_cleaned_completed')
            nimble.biomechanics.writeB3D(subject_path_new, subject)


if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'3000'}.pt"
    # opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'test_data_loader3'}/weights/train-{'3000'}.pt"

    loop_all(opt)
