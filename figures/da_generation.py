from args import parse_opt, set_with_arm_opt
import torch
import numpy as np
from model.model import MotionModel
import os
from fig_utils import show_skeletons


def generate_windows(name_, value_, win_num_per_condition=1):
    windows_list = []
    if name_ == 'speed':
        for speed in value_:
            windows = [torch.zeros([opt.window_len, len(opt.model_states_column_names)]) for _ in range(win_num_per_condition)]
            for i_win in range(win_num_per_condition):
                windows[i_win][:, opt.model_states_column_names.index('pelvis_tx')] = speed
                windows[i_win][:, opt.model_states_column_names.index('pelvis_ty')] = 0
            windows_list.append((name_, speed, windows))
    elif name_ == 'trunk_sway':
        pass

    return windows_list


if __name__ == "__main__":
    # skel_num = 2
    opt = parse_opt()

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'5000'}.pt"
    # opt.checkpoint = opt.data_path_parent + f"/../code/runs/train/{'test_data_loader3'}/weights/train-{'3000'}.pt"

    repr_dim = len(opt.model_states_column_names)
    # set_with_arm_opt(opt, False)
    model = MotionModel(opt, repr_dim)

    condition_on_speed = [round(speed, 2) for speed in np.arange(0.6, 2, 0.4)]
    # condition_on_trunk_sway = []
    windows_list = generate_windows('speed', condition_on_speed)

    state_pred_list = []
    for name_, value_, windows in windows_list:
        state_true = torch.stack(windows)
        state_true = model.normalizer.normalize(state_true.reshape(-1, state_true.shape[2])).reshape(-1, opt.window_len, state_true.shape[2])

        # For GRF estimation
        masks = torch.zeros_like(state_true)      # 0 for masking, 1 for unmasking
        unmask_col_loc = [opt.model_states_column_names.index(param_) for param_ in ['pelvis_tx', 'pelvis_ty']]
        masks[:, :, unmask_col_loc] = 1

        state_pred_list_batch = model.eval_loop(opt, state_true, masks)
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0])
        state_pred_list.extend(state_pred_list_batch)
    name_states_dict = {name_: state_pred[0] for name_, state_pred in zip(condition_on_speed, state_pred_list)}
    show_skeletons(opt, name_states_dict)

























