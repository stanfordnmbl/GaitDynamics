from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from fig_utils import show_skeletons, set_up_gui
import matplotlib.pyplot as plt
from model.utils import inverse_convert_addb_state_to_model_input


class MotionDatasetManipulated(MotionDataset):
    def customized_param_manipulation(self, trial_df):
        trial_df['lumbar_bending'] = trial_df['lumbar_bending'] * 3
        return trial_df


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)

    model = MotionModel(opt, repr_dim)

    max_trial_num = 1
    trial_start_num = 26
    test_dataset = MotionDataset(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num,
    )
    windows_original = test_dataset.get_all_wins_within_gait_cycle()

    test_dataset_mani = MotionDatasetManipulated(
        data_path=b3d_path,
        train=False,
        normalizer=model.normalizer,
        opt=opt,
        divide_jittery=False,
        max_trial_num=max_trial_num,
        trial_start_num=trial_start_num
    )
    windows_manipulated = test_dataset_mani.get_all_wins_within_gait_cycle()

    assert len(windows_original) == len(windows_manipulated)

    manipulated_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'lumbar' in col]

    state_pred_list = [[] for _ in range(skel_num-1)]
    for i_win in range(0, len(windows_manipulated), opt.batch_size_inference):

        state_manipulated = [win[0] for win in windows_manipulated[i_win:i_win+opt.batch_size_inference]]
        state_manipulated = torch.stack(state_manipulated)

        masks = torch.zeros_like(state_manipulated)      # 0 for masking, 1 for unmasking
        masks[:, :, manipulated_col_loc] = 1

        value_diff_weight = torch.ones([len(opt.model_states_column_names)])
        value_diff_thd = torch.zeros([len(opt.model_states_column_names)])
        value_diff_thd[:] = 0.1         # large value for no constraint
        value_diff_thd[manipulated_col_loc] = 0

        state_pred_list_batch = model.eval_loop(opt, state_manipulated, masks, value_diff_thd, value_diff_weight,
                                                num_of_generation_per_window=skel_num - 1)
        # pos_vec = np.array([test_dataset.trials[windows_original[i_win_vec][2]].pos_vec_for_pos_alignment
        #                     for i_win_vec in range(len(windows_original[i_win:i_win+opt.batch_size_inference]))])
        state_pred_list_batch = inverse_convert_addb_state_to_model_input(
            state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, [0, 0, 0])

        for i_skel in range(skel_num-1):
            state_pred_list[i_skel] += state_pred_list_batch[i_skel]

    for i_skel in range(skel_num-1):
        assert len(state_pred_list[i_skel]) == len(windows_manipulated)

    names = ['baseline', '3x trunk sway']
    gui, skels = set_up_gui(opt, names)
    for i_win, (win, state_pred), in enumerate(zip(windows_original, state_pred_list[0])): # !!! only the first one is used, in the future make use all
        # trial = test_dataset.trials[win[2]]
        true_val = inverse_convert_addb_state_to_model_input(
            model.normalizer.unnormalize(win[0].unsqueeze(0)), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, [0, 0, 0]).squeeze().numpy()

        name_states_dict = {names[0]: true_val, names[1]: state_pred.detach().numpy()}
        show_skeletons(opt, name_states_dict, gui, skels)

        # for param in ['calcn_l_force_vz', 'calcn_r_force_vz']:
        # # for param in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz', 'calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']:
        #     # , 'subtalar_angle_r', 'mtp_angle_r', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
        #     #           'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l'
        #     col_loc = opt.osim_dof_columns.index(param)
        #     plt.figure()
        #     plt.plot(true_val[:, col_loc])
        #     plt.plot(state_pred.detach().numpy()[:, col_loc])
        #     plt.title(param)
        # plt.show()


li, camargo, carter, falisse, moore, tan2021, tan2022 = 'li', 'camargo', 'carter', 'falisse', 'moore', 'tan2021', 'tan2022'
uhlrich, santos, vanderzee, wang = 'uhlrich', 'santos', 'vanderzee', 'wang'
b3d_path = f'/mnt/d/Local/Data/MotionPriorData/{wang}_dset/'

""" To use this code,
1. in load_addb, manipulate channels
2. in this script, change manipulated_col_loc accordingly
"""

if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    opt.guide_x_start_the_beginning_step = 1000

    opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/train-{'100'}.pt"

    loop_all(opt)










