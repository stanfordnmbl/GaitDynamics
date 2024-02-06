from alant.args import parse_opt, set_with_arm_opt
import torch
from alant.alan_consts import DATASETS
from model.alan_model import MotionModel, MotionDataset
import nimblephysics as nimble
from model.alan_model import inverse_convert_addb_state_to_model_input
import os
import numpy as np
import pickle


def loop_all(opt):
    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, repr_dim == 56)

    model = MotionModel(opt, repr_dim)
    dset_list = DATASETS
    results_true, results_pred = {}, {}

    for dset in dset_list:
        test_dataset = MotionDataset(
            data_path=opt.data_path_test,
            train=False,
            joints_3d=opt.joints_3d,
            target_sampling_rate=opt.target_sampling_rate,
            normalizer=model.normalizer,
            opt=opt,
            divide_jittery=False,
            specific_dset=dset,
        )
        wins = test_dataset.get_all_trials()
        if len(wins) == 0:
            continue

        results_true.update({test_dataset.trials[0].dset_name: {}})
        results_pred.update({test_dataset.trials[0].dset_name: {}})

        state_pred_list = [[] for _ in range(skel_num-1)]
        for i_win in range(0, len(wins), opt.batch_size):
            _, state_pred_list_batch = model.eval_loop(opt, wins[i_win:i_win+opt.batch_size], num_of_generation_per_window=skel_num-1)
            for i_skel in range(skel_num-1):
                state_pred_list[i_skel] += state_pred_list_batch[i_skel]

        for i_skel in range(skel_num-1):
            assert len(state_pred_list[i_skel]) == len(wins)

        current_start_win_idx = 0
        for trial_ in test_dataset.trials:
            state_true_trial = model.normalizer.unnormalize(trial_.converted_pose.unsqueeze(0))
            state_true_trial = inverse_convert_addb_state_to_model_input(
                state_true_trial, model_states_column_names=opt.model_states_column_names,
                joints_3d=opt.joints_3d, osim_dof_columns=opt.osim_dof_columns).squeeze().numpy()
            state_pred_trial = np.zeros(state_true_trial.shape)

            for i_sample in range(0, state_true_trial.shape[0], test_dataset.window_len):           #  - test_dataset.window_len + 1
                for i_skel in range(skel_num-1):
                    state_pred_trial[i_sample:i_sample+test_dataset.window_len] +=\
                        state_pred_list[i_skel][current_start_win_idx].numpy()[-min(test_dataset.window_len, state_true_trial.shape[0]-i_sample):]
                current_start_win_idx += 1
            state_pred_trial /= (skel_num - 1)

            results_true[trial_.dset_name].update({trial_.sub_and_trial_name: state_true_trial})
            results_pred[trial_.dset_name].update({trial_.sub_and_trial_name: state_pred_trial})

    pickle.dump([results_true, results_pred, opt.osim_dof_columns], open(f"figures/results/results_true_pred_{test_data_name}.pkl", "wb"))


if __name__ == "__main__":
    skel_num = 2
    opt = parse_opt()
    test_data_name = '0124'

    # opt.checkpoint = os.path.dirname(os.path.realpath(__file__)) + f"/trained_models/train-{'1000'}.pt"
    opt.checkpoint = opt.data_path_parent + f"../code/runs/train/{'norm_all3'}/weights/train-{'3000'}.pt"

    loop_all(opt)















