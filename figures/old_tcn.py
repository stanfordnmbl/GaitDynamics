

# def loop_mask_conditions_tcn():
#     n_split = 10
#     results_dict = {}
#     for mask_key, unmask_col_loc in cols_to_unmask.items():
#         print(mask_key)
#         true_sub_dict, pred_sub_dict, pred_std_sub_dict = {}, {}, {}
#         for dset in test_dataset_dict.keys():
#             print(dset)
#             if dset in dset_to_split:
#                 windows_splits, trials_splits, dset_names, s_list_splits, e_list_splits = [], [], [], [], []
#                 for i_split in range(n_split+1):
#                     start_trial = i_split * (len(test_dataset_dict[dset].trials) // n_split)
#                     end_trial = min(len(test_dataset_dict[dset].trials), (i_split + 1) * (len(test_dataset_dict[dset].trials) // n_split))
#                     if end_trial == start_trial:
#                         continue
#                     windows = test_dataset_dict[dset].get_wins_before_available_starts(unmask_col_loc, start_trial, end_trial)
#                     if len(windows) == 0:
#                         continue
#                     windows_splits.append(windows)
#                     trials_splits.append(test_dataset_dict[dset].trials)
#                     dset_names.append(dset + f'_{i_split}')
#
#             else:
#                 dset_names = [dset]
#                 windows = test_dataset_dict[dset].get_wins_before_available_starts(unmask_col_loc)
#                 if len(windows) == 0:
#                     continue
#                 windows_splits = [windows]
#                 trials_splits = [test_dataset_dict[dset].trials]
#
#             # to speed up
#             if speed_up and len(windows_splits) > 0 and len(windows_splits[-1]) > 2000:
#                 windows_splits = [windows_splits[-1][:2000]]
#                 trials_splits = [trials_splits[-1]]
#                 dset_names = [dset_names[-1]]
#
#             for trials, windows, dset_name in zip(trials_splits, windows_splits, dset_names):
#                 true_sub, pred_sub, pred_std_sub = loop_all_tcn(opt, trials, windows)
#                 true_sub_dict[dset_name] = true_sub[:, params_of_interest_col_loc]
#                 pred_sub_dict[dset_name] = pred_sub[:, params_of_interest_col_loc]
#                 pred_std_sub_dict[dset_name] = pred_std_sub[:, params_of_interest_col_loc]
#         results_dict.update({mask_key: [true_sub_dict, pred_sub_dict, pred_std_sub_dict, params_of_interest]})
#     pickle.dump(results_dict, open(f"figures/results/addb_marker_based_{model_key}.pkl", "wb"))
#
#
# def loop_all_tcn(opt, trials, windows):
#     state_pred_list = [[] for _ in range(skel_num-1)]
#     for i_win in range(0, len(windows), opt.batch_size_inference):
#         state_true = torch.stack([win.pose for win in windows[i_win:i_win+opt.batch_size_inference]])
#         masks = torch.stack([win.mask for win in windows[i_win:i_win+opt.batch_size_inference]])
#         cond = torch.stack([win.cond for win in windows[i_win:i_win+opt.batch_size_inference]])
#         height_m_tensor = torch.tensor([win.height_m for win in windows[i_win:i_win+opt.batch_size_inference]])
#
#         state_pred_list_batch = model.eval_loop(opt, state_true, masks, cond=cond, num_of_generation_per_window=skel_num-1)
#         pos_vec = np.array([trials[windows[i_win_vec].trial_id].pos_vec_for_pos_alignment
#                             for i_win_vec in range(len(windows[i_win:i_win+opt.batch_size_inference]))])
#         state_pred_list_batch = inverse_convert_addb_state_to_model_input(
#             state_pred_list_batch, opt.model_states_column_names, opt.joints_3d, opt.osim_dof_columns, pos_vec, height_m_tensor)
#         for i_skel in range(skel_num-1):
#             state_pred_list[i_skel] += state_pred_list_batch[i_skel]
#
#     for i_skel in range(skel_num-1):
#         assert len(state_pred_list[i_skel]) == len(windows)
#     state_pred_list_averaged, state_pred_list_std = [], []
#     for i_win in range(len(state_pred_list[0])):
#         win_skels = [state_pred_list[i_skel][i_win] for i_skel in range(skel_num-1)]
#         averaged = torch.mean(torch.stack(win_skels), dim=0)
#         std = torch.std(torch.stack(win_skels), dim=0)
#         state_pred_list_averaged.append(averaged)
#         state_pred_list_std.append(std)
#
#     results_true, results_pred, results_pred_std, results_s, results_e = {}, {}, {}, {}, {}
#     for i_win, (win, state_pred_mean, state_pred_std) in enumerate(zip(windows, state_pred_list_averaged, state_pred_list_std)):
#         trial = trials[win.trial_id]
#         if trial.sub_and_trial_name not in results_true.keys():
#             results_true.update({trial.sub_and_trial_name: []})
#             results_pred.update({trial.sub_and_trial_name: []})
#             results_pred_std.update({trial.sub_and_trial_name: []})
#             results_s.update({trial.sub_and_trial_name: []})
#             results_e.update({trial.sub_and_trial_name: []})
#
#         true_val = inverse_convert_addb_state_to_model_input(
#             model.normalizer.unnormalize(win.pose.unsqueeze(0)), opt.model_states_column_names,
#             opt.joints_3d, opt.osim_dof_columns, trial.pos_vec_for_pos_alignment, torch.tensor(win.height_m)).squeeze().numpy()
#         mask = win.mask.squeeze().numpy()
#         true_val = true_val * np.bool_(mask.sum(axis=1)).repeat(35).reshape((150, -1))
#         state_pred_mean = state_pred_mean * np.bool_(mask.sum(axis=1)).repeat(35).reshape((150, -1))
#
#         results_true[trial.sub_and_trial_name].append(true_val[-1:])
#         results_pred[trial.sub_and_trial_name].append(state_pred_mean.numpy()[-1:])
#
#     true_all, pred_all, pred_std_all = [], [], []
#     for sub_and_trial in results_true.keys():
#         results_true[sub_and_trial] = np.concatenate(results_true[sub_and_trial], axis=0)
#         results_pred[sub_and_trial] = np.concatenate(results_pred[sub_and_trial], axis=0)
#         results_pred_std[sub_and_trial] = np.zeros(results_pred[sub_and_trial].shape)
#
#         for trial in trials:
#             if trial.sub_and_trial_name == sub_and_trial:
#                 break
#         true_all.append(results_true[sub_and_trial])
#         pred_all.append(results_pred[sub_and_trial])
#         pred_std_all.append(results_pred_std[sub_and_trial])
#     true_all = np.concatenate(true_all, axis=0)
#     pred_all = np.concatenate(pred_all, axis=0)
#     pred_std_all = np.concatenate(pred_std_all, axis=0)
#     return true_all, pred_all, pred_std_all


def get_wins_before_available_starts(self, col_loc_to_unmask, start_trial=0, end_trial=None):
    """ Exclusive for TCN """
    if end_trial is None:
        end_trial = len(self.trials)
    windows = []
    for i_trial in range(start_trial, end_trial):
        trial_ = self.trials[i_trial]
        pose_to_slice = torch.concatenate([torch.zeros([149, len(self.opt.model_states_column_names)]), trial_.converted_pose], dim=0)
        for i_sample in range(trial_.converted_pose.shape[0]):
            data_ = pose_to_slice[i_sample:i_sample+self.opt.window_len, ...]            # !!! check if GRF is valid
            mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
            mask[:, col_loc_to_unmask] = 1
            windows.append(WindowData(data_, trial_.model_offsets, i_trial, None, mask,
                                      trial_.height_m, trial_.weight_kg, trial_.cond))
    return windows


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvolutionalArchitecture(nn.Module):
    def __init__(self, repr_dim, opt, kernel_size=4, dropout=0.3):
        super().__init__()
        self.ema = EMA(0.99)
        self.opt = opt
        self.input_dim = len(opt.kinematic_diffusion_col_loc)
        self.output_dim = repr_dim - self.input_dim
        self.input_col_loc = opt.kinematic_diffusion_col_loc
        self.output_col_loc = [i for i in range(repr_dim) if i not in self.input_col_loc]
        self.device = 'cuda'
        self.master_model = ClassForAdaptingBaselineModelInTheOriginalFramework()
        self.ema = ClassForAdaptingBaselineModelInTheOriginalFramework()
        self.model = ClassForAdaptingBaselineModelInTheOriginalFramework()

        layers = []
        num_levels = 5
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_dim if i == 0 else 500
            out_channels = 500
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.conv_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(500, self.output_dim)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def end_to_end_prediction(self, x):
        input = x[0][:, :, self.input_col_loc]
        x = input.permute(0, 2, 1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        output_pred = self.output_layer(x)
        return output_pred

    def predict_samples(self, x, constraint):
        x[0] = x[0] * constraint['mask']
        output_pred = self.end_to_end_prediction(x)
        x[0][:, :, self.output_col_loc] = output_pred
        return x[0]

    def forward(self, x, cond, t_override):
        output_true = x[0][:, :, self.output_col_loc]
        output_pred = self.end_to_end_prediction(x)
        loss_simple = torch.zeros([x[0].shape[0], 1, x[0].shape[2]]).to(x[0].device)
        loss_simple[:, :, self.output_col_loc] = F.mse_loss(output_pred[:, -1:], output_true[:, -1:], reduction="none")

        losses = [
            1. * loss_simple.mean(),
            torch.tensor(0.).to(loss_simple.device),
            torch.tensor(0.).to(loss_simple.device),
            torch.tensor(0.).to(loss_simple.device),
            torch.tensor(0).to(loss_simple.device)]
        return sum(losses), losses + [loss_simple]
