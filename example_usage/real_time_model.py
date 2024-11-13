import time

from gait_dynamics import *


class TransformerHipKnee(nn.Module):
    def __init__(self, repr_dim, opt, nlayers=6):
        super(TransformerHipKnee, self).__init__()
        self.input_dim = len(opt.kinematic_diffusion_col_loc)
        self.output_dim = repr_dim - self.input_dim
        embedding_dim = 192
        self.input_to_embedding = nn.Linear(self.input_dim, embedding_dim)
        self.encoder_layers = nn.Sequential(*[EncoderLayer(embedding_dim) for _ in range(nlayers)])
        self.embedding_to_output = nn.Linear(embedding_dim, self.output_dim)
        self.opt = opt
        self.input_col_loc = opt.kinematic_diffusion_col_loc
        self.output_col_loc = [i for i in range(repr_dim) if i not in self.input_col_loc]

    def loss_fun(self, output_pred, output_true):
        return F.mse_loss(output_pred, output_true, reduction='none')

    def end_to_end_prediction(self, x):
        input = x[0][:, :, self.input_col_loc]
        sequence = self.input_to_embedding(input)
        sequence = self.encoder_layers(sequence)
        output_pred = self.embedding_to_output(sequence)
        return output_pred

    def predict_samples(self, x, constraint):
        x[0] = x[0] * constraint['mask']
        output_pred = self.end_to_end_prediction(x)
        x[0][:, :, self.output_col_loc] = output_pred
        return x[0]

    def __str__(self):
        return 'tf'


def inverse_norm_cops_and_split_force(states, opt, grf_thd_to_zero_cop=20):
    forces = states[-1:, opt.grf_osim_col_loc]
    normed_cops = states[-1:, opt.cop_osim_col_loc]

    force_cop_two_feet = []
    for i_plate in range(2):
        force_v = forces[:, 3*i_plate:3*(i_plate+1)]
        force_v[force_v == 0] = 1e-6
        vector = normed_cops[:, 3 * i_plate:3 * (i_plate + 1)] / force_v[:, 1:2] * opt.height_m
        cops = np.nan_to_num(vector, posinf=0, neginf=0)
        # vector.clip(min=-0.4, max=0.4, out=vector)      # CoP should be within 0.4 m from the foot
        # cops = vector + foot_loc[:, 3*i_plate:3*(i_plate+1)]

        if grf_thd_to_zero_cop and force_v[0, 1] * opt.weight_kg < grf_thd_to_zero_cop:
            cops[:] = 0

        cops = torch.from_numpy(cops).to(states.dtype)
        states[:, opt.cop_osim_col_loc[3*i_plate:3*(i_plate+1)]] = cops
        force_cop_two_feet.append(torch.cat((forces[:, 3*i_plate:3*(i_plate+1)] * opt.weight_kg, cops), 1))
    return force_cop_two_feet


class BaselineModelCpu(BaselineModel):
    def __init__(
            self,
            opt,
            model_architecture_class,
            EMA=True,
    ):
        self.device = 'cpu'
        self.opt = opt
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.repr_dim = len(opt.model_states_column_names)
        self.horizon = horizon = opt.window_len
        self.accelerator.wait_for_everyone()

        self.model = model_architecture_class(self.repr_dim, opt)
        self.diffusion = DiffusionShellForAdaptingTheOriginalFramework(self.model)
        self.diffusion = self.accelerator.prepare(self.diffusion)

        print("Model has {} parameters".format(sum(y.numel() for y in self.model.parameters())))

        checkpoint = None
        if opt.checkpoint_bl != "":
            checkpoint = torch.load(
                opt.checkpoint_bl, map_location=self.device
            )
            self.normalizer = checkpoint["normalizer"]

        if opt.checkpoint_bl != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    1,
                )
            )

    def eval_loop(self, opt, state_true, masks, value_diff_thd=None, value_diff_weight=None, cond=None,
                  num_of_generation_per_window=1, mode="inpaint"):
        self.eval()
        constraint = {'mask': masks.to(self.device), 'value': state_true, 'cond': cond}
        state_true = state_true.to(self.device)
        state_pred_list = [self.diffusion.predict_samples([state_true], constraint)
                           for _ in range(num_of_generation_per_window)]
        state_pred_list = [self.normalizer.unnormalize(state_pred.detach().cpu()) for state_pred in state_pred_list]
        return torch.stack(state_pred_list)

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)


class RealTimePredictor:
    def __init__(self, opt):
        self.opt = opt
        self.model = BaselineModelCpu(opt, TransformerHipKnee, EMA=True)
        self.model.diffusion.to('cpu')

        customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(opt.subject_osim_model, self.opt.geometry_folder)
        skel = customOsim.skeleton

        self.skel = skel
        self.data_window_buffer = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])

        self.column_index = opt.kinematic_diffusion_col_loc
        self.mask = torch.ones([self.opt.window_len, len(self.opt.model_states_column_names)])      # No mask, just to be compatible with the model
        self.feature_len = len(self.opt.model_states_column_names)

    def predict_grf(self, right_hip_pos, right_knee_pos, left_hip_pos, left_knee_pos,
                    right_hip_vel, right_knee_vel, left_hip_vel, left_knee_vel):

        new_row = torch.zeros([1, self.feature_len])
        new_row[0, self.column_index] = torch.tensor([right_hip_pos, right_knee_pos, left_hip_pos, left_knee_pos, right_hip_vel, right_knee_vel, left_hip_vel, left_knee_vel])
        new_row = self.model.normalizer.normalize(new_row)
        self.data_window_buffer = torch.cat((self.data_window_buffer[1:], new_row), 0)
        states = self.model.eval_loop(opt, self.data_window_buffer.unsqueeze(0), self.mask)[0, 0]
        f_r, f_l = inverse_norm_cops_and_split_force(states, self.opt)
        return f_r, f_l


def update_opt(opt, current_folder, osim_model_path, height_m, weight_kg):
    opt.subject_osim_model = current_folder + osim_model_path
    opt.height_m = height_m
    opt.weight_kg = weight_kg

    opt.geometry_folder = current_folder + '/Geometry/'
    opt.checkpoint_bl = current_folder + '/GaitDynamicsRefinement.pt'
    columns_to_keep = ['hip_flexion_r', 'knee_angle_r', 'hip_flexion_l', 'knee_angle_l'] + KINETICS_ALL + \
                      ['hip_flexion_r_vel', 'knee_angle_r_vel', 'hip_flexion_l_vel', 'knee_angle_l_vel']
    opt.model_states_column_names = columns_to_keep
    # opt.kinematic_diffusion = [col for col in columns_to_keep if 'force' not in col]
    opt.kinematic_diffusion_col_loc = [columns_to_keep.index(col) for col in columns_to_keep if 'force' not in col]
    opt.grf_osim_col_loc = [columns_to_keep.index(col) for col in columns_to_keep if ('force' in col) and ('cop' not in col)]
    opt.cop_osim_col_loc = [columns_to_keep.index(col) for col in columns_to_keep if 'cop' in col]
    opt.checkpoint_bl = current_folder + '/GaitDynamicsRefinementHipKnee.pt'
    return opt


if __name__ == '__main__':
    """
    TODO before running the code:
    1. Confirm that sampling rate is 100 Hz, force output in N, CoP output in m and being relative to calcaneus.
    2. Confirm that angles are in rad and angular velocities are in rad/s.
    3. Update variables of update_opt, see comment 'TODO: AAAA'.
    4. Update code below 'TODO: BBBB'.
    """

    current_folder = os.getcwd()
    opt = parse_opt()
    # TODO: AAAA
    opt = update_opt(opt, current_folder, osim_model_path='/Scaled_generic_no_arm.osim', height_m=1.84, weight_kg=92.9)
    predictor = RealTimePredictor(opt)

    # TODO: BBBB
    data_df = pd.read_csv('fpa_segment_1_ik.mot', sep='\t', skiprows=10)
    angle_df = data_df[['hip_flexion_r', 'knee_angle_r', 'hip_flexion_l', 'knee_angle_l']]
    angle_vel_df = angle_df.diff().fillna(0)
    angle_vel_df.columns = ['hip_flexion_r_vel', 'knee_angle_r_vel', 'hip_flexion_l_vel', 'knee_angle_l_vel']
    combined_df = pd.concat([angle_df, angle_vel_df], axis=1)
    f_r_list, f_l_list = [], []
    iter_start_time = time.time()
    for i_row in range(0, combined_df.shape[0]):
        f_r, f_l = predictor.predict_grf(*combined_df.iloc[i_row])
        f_r_list.append(f_r)
        f_l_list.append(f_l)
    iter_end_time = time.time()
    print('Time taken for a', combined_df.shape[0]/100, ' s trial: ', iter_end_time - iter_start_time)
    good_prediction = pd.read_csv('fpa_segment_1_ik_pred.mot', sep='\t', skiprows=6)
    plt.figure()
    plt.plot(good_prediction[['force1_vy', 'force2_vy']])
    plt.plot(torch.concatenate(f_r_list)[:, 1], '--')
    plt.plot(torch.concatenate(f_l_list)[:, 1], '--')
    plt.show()











