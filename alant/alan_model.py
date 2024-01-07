import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from pathlib import Path
from alant.preprocess import increment_path, Normalizer
from model.adan import Adan
from model.dance_decoder import DanceDecoder
from alant.vis import SMPLSkeleton
import os
import pickle
import torch
import copy
import torch.nn as nn
from model.utils import extract, make_beta_schedule, linear_resample_data
from typing import Any, List
import nimblephysics as nimble
from alant.alan_consts import *
from alant.quaternion import euler_from_6v, euler_to_6v
from alant.alan_osim_fk import get_model_offsets, get_knee_rotation_coefficients, forward_kinematics


def convert_addb_state_to_model_input(pose_df, joints_3d):
    # shift root position to start in (x,y) = (0,0)
    pose_df['pelvis_tx'] = pose_df['pelvis_tx'] - pose_df['pelvis_tx'][0]
    pose_df['pelvis_ty'] = pose_df['pelvis_ty'] - pose_df['pelvis_ty'][0]
    pose_df['pelvis_tz'] = pose_df['pelvis_tz'] - pose_df['pelvis_tz'][0]

    # remove frozen dof
    for frozen_col in FROZEN_DOFS:
        pose_df = pose_df.drop(frozen_col, axis=1)

    # convert euler to 6v
    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_6v = euler_to_6v(torch.tensor(pose_df[joints_with_3_dof].values), "ZXY").numpy()
        for joints_euler_name in joints_with_3_dof:
            pose_df = pose_df.drop(joints_euler_name, axis=1)
        for i in range(6):
            pose_df[joint_name + '_' + str(i)] = joint_6v[:, i]

    # TODO: Might need to check if 1D joints are continuous
    # <!--Orientation of the joint in the parent body specified in the parent reference frame.
    # Euler XYZ body-fixed rotation angles are used to express the orientation. Default is (0,0,0).-->
    return pose_df


def inverse_convert_addb_state_to_model_input(model_states, model_states_column_names, joints_3d, osim_dof_columns):
    model_states_dict = {col: model_states[..., i] for i, col in enumerate(model_states_column_names) if col in osim_dof_columns}

    # convert 6v to euler
    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_name_6v = [joint_name + '_' + str(i) for i in range(6)]
        index_ = [model_states_column_names.index(joint_name_6v[i]) for i in range(6)]
        joint_euler = euler_from_6v(model_states[..., index_], "ZXY")

        for i, joints_euler_name in enumerate(joints_with_3_dof):
            model_states_dict[joints_euler_name] = joint_euler[..., i]

    # add frozen dof back
    for frozen_col in FROZEN_DOFS:
        model_states_dict[frozen_col] = torch.zeros(model_states.shape[:2]).to(model_states.device)

    osim_states = torch.stack([model_states_dict[col] for col in osim_dof_columns], dim=2).float()
    return osim_states


def identity(t, *args, **kwargs):
    return t


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class MotionDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            backup_path: str,
            train: bool,
            joints_3d: dict,
            osim_dof_columns: List[str],
            normalizer: Any = None,
            force_reload: bool = False,
    ):
        self.data_path = data_path
        self.raw_fps = 120
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.train = train
        self.name = "Train" if self.train else "Test"

        self.normalizer = normalizer

        pickle_name = "processed_train_data.pkl" if train else "processed_test_data.pkl"

        backup_path = Path(backup_path)
        # save normalizer
        if not train:
            pickle.dump(
                normalizer, open(os.path.join(backup_path, "normalizer.pkl"), "wb")
            )
        print("Loading dataset...")
        self.load_addb(joints_3d, osim_dof_columns)  # Call this last
        self.trial_num = len(self.trials)
        data_concat = torch.cat([self[0][0] for _ in range(1000)], dim=0)      # [!] randomly get 1000 samples for normalization
        print("normalizing training data")
        self.normalizer = Normalizer(data_concat)
        for i in range(self.trial_num):
            self.trials[i].converted_pose = torch.tensor(self.normalizer.normalize(self.trials[i].converted_pose)).float().detach()

    def __len__(self):
        return 20000        # [!] This num affects # of iterations in each epoch

    def __getitem__(self, idx):
        # idx_ = torch.multinomial(self.trial_length_probability, 1).item()
        idx_ = torch.randint(0, self.trial_num, (1,)).item()        # a random trial regardless of its length
        trial_length = self.trials[idx_].length
        slice_index = torch.randint(0, trial_length - 61, (1,)).item()
        return self.trials[idx_].converted_pose[slice_index:slice_index+60, ...], self.trials[idx_].model_offsets

    def load_addb(self, joints_3d, osim_dof_columns):
        subject_paths = []
        if os.path.isdir(self.data_path):
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith(".b3d"):
                        subject_paths.append(os.path.join(root, file))

        self.trials = []
        for i_sub, subject_path in enumerate(subject_paths):
            # Add the skeleton to the list of skeletons
            subject = nimble.biomechanics.SubjectOnDisk(subject_path)

            skel = subject.readSkel(0, geometryFolder=os.path.dirname(os.path.realpath(__file__)) + "/../../data/Geometry/")
            model_offsets = get_model_offsets(skel).float()
            subject_name = subject_path.split('/')[-1].split('.')[0]

            for trial_index in range(subject.getNumTrials()):
                trial_name = subject.getTrialName(trial_index)
                trial_length = subject.getTrialLength(trial_index)
                probably_missing: List[bool] = [reason != nimble.biomechanics.MissingGRFReason.notMissingGRF for reason
                                                in subject.getMissingGRF(trial_index)]

                frames: nimble.biomechanics.FrameList = subject.readFrames(trial_index, 0, trial_length,
                                                                           includeSensorData=False,
                                                                           includeProcessingPasses=True)
                try:
                    first_passes: List[nimble.biomechanics.FramePass] = [frame.processingPasses[0] for frame in frames]
                except IndexError:
                    print(f'{subject_name}, {trial_index} has no processing passes, skipping')
                    continue
                pose = [frame.pos for frame in first_passes]

                sampling_rate = 1 / subject.getTrialTimestep(trial_index)
                if (len(pose) / sampling_rate * 20) < 62:
                    print(f'Subject {subject_name} trial {trial_name} has {round(len(pose) / sampling_rate * 20, 1)} s data (less than 3 s), skipping')
                    continue
                pose = linear_resample_data(np.array(pose), sampling_rate, 20)

                pose_df = pd.DataFrame(pose, columns=osim_dof_columns)
                pose_df = convert_addb_state_to_model_input(pose_df, joints_3d)

                pose = torch.tensor(pose_df.values)

                current_trial = TrialData(pose, model_offsets, probably_missing, trial_name, subject_name)
                # TODO: check smoothness of the trial
                self.trials.append(current_trial)
        total_hour = sum([trial.length for trial in self.trials]) / 20 / 60 / 60
        total_clip_num = sum([trial.length for trial in self.trials]) / 20 / 3
        print(f"In total, {len(self.trials)} trials, {total_hour} hours, {total_clip_num} clips")
        # self.trial_length_probability = torch.tensor([1000 / trial.length for trial in self.trials]).float()


class TrialData:
    def __init__(self, converted_pose, model_offsets, probably_missing_grf, trial_name, subject_name):
        self.converted_pose = converted_pose
        self.model_offsets = model_offsets
        self.probably_missing_grf = probably_missing_grf
        self.trial_name = trial_name
        self.subject_name = subject_name
        self.length = converted_pose.shape[0]


class MotionModel:
    def __init__(
            self,
            opt,
            repr_dim,
            normalizer=None,
            EMA=True,
            learning_rate=4e-4,
            weight_decay=0.02,
    ):
        self.opt = opt
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = opt.feature_type == "baseline"

        self.repr_dim = repr_dim

        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 3
        FPS = 20
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if opt.checkpoint != "":
            checkpoint = torch.load(
                opt.checkpoint, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        smpl = SMPLSkeleton(self.accelerator.device)
        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            smpl,
            opt,
            # schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,          # predict epsilon is usually True for diffusion models right???
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        print(
            "Model has {} parameters".format(sum(y.numel() for y in model.parameters()))
        )

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if opt.checkpoint != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"train_tensor_dataset.pkl"
        )
        test_tensor_dataset_path = os.path.join(
            opt.processed_data_dir, f"test_tensor_dataset.pkl"
        )
        if (
                not opt.no_cache
                and os.path.isfile(train_tensor_dataset_path)
                and os.path.isfile(test_tensor_dataset_path)
        ):
            train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))
        else:
            train_dataset = MotionDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=True,
                force_reload=opt.force_reload,
                joints_3d=opt.joints_3d,
                osim_dof_columns=opt.osim_dof_columns,
            )
        # set normalizer
        self.normalizer = train_dataset.normalizer
        self.diffusion.set_normalizer(self.normalizer)

        # data loaders
        # decide number of workers based on cpu count
        num_cpus = multiprocessing.cpu_count()
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            # num_workers=min(int(num_cpus * 0.75), 32),            # causes error on slurm
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            if opt.log_with_wandb:
                wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, dir="wandb_logs")
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            print(f'epoch: {epoch} / {opt.epochs}')
            avg_loss_simple = 0
            avg_loss_vel = 0
            avg_loss_fk = 0
            avg_loss_drift = 0
            avg_loss_slide = 0
            # train
            self.train()            # switch to train mode.

            for step, x in enumerate(
                    load_loop(train_data_loader)
            ):
                total_loss, losses = self.diffusion(
                    x, None, t_override=None
                )
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    avg_loss_simple += losses[0].detach().cpu().numpy()
                    avg_loss_vel += losses[1].detach().cpu().numpy()
                    avg_loss_fk += losses[2].detach().cpu().numpy()
                    avg_loss_drift += losses[3].detach().cpu().numpy()
                    avg_loss_slide += losses[4].detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(
                            self.diffusion.master_model, self.diffusion.model
                        )

            self.accelerator.wait_for_everyone()

            if (epoch % 1) == 0:
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    loss_val_name_pairs = [
                        (avg_loss_simple, 'simple'), (avg_loss_vel, 'vel'), (avg_loss_fk, 'forward kinematics'),
                        (avg_loss_drift, 'drift'), (avg_loss_slide, 'slide')]
                    if opt.log_with_wandb:
                        log_dict = {name: loss / len(train_data_loader) for loss, name in loss_val_name_pairs}
                        wandb.log(log_dict)
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "normalizer": self.normalizer,
                    }

            # if epoch % 100 == 0 or epoch == opt.epochs:
            if (epoch % 100) == 0:
                torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process and opt.log_with_wandb:
            wandb.run.finish()


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
                current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,      # this is DanceDecoder
            horizon,
            repr_dim,
            smpl,
            opt,
            n_timestep=1000,
            schedule="linear",
            loss_type="l1",
            clip_denoised=True,
            predict_epsilon=True,
            guidance_weight=3,
            use_p2=False,
            cond_drop_prob=0.2,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)
        self.opt = opt

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module
        self.smpl = smpl

        betas = torch.Tensor(
            make_beta_schedule(schedule=schedule, n_timestep=n_timestep)
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timestep = int(n_timestep)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.guidance_weight = guidance_weight

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
            )

        # p2 weighting
        self.p2_loss_weight_k = 1
        self.p2_loss_weight_gamma = 0.5 if use_p2 else 0
        self.register_buffer(
            "p2_loss_weight",
            (self.p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -self.p2_loss_weight_gamma,
            )

        ## get loss coefficients and initialize objective
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, cond, t, weight=None, clip_x_start = False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, t, weight)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # guidance clipping
        if t[0] > 1.0 * self.n_timestep:
            weight = min(self.guidance_weight, 0)
        elif t[0] < 0.1 * self.n_timestep:
            weight = min(self.guidance_weight, 1)
        else:
            weight = self.guidance_weight

        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model.guided_forward(x, cond, t, weight)
        )

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, cond=cond, t=t
        )
        noise = torch.randn_like(model_mean)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(
            b, *((1,) * (len(noise.shape) - 1))
        )
        x_out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return x_out, x_start

    @torch.no_grad()
    def p_sample_loop(          # Only used during inference
            self,
            shape,
            cond,
            noise=None,
            constraint=None,
            return_diffusion=False,
            start_point=None,
    ):
        device = self.betas.device

        # default to diffusion over whole timescale
        start_point = self.n_timestep if start_point is None else start_point
        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)

        if return_diffusion:
            diffusion = [x]

        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        return x

    @torch.no_grad()
    def long_ddim_sample(self, shape, cond, **kwargs):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1

        if batch == 1:
            return self.ddim_sample(shape, cond)

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
        time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device = device)
        cond = cond.to(device)

        assert batch > 1
        assert x.shape[1] % 2 == 0
        half = x.shape[1] // 2

        x_start = None

        for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised)

            if time_next < 0:
                x = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            if time > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]
        return x

    @torch.no_grad()
    def inpaint_loop(
            self,
            shape,
            cond,
            noise=None,
            constraint=None,
            return_diffusion=False,
            start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            value_ = self.q_sample(value, timesteps - 1) if (i > 0) else x
            x = value_ * mask + (1.0 - mask) * x

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def long_inpaint_loop(
            self,
            shape,
            cond,
            noise=None,
            constraint=None,
            return_diffusion=False,
            start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = cond.to(device)
        if return_diffusion:
            diffusion = [x]

        assert x.shape[1] % 2 == 0
        if batch_size == 1:
            # there's no continuation to do, just do normal
            return self.p_sample_loop(
                shape,
                cond,
                noise=noise,
                constraint=constraint,
                return_diffusion=return_diffusion,
                start_point=start_point,
            )
        assert batch_size > 1
        half = x.shape[1] // 2

        start_point = self.n_timestep if start_point is None else start_point
        for i in tqdm(reversed(range(0, start_point))):
            # fill with i
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # sample x from step i to step i-1
            x, _ = self.p_sample(x, cond, timesteps)
            # enforce constraint between each denoising step
            if i > 0:
                # the first half of each sequence is the second half of the previous one
                x[1:, :half] = x[:-1, half:]

            if return_diffusion:
                diffusion.append(x)

        if return_diffusion:
            return x, diffusion
        else:
            return x

    @torch.no_grad()
    def conditional_sample(
            self, shape, cond, constraint=None, *args, horizon=None, **kwargs
    ):
        """
            conditions : [ (time, state), ... ]
        """
        device = self.betas.device
        horizon = horizon or self.horizon

        return self.p_sample_loop(shape, cond, *args, **kwargs)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):     # blend noise into state variables
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x, cond, t):
        x_start, model_offsets = x
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t, cond_drop_prob=self.cond_drop_prob)
        assert noise.shape == x_recon.shape

        model_out = x_recon
        if self.predict_epsilon:
            target = noise
        else:
            target = x_start

        # full reconstruction loss
        loss_simple = self.loss_fn(model_out, target, reduction="none")
        loss_simple = loss_simple * extract(self.p2_loss_weight, t, loss_simple.shape)

        loss_vel = self.loss_fn(model_out[:, 1:, 3:] - model_out[:, :-1, 3:], target[:, 1:, 3:] - target[:, :-1, 3:], reduction="none")
        loss_vel = loss_vel * extract(self.p2_loss_weight, t, loss_vel.shape)

        osim_states_pred = self.normalizer.unnormalize(model_out)
        osim_states_pred = inverse_convert_addb_state_to_model_input(osim_states_pred, self.opt.model_states_column_names, self.opt.joints_3d, self.opt.osim_dof_columns)
        foot_locations_pred, joint_locations_pred, segment_orientations_pred = forward_kinematics(osim_states_pred, model_offsets)
        osim_states_true = self.normalizer.unnormalize(target)
        osim_states_true = inverse_convert_addb_state_to_model_input(osim_states_true, self.opt.model_states_column_names, self.opt.joints_3d, self.opt.osim_dof_columns)
        foot_locations_true, joint_locations_true, segment_orientations_true = forward_kinematics(osim_states_true, model_offsets)

        loss_fk = self.loss_fn(joint_locations_pred, joint_locations_true, reduction="none")
        # loss_fk = reduce(loss_fk, "b ... -> b (...)", "mean")
        loss_fk = loss_fk * extract(self.p2_loss_weight, t, loss_fk.shape[1:])

        loss_drift = self.loss_fn(model_out[..., :3], target[..., :3], reduction="none")
        loss_drift = loss_drift * extract(self.p2_loss_weight, t, loss_drift.shape)

        # loss_floor_penetration = self.loss_fn(foot_locations_pred[..., 1], foot_locations_true[..., 1], reduction="none")
        # loss_floor_penetration = loss_floor_penetration * extract(self.p2_loss_weight, t, loss_floor_penetration.shape)

        foot_vel_pred = (foot_locations_pred[..., 1:, :] - foot_locations_pred[..., :-1, :]).abs()
        stance_based_on_foot_vel = (torch.norm(foot_vel_pred, dim=-1) < 0.001)[..., None].expand(-1, -1, -1, 3)       # TODO tune this. Tom used 0.003 (0.3 m/s)
        foot_vel_pred[~stance_based_on_foot_vel] = 0
        loss_slide = self.loss_fn(foot_vel_pred, foot_vel_pred * 0, reduction="none")
        loss_slide = loss_slide * extract(self.p2_loss_weight, t, loss_slide.shape[1:])

        losses = (
            1. * loss_simple.mean(),         # TODO tune this
            1. * loss_vel.mean(),
            1. * loss_fk.mean(),
            1. * loss_drift.mean(),
            # 1. * loss_floor_penetration.mean(),
            100. * loss_slide.mean(),
        )
        return sum(losses), losses

    def loss(self, x, cond, t_override=None):
        batch_size = len(x[0])
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x[0].device).long()        # randomly select a timestep to train
        else:
            t = torch.full((batch_size,), t_override, device=x[0].device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

    def partial_denoise(self, x, cond, t):
        x_noisy = self.noise_to_t(x, t)
        return self.p_sample_loop(x.shape, cond, noise=x_noisy, start_point=t)

    def noise_to_t(self, x, timestep):
        batch_size = len(x)
        t = torch.full((batch_size,), timestep, device=x.device).long()
        return self.q_sample(x, t) if timestep > 0 else x

    def generate_samples(
            self,
            shape,
            cond,
            normalizer,
            opt,
            mode="normal",
            noise=None,
            constraint=None,
            start_point=None,
    ):
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_loop
            elif mode == "normal":
                func_class = self.ddim_sample
            elif mode == "long":
                func_class = self.long_ddim_sample
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    cond,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples.detach().cpu())
        samples = inverse_convert_addb_state_to_model_input(samples, opt.model_states_column_names,
                                                            opt.joints_3d, opt.osim_dof_columns)
        return samples














