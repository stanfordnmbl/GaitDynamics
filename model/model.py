import numpy as np
from functools import partial
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from data.preprocess import increment_path
from data.quaternion import euler_from_6v, euler_to_6v
from model.adan import Adan
from model.dance_decoder import DanceDecoder
import os
import torch
import torch.nn as nn
from model.rotary_embedding_torch import RotaryEmbedding
from model.utils import extract, make_beta_schedule, fix_seed, identity, maybe_wrap, \
    inverse_convert_addb_state_to_model_input
from consts import *
from data.osim_fk import forward_kinematics
import matplotlib.pyplot as plt
from torch import Tensor
import math


fix_seed()


class MotionModel:
    def __init__(
            self,
            opt,
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

        self.repr_dim = len(opt.model_states_column_names)

        self.horizon = horizon = opt.window_len

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if opt.checkpoint != "":
            checkpoint = torch.load(
                opt.checkpoint, map_location=self.accelerator.device
            )
            self.normalizer = checkpoint["normalizer"]

        # !!! change it back after the last run
        # model = DanceDecoder(
        #     nfeats=repr_dim,
        #     seq_len=horizon,
        #     latent_dim=512,
        #     ff_size=1024,
        #     num_layers=8,
        #     num_heads=8,
        #     dropout=0.1,
        #     activation=F.gelu,
        # )
        model = DanceDecoder(
            nfeats=self.repr_dim,
            seq_len=horizon,
            latent_dim=256,
            ff_size=1024,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            activation=F.gelu,
        )

        diffusion = GaussianDiffusion(
            model,
            horizon,
            self.repr_dim,
            opt,
            # schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.,
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

    def train_loop(self, opt, train_dataset):
        # set normalizer
        self.normalizer = train_dataset.normalizer
        self.diffusion.set_normalizer(self.normalizer)

        # data loaders
        # decide number of workers based on cpu count
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        train_data_loader = self.accelerator.prepare(train_data_loader)
        # boot up multi-gpu training. test dataloader is only on main process
        load_loop = (
            partial(tqdm, position=1, desc="Batch")     # , miniters=int(223265/100)
            if self.accelerator.is_main_process
            else lambda x: x
        )
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        sub_and_trial_names, dset_names = train_dataset.get_attributes_of_trials()
        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            print(f'epoch: {epoch} / {opt.epochs}')
            avg_loss_simple = 0
            avg_loss_vel = 0
            avg_loss_fk = 0
            avg_loss_drift = 0
            avg_loss_slide = 0

            joint_loss = np.zeros([opt.model_states_column_names.__len__()])
            dataset_loss_record = np.zeros([train_dataset.trial_num])

            self.train()

            for step, x in enumerate(load_loop(train_data_loader)):
                cond = x[5]
                total_loss, losses = self.diffusion(x, cond, t_override=None)
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)

                self.optim.step()

                if opt.log_with_wandb:
                    # ema update and train loss update only on main
                    if self.accelerator.is_main_process:
                        avg_loss_simple += losses[0].detach().cpu().numpy()
                        avg_loss_vel += losses[1].detach().cpu().numpy()
                        avg_loss_fk += losses[2].detach().cpu().numpy()
                        avg_loss_drift += losses[3].detach().cpu().numpy()
                        avg_loss_slide += losses[4].detach().cpu().numpy()

                        joint_loss += losses[5].detach().mean(axis=0).mean(axis=0).cpu().numpy()
                        dataset_loss_record[x[2].cpu()] += losses[5].detach().mean(axis=1).mean(axis=1).cpu().numpy()

                if step % opt.ema_interval == 0:
                    self.diffusion.ema.update_model_average(self.diffusion.master_model, self.diffusion.model)

            self.accelerator.wait_for_everyone()

            if epoch > -1:
                self.accelerator.wait_for_everyone()
                # save only if on main thread
                if self.accelerator.is_main_process:
                    self.eval()
                    ckpt = {
                        "ema_state_dict": self.diffusion.master_model.state_dict(),
                        "model_state_dict": self.accelerator.unwrap_model(
                            self.model
                        ).state_dict(),
                        "normalizer": self.normalizer,
                    }
                    if opt.log_with_wandb:
                        loss_val_name_pairs_terms = [
                            (avg_loss_simple / len(train_data_loader), 'simple'), (avg_loss_vel / len(train_data_loader), 'vel'),
                            (avg_loss_fk / len(train_data_loader), 'forward kinematics'), (avg_loss_drift / len(train_data_loader), 'drift'),
                            (avg_loss_slide / len(train_data_loader), 'slide')]

                        loss_val_name_pairs_joints = [(loss_, joint_) for joint_, loss_ in zip(opt.model_states_column_names, joint_loss)]

                        loss_dset = {dset: [0, 0] for dset in train_dataset.dset_set}
                        for dset_name, trial_loss in zip(dset_names, dataset_loss_record):
                            if trial_loss == 0:
                                continue        # if 0, then the trial is not used for training, thus skipping
                            loss_dset[dset_name] = [loss_dset[dset_name][0] + trial_loss, loss_dset[dset_name][1] + 1]
                        loss_val_name_pairs_dset = []
                        for dset_name, loss_count in loss_dset.items():
                            if loss_count[1] != 0:
                                loss_val_name_pairs_dset.append((loss_count[0] / loss_count[1], dset_name))
                        log_dict = {name: loss for loss, name in loss_val_name_pairs_dset + loss_val_name_pairs_joints + loss_val_name_pairs_terms}

                        wandb.log(log_dict)

            if (epoch % 7) == 0 or epoch == opt.epochs:
                ckpt.pop("model_state_dict")
                torch.save(ckpt, os.path.join(wdir, f"train-{epoch}_{self.model}.pt"))
                print(f"[MODEL SAVED at Epoch {epoch}]")
        if self.accelerator.is_main_process and opt.log_with_wandb:
            wandb.run.finish()

    def eval_loop(self, opt, state_true, masks, value_diff_thd=None, value_diff_weight=None, cond=None,
                  num_of_generation_per_window=1, mode="inpaint"):
        self.eval()
        if value_diff_thd is None:
            value_diff_thd = torch.zeros([state_true.shape[2]])
        if value_diff_weight is None:
            value_diff_weight = torch.ones([state_true.shape[2]])
        if cond is None:
            cond = torch.ones([6])

        constraint = {'mask': masks, 'value': state_true.clone(), 'value_diff_thd': value_diff_thd,
                      'value_diff_weight': value_diff_weight, 'cond': cond}

        shape = (state_true.shape[0], self.horizon, self.repr_dim)
        state_pred_list = [self.diffusion.generate_samples(
            shape,
            self.normalizer,
            opt,
            mode=mode,
            constraint=constraint)
            for _ in range(num_of_generation_per_window)]
        return torch.stack(state_pred_list)


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
            opt,
            n_timestep=1000,
            schedule="linear",
            loss_type="l1",
            clip_denoised=False,
            predict_epsilon=True,
            guidance_weight=1,
            use_p2=False,
            cond_drop_prob=0.,
    ):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = repr_dim
        self.model = model
        self.ema = EMA(0.99)
        self.master_model = copy.deepcopy(self.model)
        self.opt = opt

        self.cond_drop_prob = cond_drop_prob

        # make a SMPL instance for FK module

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

    def model_predictions(self, x, cond, time_cond, weight=None, clip_x_start=False):
        weight = weight if weight is not None else self.guidance_weight
        model_output = self.model.guided_forward(x, cond, time_cond, weight)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, time_cond, x_start)

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

    # @torch.no_grad()
    # def ddim_sample(self, shape, cond, **kwargs):
    #     batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
    #
    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    #
    #     x = torch.randn(shape, device = device)
    #     cond = cond.to(device)
    #
    #     x_start = None
    #
    #     for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
    #         time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
    #         pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start = self.clip_denoised)
    #
    #         if time_next < 0:
    #             x = x_start
    #             continue
    #
    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]
    #
    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         c = (1 - alpha_next - sigma ** 2).sqrt()
    #
    #         noise = torch.randn_like(x)
    #
    #         x = x_start * alpha_next.sqrt() + \
    #             c * pred_noise + \
    #             sigma * noise
    #     return x

    # @torch.no_grad()
    # def long_ddim_sample(self, shape, cond, **kwargs):
    #     batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 1
    #
    #     if batch == 1:
    #         return self.ddim_sample(shape, cond)
    #
    #     times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    #     times = list(reversed(times.int().tolist()))
    #     weights = np.clip(np.linspace(0, self.guidance_weight * 2, sampling_timesteps), None, self.guidance_weight)
    #     time_pairs = list(zip(times[:-1], times[1:], weights)) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    #
    #     x = torch.randn(shape, device = device)
    #     cond = cond.to(device)
    #
    #     assert batch > 1
    #     assert x.shape[1] % 2 == 0
    #     half = x.shape[1] // 2
    #
    #     x_start = None
    #
    #     for time, time_next, weight in tqdm(time_pairs, desc = 'sampling loop time step'):
    #         time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
    #         pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, weight=weight, clip_x_start = self.clip_denoised)
    #
    #         if time_next < 0:
    #             x = x_start
    #             continue
    #
    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]
    #
    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         c = (1 - alpha_next - sigma ** 2).sqrt()
    #
    #         noise = torch.randn_like(x)
    #
    #         x = x_start * alpha_next.sqrt() + \
    #             c * pred_noise + \
    #             sigma * noise
    #
    #         if time > 0:
    #             # the first half of each sequence is the second half of the previous one
    #             x[1:, :half] = x[:-1, half:]
    #     return x

    @torch.no_grad()
    def inpaint_ddim_guided(self, shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 0

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        cond = constraint["cond"].to(device)

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels
        value_diff_thd = constraint["value_diff_thd"].to(device)  # channels
        value_diff_weight = constraint["value_diff_weight"].to(device)  # channels

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            if self.opt.guide_x_start_the_end_step <= time <= self.opt.guide_x_start_the_beginning_step:
                x.requires_grad_()
                with torch.enable_grad():
                    for step_ in range(self.opt.n_guided_steps):
                        pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
                        value_diff = torch.subtract(x_start, value)
                        loss = torch.relu(value_diff.abs() - value_diff_thd) * value_diff_weight
                        grad = torch.autograd.grad([loss.sum()], [x])[0]
                        x = x - self.opt.guidance_lr * grad

            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
            if time_next < 0:
                x = x_start
                x = value * mask + (1.0 - mask) * x
                return x

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            timesteps = torch.full((batch,), time_next, device=device, dtype=torch.long)
            value_ = self.q_sample(value, timesteps) if (time > 0) else x
            x = value_ * mask + (1.0 - mask) * x
        return x

    @torch.no_grad()
    def inpaint_ddpm_loop_hip_6v(
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
        assert batch_size == 1

        tx_col_loc = self.opt.model_states_column_names.index('pelvis_tx')

        value = constraint["value"].to(device)  # batch x horizon x channels
        timesteps = torch.full((batch_size,), 1, device=device, dtype=torch.long)
        x = self.q_sample(value, timesteps - 1)

        x_unnormed = self.normalizer.unnormalize(x)
        # convert 6v to euler
        joint_euler, joint_euler_col_loc = [], []
        for joint_name, joints_with_3_dof in self.opt.joints_3d.items():
            joint_name_6v = [joint_name + '_' + str(i) for i in range(6)]
            index_ = [self.opt.model_states_column_names.index(joint_name_6v[i]) for i in range(6)]
            joint_euler.append(euler_from_6v(x_unnormed[..., index_], "ZXY"))
            joint_euler_col_loc.append(index_)
        joint_euler = torch.stack(joint_euler, dim=-1)

        joint_euler.requires_grad_()
        with torch.enable_grad():
            x_temp = torch.zeros_like(x[0])
            for i_joint in range(0, int(joint_euler.shape[-1])):
                joint_6v = euler_to_6v(joint_euler[..., i_joint], "ZXY")
                x_temp[..., joint_euler_col_loc[i_joint]] = joint_6v

            # x_temp = self.normalizer.scaler.transform(x_temp)
            x_temp = self.normalizer.normalize(x_temp)
            for i_joint in range(0, int(joint_euler.shape[-1])):
                x[..., joint_euler_col_loc[i_joint]] = x_temp[..., joint_euler_col_loc[i_joint]]
            x.requires_grad_()

            x_new, _ = self.p_sample(x, cond, timesteps)
            vel = x_new[0, :, tx_col_loc]
            grad_to_x = torch.autograd.grad([vel.sum()], [x], retain_graph=True)[0]
            grad_to_joint_euler = torch.autograd.grad([vel.sum()], [joint_euler], retain_graph=True)[0]
            x_times_grad = 40

            # hip_flexion_r_col_loc = 3
            knee_r_col_loc = self.opt.model_states_column_names.index('knee_angle_r')
            ankle_r_col_loc = self.opt.model_states_column_names.index('ankle_angle_r')
            plt.figure()
            plt.plot(x[0, :, knee_r_col_loc].detach().cpu().numpy(), color='C0')
            # plt.plot(value[0, :, knee_r_col_loc].detach().cpu().numpy(), '-.', color='C0')
            plt.plot((x[0, :, knee_r_col_loc] + x_times_grad * grad_to_x[0, :, knee_r_col_loc]).detach().cpu().numpy(), '--', color='C0')
            plt.plot(x[0, :, ankle_r_col_loc].detach().cpu().numpy(), color='C1')
            # plt.plot(value[0, :, ankle_r_col_loc].detach().cpu().numpy(), '-.', color='C1')
            plt.plot((x[0, :, ankle_r_col_loc] + x_times_grad * grad_to_x[0, :, ankle_r_col_loc]).detach().cpu().numpy(), '--', color='C1')
            plt.plot(joint_euler[0, :, 0, 1].detach().cpu().numpy() * 10, color='C2')
            plt.plot((joint_euler[0, :, 0, 1] * 10 + grad_to_joint_euler[0, :, 0, 1]).detach().cpu().numpy(), '--', color='C2')
            plt.show()

    @torch.no_grad()
    def inpaint_ddim_loop(self, shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 0

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = constraint["cond"].to(device)

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
            if time_next < 0:
                x = x_start
                return x

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            timesteps = torch.full((batch,), time_next, device=device, dtype=torch.long)
            value_ = self.q_sample(value, timesteps) if (time > 0) else x
            x = value_ * mask + (1.0 - mask) * x
        return x

    @torch.no_grad()
    def noise_denoise_at_each_t(self, shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, constraint['total_timesteps'], 10, 0
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        cond = constraint["cond"].to(device)
        value = constraint["value"].to(device)

        x_list = []
        for time_pairs_start in range(len(time_pairs)):
            time = time_pairs[time_pairs_start][0]
            x = self.q_sample(value, torch.full((batch,), time, device=device, dtype=torch.long))

            for time, time_next in tqdm(time_pairs[time_pairs_start:], desc='sampling loop time step'):
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

                pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)
                if time_next < 0:
                    x = x_start
                    break

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(x)

                x = x_start * alpha_next.sqrt() + \
                    c * pred_noise + \
                    sigma * noise

            x_list.append(x)
        return x_list

    @torch.no_grad()
    def noise_denoise_at_each_t_ddpm(self, shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        device = self.betas.device
        batch_size = shape[0]
        cond = constraint["cond"].to(device)
        value = constraint["value"].to(device)

        start_point = constraint['total_timesteps']
        x = self.q_sample(value, torch.full((batch_size,), start_point, device=device, dtype=torch.long))
        for i in tqdm(reversed(range(0, start_point))):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x, _ = self.p_sample(x, cond, timesteps)
        x_list = [x]
        return x_list

    @torch.no_grad()
    def extract_last_hidden_layer_loop(self, shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 0

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        x = torch.randn(shape, device=device)
        cond = constraint["cond"].to(device)

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        for time, time_next in tqdm(time_pairs[:-1], desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

            pred_noise, x_start, *_ = self.model_predictions(x, cond, time_cond, clip_x_start=self.clip_denoised)

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x)

            x = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

            timesteps = torch.full((batch,), time_next, device=device, dtype=torch.long)
            value_ = self.q_sample(value, timesteps) if (time > 0) else x
            x = value_ * mask + (1.0 - mask) * x

        time, time_next = time_pairs[-1]
        time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

        x = self.model.input_projection(x)
        x = self.model.abs_pos_encoding(x)
        t_hidden = self.model.time_mlp(time_cond)
        t = self.model.to_time_cond(t_hidden)
        output_last_hidden = self.model.seqTransDecoder(x, None, t)
        output = self.model.final_layer(output_last_hidden)
        return output_last_hidden

    @torch.no_grad()
    def inpaint_ddpm_loop(
            self,
            shape,
            noise=None,
            constraint=None,
            return_diffusion=False,
            start_point=None,
    ):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        cond = constraint["cond"].to(device)
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
        x_start, model_offsets, _, _, height_m, _ = x
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # reconstruct
        x_recon = self.model(x_noisy, cond, t)
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

        loss_drift = self.loss_fn(model_out[..., :3], target[..., :3], reduction="none")
        loss_drift = loss_drift * extract(self.p2_loss_weight, t, loss_drift.shape)

        osim_states_pred = self.normalizer.unnormalize(model_out)
        osim_states_pred = inverse_convert_addb_state_to_model_input(
            osim_states_pred, self.opt.model_states_column_names, self.opt.joints_3d, self.opt.osim_dof_columns,
            pos_vec=[0, 0, 0], height_m=height_m)
        _, joint_locations_pred, _, _ = forward_kinematics(osim_states_pred, model_offsets)
        osim_states_true = self.normalizer.unnormalize(target)
        osim_states_true = inverse_convert_addb_state_to_model_input(
            osim_states_true, self.opt.model_states_column_names, self.opt.joints_3d, self.opt.osim_dof_columns,
            pos_vec=[0, 0, 0], height_m=height_m)
        foot_locations_pred, joint_locations_true, _, _ = forward_kinematics(osim_states_true, model_offsets)

        loss_fk = self.loss_fn(joint_locations_pred, joint_locations_true, reduction="none")
        # loss_fk = reduce(loss_fk, "b ... -> b (...)", "mean")
        loss_fk = loss_fk * extract(self.p2_loss_weight, t, loss_fk.shape[1:])

        # loss_floor_penetration = self.loss_fn(foot_locations_pred[..., 1], foot_locations_true[..., 1], reduction="none")
        # loss_floor_penetration = loss_floor_penetration * extract(self.p2_loss_weight, t, loss_floor_penetration.shape)

        foot_acc_pred = (foot_locations_pred[..., 2:, :] - 2 * foot_locations_pred[..., 1:-1, :] + foot_locations_pred[..., :-2, :]).abs() * self.opt.target_sampling_rate ** 2
        stance_based_on_foot_vel = (torch.norm(foot_acc_pred, dim=-1) < 0.3)[..., None].expand(-1, -1, -1, 3)
        foot_acc_pred[~stance_based_on_foot_vel] = 0
        loss_slide = self.loss_fn(foot_acc_pred, foot_acc_pred * 0, reduction="none")
        loss_slide = loss_slide * extract(self.p2_loss_weight, t, loss_slide.shape[1:])

        losses = [
            1. * loss_simple.mean(),
            0 * loss_vel.mean(),
            0. * loss_fk.mean(),
            0. * loss_drift.mean(),
            0. * loss_slide.mean()]
        return sum(losses), losses + [loss_simple]

    def loss(self, x, cond, t_override=None):
        batch_size = len(x[0])
        if t_override is None:
            t = torch.randint(0, self.n_timestep, (batch_size,), device=x[0].device).long()        # randomly select a timestep to train
        else:
            t = torch.full((batch_size,), t_override, device=x[0].device).long()
        return self.p_losses(x, cond, t)

    def forward(self, x, cond, t_override=None):
        return self.loss(x, cond, t_override)

    def generate_samples(
            self,
            shape,
            normalizer,
            opt,
            mode,
            noise=None,
            constraint=None,
            start_point=None,
    ):
        torch.manual_seed(torch.randint(0, 2 ** 32, (1,)).item())
        if isinstance(shape, tuple):
            if mode == "inpaint":
                func_class = self.inpaint_ddim_loop
            elif mode == "inpaint_ddim_guided":
                func_class = self.inpaint_ddim_guided
            elif mode == "extract_last_hidden_layer":
                func_class = self.extract_last_hidden_layer_loop
            elif mode == "noise_denoise_at_each_t":
                func_class = self.noise_denoise_at_each_t
            else:
                assert False, "Unrecognized inference mode"
            samples = (
                func_class(
                    shape,
                    noise=noise,
                    constraint=constraint,
                    start_point=start_point,
                )
            )
        else:
            samples = shape

        samples = normalizer.unnormalize(samples.detach().cpu())
        samples = samples.detach().cpu()
        return samples


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(max_len * 2) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        try:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        except RuntimeError:
            pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]

        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8, d_ff=512, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.rotary = RotaryEmbedding(dim=d_model)
        self.use_rotary = self.rotary is not None

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        attn_output, _ = self.self_attn(qk, qk, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoderArchitecture(nn.Module):
    def __init__(self, repr_dim, opt, nlayers=6):
        super(TransformerEncoderArchitecture, self).__init__()
        self.input_dim = len(opt.kinematic_diffusion_col_loc)
        self.output_dim = repr_dim - self.input_dim
        embedding_dim = 192
        self.input_to_embedding = nn.Linear(self.input_dim, embedding_dim)
        self.encoder_layers = nn.Sequential(*[EncoderLayer(embedding_dim) for _ in range(nlayers)])
        self.embedding_to_output = nn.Linear(embedding_dim, self.output_dim)
        self.opt = opt
        self.input_col_loc = opt.kinematic_diffusion_col_loc
        self.output_col_loc = [i for i in range(repr_dim) if i not in self.input_col_loc]

    def get_optimizer(self):
        return Adan(self.parameters(), lr=4e-4, weight_decay=0.02)

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


class DiffusionShellForAdaptingTheOriginalFramework(nn.Module):
    def __init__(self, model):
        super(DiffusionShellForAdaptingTheOriginalFramework, self).__init__()
        self.device = 'cuda'
        self.model = model
        self.ema = EMA(0.99)
        self.master_model = copy.deepcopy(self.model)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def predict_samples(self, x, constraint):
        x[0] = x[0] * constraint['mask']
        output_pred = self.model.end_to_end_prediction(x)
        x[0][:, :, self.model.output_col_loc] = output_pred
        return x[0]

    def forward(self, x, cond, t_override):
        output_true = x[0][:, :, self.model.output_col_loc]
        output_pred = self.model.end_to_end_prediction(x)
        loss_simple = torch.zeros(x[0].shape).to(x[0].device)
        loss_simple[:, :, self.model.output_col_loc] = self.model.loss_fun(output_pred, output_true)

        losses = [
            1. * loss_simple.mean(),
            torch.tensor(0.).to(loss_simple.device),
            torch.tensor(0.).to(loss_simple.device),
            torch.tensor(0.).to(loss_simple.device),
            torch.tensor(0).to(loss_simple.device)]
        return sum(losses), losses + [loss_simple]


class BaselineModel(MotionModel):
    def __init__(
            self,
            opt,
            model_architecture_class,
            EMA=True,
    ):
        self.opt = opt
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.repr_dim = len(opt.model_states_column_names)
        self.horizon = horizon = opt.window_len
        self.accelerator.wait_for_everyone()

        self.model = model_architecture_class(self.repr_dim, opt)
        self.diffusion = DiffusionShellForAdaptingTheOriginalFramework(self.model)
        self.diffusion = self.accelerator.prepare(self.diffusion)
        optim = self.model.get_optimizer()
        self.optim = self.accelerator.prepare(optim)

        print("Model has {} parameters".format(sum(y.numel() for y in self.model.parameters())))

        checkpoint = None
        if opt.checkpoint_bl != "":
            checkpoint = torch.load(
                opt.checkpoint_bl, map_location=self.accelerator.device
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
        constraint = {'mask': masks.to(self.accelerator.device), 'value': state_true, 'cond': cond}
        state_true = state_true.to(self.accelerator.device)
        state_pred_list = [self.diffusion.predict_samples([state_true], constraint)
                           for _ in range(num_of_generation_per_window)]
        state_pred_list = [self.normalizer.unnormalize(state_pred.detach().cpu()) for state_pred in state_pred_list]
        return torch.stack(state_pred_list)










