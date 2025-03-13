# Define functions and run GaitDynamics.
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
from torch import nn
from torch import Tensor
import argparse
from typing import Any
import nimblephysics as nimble
import scipy.interpolate as interpo
from scipy.interpolate import interp1d
import random
from scipy.signal import filtfilt, butter
from accelerate import Accelerator, DistributedDataParallelKwargs
import os
from inspect import isfunction
from math import pi
from einops import rearrange, repeat
import math
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial
from accelerate.state import AcceleratorState


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

""" ============================ Start scaler.py ============================ """


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    # if we are fitting on 1D arrays, scale might be a scalar
    if constant_mask is None:
        # Detect near constant values to avoid dividing by a very small
        # value that could lead to surprising results and numerical
        # stability issues.
        constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

    if copy:
        # New array to avoid side-effects
        scale = scale.clone()
    scale[constant_mask] = 1.0
    return scale


class MinMaxScaler:
    _parameter_constraints: dict = {
        "feature_range": [tuple],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, feature_range=(-1, 1), *, copy=True, clip=True):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X):
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        data_min = torch.min(X, axis=0)[0]
        data_max = torch.max(X, axis=0)[0]

        self.n_samples_seen_ = X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        X *= self.scale_.to(X.device)
        X += self.min_.to(X.device)
        return X

    def inverse_transform(self, X):
        X -= self.min_.to(X.device)
        X /= self.scale_.to(X.device)
        return X


class Normalizer:
    def __init__(self, scaler, cols_to_normalize):
        self.scaler = scaler
        self.cols_to_normalize = cols_to_normalize

    def normalize(self, x):
        x = x.clone()
        x[:, self.cols_to_normalize] = self.scaler.transform(x[:, self.cols_to_normalize])
        return x

    def unnormalize(self, x):
        batch, seq, ch = x.shape
        x = x.clone()
        x = x.reshape(-1, ch)
        x[:, self.cols_to_normalize] = self.scaler.inverse_transform(x[:, self.cols_to_normalize])
        return x.reshape((batch, seq, ch))


""" ============================ End scaler.py ============================ """


""" ============================ Start consts.py ============================ """

JOINTS_3D_ALL = {
    'pelvis': ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation'],
    'hip_r': ['hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r'],
    'hip_l': ['hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l'],
    'lumbar': ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation'],
    'arm_r': ['arm_flex_r', 'arm_add_r', 'arm_rot_r'],
    'arm_l': ['arm_flex_l', 'arm_add_l', 'arm_rot_l']}

OSIM_DOF_ALL = [
    'pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'hip_flexion_r',
    'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
    'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l',
    'mtp_angle_l', 'lumbar_extension', 'lumbar_bending', 'lumbar_rotation', 'arm_flex_r', 'arm_add_r', 'arm_rot_r',
    'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r', 'arm_flex_l', 'arm_add_l', 'arm_rot_l',
    'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l']


KINETICS_ALL = [body + modality for body in ['calcn_r', 'calcn_l'] for modality in
                ['_force_vx', '_force_vy', '_force_vz', '_force_normed_cop_x', '_force_normed_cop_y', '_force_normed_cop_z']]

MODEL_STATES_COLUMN_NAMES_WITH_ARM = [
                                         'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'knee_angle_r', 'ankle_angle_r', 'subtalar_angle_r',
                                         'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'elbow_flex_r', 'pro_sup_r', 'elbow_flex_l', 'pro_sup_l'
                                     ] + KINETICS_ALL + [
                                         'pelvis_0', 'pelvis_1', 'pelvis_2', 'pelvis_3', 'pelvis_4', 'pelvis_5',
                                         'hip_r_0', 'hip_r_1', 'hip_r_2', 'hip_r_3', 'hip_r_4', 'hip_r_5',
                                         'hip_l_0', 'hip_l_1', 'hip_l_2', 'hip_l_3', 'hip_l_4', 'hip_l_5',
                                         'lumbar_0', 'lumbar_1', 'lumbar_2', 'lumbar_3', 'lumbar_4', 'lumbar_5',
                                         'arm_r_0', 'arm_r_1', 'arm_r_2', 'arm_r_3', 'arm_r_4', 'arm_r_5',        # only for with arm
                                         'arm_l_0', 'arm_l_1', 'arm_l_2', 'arm_l_3', 'arm_l_4', 'arm_l_5'        # only for with arm
                                     ]

MODEL_STATES_COLUMN_NAMES_NO_ARM = copy.deepcopy(MODEL_STATES_COLUMN_NAMES_WITH_ARM)
for name_ in ['elbow_flex_r', 'pro_sup_r', 'elbow_flex_l', 'pro_sup_l', 'arm_r_0', 'arm_r_1', 'arm_r_2', 'arm_r_3',
              'arm_r_4', 'arm_r_5', 'arm_l_0', 'arm_l_1', 'arm_l_2', 'arm_l_3', 'arm_l_4', 'arm_l_5']:
    MODEL_STATES_COLUMN_NAMES_NO_ARM.remove(name_)

FROZEN_DOFS = ['mtp_angle_r', 'mtp_angle_l',
               'wrist_flex_r', 'wrist_dev_r', 'wrist_flex_l', 'wrist_dev_l']

FULL_OSIM_DOF = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                 'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
                 'subtalar_angle_r', 'mtp_angle_r', 'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
                 'knee_angle_l', 'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l', 'lumbar_extension',
                 'lumbar_bending', 'lumbar_rotation']


""" ============================ End consts.py ============================ """


""" ============================ Start model.py ============================ """


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            model,
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

        for time, time_next in time_pairs:
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
    def inpaint_ddim_loop(self, shape, noise=None, constraint=None, return_diffusion=False, start_point=None):
        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.betas.device, self.n_timestep, 50, 0

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x = torch.randn(shape, device=device)
        cond = constraint["cond"].to(device)

        mask = constraint["mask"].to(device)  # batch x horizon x channels
        value = constraint["value"].to(device)  # batch x horizon x channels

        for time, time_next in time_pairs:
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

    def q_sample(self, x_start, t, noise=None):     # blend noise into state variables
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

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


class FillingBase:
    def fill_param(self, windows, diffusion_model_for_filling):
        return self.filling(windows, diffusion_model_for_filling, self.update_kinematics_and_masks_for_masking_column)

    def fill_temporal(self, windows, diffusion_model_for_filling, mask_original):
        self.mask_original = mask_original
        return self.filling(windows, diffusion_model_for_filling, self.update_kinematics_and_masks_for_masking_temporal)

    @staticmethod
    def update_kinematics_and_masks_for_masking_column(windows, samples, i_win, masks):
        unmasked_samples_in_temporal_dim = (masks.sum(axis=2)).bool()
        for j_win in range(len(samples)):
            windows[j_win+i_win].pose = samples[j_win]
            updated_mask = windows[j_win+i_win].mask
            updated_mask[unmasked_samples_in_temporal_dim[j_win], :] = 1
            updated_mask[:, opt.kinetic_diffusion_col_loc] = 0
            windows[j_win+i_win].mask = updated_mask
        return windows

    def update_kinematics_and_masks_for_masking_temporal(self, windows, samples, i_win, masks):
        for j_win in range(len(samples)):
            windows[j_win+i_win].pose = samples[j_win]
            windows[j_win+i_win].mask = self.mask_original[j_win+i_win]
        return windows

    @staticmethod
    def filling(windows, diffusion_model_for_filling, windows_update_func):
        raise NotImplementedError


class DiffusionFilling(FillingBase):
    @staticmethod
    def filling(windows, diffusion_model_for_filling, windows_update_func):
        windows = copy.deepcopy(windows)
        for i_win in range(0, len(windows), opt.batch_size_inference):
            state_true = torch.stack([win.pose for win in windows[i_win:i_win+opt.batch_size_inference]])
            masks = torch.stack([win.mask for win in windows[i_win:i_win+opt.batch_size_inference]])
            cond = torch.ones([6])

            constraint = {'mask': masks, 'value': state_true.clone(), 'cond': cond}
            shape = (state_true.shape[0], state_true.shape[1], state_true.shape[2])
            samples = (diffusion_model_for_filling.diffusion.inpaint_ddim_loop(shape, constraint=constraint))
            samples = state_true * masks + (1.0 - masks) * samples.to(state_true.device)
            # samples[:, :, opt.kinetic_diffusion_col_loc] = state_true[:, :, opt.kinetic_diffusion_col_loc]

            windows = windows_update_func(windows, samples, i_win, masks)
        return windows

    def __str__(self):
        return 'diffusion_filling'


def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


def exists(val):
    return val is not None


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatentation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert (
            rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = (
        t[..., :start_index],
        t[..., start_index:end_index],
        t[..., end_index:],
    )
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim,
            custom_freqs=None,
            freqs_for="lang",
            theta=10000,
            max_freq=10,
            num_freqs=1,
            learned_freq=False,
    ):
        super().__init__()
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (
                    theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.cache = dict()

        if learned_freq:
            self.freqs = nn.Parameter(freqs)
        else:
            self.register_buffer("freqs", freqs)

    def rotate_queries_or_keys(self, t, seq_dim=-2):
        device = t.device
        seq_len = t.shape[seq_dim]
        freqs = self.forward(
            lambda: torch.arange(seq_len, device=device), cache_key=seq_len
        )
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key=None):
        if exists(cache_key) and cache_key in self.cache:
            return self.cache[cache_key]

        if isfunction(t):
            t = t()

        freqs = self.freqs

        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if exists(cache_key):
            self.cache[cache_key] = freqs

        return freqs

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

    def loss_fun(self, output_pred, output_true):
        return F.mse_loss(output_pred, output_true, reduction='none')

    def end_to_end_prediction(self, x):
        input = x[0][:, :, self.input_col_loc]
        sequence = self.input_to_embedding(input)
        sequence = self.encoder_layers(sequence)
        output_pred = self.embedding_to_output(sequence)
        return output_pred

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


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward=2048,
            dropout=0.1,
            activation=F.relu,
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=True,
            device=None,
            dtype=None,
            rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
            self,
            tgt,
            memory,
            t,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t):
        for layer in self.stack:
            x = layer(x, cond, t)
        return x


class DanceDecoder(nn.Module):
    def __init__(
            self,
            nfeats: int,
            seq_len: int = 150,  # 5 seconds, 30 fps
            latent_dim: int = 256,
            ff_size: int = 1024,
            num_layers: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            # cond_feature_dim: int = 6,
            activation=F.gelu,
            use_rotary=True,
            **kwargs
    ) -> None:

        super().__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(        # decoder layers
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def guided_forward(self, x, cond_embed, time_cond, guidance_weight):
        return self.forward(x, cond_embed, time_cond)

    def __str__(self):
        return 'diffusion'

    # No conditioning version
    def forward(self, x: Tensor, cond_embed: Tensor, time_cond: Tensor, cond_drop_prob: float = 0.0):
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)
        t_hidden = self.time_mlp(time_cond)
        t = self.to_time_cond(t_hidden)
        output = self.seqTransDecoder(x, None, t)
        output = self.final_layer(output)
        return output


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
                opt.checkpoint, map_location=self.accelerator.device,
                weights_only=False
            )
            self.normalizer = checkpoint["normalizer"]

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

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)

        if opt.checkpoint != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

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


class BaselineModel:
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

        checkpoint = None
        if opt.checkpoint_bl != "":
            checkpoint = torch.load(
                opt.checkpoint_bl, map_location=self.accelerator.device,
                weights_only=False
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

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)


""" ============================ End model.py ============================ """


""" ============================ Start util.py ============================ """


def load_diffusion_model(opt):
    opt.checkpoint = opt.subject_data_path + '/GaitDynamics/example_usage/GaitDynamicsDiffusion.pt'
    model = MotionModel(opt)
    model_key = 'diffusion'
    return model, model_key


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def convertDfToGRFMot(df, out_folder, dt, max_time=None):
    numFrames = df.shape[0]
    for key in df.keys():
        if key == 'TimeStamp':
            continue

    out_file = open(out_folder, 'w')
    out_file.write('nColumns=9\n')
    out_file.write('nRows='+str(numFrames)+'\n')
    out_file.write('DataType=double\n')
    out_file.write('version=3\n')
    out_file.write('OpenSimVersion=4.1\n')
    out_file.write('endheader\n')
    out_file.write('time')

    plate_num = 2

    for i in range(1, 1 + plate_num):
        out_file.write('\t' + f'force{i}_vx')
        out_file.write('\t' + f'force{i}_vy')
        out_file.write('\t' + f'force{i}_vz')
        out_file.write('\t' + f'force{i}_px')
        out_file.write('\t' + f'force{i}_py')
        out_file.write('\t' + f'force{i}_pz')
        out_file.write('\t' + f'torque{i}_x')
        out_file.write('\t' + f'torque{i}_y')
        out_file.write('\t' + f'torque{i}_z')

    out_file.write('\n')
    for i in range(numFrames):
        out_file.write(str(round(dt * i, 5)))
        for side in ['r', 'l']:
            out_file.write('\t' + str(df[f'calcn_{side}_force_vx'][i]))
            out_file.write('\t' + str(df[f'calcn_{side}_force_vy'][i]))
            out_file.write('\t' + str(df[f'calcn_{side}_force_vz'][i]))
            out_file.write('\t' + str(df[f'calcn_{side}_force_normed_cop_x'][i]))
            out_file.write('\t' + str(df[f'calcn_{side}_force_normed_cop_y'][i]))
            out_file.write('\t' + str(df[f'calcn_{side}_force_normed_cop_z'][i]))
            out_file.write('\t' + str(0))
            out_file.write('\t' + str(0))
            out_file.write('\t' + str(0))
        out_file.write('\n')
    out_file.close()
    print('GRF file exported to ' + out_folder)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def euler_from_6v(q, convention="XYZ"):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    eul = matrix_to_euler_angles(mat, convention)
    return eul


def euler_to_6v(q, convention="XYZ"):
    assert q.shape[-1] == 3
    mat = euler_angles_to_matrix(q, convention)
    mat = matrix_to_rotation_6d(mat)
    return mat


def second_order_poly(coeff, x):
    y = coeff[...,0] * x**2 + coeff[...,1] * x + coeff[...,2]
    return y


def batch_identity(batch_shape, size):
    batch_identity = torch.eye(size)
    output_shape = batch_shape.copy()
    output_shape.append(size)
    output_shape.append(size)
    batch_identity_out = batch_identity.view(*(1,) * (len(output_shape) - batch_identity.ndim),*batch_identity.shape).expand(output_shape)
    return batch_identity_out.clone()


def get_knee_rotation_coefficients():

    knee_Z_rotation_function = np.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                          1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                         [0, 0.0126809, 0.0226969, 0.0296054, 0.0332049, 0.0335354, 0.0308779,
                                          0.0257548, 0.0189295, 0.011407, 0.00443314, -0.00050475, -0.0016782]]).T
    polyfit_knee_Z_rotation = np.polyfit(knee_Z_rotation_function[:,0], knee_Z_rotation_function[:,1], deg=2, full = True)
    coefficients_knee_Z_rotation = polyfit_knee_Z_rotation[0]

    knee_Y_rotation_function = np.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                          1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                         [0, 0.059461, 0.109399, 0.150618, 0.18392, 0.210107, 0.229983, 0.24435, 0.254012, 0.25977, 0.262428, 0.262788, 0.261654]]).T
    polyfit_knee_Y_rotation = np.polyfit(knee_Y_rotation_function[:, 0], knee_Y_rotation_function[:, 1], deg=2, full=True)
    coefficients_knee_Y_rotation = polyfit_knee_Y_rotation[0]

    knee_X_translation_function = np.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                             1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                            [0, 5.3e-05, 0.000188, 0.000378, 0.000597, 0.000825, 0.001045, 0.001247, 0.00142, 0.001558, 0.001661, 0.001728, 0.00176]]).T
    polyfit_knee_X_translation = np.polyfit(knee_X_translation_function[:, 0], knee_X_translation_function[:, 1], deg=2,
                                            full=True)
    coefficients_knee_X_translation = polyfit_knee_X_translation[0]


    knee_Y_translation_function = np.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                             1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                            [0, 0.000301, 0.000143, -0.000401, -0.001233, -0.002243, -0.003316, -0.004346, -0.005239, -0.005924, -0.006361, -0.006539, -0.00648]]).T
    polyfit_knee_Y_translation = np.polyfit(knee_Y_translation_function[:, 0], knee_Y_translation_function[:, 1], deg=2,
                                            full=True)
    coefficients_knee_Y_translation = polyfit_knee_Y_translation[0]

    knee_Z_translation_function = np.array([[0, 0.174533, 0.349066, 0.523599, 0.698132, 0.872665, 1.0472, 1.22173,
                                             1.39626, 1.5708, 1.74533, 1.91986, 2.0944],
                                            [0, 0.001055, 0.002061, 0.00289, 0.003447, 0.003676, 0.003559, 0.00311, 0.002373, 0.001418, 0.000329, -0.000805, -0.001898]]).T
    polyfit_knee_Z_translation = np.polyfit(knee_Z_translation_function[:, 0], knee_Z_translation_function[:, 1], deg=2,
                                            full=True)
    coefficients_knee_Z_translation = polyfit_knee_Z_translation[0]

    walker_knee_coefficients = np.stack((coefficients_knee_Y_rotation, coefficients_knee_Z_rotation, coefficients_knee_X_translation, coefficients_knee_Y_translation, coefficients_knee_Z_translation), axis=1)

    return walker_knee_coefficients


walker_knee_coefficients = get_knee_rotation_coefficients()
walker_knee_coefficients = torch.tensor(walker_knee_coefficients).to(torch.device('cuda:0'))         # Bugprone


def forward_kinematics(pose, offsets, with_arm=False):
    """
    Pose indices
    0-5: pelvis orientation + translation
    6-8: hip_r
    9: knee_r
    10: ankle_r
    11: subtalar_r
    12: mtp_r
    13-15: hip_l
    16: knee_l
    17: ankle_l
    18: subtalar_l
    19: mtp_l
    20-22: lumbar
    23-25: shoulder_r
    26: elbow_r
    27: radioulnar
    28-30: shoulder_l
    31: elbow_l
    32: radioulnar_l
    """
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)
    pose = pose.to(torch.device('cuda:0'))      # Error-prone
    if len(pose.shape) == 2:
        pose = pose[None, ...]
    if len(offsets.shape) == 3:
        offsets = offsets[None, ...]

    offsets = offsets.to(pose.device)
    offsets = offsets[:, None, ...]
    batch_shape = pose.shape[:-1]
    batch_shape_list = []
    for i in range(pose.dim()-1):
        batch_shape_list.append(int(batch_shape[i]))
    batch_shape = batch_shape_list
    if batch_shape == ():
        batch_shape = (1,)

    coefficients_knee_Y_rotation = walker_knee_coefficients[..., 0]
    coefficients_knee_Z_rotation = walker_knee_coefficients[..., 1]
    coefficients_knee_X_translation = walker_knee_coefficients[..., 2]
    coefficients_knee_Y_translation = walker_knee_coefficients[..., 3]
    coefficients_knee_Z_translation = walker_knee_coefficients[..., 4]

    knee_r_Y_rot = second_order_poly(coefficients_knee_Y_rotation, pose[..., 9])
    knee_r_Z_rot = second_order_poly(coefficients_knee_Z_rotation, pose[..., 9])
    knee_r_X_trans = second_order_poly(coefficients_knee_X_translation, pose[..., 9])
    knee_r_Y_trans = second_order_poly(coefficients_knee_Y_translation, pose[..., 9])
    knee_r_Z_trans = second_order_poly(coefficients_knee_Z_translation, pose[..., 9])

    knee_l_Y_rot = second_order_poly(coefficients_knee_Y_rotation, pose[..., 16])
    knee_l_Z_rot = second_order_poly(coefficients_knee_Z_rotation, pose[..., 16])
    knee_l_X_trans = second_order_poly(coefficients_knee_X_translation, pose[..., 16])
    knee_l_Y_trans = second_order_poly(coefficients_knee_Y_translation, pose[..., 16])
    knee_l_Z_trans = second_order_poly(coefficients_knee_Z_translation, pose[..., 16])

    # Pelvis
    pelvis_transform = batch_identity(batch_shape, 4).to(pose.device)
    pelvis_transform[..., :3, :3] = euler_angles_to_matrix(pose[..., 0:3], 'ZXY')
    pelvis_transform[..., :3, 3] = pose[..., 3:6].clone().detach()

    # Get offsets (model and model scaling dependent)
    offset_hip_pelvis_r = offsets[..., 0]
    femur_offset_in_hip_r = offsets[..., 1]
    knee_offset_in_femur_r = offsets[..., 2]
    tibia_offset_in_knee_r = offsets[..., 3]
    ankle_offset_in_tibia_r = offsets[..., 4]
    talus_offset_in_ankle_r = offsets[..., 5]
    subtalar_offset_in_talus_r = offsets[..., 6]
    calcaneus_offset_in_subtalar_r = offsets[..., 7]
    mtp_offset_in_calcaneus_r = offsets[..., 8]
    offset_hip_pelvis_l = offsets[..., 9]
    femur_offset_in_hip_l = offsets[..., 10]
    knee_offset_in_femur_l = offsets[..., 11]
    tibia_offset_in_knee_l = offsets[..., 12]
    ankle_offset_in_tibia_l = offsets[..., 13]
    talus_offset_in_ankle_l = offsets[..., 14]
    subtalar_offset_in_talus_l = offsets[..., 15]
    calcaneus_offset_in_subtalar_l = offsets[..., 16]
    mtp_offset_in_calcaneus_l = offsets[..., 17]
    lumbar_offset_in_pelvis = offsets[..., 18]
    torso_offset_in_lumbar = offsets[..., 19]
    if with_arm:
        shoulder_offset_in_torso_r = offsets[..., 20]
        humerus_offset_in_shoulder_r = offsets[..., 21]
        elbow_offset_in_humerus_r = offsets[..., 22]
        ulna_offset_in_elbow_r = offsets[..., 23]
        radioulnar_offset_in_radius_r = offsets[..., 24]
        radius_offset_in_radioulnar_r = offsets[..., 25]
        wrist_offset_in_radius_r = offsets[..., 26]
        hand_offset_in_wrist_r = offsets[..., 27]
        shoulder_offset_in_torso_l = offsets[..., 28]
        humerus_offset_in_shoulder_l = offsets[..., 29]
        elbow_offset_in_humerus_l = offsets[..., 30]
        ulna_offset_in_elbow_l = offsets[..., 31]
        radioulnar_offset_in_radius_l = offsets[..., 32]
        radius_offset_in_radioulnar_l = offsets[..., 33]
        wrist_offset_in_radius_l = offsets[..., 34]
        hand_offset_in_wrist_l = offsets[..., 35]

    # Coordinates to transformation matrix
    hip_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    knee_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    ankle_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    subtalar_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    mtp_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
    hip_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    knee_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    ankle_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    subtalar_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    mtp_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
    lumbar_coordinates_transform = batch_identity(batch_shape, 4).to(pose.device)
    if with_arm:
        shoulder_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        elbow_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        radioulnar_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        wrist_coordinates_transform_r = batch_identity(batch_shape, 4).to(pose.device)
        shoulder_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
        elbow_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
        radioulnar_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)
        wrist_coordinates_transform_l = batch_identity(batch_shape, 4).to(pose.device)

    # Knee axis translation
    knee_coordinates_transform_r[..., :3, -1] = torch.stack((knee_r_X_trans, knee_r_Y_trans, knee_r_Z_trans), dim=-1)
    knee_coordinates_transform_l[..., :3, -1] = torch.stack((knee_l_X_trans, knee_l_Y_trans, -knee_l_Z_trans), dim=-1)

    # Joint rotations
    zero_2_shape = batch_shape.copy()
    zero_2_shape.append(2)
    zero_2 = torch.zeros(tuple(zero_2_shape), device=pose.device)
    zero_3_shape = batch_shape.copy()
    zero_3_shape.append(3)
    zero_3 = torch.zeros(tuple(zero_3_shape), device=pose.device)
    hip_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix((pose[..., 6:9]), 'ZXY')
    knee_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
        torch.stack((pose[..., 9], knee_r_Y_rot, knee_r_Z_rot), dim=-1), 'XYZ')
    ankle_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
        torch.cat((pose[..., 10:11], zero_2), dim=-1), 'ZXY')
    subtalar_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
        torch.cat((pose[..., 11:12], zero_2), dim=-1), 'ZXY')
    mtp_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
        torch.cat((pose[..., 12:13], zero_2), dim=-1), 'ZXY')
    hip_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
        torch.cat(([pose[..., 13:14], -pose[..., 14:15], -pose[..., 15:16]]), dim=-1), 'ZXY')
    knee_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
        torch.stack((-pose[..., 16], -knee_l_Y_rot, knee_l_Z_rot), dim=-1), 'XYZ')
    ankle_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
        torch.cat((pose[..., 17:18], zero_2), dim=-1), 'ZXY')
    subtalar_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
        torch.cat((pose[..., 18:19], zero_2), dim=-1), 'ZXY')
    mtp_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
        torch.cat((pose[..., 19:20], zero_2), dim=-1), 'ZXY')
    lumbar_coordinates_transform[..., :3, :3] = euler_angles_to_matrix(
        torch.cat((pose[..., 20:21], pose[..., 21:22], pose[..., 22:23]), dim=-1), 'ZXY')
    if with_arm:
        shoulder_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
            torch.cat((pose[..., 23:24], pose[..., 24:25], pose[..., 25:26]), dim=-1), 'ZXY')
        elbow_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
            torch.cat((pose[..., 26:27], zero_2), dim=-1), 'ZXY')
        radioulnar_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
            torch.cat((pose[..., 27:28], zero_2), dim=-1), 'ZXY')
        wrist_coordinates_transform_r[..., :3, :3] = euler_angles_to_matrix(
            zero_3, 'ZXY')
        shoulder_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
            torch.cat((pose[..., 28:29], -pose[..., 29:30], -pose[..., 30:31]), dim=-1), 'ZXY')
        elbow_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
            torch.cat((pose[..., 31:32], zero_2), dim=-1), 'ZXY')
        radioulnar_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
            torch.cat((pose[..., 32:33], zero_2), dim=-1), 'ZXY')
        wrist_coordinates_transform_l[..., :3, :3] = euler_angles_to_matrix(
            zero_3, 'ZXY')

    # Forward kinematics for the lower body
    hip_transform_r = torch.matmul(torch.matmul(pelvis_transform, offset_hip_pelvis_r), hip_coordinates_transform_r)
    femur_transform_r = torch.matmul(hip_transform_r, femur_offset_in_hip_r)
    knee_transform_r = torch.matmul(torch.matmul(femur_transform_r, knee_offset_in_femur_r),
                                    knee_coordinates_transform_r)
    tibia_transform_r = torch.matmul(knee_transform_r, tibia_offset_in_knee_r)
    ankle_transform_r = torch.matmul(torch.matmul(tibia_transform_r, ankle_offset_in_tibia_r),
                                     ankle_coordinates_transform_r)
    talus_transform_r = torch.matmul(ankle_transform_r, talus_offset_in_ankle_r)
    subtalar_transform_r = torch.matmul(torch.matmul(talus_transform_r, subtalar_offset_in_talus_r),
                                        subtalar_coordinates_transform_r)
    calcaneus_transform_r = torch.matmul(subtalar_transform_r, calcaneus_offset_in_subtalar_r)
    mtp_offset_transform_r = torch.matmul(torch.matmul(calcaneus_transform_r, mtp_offset_in_calcaneus_r),
                                          mtp_coordinates_transform_r)

    hip_transform_l = torch.matmul(torch.matmul(pelvis_transform, offset_hip_pelvis_l), hip_coordinates_transform_l)
    femur_transform_l = torch.matmul(hip_transform_l, femur_offset_in_hip_l)

    knee_transform_l = torch.matmul(torch.matmul(femur_transform_l, knee_offset_in_femur_l),
                                    knee_coordinates_transform_l)
    tibia_transform_l = torch.matmul(knee_transform_l, tibia_offset_in_knee_l)

    ankle_transform_l = torch.matmul(torch.matmul(tibia_transform_l, ankle_offset_in_tibia_l),
                                     ankle_coordinates_transform_l)
    talus_transform_l = torch.matmul(ankle_transform_l, talus_offset_in_ankle_l)

    subtalar_transform_l = torch.matmul(torch.matmul(talus_transform_l, subtalar_offset_in_talus_l),
                                        subtalar_coordinates_transform_l)
    calcaneus_transform_l = torch.matmul(subtalar_transform_l, calcaneus_offset_in_subtalar_l)

    mtp_offset_transform_l = torch.matmul(torch.matmul(calcaneus_transform_l, mtp_offset_in_calcaneus_l),
                                          mtp_coordinates_transform_l)

    # Forward kinematics for the upper body
    lumbar_transform = torch.matmul(torch.matmul(pelvis_transform, lumbar_offset_in_pelvis), lumbar_coordinates_transform)
    torso_transform = torch.matmul(lumbar_transform, torso_offset_in_lumbar)
    if with_arm:
        shoulder_transform_r = torch.matmul(torch.matmul(torso_transform, shoulder_offset_in_torso_r), shoulder_coordinates_transform_r)
        humerus_transform_r = torch.matmul(shoulder_transform_r, humerus_offset_in_shoulder_r)
        elbow_transform_r = torch.matmul(torch.matmul(humerus_transform_r, elbow_offset_in_humerus_r), elbow_coordinates_transform_r)
        ulna_transform_r = torch.matmul(elbow_transform_r, ulna_offset_in_elbow_r)
        radioulnar_transform_r = torch.matmul(torch.matmul(ulna_transform_r, radioulnar_offset_in_radius_r), radioulnar_coordinates_transform_r)
        radius_transform_r = torch.matmul(radioulnar_transform_r, radius_offset_in_radioulnar_r)
        wrist_transform_r = torch.matmul(torch.matmul(ulna_transform_r, wrist_offset_in_radius_r), wrist_coordinates_transform_r)
        hand_transform_r = torch.matmul(wrist_transform_r, hand_offset_in_wrist_r)
        shoulder_transform_l = torch.matmul(torch.matmul(torso_transform, shoulder_offset_in_torso_l), shoulder_coordinates_transform_l)
        humerus_transform_l = torch.matmul(shoulder_transform_l, humerus_offset_in_shoulder_l)
        elbow_transform_l = torch.matmul(torch.matmul(humerus_transform_l, elbow_offset_in_humerus_l), elbow_coordinates_transform_l)
        ulna_transform_l = torch.matmul(elbow_transform_l, ulna_offset_in_elbow_l)
        radioulnar_transform_l = torch.matmul(torch.matmul(ulna_transform_l, radioulnar_offset_in_radius_l), radioulnar_coordinates_transform_l)
        radius_transform_l = torch.matmul(radioulnar_transform_l, radius_offset_in_radioulnar_l)
        wrist_transform_l = torch.matmul(torch.matmul(ulna_transform_l, wrist_offset_in_radius_l), wrist_coordinates_transform_l)
        hand_transform_l = torch.matmul(wrist_transform_l, hand_offset_in_wrist_l)

    joint_locations = torch.stack((pelvis_transform[..., :3, 3],
                                   hip_transform_r[..., :3, 3], knee_transform_r[..., :3, 3],
                                   ankle_transform_r[..., :3, 3], calcaneus_transform_r[..., :3, 3],
                                   mtp_offset_transform_r[..., :3, 3],
                                   hip_transform_l[..., :3, 3], knee_transform_l[..., :3, 3],
                                   ankle_transform_l[..., :3, 3], calcaneus_transform_l[..., :3, 3],
                                   mtp_offset_transform_l[..., :3, 3]))
    if with_arm:
        joint_locations = torch.stack((*[joint_locations[i] for i in range(joint_locations.shape[0])],
                                       lumbar_transform[..., :3, 3],
                                       shoulder_transform_r[..., :3, 3],
                                       elbow_transform_r[..., :3, 3], wrist_transform_r[..., :3, 3],
                                       shoulder_transform_l[..., :3, 3],
                                       elbow_transform_l[..., :3, 3], wrist_transform_l[..., :3, 3]))
    if torch.isnan(joint_locations).any():
        print('NAN in joint locations')

    foot_locations = torch.stack((calcaneus_transform_r[..., :3, 3], mtp_offset_transform_r[..., :3, 3],
                                  calcaneus_transform_l[..., :3, 3], mtp_offset_transform_l[..., :3, 3]))

    segment_orientations = torch.stack((pelvis_transform[..., :3, :3],
                                        femur_transform_r[..., :3, :3], tibia_transform_r[..., :3, :3],
                                        talus_transform_r[..., :3, :3], calcaneus_transform_r[..., :3, :3],
                                        femur_transform_l[..., :3, :3], tibia_transform_l[..., :3, :3],
                                        talus_transform_l[..., :3, :3], calcaneus_transform_l[..., :3, :3],
                                        torso_transform[..., :3, :3]))
    if with_arm:
        segment_orientations = torch.stack((*[segment_orientations[i] for i in range(segment_orientations.shape[0])],
                                            humerus_transform_r[..., :3, :3], ulna_transform_r[..., :3, :3],
                                            radius_transform_r[..., :3, :3],
                                            humerus_transform_l[..., :3, :3], ulna_transform_l[..., :3, :3],
                                            radius_transform_l[..., :3, :3]))

    joint_names = ['pelvis', 'hip_r', 'knee_r', 'ankle_r', 'calcn_r', 'mtp_r',
                   'hip_l', 'knee_l', 'ankle_l', 'calcn_l', 'mtp_l']
    return foot_locations, joint_locations, joint_names, segment_orientations


def get_model_offsets(skeleton, with_arm=False):
    pelvis = skeleton.getBodyNode(0)
    hip_r_joint = pelvis.getChildJoint(0)
    femur_r = hip_r_joint.getChildBodyNode()
    knee_r_joint = femur_r.getChildJoint(0)
    tibia_r = knee_r_joint.getChildBodyNode()
    ankle_r_joint = tibia_r.getChildJoint(0)
    talus_r = ankle_r_joint.getChildBodyNode()
    subtalar_r_joint = talus_r.getChildJoint(0)
    calcn_r = subtalar_r_joint.getChildBodyNode()
    mtp_r_joint = calcn_r.getChildJoint(0)

    hip_l_joint = pelvis.getChildJoint(1)
    femur_l = hip_l_joint.getChildBodyNode()
    knee_l_joint = femur_l.getChildJoint(0)
    tibia_l = knee_l_joint.getChildBodyNode()
    ankle_l_joint = tibia_l.getChildJoint(0)
    talus_l = ankle_l_joint.getChildBodyNode()
    subtalar_l_joint = talus_l.getChildJoint(0)
    calcn_l = subtalar_l_joint.getChildBodyNode()
    mtp_l_joint = calcn_l.getChildJoint(0)

    lumbar_joint = pelvis.getChildJoint(2)
    torso = lumbar_joint.getChildBodyNode()

    if with_arm:
        shoulder_r_joint = torso.getChildJoint(0)
        humerus_r = shoulder_r_joint.getChildBodyNode()
        elbow_r_joint = humerus_r.getChildJoint(0)
        ulna_r = elbow_r_joint.getChildBodyNode()
        radioulnar_r_joint = ulna_r.getChildJoint(0)
        radius_r = radioulnar_r_joint.getChildBodyNode()
        wrist_r_joint = radius_r.getChildJoint(0)
        hand_r = wrist_r_joint.getChildBodyNode()

        shoulder_l_joint = torso.getChildJoint(1)
        humerus_l = shoulder_l_joint.getChildBodyNode()
        elbow_l_joint = humerus_l.getChildJoint(0)
        ulna_l = elbow_l_joint.getChildBodyNode()
        radioulnar_l_joint = ulna_l.getChildJoint(0)
        radius_l = radioulnar_l_joint.getChildBodyNode()
        wrist_l_joint = radius_l.getChildJoint(0)
        hand_l = wrist_l_joint.getChildBodyNode()

    # hip offset
    hip_offset_r = torch.eye(4)
    hip_offset_r[:3, :3] = torch.tensor(hip_r_joint.getTransformFromParentBodyNode().rotation())
    hip_offset_r[:3, 3] = torch.tensor(hip_r_joint.getTransformFromParentBodyNode().translation())
    hip_offset_l = torch.eye(4)
    hip_offset_l[:3, :3] = torch.tensor(hip_l_joint.getTransformFromParentBodyNode().rotation())
    hip_offset_l[:3, 3] = torch.tensor(hip_l_joint.getTransformFromParentBodyNode().translation())

    # femur offset
    femur_offset_to_knee_in_femur_r = -torch.tensor(hip_r_joint.getTransformFromChildBodyNode().translation())
    femur_rotation_to_knee_in_femur_r = torch.inverse(torch.tensor(hip_r_joint.getTransformFromChildBodyNode().rotation()))
    femur_offset_rotation_r = torch.eye(4)
    femur_offset_rotation_r[:3, :3] = femur_rotation_to_knee_in_femur_r
    femur_offset_translation_r = torch.eye(4)
    femur_offset_translation_r[:3, 3] = femur_offset_to_knee_in_femur_r
    femur_offset_r = torch.matmul(femur_offset_rotation_r, femur_offset_translation_r)

    femur_offset_to_knee_in_femur_l = -torch.tensor(hip_l_joint.getTransformFromChildBodyNode().translation())
    femur_rotation_to_knee_in_femur_l = torch.inverse(torch.tensor(hip_l_joint.getTransformFromChildBodyNode().rotation()))
    femur_offset_rotation_l = torch.eye(4)
    femur_offset_rotation_l[:3, :3] = femur_rotation_to_knee_in_femur_l
    femur_offset_translation_l = torch.eye(4)
    femur_offset_translation_l[:3, 3] = femur_offset_to_knee_in_femur_l
    femur_offset_l = torch.matmul(femur_offset_rotation_l, femur_offset_translation_l)

    # knee offset
    knee_offset_r = torch.eye(4)
    knee_offset_r[:3, :3] = torch.tensor(knee_r_joint.getTransformFromParentBodyNode().rotation())
    knee_offset_r[:3, 3] = torch.tensor(knee_r_joint.getTransformFromParentBodyNode().translation())

    knee_offset_l = torch.eye(4)
    knee_offset_l[:3, :3] = torch.tensor(knee_l_joint.getTransformFromParentBodyNode().rotation())
    knee_offset_l[:3, 3] = torch.tensor(knee_l_joint.getTransformFromParentBodyNode().translation())

    # tibia offset
    tibia_offset_to_knee_in_tibia_r = -torch.tensor(knee_r_joint.getTransformFromChildBodyNode().translation())
    tibia_rotation_to_knee_in_tibia_r = torch.inverse(torch.tensor(knee_r_joint.getTransformFromChildBodyNode().rotation()))
    tibia_offset_rotation_r = torch.eye(4)
    tibia_offset_rotation_r[:3, :3] = tibia_rotation_to_knee_in_tibia_r
    tibia_offset_translation_r = torch.eye(4)
    tibia_offset_translation_r[:3, 3] = tibia_offset_to_knee_in_tibia_r
    tibia_offset_r = torch.matmul(tibia_offset_rotation_r, tibia_offset_translation_r)

    tibia_offset_to_knee_in_tibia_l = -torch.tensor(knee_l_joint.getTransformFromChildBodyNode().translation())
    tibia_rotation_to_knee_in_tibia_l = torch.inverse(torch.tensor(knee_l_joint.getTransformFromChildBodyNode().rotation()))
    tibia_offset_rotation_l = torch.eye(4)
    tibia_offset_rotation_l[:3, :3] = tibia_rotation_to_knee_in_tibia_l
    tibia_offset_translation_l = torch.eye(4)
    tibia_offset_translation_l[:3, 3] = tibia_offset_to_knee_in_tibia_l
    tibia_offset_l = torch.matmul(tibia_offset_rotation_l, tibia_offset_translation_l)

    # ankle offset
    ankle_offset_r = torch.eye(4)
    ankle_offset_r[:3,:3] = torch.tensor(ankle_r_joint.getTransformFromParentBodyNode().rotation())
    ankle_offset_r[:3, 3] = torch.tensor(ankle_r_joint.getTransformFromParentBodyNode().translation())

    ankle_offset_l = torch.eye(4)
    ankle_offset_l[:3, :3] = torch.tensor(ankle_l_joint.getTransformFromParentBodyNode().rotation())
    ankle_offset_l[:3, 3] = torch.tensor(ankle_l_joint.getTransformFromParentBodyNode().translation())

    # talus offset
    talus_offset_to_ankle_in_talus_r = -torch.tensor(ankle_r_joint.getTransformFromChildBodyNode().translation())
    talus_rotation_to_ankle_in_talus_r = torch.inverse(torch.tensor(ankle_r_joint.getTransformFromChildBodyNode().rotation()))
    talus_offset_rotation_r = torch.eye(4)
    talus_offset_rotation_r[:3, :3] = talus_rotation_to_ankle_in_talus_r
    talus_offset_translation_r = torch.eye(4)
    talus_offset_translation_r[:3, 3] = talus_offset_to_ankle_in_talus_r
    talus_offset_r = torch.matmul(talus_offset_rotation_r, talus_offset_translation_r)

    talus_offset_to_ankle_in_talus_l = -torch.tensor(ankle_l_joint.getTransformFromChildBodyNode().translation())
    talus_rotation_to_ankle_in_talus_l = torch.inverse(torch.tensor(ankle_l_joint.getTransformFromChildBodyNode().rotation()))
    talus_offset_rotation_l = torch.eye(4)
    talus_offset_rotation_l[:3, :3] = talus_rotation_to_ankle_in_talus_l
    talus_offset_translation_l = torch.eye(4)
    talus_offset_translation_l[:3, 3] = talus_offset_to_ankle_in_talus_l
    talus_offset_l = torch.matmul(talus_offset_rotation_l, talus_offset_translation_l)

    # subtalar offset
    subtalar_offset_r = torch.eye(4)
    subtalar_offset_r[:3,:3] = torch.tensor(subtalar_r_joint.getTransformFromParentBodyNode().rotation())
    subtalar_offset_r[:3, 3] = torch.tensor(subtalar_r_joint.getTransformFromParentBodyNode().translation())

    subtalar_offset_l = torch.eye(4)
    subtalar_offset_l[:3, :3] = torch.tensor(subtalar_l_joint.getTransformFromParentBodyNode().rotation())
    subtalar_offset_l[:3, 3] = torch.tensor(subtalar_l_joint.getTransformFromParentBodyNode().translation())

    # calcaneus offset
    calcaneus_offset_to_subtalar_in_calcaneus_r = -torch.tensor(subtalar_r_joint.getTransformFromChildBodyNode().translation())
    calcaneus_rotation_to_subtalar_in_calcaneus_r = torch.inverse(torch.tensor(subtalar_r_joint.getTransformFromChildBodyNode().rotation()))
    calcaneus_offset_rotation_r = torch.eye(4)
    calcaneus_offset_rotation_r[:3, :3] = calcaneus_rotation_to_subtalar_in_calcaneus_r
    calcaneus_offset_translation_r = torch.eye(4)
    calcaneus_offset_translation_r[:3, 3] = calcaneus_offset_to_subtalar_in_calcaneus_r
    calcaneus_offset_r = torch.matmul(calcaneus_offset_rotation_r, calcaneus_offset_translation_r)

    calcaneus_offset_to_subtalar_in_calcaneus_l = -torch.tensor(subtalar_l_joint.getTransformFromChildBodyNode().translation())
    calcaneus_rotation_to_subtalar_in_calcaneus_l = torch.inverse(torch.tensor(subtalar_l_joint.getTransformFromChildBodyNode().rotation()))
    calcaneus_offset_rotation_l = torch.eye(4)
    calcaneus_offset_rotation_l[:3, :3] = calcaneus_rotation_to_subtalar_in_calcaneus_l
    calcaneus_offset_translation_l = torch.eye(4)
    calcaneus_offset_translation_l[:3, 3] = calcaneus_offset_to_subtalar_in_calcaneus_l
    calcaneus_offset_l = torch.matmul(calcaneus_offset_rotation_l, calcaneus_offset_translation_l)

    # mtp offset
    mtp_offset_r = torch.eye(4)
    mtp_offset_r[:3, :3] = torch.tensor(mtp_r_joint.getTransformFromParentBodyNode().rotation())
    mtp_offset_r[:3, 3] = torch.tensor(mtp_r_joint.getTransformFromParentBodyNode().translation())

    mtp_offset_l = torch.eye(4)
    mtp_offset_l[:3, :3] = torch.tensor(mtp_l_joint.getTransformFromParentBodyNode().rotation())
    mtp_offset_l[:3, 3] = torch.tensor(mtp_l_joint.getTransformFromParentBodyNode().translation())

    # toes offset
    toes_offset_to_mtp_in_toes_r = -torch.tensor(mtp_r_joint.getTransformFromChildBodyNode().translation())
    toes_rotation_to_mtp_in_toes_r = torch.inverse(torch.tensor(mtp_r_joint.getTransformFromChildBodyNode().rotation()))
    toes_offset_rotation_r = torch.eye(4)
    toes_offset_rotation_r[:3, :3] = toes_rotation_to_mtp_in_toes_r
    toes_offset_translation_r = torch.eye(4)
    toes_offset_translation_r[:3, 3] = toes_offset_to_mtp_in_toes_r
    toes_offset_r = torch.matmul(toes_offset_rotation_r, toes_offset_translation_r)

    toes_offset_to_mtp_in_toes_l = -torch.tensor(mtp_l_joint.getTransformFromChildBodyNode().translation())
    toes_rotation_to_mtp_in_toes_l = torch.inverse(torch.tensor(mtp_l_joint.getTransformFromChildBodyNode().rotation()))
    toes_offset_rotation_l = torch.eye(4)
    toes_offset_rotation_l[:3, :3] = toes_rotation_to_mtp_in_toes_l
    toes_offset_translation_l = torch.eye(4)
    toes_offset_translation_l[:3, 3] = toes_offset_to_mtp_in_toes_l
    toes_offset_l = torch.matmul(toes_offset_rotation_l, toes_offset_translation_l)


    # lumbar offset
    lumbar_offset = torch.eye(4)
    lumbar_offset[:3, :3] = torch.tensor(lumbar_joint.getTransformFromParentBodyNode().rotation())
    lumbar_offset[:3, 3] = torch.tensor(lumbar_joint.getTransformFromParentBodyNode().translation())

    # torso offset
    torso_offset_to_lumbar_in_torso = -torch.tensor(lumbar_joint.getTransformFromChildBodyNode().translation())
    torso_offset_rotation_to_lumbar_in_torso = torch.inverse(torch.tensor(lumbar_joint.getTransformFromChildBodyNode().rotation()))
    torso_offset_rotation = torch.eye(4)
    torso_offset_rotation[:3, :3] = torso_offset_rotation_to_lumbar_in_torso
    torso_offset_translation = torch.eye(4)
    torso_offset_translation[:3, 3] = torso_offset_to_lumbar_in_torso
    torso_offset = torch.matmul(torso_offset_rotation, torso_offset_translation)

    if with_arm:
        # shoulder offset
        shoulder_offset_r = torch.eye(4)
        shoulder_offset_r[:3, :3] = torch.tensor(shoulder_r_joint.getTransformFromParentBodyNode().rotation())
        shoulder_offset_r[:3, 3] = torch.tensor(shoulder_r_joint.getTransformFromParentBodyNode().translation())

        shoulder_offset_l = torch.eye(4)
        shoulder_offset_l[:3, :3] = torch.tensor(shoulder_l_joint.getTransformFromParentBodyNode().rotation())
        shoulder_offset_l[:3, 3] = torch.tensor(shoulder_l_joint.getTransformFromParentBodyNode().translation())

        # humerus offset
        humerus_offset_to_shoulder_in_humerus_r = -torch.tensor(shoulder_r_joint.getTransformFromChildBodyNode().translation())
        humerus_offset_rotation_to_shoulder_in_humerus_r = torch.inverse(torch.tensor(shoulder_r_joint.getTransformFromChildBodyNode().rotation()))
        humerus_offset_rotation_r = torch.eye(4)
        humerus_offset_rotation_r[:3, :3] = humerus_offset_rotation_to_shoulder_in_humerus_r
        humerus_offset_translation_r = torch.eye(4)
        humerus_offset_translation_r[:3, 3] = humerus_offset_to_shoulder_in_humerus_r
        humerus_offset_r = torch.matmul(humerus_offset_rotation_r, humerus_offset_translation_r)

        humerus_offset_to_shoulder_in_humerus_l = -torch.tensor(shoulder_l_joint.getTransformFromChildBodyNode().translation())
        humerus_offset_rotation_to_shoulder_in_humerus_l = torch.inverse(torch.tensor(shoulder_l_joint.getTransformFromChildBodyNode().rotation()))
        humerus_offset_rotation_l = torch.eye(4)
        humerus_offset_rotation_l[:3, :3] = humerus_offset_rotation_to_shoulder_in_humerus_l
        humerus_offset_translation_l = torch.eye(4)
        humerus_offset_translation_l[:3, 3] = humerus_offset_to_shoulder_in_humerus_l
        humerus_offset_l = torch.matmul(humerus_offset_rotation_l, humerus_offset_translation_l)

        # elbow offset
        elbow_offset_r = torch.eye(4)
        elbow_offset_r[:3, :3] = torch.tensor(elbow_r_joint.getTransformFromParentBodyNode().rotation())
        elbow_offset_r[:3, 3] = torch.tensor(elbow_r_joint.getTransformFromParentBodyNode().translation())

        elbow_offset_l = torch.eye(4)
        elbow_offset_l[:3, :3] = torch.tensor(elbow_l_joint.getTransformFromParentBodyNode().rotation())
        elbow_offset_l[:3, 3] = torch.tensor(elbow_l_joint.getTransformFromParentBodyNode().translation())

        # ulna offset
        ulna_offset_to_elbow_in_ulna_r = -torch.tensor(elbow_r_joint.getTransformFromChildBodyNode().translation())
        ulna_offset_rotation_to_elbow_in_ulna_r = torch.inverse(torch.tensor(elbow_r_joint.getTransformFromChildBodyNode().rotation()))
        ulna_offset_rotation_r = torch.eye(4)
        ulna_offset_rotation_r[:3, :3] = ulna_offset_rotation_to_elbow_in_ulna_r
        ulna_offset_translation_r = torch.eye(4)
        ulna_offset_translation_r[:3, 3] = ulna_offset_to_elbow_in_ulna_r
        ulna_offset_r = torch.matmul(ulna_offset_rotation_r, ulna_offset_translation_r)

        ulna_offset_to_elbow_in_ulna_l = -torch.tensor(elbow_l_joint.getTransformFromChildBodyNode().translation())
        ulna_offset_rotation_to_elbow_in_ulna_l = torch.inverse(torch.tensor(elbow_l_joint.getTransformFromChildBodyNode().rotation()))
        ulna_offset_rotation_l = torch.eye(4)
        ulna_offset_rotation_l[:3, :3] = ulna_offset_rotation_to_elbow_in_ulna_l
        ulna_offset_translation_l = torch.eye(4)
        ulna_offset_translation_l[:3, 3] = ulna_offset_to_elbow_in_ulna_l
        ulna_offset_l = torch.matmul(ulna_offset_rotation_l, ulna_offset_translation_l)

        # radioulnar offset
        radioulnar_offset_r = torch.eye(4)
        radioulnar_offset_r[:3, :3] = torch.tensor(radioulnar_r_joint.getTransformFromParentBodyNode().rotation())
        radioulnar_offset_r[:3, 3] = torch.tensor(radioulnar_r_joint.getTransformFromParentBodyNode().translation())

        radioulnar_offset_l = torch.eye(4)
        radioulnar_offset_l[:3, :3] = torch.tensor(radioulnar_l_joint.getTransformFromParentBodyNode().rotation())
        radioulnar_offset_l[:3, 3] = torch.tensor(radioulnar_l_joint.getTransformFromParentBodyNode().translation())

        # radius offset
        radius_offset_to_radioulnar_in_radius_r = -torch.tensor(radioulnar_r_joint.getTransformFromChildBodyNode().translation())
        radius_offset_rotation_to_radioulnar_in_radius_r = torch.inverse(torch.tensor(radioulnar_r_joint.getTransformFromChildBodyNode().rotation()))
        radius_offset_rotation_r = torch.eye(4)
        radius_offset_rotation_r[:3, :3] = radius_offset_rotation_to_radioulnar_in_radius_r
        radius_offset_translation_r = torch.eye(4)
        radius_offset_translation_r[:3, 3] = radius_offset_to_radioulnar_in_radius_r
        radius_offset_r = torch.matmul(radius_offset_rotation_r, radius_offset_translation_r)

        radius_offset_to_radioulnar_in_radius_l = -torch.tensor(radioulnar_l_joint.getTransformFromChildBodyNode().translation())
        radius_offset_rotation_to_radioulnar_in_radius_l = torch.inverse(torch.tensor(radioulnar_l_joint.getTransformFromChildBodyNode().rotation()))
        radius_offset_rotation_l = torch.eye(4)
        radius_offset_rotation_l[:3, :3] = radius_offset_rotation_to_radioulnar_in_radius_l
        radius_offset_translation_l = torch.eye(4)
        radius_offset_translation_l[:3, 3] = radius_offset_to_radioulnar_in_radius_l
        radius_offset_l = torch.matmul(radius_offset_rotation_l, radius_offset_translation_l)

        # wrist offset
        wrist_offset_r = torch.eye(4)
        wrist_offset_r[:3, :3] = torch.tensor(wrist_r_joint.getTransformFromParentBodyNode().rotation())
        wrist_offset_r[:3, 3] = torch.tensor(wrist_r_joint.getTransformFromParentBodyNode().translation())

        wrist_offset_l = torch.eye(4)
        wrist_offset_l[:3, :3] = torch.tensor(wrist_l_joint.getTransformFromParentBodyNode().rotation())
        wrist_offset_l[:3, 3] = torch.tensor(wrist_l_joint.getTransformFromParentBodyNode().translation())

        # hand offset
        hand_offset_to_wrist_in_hand_r = -torch.tensor(wrist_r_joint.getTransformFromChildBodyNode().translation())
        hand_offset_rotation_to_wrist_in_hand_r = torch.inverse(torch.tensor(wrist_r_joint.getTransformFromChildBodyNode().rotation()))
        hand_offset_rotation_r = torch.eye(4)
        hand_offset_rotation_r[:3, :3] = hand_offset_rotation_to_wrist_in_hand_r
        hand_offset_translation_r = torch.eye(4)
        hand_offset_translation_r[:3, 3] = hand_offset_to_wrist_in_hand_r
        hand_offset_r = torch.matmul(hand_offset_rotation_r, hand_offset_translation_r)

        hand_offset_to_wrist_in_hand_l = -torch.tensor(wrist_l_joint.getTransformFromChildBodyNode().translation())
        hand_offset_rotation_to_wrist_in_hand_l = torch.inverse(torch.tensor(wrist_l_joint.getTransformFromChildBodyNode().rotation()))
        hand_offset_rotation_l = torch.eye(4)
        hand_offset_rotation_l[:3, :3] = hand_offset_rotation_to_wrist_in_hand_l
        hand_offset_translation_l = torch.eye(4)
        hand_offset_translation_l[:3, 3] = hand_offset_to_wrist_in_hand_l
        hand_offset_l = torch.matmul(hand_offset_rotation_l, hand_offset_translation_l)

    offsets = torch.stack((hip_offset_r, femur_offset_r, knee_offset_r, tibia_offset_r,
                           ankle_offset_r, talus_offset_r, subtalar_offset_r,
                           calcaneus_offset_r, mtp_offset_r,
                           hip_offset_l, femur_offset_l, knee_offset_l, tibia_offset_l,
                           ankle_offset_l, talus_offset_l, subtalar_offset_l,
                           calcaneus_offset_l, mtp_offset_l,
                           lumbar_offset, torso_offset), dim=2)
    if with_arm:
        offsets = torch.stack((*[offsets[i] for i in range(offsets.shape[0])],
                               shoulder_offset_r, humerus_offset_r, elbow_offset_r, ulna_offset_r,
                               radioulnar_offset_r, radius_offset_r, wrist_offset_r, hand_offset_r,
                               shoulder_offset_l, humerus_offset_l, elbow_offset_l, ulna_offset_l,
                               radioulnar_offset_l, radius_offset_l, wrist_offset_l, hand_offset_l,toes_offset_r, toes_offset_l), dim=2)
    return offsets


def euler_to_angular_velocity(q, sampling_fre, convention="XYZ"):
    assert q.shape[-1] == 3
    mat = euler_angles_to_matrix(q, convention)
    mat_diff = mat.clone()
    mat_diff[:-1, :, :] = mat_diff[:-1, :, :] - mat_diff[1:, :, :]
    mat_diff[-1, :, :] = mat_diff[-2, :, :]

    angular_velocity = torch.stack([mat_diff[:, 2, 1] - mat_diff[:, 1, 2],
                                    mat_diff[:, 0, 2] - mat_diff[:, 2, 0],
                                    mat_diff[:, 1, 0] - mat_diff[:, 0, 1]], dim=1) * sampling_fre * 0.5
    return angular_velocity


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def _angle_from_tan(
        axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def get_multi_body_loc_using_nimble_by_body_names(body_names, skel, poses):
    body_ids = [skel.getBodyNode(name) for name in body_names]
    return get_multi_body_loc_using_nimble_by_body_nodes(body_ids, skel, poses)


def get_multi_body_loc_using_nimble_by_body_nodes(body_nodes, skel, poses):
    body_loc = []
    for i_frame in range(len(poses)):
        skel.setPositions(poses[i_frame])
        body_loc.append(np.concatenate([body_node.getWorldTransform().translation() for body_node in body_nodes]))
    body_loc = np.array(body_loc)
    return body_loc


def inverse_norm_cops(skel, states, opt, sub_mass, height_m, grf_thd_to_zero_cop=20):
    poses = states[:, opt.kinematic_osim_col_loc]
    forces = states[:, opt.grf_osim_col_loc]
    normed_cops = states[:, opt.cop_osim_col_loc]
    if len(skel.getDofs()) != poses.shape[1]:        # With Arm model
        print('With Arm model is used. Adding 6 zeros to the end of the poses.')
        poses = np.concatenate([poses, np.zeros((poses.shape[0], 10))], axis=-1)
    foot_loc = get_multi_body_loc_using_nimble_by_body_names(('calcn_r', 'calcn_l'), skel, poses)

    for i_plate in range(2):
        force_v = forces[:, 3*i_plate:3*(i_plate+1)]
        force_v[force_v == 0] = 1e-6
        vector = normed_cops[:, 3 * i_plate:3 * (i_plate + 1)] / force_v[:, 1:2] * height_m
        vector = np.nan_to_num(vector, posinf=0, neginf=0)
        # vector.clip(min=-0.4, max=0.4, out=vector)      # CoP should be within 0.4 m from the foot
        cops = vector + foot_loc[:, 3*i_plate:3*(i_plate+1)]

        if grf_thd_to_zero_cop:
            cops[force_v[:, 1] * sub_mass < grf_thd_to_zero_cop] = 0

        if isinstance(states, torch.Tensor):
            cops = torch.from_numpy(cops).to(states.dtype)
        else:
            cops = cops.astype(states.dtype)
        states[:, opt.cop_osim_col_loc[3*i_plate:3*(i_plate+1)]] = cops
    return states


def convert_addb_state_to_model_input(pose_df, joints_3d, sampling_fre):
    # shift root position to start in (x,y) = (0,0)
    pos_vec = [pose_df['pelvis_tx'][0], pose_df['pelvis_ty'][0], pose_df['pelvis_tz'][0]]
    pose_df['pelvis_tx'] = pose_df['pelvis_tx'] - pose_df['pelvis_tx'][0]
    pose_df['pelvis_ty'] = pose_df['pelvis_ty'] - pose_df['pelvis_ty'][0]
    pose_df['pelvis_tz'] = pose_df['pelvis_tz'] - pose_df['pelvis_tz'][0]

    # remove frozen dof
    for frozen_col in FROZEN_DOFS:
        if frozen_col in pose_df.columns:
            pose_df = pose_df.drop(frozen_col, axis=1)

    # convert euler to 6v
    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_6v = euler_to_6v(torch.tensor(pose_df[joints_with_3_dof].values), "ZXY").numpy()
        for i in range(6):
            pose_df[joint_name + '_' + str(i)] = joint_6v[:, i]

    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_angular_v = euler_to_angular_velocity(torch.tensor(pose_df[joints_with_3_dof].values), sampling_fre, "ZXY").numpy()
        joint_angular_v = data_filter(joint_angular_v, 15, sampling_fre, 4)
        for joints_euler_name in joints_with_3_dof:
            pose_df = pose_df.drop(joints_euler_name, axis=1)
        for i, axis in enumerate(['x', 'y', 'z']):
            pose_df[joint_name + '_' + axis + '_angular' + '_vel'] = joint_angular_v[:, i]

    vel_col_loc = [i for i, col in enumerate(pose_df.columns) if not np.sum([term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5']])]
    vel_col_names = [f'{col}_vel' for i, col in enumerate(pose_df.columns) if not np.sum([term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5']])]
    kinematics_np = pose_df.iloc[:, vel_col_loc].to_numpy().copy()
    kinematics_np_filtered = data_filter(kinematics_np, 15, sampling_fre, 4)
    kinematics_vel = np.stack([spline_fitting_1d(kinematics_np_filtered[:, i_col], range(kinematics_np_filtered.shape[0]), 1).ravel()
                               for i_col in range(kinematics_np_filtered.shape[1])]).T

    pose_vel_df = pd.DataFrame(np.concatenate([pose_df.values, kinematics_vel], axis=1), columns=list(pose_df.columns)+vel_col_names)
    return pose_vel_df, pos_vec


def inverse_convert_addb_state_to_model_input(model_states, model_states_column_names, joints_3d, osim_dof_columns, pos_vec, height_m, sampling_fre=100):
    model_states_dict = {col: model_states[..., i] for i, col in enumerate(model_states_column_names) if
                         col in osim_dof_columns}

    for i_col, col in enumerate(['pelvis_tx', 'pelvis_ty', 'pelvis_tz']):
        model_states_dict[col] = model_states_dict[col] * height_m.unsqueeze(-1).expand(model_states_dict[col].shape)
        model_states_dict[col] = torch.cumsum(model_states_dict[col], dim=-1) / sampling_fre

    # convert 6v to euler
    for joint_name, joints_with_3_dof in joints_3d.items():
        joint_name_6v = [joint_name + '_' + str(i) for i in range(6)]
        index_ = [model_states_column_names.index(joint_name_6v[i]) for i in range(6)]
        joint_euler = euler_from_6v(model_states[..., index_], "ZXY")

        for i, joints_euler_name in enumerate(joints_with_3_dof):
            model_states_dict[joints_euler_name] = joint_euler[..., i]

    # add frozen dof back
    for frozen_col in FROZEN_DOFS:
        model_states_dict[frozen_col] = torch.zeros(model_states.shape[:len(model_states.shape)-1]).to(model_states.device)

    pos_vec = torch.tensor(pos_vec)
    pos_vec_torch = pos_vec.unsqueeze(-1).repeat(*[1 for _ in range(len(pos_vec.shape))], model_states.shape[-2]).to(model_states.device)
    for i_col, col in enumerate(['pelvis_tx', 'pelvis_ty', 'pelvis_tz']):
        model_states_dict[col] += pos_vec_torch[..., i_col, :]

    osim_states = torch.stack([model_states_dict[col] for col in osim_dof_columns], dim=len(model_states.shape)-1).float()
    return osim_states


def align_moving_direction(poses, column_names):
    if isinstance(poses, np.ndarray):
        poses = torch.from_numpy(poses)
    pose_clone = poses.clone().float()
    pelvis_orientation_col_loc = [column_names.index(col) for col in JOINTS_3D_ALL['pelvis']]
    p_pos_col_loc = [column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
    r_grf_col_loc = [column_names.index(col) for col in ['calcn_r_force_vx', 'calcn_r_force_vy', 'calcn_r_force_vz']]
    l_grf_col_loc = [column_names.index(col) for col in ['calcn_l_force_vx', 'calcn_l_force_vy', 'calcn_l_force_vz']]
    r_cop_col_loc = [column_names.index(col) for col in [f'calcn_r_force_normed_cop_{x}' for x in ['x', 'y', 'z']]]
    l_cop_col_loc = [column_names.index(col) for col in [f'calcn_l_force_normed_cop_{x}' for x in ['x', 'y', 'z']]]

    if len(pelvis_orientation_col_loc) != 3 or len(p_pos_col_loc) != 3 or len(r_grf_col_loc) != 3 or len(
            l_grf_col_loc) != 3:
        raise ValueError('check column names')

    pelvis_orientation = pose_clone[:, pelvis_orientation_col_loc]
    pelvis_orientation = euler_angles_to_matrix(pelvis_orientation, "ZXY")
    p_pos = pose_clone[:, p_pos_col_loc]
    r_grf = pose_clone[:, r_grf_col_loc]
    l_grf = pose_clone[:, l_grf_col_loc]
    r_cop = pose_clone[:, r_cop_col_loc]
    l_cop = pose_clone[:, l_cop_col_loc]

    angles = np.arctan2(- pelvis_orientation[:, 0, 2], pelvis_orientation[:, 2, 2])
    if np.rad2deg(angles.max() - angles.min()) > 45:
        return False, None
    angle = angles.median()
    rot_mat = torch.tensor([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]).float()

    pelvis_orientation_rotated = torch.matmul(rot_mat, pelvis_orientation)
    p_pos_rotated = torch.matmul(rot_mat, p_pos.unsqueeze(2)).squeeze(2)
    r_grf_rotated = torch.matmul(rot_mat, r_grf.unsqueeze(2)).squeeze(2)
    l_grf_rotated = torch.matmul(rot_mat, l_grf.unsqueeze(2)).squeeze(2)
    r_cop_rotated = torch.matmul(rot_mat, r_cop.unsqueeze(2)).squeeze(2)
    l_cop_rotated = torch.matmul(rot_mat, l_cop.unsqueeze(2)).squeeze(2)

    pose_clone[:, pelvis_orientation_col_loc] = matrix_to_euler_angles(pelvis_orientation_rotated.float(), "ZXY")
    pose_clone[:, p_pos_col_loc] = p_pos_rotated.float()
    pose_clone[:, r_grf_col_loc] = r_grf_rotated.float()
    pose_clone[:, l_grf_col_loc] = l_grf_rotated.float()
    pose_clone[:, r_cop_col_loc] = r_cop_rotated.float()
    pose_clone[:, l_cop_col_loc] = l_cop_rotated.float()

    return pose_clone, rot_mat


def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered


def fix_seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def linear_resample_data(trial_data, original_fre, target_fre):
    x, step = np.linspace(0., 1., trial_data.shape[0], retstep=True)
    new_x = np.arange(0., 1., step * original_fre / target_fre)
    f = interp1d(x, trial_data, axis=0)
    trial_data_resampled = f(new_x)
    return trial_data_resampled


def spline_fitting_1d(data_, step_to_resample, der=0):
    assert len(data_.shape) == 1
    data_ = data_.reshape(1, -1)
    tck, step = interpo.splprep(data_, u=range(data_.shape[1]), s=0)
    data_resampled = interpo.splev(step_to_resample, tck, der=der)
    data_resampled = np.column_stack(data_resampled)
    return data_resampled


def convert_overlapped_list_to_array(trial_len, win_list, s_, e_, fun=np.nanmedian, max_size=150):
    array_val_expand = np.full((max_size, trial_len, win_list[0].shape[1]), np.nan)
    for i_win, (win, s, e) in enumerate(zip(win_list, s_, e_)):
        array_val_expand[i_win%max_size, s:e] = win[:e-s]
    array_val = fun(array_val_expand, axis=0)
    std_val = np.nanstd(array_val_expand, axis=0)
    return array_val, std_val


def identity(t, *args, **kwargs):
    return t


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
                torch.linspace(
                    linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
                )
                ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
                torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
                ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


""" ============================ End util.py ============================ """


""" ============================ Start args.py ============================ """


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="new_orientation_alignment_diffusion", help="save to project/name")
    parser.add_argument("--with_arm", type=bool, default=False, help="whether osim model has arm DoFs")
    parser.add_argument("--with_kinematics_vel", type=bool, default=True, help="whether to include 1st derivative of kinematics")
    parser.add_argument("--epochs", type=int, default=7680)
    parser.add_argument("--target_sampling_rate", type=int, default=100)
    parser.add_argument("--window_len", type=int, default=150)
    parser.add_argument("--guide_x_start_the_beginning_step", type=int, default=-10)      # negative value means no guidance

    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="dataset_backups/",
        help="Dataset backup path",
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--batch_size_inference", type=int, default=32, help="batch size during inference")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    # parser.add_argument(
    #     "--checkpoint_bl", type=str, default="", help="trained checkpoint path (optional)"
    # )
    opt = parser.parse_args(args=[])
    set_no_arm_opt(opt)
    current_folder = os.getcwd()
    opt.subject_data_path = current_folder
    opt.geometry_folder = current_folder + '/Geometry/'
    opt.checkpoint_bl = current_folder + '/GaitDynamics/example_usage/GaitDynamicsRefinement.pt'
    return opt


def set_no_arm_opt(opt):
    opt.with_arm = False
    opt.osim_dof_columns = copy.deepcopy(OSIM_DOF_ALL[:23] + KINETICS_ALL)
    opt.joints_3d = {key_: value_ for key_, value_ in JOINTS_3D_ALL.items() if key_ in ['pelvis', 'hip_r', 'hip_l', 'lumbar']}
    opt.model_states_column_names = copy.deepcopy(MODEL_STATES_COLUMN_NAMES_NO_ARM)
    for joint_name, joints_with_3_dof in opt.joints_3d.items():
        opt.model_states_column_names = opt.model_states_column_names + [
            joint_name + '_' + axis + '_angular_vel' for axis in ['x', 'y', 'z']]

    if opt.with_kinematics_vel:
        opt.model_states_column_names = opt.model_states_column_names + [
            f'{col}_vel' for i_col, col in enumerate(opt.model_states_column_names)
            if not sum([term in col for term in ['force', 'pelvis_', '_vel', '_0', '_1', '_2', '_3', '_4', '_5']])]


    opt.knee_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'knee' in col]
    opt.ankle_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'ankle' in col]
    opt.hip_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'hip' in col]
    opt.kinematic_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if 'force' not in col]
    opt.kinetic_diffusion_col_loc = [i_col for i_col, col in enumerate(opt.model_states_column_names) if i_col not in opt.kinematic_diffusion_col_loc]
    opt.grf_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if 'force' in col and '_cop_' not in col]
    opt.cop_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if '_cop_' in col]
    opt.kinematic_osim_col_loc = [i_col for i_col, col in enumerate(opt.osim_dof_columns) if 'force' not in col]

""" ============================ End args.py ============================ """


""" ============================ Start dataset.py ============================ """


class MotionDataset(Dataset):
    def __init__(
            self,
            opt,
            align_moving_direction_flag: bool = True,
            normalizer: Any = None,
            max_trial_num=None,
            check_cop_to_calcn_distance=True,
    ):
        self.data_path = opt.subject_data_path
        self.subject_osim_model = opt.subject_osim_model
        if opt.target_sampling_rate != 100:
            raise ValueError('100 Hz sampling rate is not confirmed. Confirm by setting opt.target_sampling_rate = 100')
        self.target_sampling_rate = opt.target_sampling_rate
        self.window_len = opt.window_len
        self.align_moving_direction_flag = align_moving_direction_flag
        self.opt = opt
        self.check_cop_to_calcn_distance = check_cop_to_calcn_distance
        self.skel = None
        self.dset_set = set()

        print("Loading dataset...")
        self.load_addb(opt)

        self.guess_vel_and_replace_txtytz()

        if not len(self.trials):
            print("No trials loaded")
            return
        self.normalizer = normalizer
        for i_trial in range(len(self.trials)):
            self.trials[i_trial].converted_pose = self.normalizer.normalize(self.trials[i_trial].converted_pose).clone().detach().float()

    def customized_param_manipulation(self, states_df):
        # for guided diffusion
        return states_df

    def guess_vel_and_replace_txtytz(self):
        pelvis_pos_loc = [self.opt.model_states_column_names.index(col) for col in [f'pelvis_t{x}' for x in ['x', 'y', 'z']]]
        for i_trial, trial in enumerate(self.trials):
            body_center = trial.converted_pose[:, 0:3]
            body_center = data_filter(body_center, 10, self.target_sampling_rate).astype(np.float32)
            vel_from_t = np.diff(body_center, axis=0) * self.target_sampling_rate
            vel_from_t = np.concatenate([vel_from_t, vel_from_t[-1][None, :]], axis=0)
            if self.opt.treadmill_speed is None:
                raise ValueError('Treadmill speed is not set. For overground walking, set opt.treadmill_speed = 0')
            vel_from_t[:, 0] = vel_from_t[:, 0] + self.opt.treadmill_speed
            walking_vel = vel_from_t

            self.trials[i_trial].converted_pose[:, pelvis_pos_loc] = torch.from_numpy(walking_vel) / self.trials[i_trial].height_m

    def __len__(self):
        return self.opt.pseudo_dataset_len

    def get_overlapping_wins(self, col_loc_to_unmask, step_len, start_trial=0, end_trial=None, including_shorter_than_window_len=False):
        if end_trial is None:
            end_trial = len(self.trials)
        windows, s_list, e_list = [], [], []
        for i_trial in range(start_trial, end_trial):
            trial_ = self.trials[i_trial]
            trial_len = trial_.converted_pose.shape[0]
            if including_shorter_than_window_len:
                e_of_trial = trial_len
            else:
                e_of_trial = trial_len - self.opt.window_len + step_len
            for i in range(0, e_of_trial, step_len):
                s = max(0, i)
                e = min(trial_len, i + self.opt.window_len)
                s_list.append(s)
                e_list.append(e)
                mask = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
                mask[:, col_loc_to_unmask] = 1
                mask[e-s:, :] = 0
                data_ = torch.zeros([self.opt.window_len, len(self.opt.model_states_column_names)])
                data_[:e-s] = trial_.converted_pose[s:e, ...]
                windows.append(WindowData(data_, trial_.model_offsets, i_trial, None, mask,
                                          trial_.height_m, trial_.weight_kg, trial_.missing_col))
        return windows, s_list, e_list

    def load_addb(self, opt):
        file_paths = opt.file_paths
        customOsim: nimble.biomechanics.OpenSimFile = nimble.biomechanics.OpenSimParser.parseOsim(self.subject_osim_model, self.opt.geometry_folder)
        skel = customOsim.skeleton
        self.skel = skel

        self.trials, self.file_names = [], []
        for i_file, file_path in enumerate(file_paths):
            model_offsets = get_model_offsets(skel).float()
            poses_df = pd.read_csv(file_path, sep='\t', skiprows=10)
            with open(file_path) as f:
                for _ in range(10):
                    header = f.readline()
                    if 'inDegrees' in header:
                        if 'yes' in header and 'no' not in header:
                            angle_scale = np.pi / 180
                        elif 'no' in header and 'yes' not in header:
                            angle_scale = 1
                        else:
                            raise ValueError('No inDegrees keyword in the header, cannot determine the unit of angles. '
                                             'Here is an example header: \nCoordinates\nversion=1\nnRows=1380'
                                             '\nnColumns=26\ninDegrees=yes\n')
                        break

            if 'time' not in poses_df.columns:
                raise ValueError(f'{file_path} does not have time column. Necessary for compuing sampling rate')
            sampling_rate = round((poses_df.shape[0] - 1) / (poses_df['time'].iloc[-1] - poses_df['time'].iloc[0]))

            missing_col = []
            for col in FULL_OSIM_DOF:
                if col not in poses_df.columns:
                    poses_df[col] = 0.
                    missing_col.append(col)

            poses_df = poses_df.astype(float)
            col_list = list(poses_df.columns)
            col_loc = [col_list.index(col) for col in OSIM_DOF_ALL[:23]]
            angle_col_loc = [col_list.index(col) for col in OSIM_DOF_ALL[:23] if col not in ['pelvis_tx', 'pelvis_ty', 'pelvis_tz']]
            poses_df.iloc[:, angle_col_loc] = poses_df.iloc[:, angle_col_loc] * angle_scale
            poses = poses_df.values[:, col_loc]

            all_zeros = np.zeros([poses_df.shape[0], 12])       # used to fill in the missing GRF in normalization
            states = np.concatenate([np.array(poses), all_zeros], axis=1)
            if not self.is_lumbar_rotation_reasonable(np.array(states), opt.osim_dof_columns):
                Warning(f'Warning: {file_path} has unreasonable lumbar rotation, don\'t trust this trial.')

            if self.align_moving_direction_flag:
                states_aligned, rot_mat = align_moving_direction(states, opt.osim_dof_columns)
                if states_aligned is False:
                    print(f'Warning: {file_path} Pelvis orientation changed by more than 45 deg, don\'t trust this trial')
                    rot_mat = torch.eye(3).float()
                else:
                    states = states_aligned
            else:
                rot_mat = torch.eye(3).float()

            file_name = file_path.split('/')[-1]
            if states.shape[0] / sampling_rate * self.target_sampling_rate < self.window_len + 2:
                print(f'Warning: {file_name} is shorter than 1.5s, skipping.')
                continue
            if sampling_rate != self.target_sampling_rate:
                states = linear_resample_data(states, sampling_rate, self.target_sampling_rate)

            states_df = pd.DataFrame(states, columns=opt.osim_dof_columns)
            states_df = self.customized_param_manipulation(states_df)
            states_df, pos_vec = convert_addb_state_to_model_input(states_df, opt.joints_3d, self.target_sampling_rate)

            assert self.opt.model_states_column_names == list(states_df.columns)
            converted_states = torch.tensor(states_df.values).float()

            trial_data = TrialData(converted_states, model_offsets, opt.height_m, opt.weight_kg, rot_mat, pos_vec,
                                   self.window_len, missing_col)
            self.trials.append(trial_data)
            self.file_names.append(file_name)

    @staticmethod
    def is_lumbar_rotation_reasonable(states, column_names):
        lumbar_rotation_col_loc = column_names.index('lumbar_rotation')
        if np.abs(np.mean(states[:, lumbar_rotation_col_loc])) > np.deg2rad(45):
            return False
        else:
            return True


class WindowData:
    def __init__(self, pose, model_offsets, trial_id, gait_phase_label, mask, height_m, weight_kg, missing_col):
        self.pose = pose
        self.model_offsets = model_offsets
        self.trial_id = trial_id
        self.gait_phase_label = gait_phase_label
        self.mask = mask
        self.height_m = height_m
        self.weight_kg = weight_kg
        self.missing_col = missing_col


class TrialData:
    def __init__(self, converted_states, model_offsets, height_m, weight_kg, rot_mat_for_moving_direction_alignment,
                 pos_vec_for_pos_alignment, window_len, missing_col):
        self.converted_pose = converted_states
        self.model_offsets = model_offsets
        self.height_m = height_m
        self.weight_kg = weight_kg
        self.length = converted_states.shape[0]
        self.rot_mat_for_moving_direction_alignment = rot_mat_for_moving_direction_alignment
        self.pos_vec_for_pos_alignment = pos_vec_for_pos_alignment
        self.window_len = window_len
        self.missing_col = missing_col


""" ============================ End args.py ============================ """


def usr_inputs():
    opt = parse_opt()

    # # [DEBUG]
    # opt.height_m = 1.84
    # opt.weight_kg = 92.9
    # opt.treadmill_speed = 1.15
    # opt.subject_osim_model = opt.subject_data_path + '/Scaled_generic_no_arm.osim'

    input("Upload .mot files and a .osim file to Colab via \n \
    1) clicking the folder button on the left side of the window \n \
    2) upload .mot files by clicking \"Upload to session storage\" button and \n \
    3) upload .osim file by clicking \"Upload to session storage\" button. \n \
    Please confirm completion and then entering anything ")

    file_paths = []
    for file in os.listdir(opt.subject_data_path):
        file_path = os.path.join(opt.subject_data_path, file)
        if file.endswith(".mot") and '_pred___' not in file:
            file_paths.append(file_path)
    if len(file_paths) == 0:
        raise RuntimeError(f'No .mot file found. Upload the .mot file to Colab')
    opt.file_paths = file_paths

    osim_paths = []
    for root, dirs, files in os.walk(opt.subject_data_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".osim") and 'example_opensim_model.osim' not in file:
                osim_paths.append(file_path)
    if len(osim_paths) > 1:
        print(f'Multiple .osim files found.')
        [print(f'{i}: {file_path}') for i, file_path in enumerate(osim_paths)]
        i_file = int(input(f'Choose the .osim file entering its index (between 0 and {len(osim_paths)-1}): '))
        opt.subject_osim_model = osim_paths[i_file]
    elif len(osim_paths) == 0:
        raise RuntimeError(f'No .osim file found. Upload the .osim file to Colab')
    else:
        opt.subject_osim_model = osim_paths[0]
    print()

    while True:
        try:
            opt.height_m = float(input("Enter the subject's height in meter: "))
            if opt.height_m > 2.5 or opt.height_m < 1:
                raise ValueError()
            print()
            break
        except ValueError:
            print("Invalid input. Please enter a floating-point number between 1.0 and 2.5.")
    while True:
        try:
            opt.weight_kg = float(input("Enter the subject's weight in Kg: "))
            if opt.weight_kg > 200 or opt.weight_kg < 30:
                raise ValueError()
            print()
            break
        except ValueError:
            print("Invalid input. Please enter a floating-point number between 30 and 200.")
    while True:
        try:
            opt.treadmill_speed = float(input("Enter treadmill speed in m/s (enter 0 for overground gait): "))
            if opt.treadmill_speed > 10 or opt.treadmill_speed < 0:
                raise ValueError()
            print()
            break
        except ValueError:
            print("Invalid input. Please enter a floating-point number between 0 and 10.")

    return opt


def predict_grf(opt):
    refinement_model = BaselineModel(opt, TransformerEncoderArchitecture)
    dataset = MotionDataset(opt, normalizer=refinement_model.normalizer)
    diffusion_model_for_filling = None
    filling_method = DiffusionFilling()
    for i_trial in range(len(dataset.trials)):
        windows, s_list, e_list = dataset.get_overlapping_wins(opt.kinematic_diffusion_col_loc, 20, i_trial, i_trial+1)
        if len(windows) == 0:
            continue

        if len(windows[0].missing_col) > 0:
            print(f'File {dataset.file_names[i_trial]} do not have {windows[0].missing_col}. '
                  f'\nGenerating missing kinematics for {dataset.file_names[i_trial]}')
            if diffusion_model_for_filling is None:
                diffusion_model_for_filling, _ = load_diffusion_model(opt)
            windows_reconstructed = filling_method.fill_param(windows, diffusion_model_for_filling)
        else:
            windows_reconstructed = windows

        state_pred_list = []
        print(f'Running GaitDynamics on file {dataset.file_names[i_trial]} for external force prediction.')
        for i_win in range(0, len(windows), opt.batch_size_inference):
            state_true = torch.stack([win.pose for win in windows_reconstructed[i_win:i_win+opt.batch_size_inference]])
            masks = torch.stack([win.mask for win in windows_reconstructed[i_win:i_win+opt.batch_size_inference]])

            state_pred_list_batch = refinement_model.eval_loop(opt, state_true, masks, num_of_generation_per_window=1)[0]
            state_pred_list.append(state_pred_list_batch)

        state_pred = torch.cat(state_pred_list, dim=0)
        trial_len = dataset.trials[i_trial].converted_pose.shape[0]

        results_pred, _ = convert_overlapped_list_to_array(
            trial_len, state_pred, s_list, e_list)

        height_m_tensor = torch.tensor([windows[0].height_m])
        results_pred = inverse_convert_addb_state_to_model_input(
            torch.from_numpy(results_pred).unsqueeze(0), opt.model_states_column_names,
            opt.joints_3d, opt.osim_dof_columns, dataset.trials[i_trial].pos_vec_for_pos_alignment, height_m_tensor)[0].numpy()
        results_pred = inverse_norm_cops(dataset.skel, results_pred, opt, windows[0].weight_kg, windows[0].height_m)

        results_pred[:, -12:-9] = results_pred[:, -12:-9] * opt.weight_kg  # convert to N
        results_pred[:, -6:-3] = results_pred[:, -6:-3] * opt.weight_kg  # convert to N
        df = pd.DataFrame(results_pred, columns=opt.osim_dof_columns)

        trial_save_path = f'{dataset.file_names[i_trial][:-4]}_pred___.mot'
        convertDfToGRFMot(df, trial_save_path, round(1 / opt.target_sampling_rate, 3))



if __name__ == '__main__':
    opt = usr_inputs()
    predict_grf(opt)
