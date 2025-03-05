import torch
import os
from example_usage.real_time_model import TransformerHipKnee, update_opt
from model.dance_decoder import DanceDecoder
from model.model import TransformerEncoderArchitecture
from args import parse_opt, set_with_arm_opt



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

def compress_model(checkpoint_path, model_class, new_model_name):
    checkpoint = torch.load(checkpoint_path)
    new_model_object = model_class(len(opt.model_states_column_names), opt)
    new_model_object.load_state_dict(checkpoint["ema_state_dict"])

    normalizer = checkpoint["normalizer"]
    min_max_scaler = MinMaxScaler(feature_range=normalizer.scaler.feature_range)
    min_max_scaler.n_samples_seen_ = normalizer.scaler.n_samples_seen_
    min_max_scaler.scale_ = normalizer.scaler.scale_
    min_max_scaler.min_ = normalizer.scaler.min_
    min_max_scaler.data_min_ = normalizer.scaler.data_min_
    min_max_scaler.data_max_ = normalizer.scaler.data_max_
    min_max_scaler.data_range_ = normalizer.scaler.data_range_
    new_noramlizer = Normalizer(min_max_scaler, normalizer.cols_to_normalize)

    new_check_point = {'ema_state_dict': new_model_object.state_dict(), 'normalizer': new_noramlizer}
    torch.save(new_check_point, new_model_name + '.pt')


if __name__ == '__main__':
    opt = parse_opt()
    set_with_arm_opt(opt, False)

    # diffusion
    checkpoint_path = os.getcwd() + '/../trained_models/train-2560_diffusion.pt'
    compress_model(checkpoint_path, DanceDecoder, 'GaitDynamicsDiffusion')

    # # full-body tf
    # checkpoint_path = os.getcwd() + '/../trained_models/train-7680_tf.pt'
    # compress_model(checkpoint_path, TransformerEncoderArchitecture, 'GaitDynamicsRefinement')

    # # hip-knee tf
    # opt = update_opt(opt, '', osim_model_path='', height_m=0., weight_kg=0.)
    # checkpoint_path = os.getcwd() + '/../trained_models/GaitDynamicsRefinementHipKnee.pt'
    # compress_model(checkpoint_path, TransformerHipKnee, 'GaitDynamicsRefinementHipKnee')

