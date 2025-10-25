import torch


def _handle_zeros_in_scale(scale, copy=True, constant_mask=None):
    if constant_mask is None:
        constant_mask = scale < 10 * torch.finfo(scale.dtype).eps

    if copy:
        scale = scale.clone()
    scale[constant_mask] = 1.0
    return scale


class StandardScaler:
    _parameter_constraints: dict = {
        "feature_range": [tuple],
        "copy": ["boolean"],
        "clip": ["boolean"],
    }

    def __init__(self, copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        if hasattr(self, "scale_"):
            del self.scale_
            del self.data_mean_

    def fit(self, X):
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):

        self.data_mean_ = torch.mean(X, axis=0)
        self.scale_ = torch.std(X, axis=0)

        return self

    def transform(self, X):
        X -= self.data_mean_.to(X.device)
        X /= self.scale_.to(X.device)
        return X

    def inverse_transform(self, X):
        X *= self.scale_.to(X.device)
        X += self.data_mean_.to(X.device)
        return X


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
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
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
