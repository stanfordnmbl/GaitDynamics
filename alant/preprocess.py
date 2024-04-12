import glob
import os
import re
from pathlib import Path
import torch
from alant.scaler import StandardScaler, MinMaxScaler


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == "" else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


class Normalizer:
    def __init__(self, data, cols_to_normalize):
        flat = data.reshape(-1, data.shape[-1])
        # self.scaler = MinMaxScaler(feature_range=(-10, 10))      # MinMaxScaler allows clipping
        self.scaler = StandardScaler()      # StandardScaler converges faster
        self.scaler.fit(flat[:, cols_to_normalize])
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


def vectorize_many(data):
    # given a list of batch x seqlen x joints? x channels, flatten all to batch x seqlen x -1, concatenate
    batch_size = data[0].shape[0]
    seq_len = data[0].shape[1]

    out = [x.reshape(batch_size, seq_len, -1).contiguous() for x in data]

    global_pose_vec_gt = torch.cat(out, dim=2)
    return global_pose_vec_gt
