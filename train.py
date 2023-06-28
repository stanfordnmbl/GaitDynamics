

import numpy as np
import torch
from vis import SMPLSkeleton
import vis
# data_file = np.load('/home/tvwouw/Downloads/ACCAD/ACCAD/Female1General_c3d/A11 - crawl forward_poses.npz')
# data = data_file.keys()
# poses = torch.tensor(data_file['poses'])
# poses = poses[:,:66].reshape(poses.shape[0],-1,3)
# trans = torch.tensor(data_file['trans'])
# smpl = SMPLSkeleton('cpu')
# positions = smpl.forward(poses.unsqueeze(0),trans.unsqueeze(0))
# vis.skeleton_render(positions.squeeze())
from args import parse_train_opt
from EDGE import EDGE



def train(opt):
    model = EDGE(opt.feature_type, opt.checkpoint)
    opt.batch_size = 512
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    opt.checkpoint = "./runs/train/exp16/weights/train-2.pt"
    train(opt)
