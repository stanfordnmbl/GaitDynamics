import torch
import os


diffusion_pt = os.path.dirname(os.path.realpath(__file__)) + f"/../trained_models/{'train-7680_diffusion.pt'}"
checkpoint = torch.load(diffusion_pt)


xxx = 1