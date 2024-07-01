import numpy as np
from args import parse_opt, set_with_arm_opt
import torch
import os
from model.model import MotionModel
from data.addb_dataset import MotionDataset
from figures.fig_utils import show_skeletons, set_up_gui
import matplotlib.pyplot as plt


gui = set_up_gui()

if __name__ == "__main__":
    opt = parse_opt()
    carter_data_path = '/dataNAS/people/alanttan/mfm/data/b3d_no_arm/train_cleaned/Carter2023_Formatted_No_Arm/'
    opt.checkpoint = f"/dataNAS/people/alanttan/mfm/code/runs/train/t_plus_cond/weights/train-{'6993'}.pt"

    model = torch.load(opt.checkpoint)
    repr_dim = model["ema_state_dict"]["input_projection.weight"].shape[1]
    set_with_arm_opt(opt, False)
    model = MotionModel(opt, repr_dim)

    subjects = list(sorted(set([x[0].split('Carter2023_Formatted_No_Arm/')[1].split('_split')[0]
                                for x in os.walk(carter_data_path)])))
    sub_cond_pairs = []
    for i_sub, sub_ in enumerate(subjects):
        if 'P0' not in sub_:
            continue

        for split in range(5):
            test_dataset = MotionDataset(
                data_path=carter_data_path+sub_+f"_split{split}",
                train=False,
                normalizer=model.normalizer,
                opt=opt,
                divide_jittery=False,
                max_trial_num=1,
                trial_start_num=0,
            )
            if len(test_dataset.trials) > 0:
                for i_cond in range(6):
                    sub_cond_pairs.append((i_sub, test_dataset.trials[0].cond[i_cond], i_cond))

    for i_cond in range(6):
        plt.figure()
        plt.scatter([x[0] for x in sub_cond_pairs if x[2] == i_cond], [x[1] for x in sub_cond_pairs if x[2] == i_cond])
        plt.savefig(f'cond_{i_cond}.png')
        plt.show()


















