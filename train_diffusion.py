from args import parse_opt
from model.model import MotionModel, BaselineModel, TransformerEncoderArchitecture
from model_baseline.grf_baseline import GroundLinkArchitecture, SugaiNetArchitecture
from data.addb_dataset import MotionDataset
from torch.nn import functional as F
import wandb


def train(opt):
    # model = MotionModel(opt)
    model = BaselineModel(opt, TransformerEncoderArchitecture)
    # model = BaselineModel(opt, GroundLinkArchitecture)
    # model = BaselineModel(opt, SugaiNetArchitecture)

    if opt.log_with_wandb:
        wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, dir="wandb_logs")
        wandb.watch(model.diffusion, F.mse_loss, log='all', log_freq=200)
    train_dataset = MotionDataset(
        data_path=opt.data_path_train,
        train=True,
        # trial_start_num=-1,
        # max_trial_num=1,
        dset_keyworks_to_exclude=['Fregly2012', 'Uhlrich2023', 'Hamner2013', 'Han2023'],
        # dset_keyworks_to_exclude=['Carter2023', 'Fregly2012', 'Falisse2017', 'Hamner2013', 'Han2023', 'Li2021', 'Santos2017', 'Tan2021', 'Uhlrich2023', 'Wang2023'],
        opt=opt,
    )
    model.train_loop(opt, train_dataset)


""" Steps:
1. Download data to the server using command: addb -d dev download --prefix "protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/"
2. Move data to train1, train2, and train3 folders for post-processing
3. Run post-processing using command: addb post-process --geometry-folder "/dataNAS/people/alanttan/mfm/data/Geometry/" --only-dynamics False "./train2/" "./train_cleaned/"
4. Manually split training and test sets.
5. Run this script to train the model.
"""


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)






















