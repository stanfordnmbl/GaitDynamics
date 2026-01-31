from args import parse_opt
from model.model import MotionModel, BaselineModel, TransformerEncoderArchitecture
from model_baseline.grf_baseline import GroundLinkArchitecture, SugaiNetArchitecture
from data.addb_dataset import MotionDataset
from torch.nn import functional as F
import wandb


def train(opt):
    model = MotionModel(opt)
    model = BaselineModel(opt, TransformerEncoderArchitecture)
    # model = BaselineModel(opt, GroundLinkArchitecture)
    # model = BaselineModel(opt, SugaiNetArchitecture)

    if opt.log_with_wandb:
        wandb.init(project=opt.wandb_pj_name, name=opt.exp_name, dir="wandb_logs")
        wandb.watch(model.diffusion, F.mse_loss, log='all', log_freq=200)
    train_dataset = MotionDataset(
        data_path=opt.data_path_train,
        train=True,
        dset_keyworks_to_exclude=['Fregly2012', 'Uhlrich2023', 'Hamner2013', 'Han2023'],
        opt=opt,
    )
    model.train_loop(opt, train_dataset)


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)






















