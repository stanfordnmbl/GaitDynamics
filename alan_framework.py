from alant.args import parse_train_opt
from model.alan_model import MotionModel


def train(opt):
    repr_dim = len(opt.osim_dof_columns) + len(opt.joints_3d) * 3 - 2
    model = MotionModel(opt, repr_dim)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)






















