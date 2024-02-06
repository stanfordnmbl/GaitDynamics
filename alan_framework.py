from alant.args import parse_opt
from model.alan_model import MotionModel


def train(opt):
    repr_dim = len(opt.model_states_column_names)
    model = MotionModel(opt, repr_dim)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)






















