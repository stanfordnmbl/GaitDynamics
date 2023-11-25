from args import parse_train_opt
from alan_model import MotionModel


def train(opt):
    model = MotionModel(opt.feature_type, opt.checkpoint)
    opt.batch_size = 1     # !!! change back to 512
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    opt.data_path = "E:/MotionModelData/train/_sliced/"
    train(opt)






















