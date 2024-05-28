from args import parse_opt
from model.model import MotionModel
from data.addb_dataset import MotionDataset


def train(opt):
    repr_dim = len(opt.model_states_column_names)
    model = MotionModel(opt, repr_dim)
    train_dataset = MotionDataset(
        data_path=opt.data_path_train,
        train=True,
        # trial_start_num=-3,
        # max_trial_num=1,            # !!!
        dset_keyworks_to_exclude=['Hammer2013'],
        opt=opt,
    )
    model.train_loop(opt, train_dataset)


""" Steps:
1. Download data to the server using command: addb -d dev download --prefix "protected/us-west-2:e013a4d2-683d-48b9-bfe5-83a0305caf87/"
2. Move data to train1, train2, and train3 folders for post-processing
3. Run post-processing using command: addb post-process --geometry-folder "/dataNAS/people/alanttan/mfm/data/Geometry/" --only-dynamics True "./train2/" "./train_cleaned/"
4. Manually split training and test sets.
5. Run this script to train the model.
"""


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)






















