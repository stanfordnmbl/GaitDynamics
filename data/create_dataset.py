import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *
from agreggate import *


def create_dataset(opt):
    # split the data according to the splits files
    # print("Creating train / test split")
    # split_data(opt.dataset_folder)

    # agreggate AMASS dataset
    path_AMASS = os.path.join(os.path.dirname(__file__),'AMASS')
    path_AMASS_agreggated = os.path.join(os.path.dirname(__file__), 'AMASS_agreggated')
    Path(path_AMASS_agreggated).mkdir(parents=True, exist_ok=True)
    # agreggate(path_AMASS, path_AMASS_agreggated)

    # slice motions into sliding windows to create training dataset
    print("Slicing train data")
    slice_AMASS(path_AMASS_agreggated)
    # print("Slicing test data")
    # slice_AMASS(f"test/motions", f"test/wavs")

    # process dataset to extract audio features
    if opt.extract_baseline:
        print("Extracting baseline features")
        baseline_extract("train/wavs_sliced", "train/baseline_feats")
        baseline_extract("test/wavs_sliced", "test/baseline_feats")
    if opt.extract_jukebox:
        print("Extracting jukebox features")
        jukebox_extract("train/wavs_sliced", "train/jukebox_feats")
        jukebox_extract("test/wavs_sliced", "test/jukebox_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
