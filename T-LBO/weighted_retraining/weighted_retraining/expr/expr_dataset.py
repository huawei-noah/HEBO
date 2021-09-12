import argparse
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent.parent)
sys.path.insert(0, ROOT_PROJECT)

from weighted_retraining.weighted_retraining.expr.expr_data import load_data_str, load_data_enc, score_function


def get_filepath(ignore_percentile, save_dir, seed: int, good_percentile=0) -> str:
    if good_percentile == 0:
        save_name = f"expr_P{ignore_percentile}_{seed}"
    else:
        save_name = f"expr_P{ignore_percentile}_{good_percentile}_{seed}"
    filepath = str(Path(save_dir) / save_name) + ".npz"
    return filepath


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore_percentile",
        type=int,
        default=65,
        help="percentile of scores to ignore"
    )
    parser.add_argument(
        "--good_percentile",
        type=int,
        default=5,
        help="percentile of good scores to take (good scores being obtained from the top-`ignore_percentile` "
             "part of the dataset)  --> if `ignore_percentile = 65` and `good_percentile = 5` the dataset will be made"
             "of the 35% worse equations and `N_good` samples of equations from the data having scores in 65 - 3% "
             "(we exclude top 3%), where `N_good = 5% * N_total`"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for reproducibility"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="weighted_retraining/assets/data/expr",
        help="directory of datasets",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="Whether to overwrite existing file",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="directory to save files in"
    )
    parser.add_argument(
        "--test_valid", action="store_true",
        help="check that the dataset is valid (i.e. triplets are well composed)"
    )

    args = parser.parse_args()
    assert args.ignore_percentile + args.good_percentile <= 100, (args.ignore_percentile, args.good_percentile)
    if args.good_percentile > 0:
        assert args.ignore_percentile + args.good_percentile <= 97, (
            args.ignore_percentile, args.good_percentile, "Protect top 3%")

    os.makedirs(args.save_dir, exist_ok=True)

    filepath = get_filepath(args.ignore_percentile, args.save_dir, args.seed, args.good_percentile)

    if os.path.exists(filepath) and not args.overwrite:
        exit(print(f'{filepath} already exists'))
    data_str = np.array(load_data_str(Path(args.data_dir)))
    data_enc = load_data_enc(Path(args.data_dir))
    data_scores = score_function(data_str)

    if args.test_valid:
        aux = set()
        for i in tqdm(range(len(data_str))):
            aux.add((str(data_str[i]), str(data_scores[i]), str(data_enc[i])))

    perc = np.percentile(data_scores, args.ignore_percentile)
    perc_idx = data_scores >= perc
    bad_data = data_enc[perc_idx]
    bad_scores = data_scores[perc_idx]  # MSE
    bad_data_str = data_str[perc_idx]

    perc_top_3 = np.percentile(data_scores, 3)
    perc_idx = np.logical_and((data_scores <= perc), (data_scores > perc_top_3))
    good_data = data_enc[perc_idx]
    good_scores = data_scores[perc_idx]  # MSE
    good_str = data_str[perc_idx]

    N_good = int(args.good_percentile * len(data_str) / 100)

    np.random.seed(args.seed)
    inds = np.random.permutation(len(bad_data))
    data = bad_data[inds]
    scores = bad_scores[inds]
    _str = bad_data_str[inds]

    if N_good > 0:
        inds = np.random.choice(np.arange(0, len(good_data)), size=N_good, replace=False)
        good_data = good_data[inds]
        good_scores = good_scores[inds]
        good_str = good_str[inds]

        inds = np.random.permutation(len(data) + N_good)
        data = np.vstack([data, good_data])[inds]
        scores = np.concatenate([scores, good_scores])[inds]
        _str = np.concatenate([_str, good_str])[inds]

    print(f"Dataset created. Total of {len(data)} points")
    print(f"Array size {data.nbytes / 1e9:.1f} Gb")
    print(f"Save {filepath}")

    if args.test_valid:
        for _ in range(100):
            i = np.random.randint(len(data))
            assert (str(_str[i]), str(scores[i]), str(data[i])) in aux

    assert data.shape[0] == _str.shape[0] == scores.shape[0]

    # Save dataset
    np.savez_compressed(filepath, data=data, scores=scores, expr=_str)
