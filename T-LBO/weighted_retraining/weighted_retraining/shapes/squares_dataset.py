"""
Makes a dataset consisting of squares
"""

import numpy as np
from tqdm.auto import trange
import argparse
from pathlib import Path

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--grid_size", type=int, default=64)
parser.add_argument("--min_size", type=int, default=2)
parser.add_argument("--max_size", type=int, default=20)
parser.add_argument("--shuffle_seed", type=int, default=None)
parser.add_argument(
    "--min_col", type=int, default=None, help="min col to start spanning"
)
parser.add_argument("--max_col", type=int, default=None, help="max col to span")
parser.add_argument(
    "--save_dir", type=str, required=True, help="directory to save files in"
)
parser.add_argument(
    "--num_replicates",
    type=int,
    default=1,
    help="Number of times to replicate each point",
)


if __name__ == "__main__":

    args = parser.parse_args()
    assert Path(args.save_dir).exists()

    # Make all the rectangles
    print("Making Dataset")
    arr_list = []
    for rect_height in trange(args.min_size, args.max_size + 1):
        for rect_width in [rect_height]:  # Because it is squares only
            for row in range(args.grid_size - rect_height + 1):

                # Specify which columns to span
                min_col = args.min_col
                if min_col is None:
                    min_col = 0
                max_col = args.max_col
                if max_col is None:
                    max_col = args.grid_size
                for col in range(
                    min_col, min(args.grid_size - rect_width + 1, max_col)
                ):
                    arr = np.zeros((args.grid_size, args.grid_size), dtype=np.uint8)
                    arr[row : row + rect_height, col : col + rect_width] = np.ones(
                        (rect_height, rect_width)
                    )

                    # Append it multiple times
                    for _ in range(args.num_replicates):
                        arr_list.append(arr)
    arr_list = np.array(arr_list)
    print(f"Dataset created. Total of {len(arr_list)} points")
    print(f"Array size {arr_list.nbytes / 1e9:.1f} Gb")

    # Possibly shuffle
    if args.shuffle_seed is None:
        shuffle_str = ""
    else:
        print("Shuffling!")
        np.random.seed(args.shuffle_seed)
        np.random.shuffle(arr_list)
        shuffle_str = f"_seed{args.shuffle_seed}"

    # Calculate areas
    areas = np.sum(arr_list, axis=(1, 2)).astype(float)

    # Save dataset
    shapes_desc = "squares"
    save_name = f"{shapes_desc}_G{args.grid_size}"
    save_name += f"_S{args.min_size}-{args.max_size}{shuffle_str}"
    save_name += f"_R{args.num_replicates}"
    if args.min_col is not None:
        save_name += f"_mnc{args.min_col}"
    if args.max_col is not None:
        save_name += f"_mxc{args.max_col}"
    np.savez_compressed(
        str(Path(args.save_dir) / save_name) + ".npz", data=arr_list, areas=areas
    )
