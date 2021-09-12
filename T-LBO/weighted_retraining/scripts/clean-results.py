"""
Script to clean results folders of all data
except the json and npz logs
"""

from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_dir", required=True, help="Root dir to traverse for cleaning"
)
parser.add_argument(
    "--dry-run",
    "-d",
    action="store_true",
    help="Flag to just print what would be deleted rather than actually deleting it",
)

if __name__ == "__main__":
    args = parser.parse_args()
    rm_list = []
    for f in Path(args.root_dir).rglob("*"):
        delete = (f.is_file() and f.name not in ["results.json", "results.npz"]) or (
            f.is_dir() and len(list(f.iterdir())) == 0
        )
        if delete:
            rm_list.append(f)

    # Do actual deletion?
    for f in rm_list:
        if args.dry_run:
            print(f"Would delete {f}")
        elif f.exists() and f.is_dir():
            f.rmdir()
        elif f.exists() and f.is_file():
            f.unlink()
