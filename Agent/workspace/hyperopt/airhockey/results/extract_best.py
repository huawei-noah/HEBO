import os
from os.path import isfile
import sys
import json
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
json_filename = "best_agent_params.json"


def json_path(d):
    return os.path.join(root, d, json_filename)


def results_path(d):
    return os.path.join(root, d, "optimization_trajectory.csv")


def dirs():
    for d in os.listdir(root):
        if not os.path.isdir(os.path.join(root, d)):
            continue

        if isfile(results_path(d)):
            yield d


def main():
    for d in dirs():
        df = pd.read_csv(results_path(d))
        df.loc[df["y"].idxmin()].to_json(json_path(d))


if __name__ == "__main__":
    main()
