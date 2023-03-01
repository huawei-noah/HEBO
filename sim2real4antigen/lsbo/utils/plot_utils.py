import os, sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = str(Path(os.path.realpath(__file__)).parent.parent.parent)

def plot_bo_results(exp_path):
    y = torch.load(os.path.join(exp_path, 'y.pt'))
    baseline = np.load(os.path.join(ROOT, f'saved_models/cnn/10_folds_mean_test_accuracy.npy'))
    plt.plot(np.maximum.accumulate(y), label='10-fold Test Accuracy with BO')
    plt.axhline(y=baseline, xmin=0, xmax=len(y), ls='--', c="black", label='CNN Baseline 10-fold Test Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join(exp_path, 'bo_results.pdf'))
    plt.close()