import os
import sys
from pathlib import Path

import torch
from gpytorch.kernels import MaternKernel

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
sys.path[0] = ROOT_PROJECT

from mcbo.models.gp.kernels import HEDKernel

def test_hed_kernel():
    base_kernel = MaternKernel()
    hed_num_embedders = 16
    n_cats_per_dim = [5, 6, 5, 6, 5, 6]
    hed_kernel = HEDKernel(base_kernel=base_kernel, hed_num_embedders=hed_num_embedders, n_cats_per_dim=n_cats_per_dim)

    x1 = torch.randint(0, 5, (2, 6))
    x2 = torch.randint(0, 5, (4, 6))

    k = hed_kernel(x1, x2)
    print(k)


if __name__ == "__main__":
    test_hed_kernel()
