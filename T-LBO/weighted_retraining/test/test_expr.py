# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
from pathlib import Path

import torch


def test_get_rec_error():
    device = 0
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
    }

    torch.cuda.set_device(device)

    model = instentiate_expr_model()
    datamodule = instentiate_expr_datamodule()

    n = 1000
    zs = torch.randn(n, model.vae.latent_dim)  # .to(**tkwargs)
    one_hots = torch.from_numpy(datamodule.data_train[:n])  # .to(**tkwargs)

    errors = get_rec_x_error(
        model=model,
        tkwargs=tkwargs,
        one_hots=one_hots,
        zs=zs
    )

    assert errors.shape == (n, 1)
    print("Passed `test_get_rec_error`")
    return 1


def test_get_rec_error_emb():
    device = 0
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
    }

    torch.cuda.set_device(device)

    model = instentiate_expr_model()
    datamodule = instentiate_expr_datamodule()
    n = 1000
    exprs = datamodule.expr_train[:n]

    errors = get_rec_error_emb(
        model=model,
        tkwargs=tkwargs,
        exprs=exprs
    )

    assert errors.shape == (len(exprs), 1), errors.shape
    print("Passed `test_get_rec_error_emb`")
    return 1


def test_get_rec_z_error():
    device = 0
    tkwargs = {
        "dtype": torch.float,
        "device": torch.device(f"cuda:{device}" if torch.cuda.is_available() and device is not None else "cpu"),
    }

    torch.cuda.set_device(device)

    model = instentiate_expr_model()
    n = 1000
    zs = torch.randn(n, model.vae.latent_dim)

    errors = get_rec_z_error(
        model=model,
        tkwargs=tkwargs,
        zs=zs
    )

    assert errors.shape == (len(zs), 1), errors.shape
    print("Passed `test_get_rec_z_error`")
    return 1


if __name__ == '__main__':
    ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent.parent)
    sys.path[0] = ROOT_PROJECT

    from weighted_retraining.weighted_retraining.expr.expr_data import get_rec_x_error, get_rec_error_emb, \
        get_rec_z_error

    from weighted_retraining.test.utils_test_expr import instentiate_expr_model, instentiate_expr_datamodule

    MAX_LEN = 15

    test_get_rec_z_error()
    test_get_rec_error()
    test_get_rec_error_emb()
