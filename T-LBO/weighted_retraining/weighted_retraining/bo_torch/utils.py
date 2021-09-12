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

import torch
from torch import Tensor


def within_bounds(X: Tensor, bounds: Tensor):
    return torch.all((X >= bounds[0]) * (X <= bounds[1]))


def put_max_in_bounds(X: Tensor, y: Tensor, bounds: Tensor):
    assert y.ndim == 1 or y.shape[-1] == 1, y.shape
    i = torch.argmax(y.flatten())
    bounds[1] = torch.maximum(X[i], bounds[1])
    bounds[0] = torch.minimum(X[i], bounds[0])
    return bounds


if __name__ == '__main__':
    X = torch.arange(12).reshape(3, 4)
    X[-1][0] = -1
    y = torch.arange(3).unsqueeze(1)

    bounds_ = torch.zeros(2, X.shape[1])
    bounds_[1] = torch.ones(X.shape[1])

    print(f"Original bounds: {bounds_}")
    print(f"Final bounds: {put_max_in_bounds(X, y, bounds_)}")
