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

import numpy as np
import torch

from weighted_retraining.weighted_retraining.metrics import ContrastiveLossTorch


def test_contrastive_loss():
    """ Test tensor contrastive loss """

    def contrast(df: float, dy: float, threshold: float) -> float:
        """ Contrastive loss for continuous labels """
        if dy > threshold:
            s = (2 - min(threshold, df) / threshold) * (dy - max(df, threshold))
        else:
            s = max(threshold, df) / threshold * (min(df, threshold) - dy)
        return max(0, s)

    ys = torch.rand(10, 1)
    embs = torch.rand(10, 3)
    constr_loss = ContrastiveLossTorch(.25)

    loss = constr_loss.build_loss_matrix(embs, ys)

    for i in range(loss.shape[0]):
        for j in range(loss.shape[1]):
            df_ = torch.linalg.norm(embs[i] - embs[j]).item()
            dy_ = torch.abs(ys[i] - ys[j]).item()
            if not np.isclose(loss[i, j].item(), contrast(df_, dy_, constr_loss.threshold)):
                return 0

    print(f"Passed: test_contrastive_loss")
    return 1


if __name__ == '__main__':
    test_contrastive_loss()
