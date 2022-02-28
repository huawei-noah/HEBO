# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved. Redistribution and use in source and binary
# forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
# following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel
from torch import Tensor


class FastStringKernel(Kernel):

    def __init__(self, seq_length: int, alphabet_size: int, gap_decay=.5, match_decay=.8,
                 max_subsequence_length: int = 3, normalize=True, **kwargs):
        super(FastStringKernel, self).__init__(has_lengthscale=False, **kwargs)

        self.register_parameter(name='match_decay', parameter=torch.nn.Parameter(torch.tensor(match_decay).to(float)))
        self.register_parameter(name='gap_decay', parameter=torch.nn.Parameter(torch.tensor(gap_decay).to(float)))
        self.register_constraint("gap_decay", Interval(0, 1))
        self.register_constraint("match_decay", Interval(0, 1))
        self.max_subsequence_length = max_subsequence_length

        # Keep track of other kernel params
        self.maxlen = seq_length
        self.alphabet_size = alphabet_size
        self.normalize = normalize

        self.tril = torch.triu(torch.ones((self.maxlen, self.maxlen), dtype=torch.double), diagonal=1).to(
            kwargs['device'])
        self.exp = torch.ones(self.maxlen, self.maxlen, dtype=int).to(kwargs['device'])
        for i in range(self.maxlen - 1):
            self.exp[i, i + 1:] = torch.arange(self.maxlen - i - 1)

    def K_diag(self, X: Tensor):
        r"""
        Due to normalisation, the diag of the SSK is (1 ... 1)
        """
        return torch.ones(X.shape[:-1], dtype=torch.double)

    def forward(self, X1, X2, diag=False, last_dim_is_batch=False, **params):
        r"""
        Calulate SSK in a batch way
        Keeping notation of Beck (2017), with S,D,Kpp,Kp

        Args:
            X1: tensors of shape (# strings , # characters)
            X2: tensors of shape (# strings , # characters)

        Return:
            matrix K containing k(X1[i], X2[j]) at entry K[i, j]
        """

        if diag:
            raise ValueError()
        if X2 is None:
            X2 = X1
            self.symmetric = True
        else:
            self.symmetric = False
        X1_shape = X1.shape[0]
        X2_shape = X2.shape[0]

        # decay tensor
        self.D = self.calc_decay_matrix().to(X1)

        # Get one-hot encoding (# strings, #characters + 1, # alphabet)
        X1 = torch.nn.functional.one_hot(X1.to(int), self.alphabet_size).to(X1)
        X2 = torch.nn.functional.one_hot(X2.to(int), self.alphabet_size).to(X2)

        # get indicies of all possible pairings from X1 and X2 to permit efficient computation on GPU
        indicies_2, indicies_1 = torch.meshgrid(torch.arange(0, X2.shape[0]), torch.arange(0, X1.shape[0]))
        indicies = torch.cat([torch.reshape(indicies_1.T, (-1, 1)), torch.reshape(indicies_2.T, (-1, 1))], axis=1)

        if self.symmetric:
            # only calc upper matrix (the rest is filled after)
            indicies = indicies[indicies[:, 1] >= indicies[:, 0]]

        X1_full = torch.repeat_interleave(X1.unsqueeze(0), len(indicies), dim=0)[
            np.arange(len(indicies)), indicies[:, 0]]
        X2_full = torch.repeat_interleave(X2.unsqueeze(0), len(indicies), dim=0)[
            np.arange(len(indicies)), indicies[:, 1]]

        if not self.symmetric:
            # supplementary evaluations for normalization
            X1_full = torch.cat([X1_full, X1, X2], 0)
            X2_full = torch.cat([X2_full, X1, X2], 0)

        # S: similarity vector (# strings, #characters, # characters)
        S = torch.matmul(X1_full, torch.transpose(X2_full, 1, 2))

        # squared match coef
        match_sq = self.match_decay ** 2

        Kp = torch.ones(*[S.shape[0], self.maxlen, self.maxlen]).to(S)

        # do the rest
        for i in torch.arange(self.max_subsequence_length - 1):
            Kp = torch.multiply(S, Kp)
            Kp = match_sq * Kp
            Kp = torch.matmul(Kp, self.D)
            Kp = torch.matmul(self.D.T, Kp)

        # ultimate computation
        Kp = torch.multiply(S, Kp)
        k = Kp.sum((-2, -1)).unsqueeze(1) * match_sq

        # build the gram matrix and normalize
        if self.symmetric:
            # only fill top triangle if symmetric
            mask = torch.triu(torch.ones((X1_shape, X2_shape)), 0).to(S)
            non_zero = mask > 0
            k_results = torch.zeros((X1_shape, X2_shape)).to(S)
            k_results[non_zero] = k.squeeze()
            # full the rest
            k_results = k_results + k_results.T - torch.diag(k_results.diag())

            # normalise
            X_diag_Ks = torch.diag(k_results)
            norm = torch.matmul(X_diag_Ks[:, None], X_diag_Ks[None, :])
            k_results = torch.divide(k_results, torch.sqrt(norm))
        else:
            # Additional kernel computation at the end of k and reshape into gram matrix

            # NEEDS to speed up when predicting, most of norm elements are in gram matrix already

            X_diag_Ks = k[X1_shape * X2_shape:X1_shape * X2_shape + X1_shape].flatten()

            X2_diag_Ks = k[-X2_shape:].flatten()

            k = k[0:X1_shape * X2_shape]
            k_results = k.reshape(X1_shape, X2_shape)

            # normalise
            norm = torch.matmul(X_diag_Ks[:, None], X2_diag_Ks[None, :])
            k_results = torch.divide(k_results, torch.sqrt(norm))

        return k_results

    def calc_decay_matrix(self):
        r"""
        Get matrix D, the decay matrix
        """
        return torch.pow(self.gap_decay * self.tril, self.exp)


if __name__ == '__main__':
    seq_length = 20
    alphabet = np.arange(20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    str_kernel_torch = FastStringKernel(seq_length=seq_length, alphabet_size=len(alphabet), max_subsequence_length=3,
                                        gap_decay=1., match_decay=1., device=device).to(device)
    X = torch.from_numpy(np.random.randint(0, 10, (10, 20))).to(float).to(device)
    X2 = torch.from_numpy(np.random.randint(0, 10, (10, 20))).to(float).to(device)
    kk = str_kernel_torch(X, X2).evaluate()
    print(kk)

    seq_length = 5
    alphabet = np.arange(6)

    str_kernel_torch = FastStringKernel(seq_length=seq_length, alphabet_size=len(alphabet), max_subsequence_length=3,
                                        gap_decay=1., match_decay=1., device=device).to(device)
    X = torch.from_numpy(np.array([[0, 0, 1, 3, 0], [0, 2, 2, 2, 2]]).astype(float)).to(device)
    X2 = torch.from_numpy(np.array([[1, 0, 1, 0, 1], [0, 1, 2, 3, 0]]).astype(float)).to(device)
    kk = str_kernel_torch(X, X2).evaluate()
    print(kk)

    kk.sum().backward()
    for p in str_kernel_torch.parameters():
        if p.requires_grad:
            print(p.grad)
    kk = str_kernel_torch(X2).evaluate()
    print(kk)
