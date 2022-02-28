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


from typing import Callable, Optional

import numpy as np
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch


class InputTransformation(ABC):

    def __init__(self, id: str, embed_dim: int):
        self.id = id
        self.embed_dim = embed_dim

    @abstractmethod
    def __call__(self, x):
        assert False, "Should implement"


class SentenceBertInputTransform(InputTransformation):

    def __init__(self, sbert_model, id: str, name_embed: bool, name_embedder: Optional[Callable[[int], str]]):
        self.sbert_model: SentenceTransformer = sbert_model
        self.name_embed = name_embed
        self.name_embedder = name_embedder
        embed_dim = self.sbert_model.get_sentence_embedding_dimension()
        super().__init__(id, embed_dim=embed_dim)

    def __call__(self, input: np.ndarray) -> Tensor:
        assert input.ndim == 2, f"Input should be of ndim 2, got shape: {input.shape}"
        if self.name_embed:
            assert self.name_embedder is not None
            input_str = np.array([", ".join(map(self.name_embedder, list(xi))) for xi in input])
        else:
            input_str = np.array([", ".join(map(str, list(xi))) for xi in input])
        return torch.from_numpy(self.sbert_model.encode(input_str))
