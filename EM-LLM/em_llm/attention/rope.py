import torch
from typing import Union, Tuple

class RotaryEmbeddingESM(torch.nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(
        self, 
        dim: int, 
        base: Union[int, float] = 10000,
        distance_scale: Union[int, float] = 1,
        ext_factors=1.0
    ):
        super().__init__()
        self.base = base
        self.distance_scale = distance_scale

        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (
            ext_factors.to(torch.device("cuda")) * base ** (torch.arange(0, dim, 2, device="cuda", dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, x, length, right, cos, sin):
        dtype = x.dtype
        if cos.dim() == 2:
            cos = cos[right-length:right, :]
            sin = sin[right-length:right, :]
        elif cos.dim() == 3:
            cos = cos[:, right-length:right, :]
            sin = sin[:, right-length:right, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, right-length:right, :]
            sin = sin[:, :, right-length:right, :]
            
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def _update_cos_sin_tables(self, x, seq_dim):
        seq_len = x.size(seq_dim)
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            if x.dim() == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif x.dim() == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif x.dim() == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]
        return self._cos_cached, self._sin_cached

    def _update_cos_sin_tables_len(self, seq_len, device, dim = None):
        if seq_len > self._seq_len_cached:
            if dim is None:
                assert self._cos_cached is not None
                dim = self._cos_cached.dim()

            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
            freqs = torch.outer(t * self.distance_scale, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)
            if dim == 2:
                self._cos_cached = emb.cos()
                self._sin_cached = emb.sin()
            elif dim == 3:
                self._cos_cached = emb.cos()[None, :, :]
                self._sin_cached = emb.sin()[None, :, :]
            elif dim == 4:
                self._cos_cached = emb.cos()[None, None, :, :]
                self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached.to(device), self._sin_cached.to(device)

    def apply_rotary_pos_emb_one_angle(
        self, x: torch.Tensor, index
    ):
        dtype = x.dtype
        cos, sin = self._update_cos_sin_tables_len(index, x.device)
        if cos.dim() == 2:
            cos = cos[index-1:index, :]
            sin = sin[index-1:index, :]
        elif cos.dim() == 3:
            cos = cos[:, index-1:index, :]
            sin = sin[:, index-1:index, :]
        elif  cos.dim() == 4:
            cos = cos[:, :, index-1:index, :]
            sin = sin[:, :, index-1:index, :]
            
        return ((x.float() * cos) + (self.rotate_half(x).float() * sin)).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim= -2) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dim=seq_dim)
        return (
            self.apply_rotary_pos_emb(q, q.size(seq_dim), k.size(seq_dim), self._cos_cached.to(q.device), self._sin_cached.to(q.device)),
            self.apply_rotary_pos_emb(k, k.size(seq_dim), k.size(seq_dim), self._cos_cached.to(k.device), self._sin_cached.to(k.device)),
            )
