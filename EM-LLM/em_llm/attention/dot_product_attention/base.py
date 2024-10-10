import torch

class MultiStageDotProductAttention:
    def __init__(
        self, 
        q_shape,
        dtype,
        device,
        output_softmax_denom,
    ):
        self.q_shape = q_shape
        self.dtype = dtype
        self.device = device
        self.end = False
        self.ret = torch.zeros(
            q_shape, dtype=dtype, device=device
        )
        self.score_list = []
        self.output_softmax_denom = output_softmax_denom
        self.softmax_denom = torch.zeros(
            q_shape[:-1] + (1,), dtype=dtype, device=device
        )

    def append(
        self, 
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        sliding_window=None, complement_sliding_window: bool = False,
        end=False, get_score=False,
        *args, **kwargs
    ):
        raise NotImplementedError


    def get_result(self):
        if self.output_softmax_denom:
            ret = self.ret, self.score_list, self.softmax_denom
        else:
            ret = self.ret, self.score_list
        return ret