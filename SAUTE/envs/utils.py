import torch 
import numpy as np
from typing import Union
Array = Union[torch.Tensor, np.ndarray]

def angle_normalize(theta:Array, is_tensor:bool=True) -> Array:
    """Normalizes an angle theta to be between -pi and pi."""
    if is_tensor:
        torch_pi = torch.Tensor(np.asarray(np.pi))
        return ((theta + torch_pi) % (2 * torch_pi)) - torch_pi
    else:
        return (((theta+np.pi) % (2*np.pi)) - np.pi)

