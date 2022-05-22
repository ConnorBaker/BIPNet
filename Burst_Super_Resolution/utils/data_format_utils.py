from typing import Any
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
import cv2 as cv
from cv2 import Mat


def numpy_to_torch(a: NDArray[Any]) -> Tensor:
    return torch.from_numpy(a).float().permute(2, 0, 1)


def torch_to_numpy(a: Tensor) -> NDArray[Any]:
    return a.permute(1, 2, 0).numpy()


def torch_to_npimage(a: Tensor, unnormalize: bool = True) -> Mat:
    # If unnormalize, multiply by 255, else stay the same
    a_np = (torch_to_numpy(a) * (int(unnormalize)*254 + 1)).astype(np.uint8)
    return cv.cvtColor(a_np, cv.COLOR_RGB2BGR)


def npimage_to_torch(
    a: NDArray[Any], normalize: bool = True, input_bgr: bool = True
) -> Tensor:
    if input_bgr:
        a = cv.cvtColor(a, cv.COLOR_BGR2RGB)
    
    # If normalize, divide by 255.0
    a_t = numpy_to_torch(a) / (int(normalize)*254.0 + 1.0)

    return a_t


def convert_dict(base_dict: dict[str, Any], batch_sz: int) -> list[dict[str, Any]]:
    out_dict: list[dict[str, Any]] = []
    for b_elem in range(batch_sz):
        b_info: dict[str, Any] = {}
        for k, v in base_dict.items():
            if isinstance(v, (list, Tensor)):
                b_info[k] = v[b_elem]
        out_dict.append(b_info)

    return out_dict
