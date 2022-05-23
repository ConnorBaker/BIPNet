from typing import Any, Literal, Optional, overload
from cv2 import Mat
import torch
from torch import Tensor
import numpy as np
import utils.data_format_utils as df_utils
from data_processing.camera_pipeline import (
    apply_gains,
    apply_ccm,
    apply_smoothstep,
    gamma_compression,
)


class SimplePostProcess:
    def __init__(
        self,
        gains: bool = True,
        ccm: bool = True,
        gamma: bool = True,
        smoothstep: bool = True,
        return_np: bool = False,
    ):
        self.gains = gains
        self.ccm = ccm
        self.gamma = gamma
        self.smoothstep = smoothstep
        self.return_np = return_np

    def process(self, image: Tensor, meta_info: dict[str, Any]) -> Tensor:
        return process_linear_image_rgb(
            image,
            meta_info,
            self.gains,
            self.ccm,
            self.gamma,
            self.smoothstep,
            self.return_np,
        )


@overload
def process_linear_image_rgb(
    image: Tensor,
    meta_info: dict[str, Any],
    gains: bool = True,
    ccm: bool = True,
    gamma: bool = True,
    smoothstep: bool = True,
    return_np: Literal[True] = True,
) -> Mat:
    ...


@overload
def process_linear_image_rgb(
    image: Tensor,
    meta_info: dict[str, Any],
    gains: bool = True,
    ccm: bool = True,
    gamma: bool = True,
    smoothstep: bool = True,
    return_np: Literal[False] = False,
) -> Tensor:
    ...


def process_linear_image_rgb(
    image: Tensor,
    meta_info: dict[str, Any],
    gains: bool = True,
    ccm: bool = True,
    gamma: bool = True,
    smoothstep: bool = True,
    return_np: bool = False,
) -> Mat | Tensor:
    if gains:
        image = apply_gains(
            image, meta_info["rgb_gain"], meta_info["red_gain"], meta_info["blue_gain"]
        )

    if ccm:
        image = apply_ccm(image, meta_info["cam2rgb"])

    if meta_info["gamma"] and gamma:
        image = gamma_compression(image)

    if meta_info["smoothstep"] and smoothstep:
        image = apply_smoothstep(image)

    image = image.clamp(0.0, 1.0)

    if return_np:
        return df_utils.torch_to_npimage(image)
    else:
        return image


class BurstSRPostProcess:
    def __init__(
        self,
        no_white_balance: bool = False,
        gamma: bool = True,
        smoothstep: bool = True,
        return_np: bool = False,
    ):
        self.no_white_balance = no_white_balance
        self.gamma = gamma
        self.smoothstep = smoothstep
        self.return_np = return_np

    def process(
        self,
        image: Tensor,
        meta_info: dict[str, Any],
        external_norm_factor: Optional[float] = None,
    ):
        return process_burstsr_image_rgb(
            image,
            meta_info,
            external_norm_factor=external_norm_factor,
            no_white_balance=self.no_white_balance,
            gamma=self.gamma,
            smoothstep=self.smoothstep,
            return_np=self.return_np,
        )


def process_burstsr_image_rgb(
    im: Tensor,
    meta_info: dict[str, Any],
    return_np: bool = False,
    external_norm_factor: Optional[float] = None,
    gamma: bool = True,
    smoothstep: bool = True,
    no_white_balance: bool = False,
) -> Mat:
    im *= meta_info.get("norm_factor", 1.0)

    if not meta_info.get("black_level_subtracted", False):
        im -= torch.tensor(meta_info["black_level"])[[0, 1, -1]].view(3, 1, 1)

    if not meta_info.get("while_balance_applied", False) and not no_white_balance:
        im *= (
            torch.tensor(meta_info["cam_wb"])[[0, 1, -1]].view(3, 1, 1)
            / torch.tensor(meta_info["cam_wb"])[1]
        )

    im_out = im

    if external_norm_factor is None:
        im_out /= im_out.max()
    else:
        im_out /= external_norm_factor

    im_out = im_out.clamp(0.0, 1.0)

    if gamma:
        im_out = im_out ** (1.0 / 2.2)

    if smoothstep:
        # Smooth curve
        im_out = 3 * im_out**2 - 2 * im_out**3

    if return_np:
        im_out = im_out.permute(1, 2, 0).numpy() * 255.0
        im_out = im_out.astype(np.uint8)

    return im_out
