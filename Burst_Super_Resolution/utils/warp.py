import torch
from torch import Tensor
import torch.nn.functional as F


def warp(
    feat: Tensor, flow: Tensor, mode: str = "bilinear", padding_mode: str = "zeros"
) -> Tensor:
    """
    warp an image/tensor (im2) back to im1, according to the optical flow im1 --> im2

    input flow must be in format (x, y) at every pixel
    feat: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow (x, y)

    """
    _, _, H, W = feat.size()

    # mesh grid
    rowv, colv = torch.meshgrid(
        [torch.arange(0.5, H + 0.5), torch.arange(0.5, W + 0.5)]
    )
    grid = torch.stack((colv, rowv), dim=0).unsqueeze(0).float().to(feat.device)
    grid = grid + flow

    # scale grid to [-1,1]
    grid_norm_c = 2.0 * grid[:, 0] / W - 1.0
    grid_norm_r = 2.0 * grid[:, 1] / H - 1.0

    grid_norm = torch.stack((grid_norm_c, grid_norm_r), dim=1)

    grid_norm = grid_norm.permute(0, 2, 3, 1)

    output = F.grid_sample(
        feat, grid_norm, mode=mode, padding_mode=padding_mode, align_corners=True
    )

    return output
