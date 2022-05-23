import math
import torch
from torch import Tensor
import torch.nn.functional as F


def gauss_1d(
    sz: int, sigma: float, center: Tensor, end_pad: int = 0, density: bool = False
) -> Tensor:
    """Returns a 1-D Gaussian"""
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma**2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(
    sz: int | tuple[int, int],
    sigma: float | tuple[float, float],
    center: list[float] | tuple[float, ...] | Tensor,
    end_pad: tuple[int, int] = (0, 0),
    density: bool = False,
) -> Tensor:
    """Returns a 2-D Gaussian"""
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    if isinstance(sz, int):
        sz = (sz, sz)

    if isinstance(center, (list, tuple)):
        center = torch.tensor(center).view(1, 2)

    return gauss_1d(sz[0], sigma[0], center[:, 0], end_pad[0], density).reshape(
        center.shape[0], 1, -1
    ) * gauss_1d(sz[1], sigma[1], center[:, 1], end_pad[1], density).reshape(
        center.shape[0], -1, 1
    )


def get_gaussian_kernel(sd: float) -> tuple[Tensor, int]:
    """Returns a Gaussian kernel with standard deviation sd"""
    ksz = int(4 * sd + 1)
    assert ksz % 2 == 1
    k_t = gauss_2d(ksz, sd, (0.0, 0.0), density=True)
    k_t /= k_t.sum()
    return k_t.unsqueeze(0), ksz


def apply_kernel(im: Tensor, ksz: int, gauss_kernel: Tensor) -> Tensor:
    shape = im.shape
    im = im.view(-1, 1, *im.shape[-2:])

    pad = [ksz // 2, ksz // 2, ksz // 2, ksz // 2]
    im = F.pad(im, pad, mode="reflect")
    im_mean = F.conv2d(im, gauss_kernel).view(shape)
    return im_mean


def match_colors(
    im_ref: Tensor, im_q: Tensor, im_test: Tensor, ksz: int, gauss_kernel: Tensor
) -> tuple[Tensor, Tensor]:
    """Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
    im_test
    """
    gauss_kernel = gauss_kernel.to(im_ref.device)
    bi = 5

    # Apply Gaussian smoothing
    im_ref_mean = apply_kernel(im_ref, ksz, gauss_kernel)[
        :, :, bi:-bi, bi:-bi
    ].contiguous()
    im_q_mean = apply_kernel(im_q, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()

    im_ref_mean_re = im_ref_mean.view(*im_ref_mean.shape[:2], -1)
    im_q_mean_re = im_q_mean.view(*im_q_mean.shape[:2], -1)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all: list[Tensor] = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        # See deprecation at https://pytorch.org/docs/master/generated/torch.lstsq.html?highlight=lstsq#torch.lstsq
        # X = torch.lstsq(B, A).solution[:A.size(1)] should be replaced with X = torch.linalg.lstsq(A, B).solution
        # So, assuming iq.t().size(1)==3, we replace
        # c = torch.lstsq(ir.t(), iq.t()).solution[:3]
        # with
        c: Tensor = torch.linalg.lstsq(iq.t(), ir.t()).solution
        c_mat_all.append(c)

    c_mat = torch.stack(c_mat_all, dim=0)
    im_q_mean_conv = (
        torch.matmul(im_q_mean_re.permute(0, 2, 1), c_mat)
        .permute(0, 2, 1)
        .view(im_q_mean.shape)
    )

    err: Tensor = ((im_q_mean_conv - im_ref_mean) * 255.0).norm(dim=1)

    thresh = 20

    # If error is larger than a threshold, ignore these pixels
    valid: Tensor = err < thresh

    pad = (im_q.shape[-1] - valid.shape[-1]) // 2
    pad = [pad, pad, pad, pad]
    valid = F.pad(valid, pad)

    upsample_factor = im_test.shape[-1] / valid.shape[-1]
    valid = F.interpolate(
        valid.unsqueeze(1).float(),
        scale_factor=upsample_factor,
        mode="bilinear",
        align_corners=True,
    )
    valid = valid > 0.9

    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)

    return im_t_conv, valid
