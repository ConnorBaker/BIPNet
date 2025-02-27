import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Any
from scipy import signal


def cal_ssim(
    img1: NDArray[Any], img2: NDArray[Any]
) -> tuple[float, NDArray[np.float64]]:
    K = [0.01, 0.03]
    L = 255
    kernelX: NDArray[np.float64] = cv2.getGaussianKernel(11, 1.5)
    window: NDArray[np.float64] = kernelX * kernelX.T

    # M,N = np.shape(img1)
    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1: NDArray[np.float64] = signal.convolve2d(img1, window, "same")
    mu2: NDArray[np.float64] = signal.convolve2d(img2, window, "same")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq: NDArray[np.float64] = (
        signal.convolve2d(img1 * img1, window, "same") - mu1_sq
    )
    sigma2_sq: NDArray[np.float64] = (
        signal.convolve2d(img2 * img2, window, "same") - mu2_sq
    )
    sigma12: NDArray[np.float64] = (
        signal.convolve2d(img1 * img2, window, "same") - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    mssim: float = ssim_map.mean()
    return mssim, ssim_map


"""
# Assuming single channel images are read. For RGB image, uncomment the following commented lines
img1 = cv2.imread('location_noisy',0)
#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('location_clean',0)
#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
"""
