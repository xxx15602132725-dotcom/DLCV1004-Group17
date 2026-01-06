import torch
import torch.nn.functional as F

"""
Sobel Edge Detection Utility Module
Used to calculate image gradient edges to support structural constraint Loss calculation.
"""


def get_sobel_kernel(device):
    """
    Create Sobel convolution kernels for extracting horizontal and vertical edges.
    Returns: kernel_x, kernel_y
    """
    # Define X-direction kernel (detects vertical edges)
    kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    # Define Y-direction kernel (detects horizontal edges)
    kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    return kernel_x, kernel_y


def detect_edges(images):
    """
    Perform differentiable edge detection on input images.
    Input: images (B, C, H, W) range [-1, 1]
    Output: edges (B, 1, H, W) range [0, 1]
    """
    # 1. If color image, convert to grayscale first (weighted average method)
    if images.shape[1] == 3:
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
    else:
        gray = images

    # 2. Get convolution kernels and compute gradients
    device = images.device
    kx, ky = get_sobel_kernel(device)

    # Keep dimensions unchanged (padding=1)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)

    # 3. Calculate gradient magnitude
    # Add epsilon 1e-8 to prevent error when calculating sqrt of 0 (derivative stability)
    edges = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    # 4. Normalize to 0-1 range for easier MSE Loss calculation
    edges = edges / (edges.max() + 1e-8)

    return edges