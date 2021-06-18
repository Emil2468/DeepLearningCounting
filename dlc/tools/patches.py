#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""Patch creation utility functions."""

# References:
# [1]
#     Dumoulin, V. & Visin, F. (2018)
#     A guide to convolution arithmetic for deep learning.
#     arXiv: https://arxiv.org/abs/1603.07285
# [2]
#   Abadi, M. et al. (2015)
#   TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.
#   URL: https://www.tensorflow.org/api_docs/python/tf/image/extract_patches

import math
from typing import Tuple


def compute_patch_slices(img_size, patch_size, stride, padding):
    """Compute patch slices assuming an interface similar to \
        that used in [2]."""
    y_size, x_size = img_size
    y_ksize, x_ksize = patch_size
    y_stride, x_stride = stride
    y_pad, x_pad = padding
    y_n = int(math.floor((y_size + 2 * y_pad - y_ksize) / y_stride + 1))
    x_n = int(math.floor((x_size + 2 * x_pad - x_ksize) / x_stride + 1))
    n = y_n * x_n
    patch_specs = []
    y_lim = y_size + y_pad
    x_lim = x_size + x_pad
    y = -y_pad
    x = -x_pad
    for _ in range(n):
        y_start = y
        y_end = min(y_start + y_ksize, y_size)
        if y_start < 0:
            y_start = 0

        x_start = x
        x_end = min(x_start + x_ksize, x_size)
        if x_start < 0:
            x_start = 0

        patch_spec = {
            "y_slice": (y_start, y_end),
            "x_slice": (x_start, x_end),
        }
        patch_specs.append(patch_spec)

        x += x_stride
        if x + x_ksize > x_lim:
            x = -x_pad
            y += y_stride
            if y + y_ksize > y_lim:
                break

    assert len(patch_specs) == n
    return patch_specs
