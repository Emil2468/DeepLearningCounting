#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""Utility functions for image feature scaling."""
#
# Implements various feature scalers.
#
# References:
# [1]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78-82
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl
# [2]
#   Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12,
#   pp. 2825-2830, 2011.
#   https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
# [3]
#   https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization

import numpy as np


def image_normalize(im, axis=(0, 1), c=1e-8):
    """Normalize to zero mean and unit standard deviation along the given axis"""
    return (im - im.mean(axis)) / (im.std(axis) + c)


def standardize_image_np(x, axis=(0, 1), c=1e-8):
    return image_normalize(x, axis=axis, c=c)


# Ported from [1] with protection against division by 0 as in [3].
def alt_standardize_image_np(x, axis=(0, 1)):
    """Standardize x to have 0 mean and unit variance."""
    n = x.shape[0] * x.shape[1]
    z = x - np.nanmean(x, axis=axis, dtype="float64")
    z /= np.maximum(np.nanstd(x, axis=axis, dtype="float64"), 1.0 / n)
    return z.astype(x.dtype)


# Similar to that of [2].
def normalize_image_np(x, axis=(0, 1), c=1e-8, clip=True):
    """Normalize x in the [0,1] range."""
    x_min = np.nanmin(x, axis=axis)
    x_max = np.nanmax(x, axis=axis)
    x_scaled = (x - x_min) / np.maximum((x_max - x_min), c)
    if clip:
        np.clip(x_scaled, 0.0, 1.0, out=x_scaled)
    return x_scaled


__all__ = ["standarize_image_np", "normalize_image_np"]
