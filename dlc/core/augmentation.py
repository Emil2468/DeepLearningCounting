#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""Augmentation utilities."""
#
# Implements various augmentation pipelines.
#
# References:
# [1]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78–82.
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl
#
# [2]
#   Yuhong Li and Xiaofan Zhang and Deming Chen (2018).
#   CSRNet: Dilated Convolutional Neural Networks for Understanding
#   the Highly Congested Scenes. CoRR, abs/1802.10062.
#   arXiv: http://arxiv.org/abs/1802.10062
#
# See also:
# - https://albumentations.ai/docs/
# - https://imgaug.readthedocs.io/en/latest/
# - https://albumentations.ai/docs/examples/tensorflow-example/

import abc
from albumentations.augmentations import transforms

import cv2
import imgaug
import albumentations
import tensorflow as tf
import numpy as np

from albumentations.core.transforms_interface import BasicTransform


class FixWeightsAnnotation(BasicTransform):
    """Transform to fix the weight annotations."""

    def __init__(self, weight=10.0, index=1, threshold=0.5, **kwargs):
        super(FixWeightsAnnotation, self).__init__(**kwargs)
        self.index = index
        self.weight = weight
        self.threshold = threshold

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, x, **_params):
        weights = x[:, :, self.index]
        weights = np.where(weights >= self.threshold, self.weight, 1.0)
        x[:, :, self.index] = weights
        return x


class AugmentationTransform(metaclass=abc.ABCMeta):
    """Define the interface of an augmentation transform."""

    @abc.abstractmethod
    def _augment(self, features, annotations):
        pass

    @tf.function
    def _augment_tf(self, features, annotations):
        return tf.numpy_function(self._augment, [features, annotations],
                                 [tf.float32, tf.float32])

    @tf.function
    def __call__(self, features, annotations):
        return self._augment_tf(features, annotations)


class SegmentationAugTransform0(AugmentationTransform):
    """Augmentation transform that implements that of [1]."""

    def __init__(self, fix_weights=True):
        # Transformations A (features and annotations).
        self._transform_a = albumentations.Compose([
            albumentations.crops.transforms.CropAndPad(
                percent=(0, 0.1),
                p=0.5,
            ),
            albumentations.imgaug.transforms.IAAPiecewiseAffine(
                0.05,
                p=0.3,
            ),
            albumentations.imgaug.transforms.Perspective(
                0.01,
                p=0.1,
            ),
        ])
        # Transformations B (features only)
        self._transform_b = albumentations.Compose([
            albumentations.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=(0.3, 1.2),
                p=0.3,
            ),
        ])
        self._weight_fixer = None
        if fix_weights:
            self._weight_fixer = FixWeightsAnnotation(always_apply=True)

    def _augment(self, features, annotations):
        # Augmentation
        augmented_a = self._transform_a(
            image=features,
            mask=annotations,
        )
        features = augmented_a["image"]
        augmented_b = self._transform_b(image=features)
        features = augmented_b["image"]
        annotations = augmented_a["mask"]
        if self._weight_fixer is not None:
            augmented_w = self._weight_fixer(image=annotations)
            annotations = augmented_w["image"]
        return features, annotations


class DensityAugTransform0(AugmentationTransform):
    """Augmentation transform that implements that of [1]."""

    def __init__(self):
        # Transformations B (features only)
        self._transform_b = albumentations.Compose([
            albumentations.RandomBrightnessContrast(
                brightness_limit=0,
                contrast_limit=(0.3, 1.2),
                p=0.3,
            ),
        ])

    def _augment(self, features, annotations):
        # Augmentation
        features = self._transform_b(image=features)["image"]
        return features, annotations


__all__ = ["SegmentationAugTransform0", "DensityAugTransform0"]
