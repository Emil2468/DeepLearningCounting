#   Author: Luis Diego García Castro and Adolfo Enrique García Castro

# References
# [1]
#   Kariryaa, A., et al. (2021). PlanetUNet codebase.
from typing import List, Union, Optional, Tuple, Mapping
from functools import partial
from multiprocessing import Lock
import pathlib

import rasterio
import rasterio.plot
import numpy as np
import tensorflow as tf

from .vendor import new_py_function
from .scalers import standardize_image_np

from .types import GeoDataFrameSource, PathSource
from .common import load_geodataframe

LocalStandardizationType = Union[Optional[float], List[Optional[float]]]


def decode_path(path: Union[str, bytes]) -> str:
    if isinstance(path, bytes):
        return path.decode()
    return path


def decode_paths(paths) -> List[List[str]]:
    decoded_paths = []
    if isinstance(paths, tf.RaggedTensor):
        row_splits = paths.row_splits.numpy()
        values = paths.values.numpy()
        for start, end in zip(row_splits, row_splits[1:]):
            decoded_paths.append(list(map(
                decode_path,
                values[start:end].tolist(),
            )))
    elif isinstance(paths, tf.Tensor):
        decoded_paths.append(list(map(decode_path, paths.numpy())))
    else:
        decoded_paths.append(list(map(decode_path, paths)))
    return decoded_paths


class ImageCache:

    def __init__(self):
        self._cache: Mapping[str, np.ndarray] = {}
        self._hits = 0
        self._misses = 0
        self._updates = 0
        self._size = 0
        self._write_lock = Lock()

    @property
    def hits(self):
        return self._hits

    @property
    def misses(self):
        return self._misses

    @property
    def updates(self):
        return self._updates

    @property
    def size(self):
        return self._size

    def read(self, key: str) -> Optional[np.ndarray]:
        if key in self._cache:
            self._hits += 1
        else:
            self._misses += 1

        return self._cache.get(key, None)

    def write(self, key: str, value: np.ndarray) -> None:
        self._write_lock.acquire()
        try:
            if key not in self._cache:
                self._updates += 1
            else:
                self._size -= self._cache[key].nbytes
            self._size += value.nbytes
            self._cache[key] = value
        finally:
            self._write_lock.release()

    def clear(self):
        self._write_lock.acquire()
        try:
            del self._cache
            self._cache = {}
            self._hits = 0
            self._misses = 0
            self._updates = 0
            self._size = 0
        finally:
            self._write_lock.release()


def load_image(
    path: str,
    *,
    masked: bool = False,
    cache: Optional[ImageCache] = None,
    dtype: Optional[str] = None,
) -> np.ndarray:
    if cache is not None:
        image = cache.read(path)
        if image is not None:
            return image
    # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
        with rasterio.open(path, "r") as src:
            image = src.read(masked=masked)
            if dtype is not None:
                image = image.astype(dtype)
            image = np.transpose(image, axes=[1, 2, 0])
            if cache is not None:
                cache.write(path, image)
            return image


def create_patch(
    image: np.ndarray,
    *,
    y_range: Tuple[int, int],
    x_range: Tuple[int, int],
    size: Tuple[int, int],
    dtype: str = "float32",
) -> np.ndarray:
    y_slice = slice(*y_range)
    x_slice = slice(*x_range)
    height, width = size
    shape = (height, width, image.shape[2])
    patch = np.zeros(shape, dtype=dtype)
    slice_height = y_range[1] - y_range[0]
    slice_width = x_range[1] - x_range[0]
    patch[0:slice_height, 0:slice_width, :] = image[y_slice, x_slice, :]
    return patch


class ImageLoader:

    def __init__(
        self,
        *,
        local_standardization_p: LocalStandardizationType = None,
        seed: Optional[int] = None,
        masked: bool = False,
        cache: Optional[ImageCache] = None,
        dtype: Optional[str] = "float32",
    ):
        self._rng = np.random.default_rng(seed)
        self._masked = masked
        self._cache = cache
        self._dtype = dtype

        if not isinstance(local_standardization_p, list):
            self._local_standardization_p = [local_standardization_p]
        else:
            self._local_standardization_p = local_standardization_p

    def load_np(self, spec):
        images_paths = decode_paths(spec["paths"])
        create_patch_fn = partial(
            create_patch,
            y_range=spec["y_slice"],
            x_range=spec["x_slice"],
            size=spec["size"],
            dtype=self._dtype,
        )
        load_image_fn = partial(
            load_image,
            masked=self._masked,
            cache=self._cache,
            dtype=self._dtype,
        )

        if len(images_paths) != len(self._local_standardization_p):
            msg = "Shape mismatch: images ({}) and local_standarization_p ({})"
            msg = msg.format(
                len(images_paths),
                len(self._local_standardization_p),
            )
            raise ValueError(msg)

        patches: List[np.ndarray] = []
        for i, image_paths in enumerate(images_paths):
            stack: List[np.ndarray] = []
            standardize_probability = self._local_standardization_p[i]
            standardize_type = None
            for path in image_paths:
                image = load_image_fn(path)
                # Define if/how to standardize the patch
                if standardize_probability is not None:
                    standardize_type = "image"
                    if self._rng.uniform(size=1) < standardize_probability:
                        standardize_type = "patch"
                # Create patch
                patch = None
                if standardize_type == "image":
                    image = standardize_image_np(image, axis=(0, 1))
                    patch = create_patch_fn(image)
                elif standardize_type == "patch":
                    patch = create_patch_fn(image)
                    patch = standardize_image_np(patch, axis=(0, 1))
                else:
                    patch = create_patch_fn(image)
                stack.append(patch)
            patches.append(np.dstack(stack))

        if len(patches) == 1:
            return patches[0]
        return patches

    def load(self, spec):
        # Wrap function to use with Tensorflow
        if isinstance(spec["paths"], tf.RaggedTensor):
            return new_py_function(
                self.load_np,
                [spec],
                [self._dtype, self._dtype],
            )
        return new_py_function(
            self.load_np,
            [spec],
            self._dtype,
        )

    def preload_cache(self, generator) -> None:
        if self._cache is None:
            raise ValueError("No cache")
        for path in generator():
            value = load_image(path, dtype=self._dtype)
            self._cache.write(str(path), value)
