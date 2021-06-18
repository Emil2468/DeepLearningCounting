#   Author: Luis Diego García Castro and Adolfo Enrique García Castro

import abc
import pathlib
from typing import Any, Optional, List, Mapping, Tuple, Union

import tensorflow as tf
import numpy as np
import geopandas as gpd

from .types import GeoDataFrameSource, PathSource
from .common import load_geodataframe
from .patches import compute_patch_slices


def summarize_spec(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return [summarize_spec(y) for y in x]
    return (x.shape, x.dtype)


class DatasetGenerator(metaclass=abc.ABCMeta):
    """Generate datasets from objects in a database (geodataframe).

    Parameters
    ----------
    db
        The objects database.
    splits
        The object splits.
    splits_map
        A mapping of names (e.g. validation) to the labels used in the splits.
    seed
        A seed for the random state.
    """

    def __init__(
        self,
        db: GeoDataFrameSource,
        splits: List[Any],
        splits_map: Mapping[str, Any],
        *,
        seed: Optional[int] = None,
    ) -> None:
        self._db = load_geodataframe(db, copy=True)
        self._db["split"] = splits
        self._splits_map = splits_map
        self._seed = seed

    def _get_split_objects(self, split: str) -> gpd.GeoDataFrame:
        if split not in self._splits_map:
            raise ValueError("Invalid split name.")

        split = self._splits_map[split]
        return self._db.query(f"split == {split}", inplace=False)


class ScalarDatasetGenerator(DatasetGenerator):
    """Generate scalar datasets from objects in a database."""

    def get(self, split: str, keys: Union[str, List[str]]) -> tf.data.Dataset:
        objects = self._get_split_objects(split)
        if isinstance(keys, str):
            return tf.data.Dataset.from_tensor_slices(objects[keys])
        datasets = tuple(
            [tf.data.Dataset.from_tensor_slices(objects[key]) for key in keys])
        return tf.data.Dataset.zip(datasets)


def get_sequential_images_generator(
    images,
    heights,
    widths,
    *,
    shuffle: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
):
    indices = np.arange(0, len(images), dtype=int)
    if verbose:
        print(f"Cardinality: {len(indices)}")

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    def _sequential_images():
        for index in indices:
            paths = images[index]
            if isinstance(paths, list):
                paths = tf.ragged.constant(paths)
            yield {
                "paths": paths,
                "y_slice": (0, int(heights[index])),
                "x_slice": (0, int(widths[index])),
                "size": (int(heights[index]), int(widths[index])),
            }

    return _sequential_images


def get_sequential_patches_generator(
    images,
    heights,
    widths,
    patch_size: Tuple[int, int],
    *,
    patch_stride: Optional[Tuple[int, int]] = None,
    patch_padding: Tuple[int, int] = (0, 0),
    shuffle: bool = False,
    seed: Optional[int] = None,
    verbose: bool = False,
):
    if patch_stride is None:
        patch_stride = patch_size

    patch_specs = []
    for paths, height, width in zip(images, heights, widths):
        if isinstance(paths, list):
            paths = tf.ragged.constant(paths)
        pad_y, pad_x = patch_padding
        stride_y, stride_x = patch_stride
        size_y, size_x = patch_size
        if size_y > height:
            pad_y = (size_y - height) // 2 + 1
            stride_y = size_y
        if size_x > width:
            pad_x = (size_x - width) // 2 + 1
            stride_x = size_x
        specs = compute_patch_slices(
            (height, width),
            patch_size,
            (stride_y, stride_x),
            (pad_y, pad_x),
        )
        for spec in specs:
            spec["y_slice"] = tuple(int(x) for x in spec["y_slice"])
            spec["x_slice"] = tuple(int(x) for x in spec["x_slice"])
            spec["size"] = patch_size
            spec["paths"] = paths
            patch_specs.append(spec)

    if verbose:
        print(f"Cardinality: {len(patch_specs)}")

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(patch_specs)

        def _sequential_patches():
            for spec in patch_specs:
                yield spec

        return _sequential_patches
    else:

        def _sequential_patches():
            for spec in patch_specs:
                yield spec

        return _sequential_patches


def get_random_patches_generator(
    images,
    heights,
    widths,
    patch_size: Tuple[int, int],
    *,
    seed: Optional[int] = None,
    verbose: bool = False,
):
    rng = np.random.default_rng(seed)

    def _random_patches():
        while True:
            index = rng.integers(0, len(images), size=1)[0]
            height = heights[index]
            if height > patch_size[0]:
                y_start = rng.integers(0, height - patch_size[0], size=1)[0]
                y_end = y_start + patch_size[0]
            else:
                y_start = 0
                y_end = height
            width = widths[index]
            if width > patch_size[1]:
                x_start = rng.integers(0, width - patch_size[1], size=1)[0]
                x_end = x_start + patch_size[1]
            else:
                x_start = 0
                x_end = width

            paths = images[index]
            if isinstance(paths, list):
                paths = tf.ragged.constant(paths)
            yield {
                "paths": paths,
                "y_slice": (int(y_start), int(y_end)),
                "x_slice": (int(x_start), int(x_end)),
                "size": patch_size,
            }

    return _random_patches


class ImageDatasetGenerator(DatasetGenerator):

    def __init__(
        self,
        db: GeoDataFrameSource,
        *,
        splits: np.ndarray,
        splits_map: Mapping[str, Any],
        image_keys: Union[List[str], Tuple[List[str], List[str]]],
        input_base_path: Optional[PathSource] = None,
        seed: Optional[int] = None,
        width_key: str = "scalars_width",
        height_key: str = "scalars_height",
    ) -> None:
        self._db = load_geodataframe(db, copy=True)
        self._db["split"] = splits
        self._splits_map = splits_map
        self._image_keys = image_keys
        self._width_key = width_key
        self._height_key = height_key
        if input_base_path is not None:
            self._input_base_path = pathlib.Path(input_base_path)
        else:
            self._input_base_path = None
        self._seed = seed

    def _get_paths(self, row, keys):
        paths = []
        for key in keys:
            path = row[key]
            if self._input_base_path is not None:
                path = str(self._input_base_path.joinpath(path))
            paths.append(path)
        return paths

    def _get_images(self, row):
        if isinstance(self._image_keys, tuple):
            images = []
            for keys in self._image_keys:
                images.append(self._get_paths(row, keys))
            return images
        else:
            return self._get_paths(row, self._image_keys)

    def _get_output_signature(self):
        if isinstance(self._image_keys, tuple):
            paths_spec = tf.RaggedTensorSpec.from_value(
                tf.ragged.constant(self._image_keys))
        else:
            paths_spec = tf.TensorSpec(
                shape=(len(self._image_keys),),
                dtype=tf.string,
            )
        return {
            "paths": paths_spec,
            "y_slice": tf.TensorSpec(shape=(2,), dtype=tf.int32),
            "x_slice": tf.TensorSpec(shape=(2,), dtype=tf.int32),
            "size": tf.TensorSpec(shape=(2,), dtype=tf.int32),
        }

    def get_sequential_images(
        self,
        split: str,
        *,
        shuffle: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> tf.data.Dataset:
        objects = self._get_split_objects(split)
        images = objects.apply(self._get_images, axis=1).tolist()
        heights = objects[self._height_key].tolist()
        widths = objects[self._width_key].tolist()
        generator = get_sequential_images_generator(
            images,
            heights,
            widths,
            shuffle=shuffle,
            seed=seed,
            verbose=verbose,
        )
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=self._get_output_signature(),
        )

    def get_sequential_patches(
        self,
        split: str,
        patch_size: Tuple[int, int],
        *,
        patch_stride: Optional[Tuple[int, int]] = None,
        patch_padding: Tuple[int, int] = (0, 0),
        shuffle: bool = False,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> tf.data.Dataset:
        objects = self._get_split_objects(split)

        images = objects.apply(self._get_images, axis=1).tolist()
        heights = objects[self._height_key].tolist()
        widths = objects[self._width_key].tolist()

        generator = get_sequential_patches_generator(
            images,
            heights,
            widths,
            patch_size,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            shuffle=shuffle,
            seed=seed,
            verbose=verbose,
        )

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=self._get_output_signature(),
        )

    def get_random_patches(
        self,
        split: str,
        patch_size: Tuple[int, int],
        *,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> tf.data.Dataset:
        objects = self._get_split_objects(split)

        images = objects.apply(self._get_images, axis=1).tolist()
        heights = objects[self._height_key].tolist()
        widths = objects[self._width_key].tolist()

        generator = get_random_patches_generator(
            images,
            heights,
            widths,
            patch_size,
            seed=seed,
            verbose=verbose,
        )

        return tf.data.Dataset.from_generator(
            generator,
            output_signature=self._get_output_signature(),
        )

    def get_cache_preload_generator(self, split: str):
        objects = self._get_split_objects(split)
        images = objects.apply(self._get_images, axis=1).tolist()

        def generator():
            for image in images:
                for path in image:
                    if isinstance(path, list):
                        for x in path:
                            yield x
                    else:
                        yield path

        return generator
