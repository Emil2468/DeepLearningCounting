#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""This module contains code to create splits."""
import abc
import math
from typing import List, Optional, Union, Any, Tuple

import numpy as np
import geopandas as gpd
import shapely

from .common import load_geodataframe
from .types import GeoDataFrameSource

ObjectSplitterResult = Tuple[List[int], Optional[Any]]


class ObjectSplitter(metaclass=abc.ABCMeta):
    """Define an object splitter interface."""

    @abc.abstractmethod
    def run(
        self,
        objects: GeoDataFrameSource,
        splits: List[float],
        *,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> ObjectSplitterResult:
        """Create the object splits.

        Parameters
        ----------
        objects
            The objects database.
        splits
            The splits distribution. Should be in [0, 1].
        seed: optional
            A seed for the random number generator.
        verbose
            Display diagnostic information.

        Returns
        -------
        ObjectSplitResult
            A list of split classes and optionally implementation-specific \
                extra information.
        """


class SimpleSplitter(ObjectSplitter):

    def __init__(self, **kwargs):
        super(SimpleSplitter, self).__init__(**kwargs)

    def run(
        self,
        objects: GeoDataFrameSource,
        splits: List[float],
        *,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> ObjectSplitterResult:

        objects = load_geodataframe(objects, copy=False)
        n = len(objects)
        pairs = [(i, np.round(n * p)) for i, p in enumerate(splits, 1)]
        split_ids = np.zeros(size=n, dtype=int)
        next = 0
        for i, m in pairs:
            split_ids[next, m] = i
            next = m
        rng = np.random.default_rng(seed)
        rng.shuffle(split_ids)
        return list(split_ids), None


class LatitudeObjectSplitter(ObjectSplitter):
    """Create latitude-balanced object splits.

    Examples
    --------
    >>> splitter = LatitudeObjectSplitter()
    >>> splits, sampled_areas = splitter.run(tiles, splits=(0.20, 0.20),
    ...                                      seed=seed)
    """

    def __init__(
        self,
        bins: Union[str, int] = "auto",
        centroid_projection: Optional[str] = "EPSG:6933",
    ) -> None:
        self.bins = bins
        self.centroid_projection = centroid_projection

    def run(
        self,
        objects: GeoDataFrameSource,
        splits: List[float],
        *,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> ObjectSplitterResult:
        # References:
        # https://gis.stackexchange.com/a/390563

        total = np.sum(splits)
        if not 0.0 <= total <= 1.0:
            raise ValueError("Invalid splits distribution.")
        objects = load_geodataframe(objects, copy=False)

        rng = np.random.default_rng(seed)

        # We compute a histogram of the centroid latitudes
        # to create "latitudinal bins" from which to sample tiles
        # uniformly at random. The target number of tiles to sample from
        # each area is proportional to the bin counts.
        if self.centroid_projection is not None:
            centroids = objects.to_crs(self.centroid_projection).centroid
            centroids = centroids.to_crs(objects.crs)
        else:
            centroids = objects.centroid
        # The latitude is the y-coordinate of the centroid
        counts, bins = np.histogram(centroids.geometry.y)
        counts = np.array(counts, dtype=float)
        # Compute probability of area selection
        p = counts / np.sum(counts)
        # Compute area boxes
        minx = objects.bounds["minx"].min()
        maxx = objects.bounds["maxx"].max()
        areas = [
            shapely.geometry.box(minx, start, maxx, end)
            for start, end in zip(bins, bins[1:])
        ]
        n_objects = len(objects)
        object_splits = np.zeros(n_objects, dtype=int)
        for split_id, split_rel_size in enumerate(splits, 1):
            if split_rel_size == 0:
                continue
            n_selected_target = math.ceil(n_objects * split_rel_size)
            target_count_per_area = np.round(p * n_selected_target).astype(int)
            area_indices = np.argsort(target_count_per_area)
            if np.sum(target_count_per_area) > n_selected_target:
                # Corrects possible rounding issue
                area_idx = area_indices[-1]
                target_count_per_area[area_idx] -= 1
            for area_idx in area_indices:
                n = target_count_per_area[area_idx]
                if n == 0:
                    continue
                area = areas[area_idx]
                intersects = centroids.within(area).to_numpy()
                available = object_splits == 0
                candidates = np.logical_and(available, intersects)
                candidates = np.squeeze(np.argwhere(candidates))
                sample = rng.choice(candidates, size=n, replace=False)
                object_splits[sample] = split_id

        sampled_areas = gpd.GeoDataFrame(
            {
                "geometry": areas,
                "count": p
            },
            crs=objects.crs,
        )

        if verbose:
            idx = object_splits != 0
            n_selected = idx.sum()
            n_percent = n_selected / n_objects
            print("Selected {}/{} ({}) objects, target was {}.".format(
                n_selected,
                n_objects,
                n_percent,
                math.ceil(n_objects * total),
            ))
        return object_splits, sampled_areas
