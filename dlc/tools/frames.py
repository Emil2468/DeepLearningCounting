#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
#   Edited by: Emil Møller Hansen
"""Frame creators and related functions."""
# References
# [1]
#   Kariryaa, A., et al. (2021). PlanetUNet codebase.
# [2]
#   Brandt, M., Tucker, C., Kariryaa, A., et al. (2020).
#   An unexpectedly large count of trees in the West African Sahara and Sahel.
#   Nature, 587(7832), 78–82.
#   DOI:  https://doi.org/10.1038/s41586-020-2824-5
#   Code: https://git.io/Jsivl

import os
import abc
import pathlib
import dataclasses
import multiprocessing
from functools import partial
from itertools import product
from typing import Callable, Any, Union, Optional, Tuple, List
from rasterio.windows import Window
import scipy.ndimage
import skimage.filters
import rasterio
import rasterio.mask
import rasterio.warp
import rasterio.features
import rasterio.plot
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from .types import GeoDataFrameSource, PathSource, CountHeuristicFuntion
from .common import load_geodataframe


def get_frame_filename(
    area_id: int,
    tile_id: int,
    suffix: str,
) -> str:
    """Get the filename for the frame."""
    return "./{}-{}_{}.tif".format(
        area_id,
        tile_id,
        suffix,
    )


@dataclasses.dataclass
class FrameDataCreatorResult:
    """Define the result of a frame data creator.

    Parameters
    ----------
    tile_id
        The tile index.
    area_id
        The area index.
    key
        The key for the result (e.g. segmentation-mask).
    payload
        The derived data.
    """

    tile_id: int
    area_id: int
    key: str
    payload: Any


class FrameDataCreator(metaclass=abc.ABCMeta):
    """Define interface of a frame data creator.

    Notes
    -----
    A frame data creator produces derived data for a particular frame \
        which is defined as the intersection of a tile and an area.
    """

    def get_tile_id(self, area: Any, tile_id: Optional[int] = None) -> int:
        """Get the tile id for a given area."""
        if area["n_tiles"] == 0:
            raise ValueError("Area has no tiles.")
        if tile_id is None:
            if area["n_tiles"] > 1:
                raise ValueError("Area has multiple tiles.")
            return area["tile_ids"][0]
        else:
            if tile_id not in area["tile_ids"]:
                raise ValueError("Invalid tile")
            return tile_id

    @abc.abstractmethod
    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        overwrite: bool = True,
        verbose: bool = True,
    ) -> FrameDataCreatorResult:
        """Run the frame creator.

        Parameters
        ----------
        area_id
            The index of the area in the database.
        tile_id: optional
            The index of the tile in the database.
        verbose
            Print diagnostic information.

        Returns
        -------
        FrameDataCreatorResult
            The result.
        """


class FrameDataRasterCreator(FrameDataCreator):

    @property
    def only_filename(self) -> bool:
        return self._only_filename

    @only_filename.setter
    def only_filename(self, value: bool) -> None:
        self._only_filename = value


class SegmentationMaskFrameCreator(FrameDataRasterCreator):
    """Create the segmentation mask for an area.

    Parameters
    ----------
    input_base_path
        The path to the folder containing tile images.
    output_base_path
        The path to the folder in which to store the created frames.
    areas
        The areas database.
    tiles
        The tiles database.
    polygons
        The polygons database.
    key
        The key for the result.

    Examples
    --------
    >>> creator = SegmentationMaskFrameCreator("./tile_images",
    ...                                        "./frames",
    ...                                        areas=areas,
    ...                                        tiles=tiles,
    ...                                        polygons=polygons,
    ...                                        )
    >>> result = creator.run(area_id)

    Notes
    -----
    Adapted from [1].
    """

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        *,
        key: str = "segmentation-mask",
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._key = key
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))
        if overwrite or not output_path.exists():
            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": window.height,
                        "width": window.width,
                        "transform": transform,
                        "dtype": np.float32,
                        "nodata": None,
                    })

                    out_shape = (window.height, window.width)
                    polygons_in_area = self._polygons["area_id"] == area_id
                    polygons_in_area = self._polygons[polygons_in_area]
                    output = np.zeros(
                        (1, window.height, window.width),
                        dtype=np.float32,
                    )
                    if len(polygons_in_area) > 0:
                        output[0, :, :] = rasterio.features.geometry_mask(
                            polygons_in_area.geometry,
                            out_shape=out_shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        ).astype(np.float32)

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(output)

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


class SegmentationBoundaryWeightsFrameCreator(FrameDataRasterCreator):
    """Create boundary weights between close polygons.

    Parameters
    ----------
    input_base_path
        The path to the folder containing tile images.
    output_base_path
        The path to the folder in which to store the created frames.
    areas
        The areas database.
    tiles
        The tiles database.
    polygons
        The polygons database.
    key: optional
        The key for the result.
    scale: optional
        The scale to resize the polygons.
    only_filename: optional
        Output the filename instead of the full path.

    Examples
    --------
    >>> creator = SegmentationBoundaryWeightsFrameCreator("./tile_images",
    ...                                                   "./frames",
    ...                                                   areas=areas,
    ...                                                   tiles=tiles,
    ...                                                   polygons=polygons,
    ...                                                   )
    >>> result = creator.run(area_id)

    Notes
    -----
    Adapted from [1].
    """

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        *,
        key: str = "segmentation-boundary-weights",
        scale: float = 1.5,
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._key = key
        self._scale = scale
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))
        if overwrite or not output_path.exists():
            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": window.height,
                        "width": window.width,
                        "transform": transform,
                        "dtype": np.float32,
                        "nodata": None,
                    })

                    out_shape = (window.height, window.width)
                    polygons_in_area = self._polygons.query(f"area_id == {area_id}",
                                                            inplace=False)
                    boundary_weights = calculate_boundary_weights(
                        polygons_in_area,
                        self._scale,
                    )
                    output = np.zeros(
                        (1, window.height, window.width),
                        dtype=np.float32,
                    )
                    if len(polygons_in_area) > 0:
                        mask = rasterio.features.geometry_mask(
                            boundary_weights.geometry,
                            out_shape=out_shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        ).astype(np.float32)
                        output[0, :, :] = mask
                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(output)

                    if verbose:
                        print("Using scale = {:.4f}".format(self._scale,))
                        background = np.sum(mask[mask == 0.0])
                        foreground = np.sum(mask[mask == 1.0])
                        print("Histogram: 0.0 -> {}, 1.0 -> {}".format(
                            background,
                            foreground,
                        ))

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


# commented out because it uses code from https://gitlab.com/rscph/planetunet which is a private repository
# class ImageFrameCreator(FrameDataRasterCreator):
#     """Create an image frame of the intersection of the tile and area."""

#     def __init__(
#         self,
#         input_base_path: PathSource,
#         output_base_path: PathSource,
#         areas: GeoDataFrameSource,
#         tiles: GeoDataFrameSource,
#         *,
#         key: str = "image",
#         resolution: Optional[Any] = None,
#         warp_mem_limit: int = 0,
#         only_filename: bool = False,
#     ):
#         self._input_base_path = input_base_path
#         self._output_base_path = output_base_path
#         self._areas = load_geodataframe(areas, copy=False)
#         self._tiles = load_geodataframe(tiles, copy=False)
#         self._key = key
#         self._resolution = resolution
#         self._warp_mem_limit = warp_mem_limit
#         self._only_filename = only_filename

#     def run(
#         self,
#         area_id: int,
#         tile_id: Optional[int] = None,
#         *,
#         verbose: bool = True,
#         overwrite: bool = True,
#     ) -> FrameDataCreatorResult:

#         # since planet_unet.preprocessing requires GDAL, which is not always avaiable import only if needed
#         from planet_unet.preprocessing import raster_copy
#         area = self._areas.iloc[area_id]
#         tile_id = self.get_tile_id(area, tile_id)

#         output_base_path = pathlib.Path(self._output_base_path)
#         output_path = output_base_path.joinpath(
#             get_frame_filename(area_id, tile_id, self._key))

#         if overwrite or not output_path.exists():
#             if self._tiles.loc[tile_id, "reproject"]:
#                 # TODO: Handle correct reprojection of the image in
#                 # the code below
#                 raise ValueError("Image requires reprojection.")

#             input_path = pathlib.Path(self._input_base_path).joinpath(
#                 self._tiles.loc[tile_id, "file_path"]
#             )
#             raster_copy(
#                 str(output_path),
#                 str(input_path),
#                 bounds=self._areas.bounds.iloc[area_id],
#             )

#         return FrameDataCreatorResult(
#             tile_id,
#             area_id,
#             self._key,
#             output_path.name if self._only_filename else str(output_path),
#         )


class AltImageFrameCreator(FrameDataRasterCreator):
    """Create an image frame of the intersection of the tile and area."""

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        *,
        key: str = "image",
        resolution: Optional[Any] = None,
        warp_mem_limit: int = 0,
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._key = key
        self._resolution = resolution
        self._warp_mem_limit = warp_mem_limit
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        # References:
        # https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
        # https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
        # https://github.com/mapbox/rasterio/blob/master/rasterio/mask.py
        # Timeit: 1 loop, best of 5: 1.87 s per loop

        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))

        if overwrite or not output_path.exists():
            if self._tiles.loc[tile_id, "reproject"]:
                # TODO: Handle correct reprojection of the image in
                # the code below
                raise ValueError("Image requires reprojection.")

            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)

                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "height": int(window.height),
                        "width": int(window.width),
                        "transform": transform,
                    })

                    with rasterio.open(output_path, "w", **out_meta) as dst:
                        for i in range(1, src.count + 1):
                            rasterio.warp.reproject(
                                source=rasterio.band(src, i),
                                destination=rasterio.band(dst, i),
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=transform,
                                dst_crs=self._areas.crs,
                                resampling=rasterio.warp.Resampling.bilinear,
                                dst_resolution=self._resolution,
                                warp_mem_limit=self._warp_mem_limit,
                            )

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


def sahel_count_heuristic(polygon: Any) -> float:
    """Compute the count heuristic for a polygon from the Sahel dataset.

    Notes
    -----
    As used in [2].
    """
    # The heuristic consists in counting 6 trees for
    # every 200 m2 of area covered
    if polygon["size"] >= 200.0:
        #      polygon["size"] / (200.0 / 6.0)
        #      polygon["size"] / (100.0 / 3.0)
        return polygon["size"] * 3e-2
    return 1.0


def rwanda_count_heuristic(polygon: Any) -> float:
    """Compute the count heuristic for a polygon from the Rwanda dataset."""
    return 1.0


def get_blob_window(mask: np.ndarray) -> Tuple[slice, slice]:
    """Get a window containing the blob(s) for a given boolean mask."""
    blob = np.argwhere(mask)
    return (
        slice(blob[:, 0].min(), blob[:, 0].max()),
        slice(blob[:, 1].min(), blob[:, 1].max()),
    )


def get_blob_centroid(mask: np.ndarray) -> np.ndarray:
    # See: https://en.wikipedia.org/wiki/Centroid#Of_a_finite_set_of_points
    xs = np.argwhere(mask)
    centroid = np.round(np.sum(xs, axis=0) / len(xs)).astype(int)
    centroid_mask = np.zeros_like(mask, dtype="float32")
    centroid_mask[centroid[0], centroid[1]] = 1.0
    return centroid_mask


class DistanceDensityFrameCreator(FrameDataRasterCreator):

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        *,
        count_heuristic: Optional[CountHeuristicFuntion] = None,
        key: str = "distance-density",
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._count_heuristic = count_heuristic
        self._key = key
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))
        if overwrite or not output_path.exists():
            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": window.height,
                        "width": window.width,
                        "transform": transform,
                        "dtype": np.float32,
                        "nodata": None,
                    })

                    out_shape = (window.height, window.width)
                    polygons_in_area = self._polygons["area_id"] == area_id
                    polygons_in_area = self._polygons[polygons_in_area]
                    output = np.zeros(
                        (1, window.height, window.width),
                        dtype=np.float32,
                    )
                    for _, polygon in polygons_in_area.iterrows():
                        mask = rasterio.features.geometry_mask(
                            [polygon.geometry],
                            out_shape=out_shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        )
                        window = get_blob_window(mask)
                        blob = scipy.ndimage.distance_transform_edt(
                            mask[window]).astype(np.float32)
                        blob /= np.sum(blob)
                        if self._count_heuristic is not None:
                            blob *= self._count_heuristic(polygon)
                        output[0, window[0], window[1]] += blob

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(output)

                    if verbose:
                        print("Tree count: {}, Polygon count: {}".format(
                            np.sum(output),
                            len(polygons_in_area),
                        ))

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


class UniformDensityFrameCreator(FrameDataRasterCreator):

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        *,
        count_heuristic: Optional[CountHeuristicFuntion] = None,
        key: str = "uniform-density",
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._count_heuristic = count_heuristic
        self._key = key
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))
        if overwrite or not output_path.exists():
            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": window.height,
                        "width": window.width,
                        "transform": transform,
                        "dtype": np.float32,
                        "nodata": None,
                    })

                    out_shape = (window.height, window.width)
                    polygons_in_area = self._polygons["area_id"] == area_id
                    polygons_in_area = self._polygons[polygons_in_area]
                    output = np.zeros(
                        (1, window.height, window.width),
                        dtype=np.float32,
                    )
                    for _, polygon in polygons_in_area.iterrows():
                        mask = rasterio.features.geometry_mask(
                            [polygon.geometry],
                            out_shape=out_shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        )
                        window = get_blob_window(mask)
                        blob = mask[window].astype(np.float32)
                        blob /= np.sum(blob)
                        if self._count_heuristic is not None:
                            blob *= self._count_heuristic(polygon)
                        output[0, window[0], window[1]] += blob

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(output)

                    if verbose:
                        print("Tree count: {}, Polygon count: {}".format(
                            np.sum(output),
                            len(polygons_in_area),
                        ))

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


class FixedGaussianDensityFrameCreator(FrameDataRasterCreator):

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        *,
        sigma: float = 1.0,
        count_heuristic: Optional[CountHeuristicFuntion] = None,
        key: str = "fixed-gaussian-density",
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._sigma = sigma
        self._count_heuristic = count_heuristic
        self._key = key
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))
        if overwrite or not output_path.exists():
            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": window.height,
                        "width": window.width,
                        "transform": transform,
                        "dtype": np.float32,
                        "nodata": None,
                    })

                    out_shape = (window.height, window.width)
                    polygons_in_area = self._polygons["area_id"] == area_id
                    polygons_in_area = self._polygons[polygons_in_area]
                    output = np.zeros(
                        (1, window.height, window.width),
                        dtype=np.float32,
                    )
                    for _, polygon in polygons_in_area.iterrows():
                        mask = rasterio.features.geometry_mask(
                            [polygon["geometry"]],
                            out_shape=out_shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        )
                        centroid_mask = get_blob_centroid(mask)
                        blob = skimage.filters.gaussian(
                            centroid_mask,
                            self._sigma,
                            mode="mirror",
                        )
                        blob_sum = np.sum(blob)
                        if blob_sum > 0.0:
                            blob /= blob_sum
                        if self._count_heuristic is not None:
                            blob *= self._count_heuristic(polygon)
                        output[0, :, :] += blob

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(output)

                    if verbose:
                        print("Tree count: {}, Polygon count: {}".format(
                            np.sum(output),
                            len(polygons_in_area),
                        ))

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


class CentroidsFrameCreator(FrameDataRasterCreator):

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        *,
        key: str = "centroids",
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._key = key
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))
        if overwrite or not output_path.exists():
            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": window.height,
                        "width": window.width,
                        "transform": transform,
                        "dtype": np.float32,
                        "nodata": None,
                    })

                    out_shape = (window.height, window.width)
                    polygons_in_area = self._polygons.query(
                        f"area_id == {area_id}",
                        inplace=False,
                    )
                    output = np.zeros(
                        (1, window.height, window.width),
                        dtype=np.float32,
                    )
                    for _, polygon in polygons_in_area.iterrows():
                        mask = rasterio.features.geometry_mask(
                            [polygon["geometry"]],
                            out_shape=out_shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        )
                        output[0, :, :] += get_blob_centroid(mask)

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(output)

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


class AltCentroidsFrameCreator(FrameDataRasterCreator):

    def __init__(
        self,
        input_base_path: PathSource,
        output_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        *,
        key: str = "alt-centroids",
        only_filename: bool = False,
    ):
        self._input_base_path = input_base_path
        self._output_base_path = output_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._key = key
        self._only_filename = only_filename

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)

        output_base_path = pathlib.Path(self._output_base_path)
        output_path = output_base_path.joinpath(
            get_frame_filename(area_id, tile_id, self._key))
        if overwrite or not output_path.exists():
            input_path = pathlib.Path(self._input_base_path).joinpath(
                self._tiles.loc[tile_id, "file_path"])
            # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
                with rasterio.open(input_path, "r") as src:
                    window = rasterio.features.geometry_window(src, [area.geometry])
                    transform = src.window_transform(window)
                    out_meta = src.meta
                    out_meta.update({
                        "driver": "GTiff",
                        "count": 1,
                        "height": window.height,
                        "width": window.width,
                        "transform": transform,
                        "dtype": np.float32,
                        "nodata": None,
                    })

                    out_shape = (window.height, window.width)
                    polygons_in_area = self._polygons.query(
                        f"area_id == {area_id}",
                        inplace=False,
                    )
                    output = np.zeros(
                        (1, window.height, window.width),
                        dtype=np.float32,
                    )
                    if len(polygons_in_area) > 0:
                        output[0, :, :] = rasterio.features.geometry_mask(
                            polygons_in_area["centroid"],
                            out_shape=out_shape,
                            transform=transform,
                            invert=True,
                            all_touched=True,
                        ).astype(np.float32)

                    with rasterio.open(output_path, "w", **out_meta) as dest:
                        dest.write(output)

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            output_path.name if self._only_filename else str(output_path),
        )


class ScalarFrameDataCreator(FrameDataCreator):
    """Compute some scalar information for an area.

    Parameters
    ----------
    input_base_path
        The path to the folder containing tile images.
    areas
        The areas database.
    tiles
        The tiles database.
    polygons
        The polygons database.
    count_heuristic: optional
        A function that computes the count heuristic given polygon data.
    key
        The key for the result.

    Examples
    --------
    >>> creator = ScalarFrameDataCreator("./tile_images",
    ...                                   areas=areas,
    ...                                   tiles=tiles,
    ...                                   polygons=polygons,
    ...                                   count_heuristic=None,
    ...                                  )
    >>> result = creator.run(area_id)
    >>> canopy_cover = result.payload["canopy]
    >>> tree_count = result.payload["trees"]
    >>> polygon_count = result.payload["polygons"]
    """

    def __init__(
        self,
        input_base_path: PathSource,
        areas: GeoDataFrameSource,
        tiles: GeoDataFrameSource,
        polygons: GeoDataFrameSource,
        count_heuristic: Optional[CountHeuristicFuntion] = None,
        key: str = "scalars",
    ):
        self._input_base_path = input_base_path
        self._areas = load_geodataframe(areas, copy=False)
        self._tiles = load_geodataframe(tiles, copy=False)
        self._polygons = load_geodataframe(polygons, copy=False)
        self._count_heuristic = count_heuristic
        self._key = key

    def run(
        self,
        area_id: int,
        tile_id: Optional[int] = None,
        *,
        verbose: bool = True,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        area = self._areas.iloc[area_id]
        tile_id = self.get_tile_id(area, tile_id)
        input_path = pathlib.Path(self._input_base_path).joinpath(
            self._tiles.loc[tile_id, "file_path"])
        # See: https://github.com/mapbox/rasterio/issues/2053#issuecomment-744579992
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN=True):
            with rasterio.open(input_path, "r") as src:
                window = rasterio.features.geometry_window(
                    src,
                    [area.geometry],
                )
                data = src.read(window=window, masked=True)
                if data.mask.ndim > 0:
                    # The channel dimension is in the axis 0 of data
                    all_nodata = np.sum(np.all(data.mask, axis=(1, 2)))
                    any_nodata = np.sum(np.any(data.mask, axis=(1, 2)))
                else:
                    all_nodata = 0
                    any_nodata = 0
                transform = src.window_transform(window)
                out_shape = (window.height, window.width)
                polygons_in_area = self._polygons.query(
                    f"area_id == {area_id}",
                    inplace=False,
                )
                polygon_count = len(polygons_in_area)
                frame = np.zeros(
                    (window.height, window.width),
                    dtype=np.float32,
                )
                tree_count = 0.0
                for _, polygon in polygons_in_area.iterrows():
                    mask = rasterio.features.geometry_mask(
                        [polygon.geometry],
                        out_shape=out_shape,
                        transform=transform,
                        invert=True,
                        all_touched=True,
                    )
                    window = get_blob_window(mask)
                    blob = mask[window].astype(np.float32)
                    blob_normalized = blob / np.sum(blob)
                    if self._count_heuristic is not None:
                        count_scaler = self._count_heuristic(polygon)
                        tree_count += np.sum(blob_normalized * count_scaler)
                    else:
                        tree_count += np.sum(blob_normalized)
                    frame[window[0], window[1]] += blob
                canopy_cover = np.average(frame)

        return FrameDataCreatorResult(
            tile_id,
            area_id,
            self._key,
            {
                "canopy": canopy_cover,
                "polygons": polygon_count,
                "trees": tree_count,
                "height": out_shape[0],
                "width": out_shape[1],
                "all_nodata": all_nodata,
                "any_nodata": any_nodata,
            },
        )


FrameFactoryJob = Tuple[Tuple[int, int], FrameDataCreator]

GeoDataFrameSourceTuple = Tuple[GeoDataFrameSource, GeoDataFrameSource]


@dataclasses.dataclass
class FrameDataFactoryResult:
    """Define the results of a frame data factory run.

    Parameters
    ----------
    frames
        The newly created or updated frames database.
    results
        The raw results collection from the frame data creators.
    """

    frames: gpd.GeoDataFrame
    results: List[FrameDataCreatorResult]


class FrameDataFactory:
    """Create frame data using multiple frame data creators.

    Parameters
    ----------
    creators: optional
        A list of frame creators.

    Examples
    --------
    >>> factory = FrameDataFactory()
    >>> factory.add_creator(image_creator)
    >>> factory.add_creator(density_creator)
    >>> results = factory.run_jobs(tiles, areas,
    ...                            output_path="./frames.geojson")

    Notes
    -----
    The output database format should be a GeoJSON file, since other formats
    limit the column name length to 10 characters.
    """

    def __init__(self, creators: Optional[List[FrameDataCreator]] = None):
        if creators is None:
            self.creators = []
        else:
            self.creators = creators

    def add_creator(self, creator: FrameDataCreator) -> None:
        """Add a creator to the factory.

        Parameters
        ----------
        creator
            The frame data creator instance.
        """
        self.creators.append(creator)

    def _run_job(
        self,
        job: FrameFactoryJob,
        overwrite: bool = True,
    ) -> FrameDataCreatorResult:
        (area_id, tile_id), creator = job

        return creator.run(
            area_id,
            tile_id,
            verbose=False,
            overwrite=overwrite,
        )

    def run_jobs(
        self,
        db: Union[GeoDataFrameSourceTuple, GeoDataFrameSource],
        *,
        output_path: Optional[Union[pathlib.Path, str]] = None,
        n_processes: Optional[int] = None,
        job_slice: Optional[slice] = None,
        dry_run: bool = False,
        overwrite: bool = True,
        verbose: bool = True,
        tile_filter_fn: Optional[Callable[[Any], bool]] = None,
        save_keys: Optional[List[str]] = None,
        driver: str = "GeoJSON",
    ) -> FrameDataFactoryResult:
        """Run the frame creation jobs.

        Parameters
        ----------
        db
            Either a pair of tiles and areas dataframes or a frames dataframe.
        output_path: optional
            A path to save the frames database.
        n_processes: optional
            The target number of CPUs to use. By default uses all available.
        job_slice: optional
            The slice of jobs to work on.
        dry_run
            Only display the work that would be done.
        overwrite
            Overwrite existing files.
        tile_filter_fn: optional
            A function that receives a tile and outputs True if it should \
                be processed.
        save_keys: optional
            A list of keys for which we wish to save results in the frames \
                database. By default all are saved.

        Returns
        -------
        FrameDataFactoryResult
            The results obtained after running all the jobs.
        """
        if isinstance(db, tuple):
            tiles, areas = db
            tiles = load_geodataframe(tiles, copy=False)
            areas = load_geodataframe(areas, copy=False)
            frames = {
                "area_id": [],
                "tile_id": [],
                "geometry": [],
            }
            for tile_id, tile in tiles.iterrows():
                if tile_filter_fn is None or tile_filter_fn(tile):
                    for area_id in tile["area_ids"]:
                        geometry = tile["geometry"].intersection(
                            areas.loc[area_id, "geometry"],)
                        frames["area_id"].append(area_id)
                        frames["tile_id"].append(tile_id)
                        frames["geometry"].append(geometry)
            if verbose:
                print("Will create new frames database.")
            frames = gpd.GeoDataFrame(frames, crs=tiles.crs)
        else:
            if verbose:
                print("Will update existing frames database.")
            frames = load_geodataframe(db, copy=True)

        jobs = list(product(
            zip(frames["area_id"], frames["tile_id"]),
            self.creators,
        ))
        n_total_jobs = len(jobs)

        if job_slice is None:
            job_slice = slice(0, len(jobs))
        jobs = jobs[job_slice]
        n_slice_jobs = len(jobs)

        if n_processes is None:
            n_processes = os.cpu_count()
        n_processes = min(n_processes, n_slice_jobs, os.cpu_count())
        if verbose:
            print("""Will run {}/{} jobs ({} frames x {} data creators) """
                  """using {} processes and {}.""".format(
                      n_slice_jobs,
                      n_total_jobs,
                      len(frames),
                      len(self.creators),
                      n_processes,
                      job_slice,
                  ))

        results: List[FrameDataCreatorResult] = []

        if not dry_run:
            if n_processes > 1:
                with multiprocessing.Pool(processes=n_processes) as pool:
                    results = pool.map(
                        partial(self._run_job, overwrite=overwrite),
                        jobs,
                    )
            else:
                results = list(map(
                    partial(self._run_job, overwrite=overwrite),
                    jobs,
                ))

            # Temporarily set (tile_id, area_id) as the database index.
            frames.set_index(
                ["tile_id", "area_id"],
                drop=True,
                inplace=True,
            )

            # We keep track of which columns are already created.
            seen_keys = set()

            def check_key(result: FrameDataCreatorResult) -> None:
                """Check if a column exists in the dataframe for the \
                    key of the given result, and create it if needed."""
                # References:
                # https://stackoverflow.com/a/57293727
                if result.key not in seen_keys and result.key not in frames:
                    if isinstance(result.payload, dict):
                        # We flatten the dictionary into individual columns
                        # Does not support nested dictionaries with depth > 1.
                        for suffix, value in result.payload.items():
                            compound_key = f"{result.key}_{suffix}"
                            if compound_key not in seen_keys:
                                dtype = type(value)
                                frames[compound_key] = pd.Series(dtype=dtype)
                                seen_keys.add(compound_key)
                    else:
                        dtype = type(result.payload)
                        frames[result.key] = pd.Series(dtype=dtype)
                    seen_keys.add(result.key)

            # Now, we iterate over our results and save them
            # to the database (dataframe).

            for result in results:
                if save_keys is None or result.key in save_keys:
                    check_key(result)
                    index = (result.tile_id, result.area_id)
                    if isinstance(result.payload, dict):
                        for key, value in result.payload.items():
                            compound_key = f"{result.key}_{key}"
                            frames.loc[index, compound_key] = value
                    else:
                        frames.loc[index, result.key] = result.payload

            # Reset the index (otherwise to_file below complains)
            frames.reset_index(inplace=True)
            if output_path is not None:
                output_path = pathlib.Path(output_path)
                frames.to_file(output_path, driver=driver)

            if verbose:
                print(f"Finished {len(results)} jobs.")

        return FrameDataFactoryResult(frames, results)


__all__ = [
    "sahel_count_heuristic",
    "rwanda_count_heuristic",
    "FrameDataCreator",
    "SegmentationMaskFrameCreator",
    "SegmentationBoundaryWeightsFrameCreator",
    "ImageFrameCreator",
    "AltImageFrameCreator",
    "DistanceDensityFrameCreator",
    "UniformDensityFrameCreator",
    "FixedGaussianDensityFrameCreator",
    "CentroidsFrameCreator",
    "QuantitativeFrameDataCreator",
    "FrameFactory",
]


def calculate_boundary_weights(polygons, scale):
    """Find boundaries between close polygons.

    Scales up each polygon, then get overlaps by intersecting. The overlaps of the scaled polygons are the boundaries.
    Returns geopandas data frame with boundary polygons.
    """
    # Scale up all polygons around their center, until they start overlapping
    # NOTE: scale factor should be matched to resolution and type of forest
    scaled_polys = gpd.GeoDataFrame(
        {
            "geometry":
                polygons.geometry.scale(xfact=scale, yfact=scale, origin="center")
        },
        crs=polygons.crs,
    )

    # Get intersections of scaled polygons, which are the boundaries.
    boundaries = []
    for i in range(len(scaled_polys)):

        # For each scaled polygon, get all nearby scaled polygons that intersect with it
        nearby_polys = scaled_polys[scaled_polys.geometry.intersects(
            scaled_polys.iloc[i].geometry)]

        # Add intersections of scaled polygon with nearby polygons [except the intersection with itself!]
        for j in range(len(nearby_polys)):
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                boundaries.append(scaled_polys.iloc[i].geometry.intersection(
                    nearby_polys.iloc[j].geometry))

    # Convert to df and ensure we only return Polygons (sometimes it can be a Point, which breaks things)
    boundaries = gpd.GeoDataFrame(
        {
            "geometry": gpd.GeoSeries(boundaries)
        },
        crs=polygons.crs,
    ).explode()
    boundaries = boundaries[boundaries.type == "Polygon"]

    # If we have boundaries, difference overlay them with original polygons to ensure boundaries don't cover labels
    if len(boundaries) > 0:
        boundaries = gpd.overlay(boundaries, polygons, how="difference")
    else:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries
