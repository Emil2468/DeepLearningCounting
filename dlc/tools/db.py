#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""This module contains functions related to manipulate tiles in a dataset."""
import abc
import pathlib
import math
from typing import Union, Optional, Tuple, List, Any

import geopandas as gpd
import shapely
import numpy as np
import geopandas as gpd
import shapely
import rasterio
import rasterio.coords
import rasterio.warp

from tqdm import tqdm

from .common import load_geodataframe
from .types import GeoDataFrameSource


def get_objects_from_images(
    input_base_path: pathlib.Path,
    search_pattern: str,
    *,
    output_path: Optional[Union[pathlib.Path, str]] = None,
    crs: Optional[str] = None,
    different_crs: str = "skip",
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    """Create a database of objects from a folder of images.

    Examples
    --------
    >>> tiles = get_objects_from_images("./tile_images", "*.tif",
    ...                                 output_path="./tiles.gpkg")
    >>> tiles.head()
    """

    # References:
    # https://shapely.readthedocs.io/en/stable/manual.html#shapely.geometry.box
    # https://geopandas.org/docs/reference/api/geopandas.GeoDataFrame.html#geopandas-geodataframe
    # https://rasterio.readthedocs.io/en/latest/quickstart.html#dataset-georeferencing
    # https://rasterio.readthedocs.io/en/latest/topics/reproject.html#reprojecting-a-geotiff-dataset
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.warp.html#rasterio.warp.transform_bounds

    dataframe: dict = {
        "file_path": [],
        "file_size": [],
        "geometry": [],
        "reproject": [],
    }
    input_base_path = pathlib.Path(input_base_path)
    for file_path in tqdm(input_base_path.glob(search_pattern)):
        relative_file_path = file_path.relative_to(input_base_path)
        reproject = False
        dataset = rasterio.open(file_path)
        bounds = dataset.bounds
        if crs is None:
            # The first file defines the CRS to use if none is passed.
            crs = dataset.crs
            if verbose:
                print(f"Setting CRS = {crs}")
        elif crs != dataset.crs:
            # All entries must have the same CRS in a geodataframe.
            if different_crs == "error":
                msg = f"{relative_file_path} with CRS {dataset.crs} should've been CRS {crs}."
                raise ValueError(msg)
            elif different_crs == "reproject":
                if verbose:
                    print(f"Reprojecting {relative_file_path} into CRS {crs}")
                reproject = True
                bounds = rasterio.coords.BoundingBox(*rasterio.warp.transform_bounds(
                    dataset.crs,
                    crs,
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                ))
            else:
                if verbose:
                    print("Skipping {} with different CRS: {}".format(
                        relative_file_path,
                        dataset.crs,
                    ))
                continue

        dataframe["file_path"].append(str(relative_file_path))
        # Store file size in GBs
        file_size = file_path.stat().st_size * 1e-9
        dataframe["file_size"].append(file_size)
        dataframe["geometry"].append(
            shapely.geometry.box(
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
            ))
        # A flag to indicate if we should reproject the file.
        dataframe["reproject"].append(reproject)

    geodataframe = gpd.GeoDataFrame(dataframe, crs=crs)

    if output_path is not None:
        output_path = pathlib.Path(output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        geodataframe.to_file(output_path)

    if verbose:
        print(f"Finished! Kept {geodataframe.count().file_path} objects.")

    return geodataframe


def get_tiles_with_areas(
    tiles: GeoDataFrameSource,
    areas: GeoDataFrameSource,
    *,
    verbose: bool = True,
):
    """Add area infomation to tiles."""
    # Adapted from:
    # https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
    tiles = load_geodataframe(tiles, copy=True)
    areas = load_geodataframe(areas, copy=False)

    if tiles.crs != areas.crs:
        raise ValueError("Tiles and areas have different CRS.")

    tiles["id"] = np.arange(len(tiles), dtype=int)
    join = gpd.sjoin(
        tiles,
        gpd.GeoDataFrame(areas.geometry),
        op="intersects",
        how="inner",
    )
    area_ids = [[] for _ in range(len(tiles))]
    for _, tile in join.iterrows():
        area_ids[tile.id].append(int(tile.index_right))
    tiles["area_ids"] = area_ids
    tiles["n_areas"] = list(map(lambda x: len(x), area_ids))

    if verbose:
        print("Found {}/{} tiles with at least one area.".format(
            len(tiles[tiles["n_areas"] > 0]),
            len(tiles),
        ))
    return tiles


def get_polygons_with_area(
    polygons: GeoDataFrameSource,
    areas: GeoDataFrameSource,
):
    """Get polygons with the corresponding area index and size in \
        square meters."""
    # Adapted from:
    # https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
    areas = load_geodataframe(areas, copy=False)
    polygons = load_geodataframe(polygons, copy=True)

    if areas.crs != polygons.crs:
        raise ValueError("Areas and polygons have different CRS.")

    polygons_equiarea = polygons.to_crs("EPSG:6933")
    polygons["size"] = polygons_equiarea.area
    polygons["centroid"] = polygons_equiarea.centroid.to_crs(polygons.crs)

    polygons = gpd.sjoin(
        polygons,
        gpd.GeoDataFrame(areas.geometry),
        op="intersects",
        how="inner",
    )
    polygons = polygons.rename(columns={"index_right": "area_id"},)
    return polygons


def get_areas_with_tile(
    areas: GeoDataFrameSource,
    tiles: GeoDataFrameSource,
    *,
    verbose: bool = True,
) -> gpd.GeoDataFrame:
    areas = load_geodataframe(areas, copy=True)
    tiles = load_geodataframe(tiles, copy=False)

    if tiles.crs != areas.crs:
        raise ValueError("Areas and tiles have different CRS.")

    areas["id"] = np.arange(len(areas), dtype=int)
    join = gpd.sjoin(
        areas,
        gpd.GeoDataFrame(tiles.geometry),
        op="intersects",
        how="inner",
    )
    tile_ids = [[] for _ in range(len(areas))]
    for _, area in join.iterrows():
        tile_ids[area.id].append(int(area.index_right))
    areas["tile_ids"] = tile_ids
    areas["n_tiles"] = list(map(lambda x: len(x), tile_ids))

    if verbose:
        print("Found {}/{} areas with at least one tile.".format(
            len(areas[areas["n_tiles"] > 0]),
            len(areas),
        ))

    return areas


__all__ = [
    "get_objects_from_images",
    "get_tiles_with_areas",
    "get_polygons_with_area",
    "get_areas_with_tile",
]
