#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""Contains plotting functions."""
import pathlib
from typing import Union, Optional, List, Tuple, Any

import geopandas as gpd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import rasterio.plot
from descartes import PolygonPatch

from .common import load_geodataframe
from .types import PathSource


def plot_object_splits(
    objects: Union[pathlib.Path, str, gpd.GeoDataFrame],
    splits: List[float],
    *,
    areas: Optional[gpd.GeoDataFrame] = None,
    colors: List[str] = ["white", "red"],
    window: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
):
    """Plot objects splits and optionally the sampled areas.

    Examples
    --------
    >>> plot_object_splits(tile_splits, areas=sampled_areas)
    """
    objects = load_geodataframe(objects, copy=True)
    objects["split"] = splits
    # References:
    # https://stackoverflow.com/a/38885389
    split_classes = set(splits)
    if len(colors) != len(split_classes):
        raise ValueError("Incorrect number of colors")
    cmap = matplotlib.colors.ListedColormap(colors, N=len(colors))
    ax = objects.plot(
        column="split",
        cmap=cmap,
        vmin=min(split_classes),
        vmax=max(split_classes),
    )
    ax = objects.boundary.plot(ax=ax, color="black")
    if window is not None:
        ax.set_ylim(*window[0])
        ax.set_xlim(*window[1])
    if areas is not None:
        ax = areas.plot(
            column="count",
            ax=ax,
            alpha=0.4,
            legend=True,
            cmap="jet",
        )
    return ax


def plot_polygons_in_area(
    areas: Union[pathlib.Path, str, gpd.GeoDataFrame],
    polygons: Union[pathlib.Path, str, gpd.GeoDataFrame],
    area_id: Optional[int] = None,
    *,
    polygon_id: Optional[int] = None,
    seed: Optional[int] = None,
    plot_type: str = "area",
    figsize: Optional[Tuple[int]] = None,
    ax=None,
    verbose: bool = True,
    plot_centroids: bool = False,
):
    """Plot an area to inspect it."""
    # References:
    # https://gitlab.com/rscph/planetunet/-/blob/master/preprocessing.py
    rng = np.random.default_rng(seed)
    areas = load_geodataframe(areas, copy=False)
    polygons = load_geodataframe(polygons, copy=False)

    if areas.crs != polygons.crs:
        raise ValueError("Areas and polygons have different CRS.")

    # If an area index is not provided, get one at random.
    if area_id is None:
        area_id = rng.integers(0, len(areas), size=1)[0]

    polygons_in_area = polygons[polygons["area_id"] == area_id]

    if plot_type == "area":
        ax = areas.iloc[area_id:area_id + 1].plot(
            color="brown",
            alpha=0.2,
            figsize=figsize,
            ax=ax,
        )
        if len(polygons_in_area) > 0:
            ax = polygons_in_area.plot(
                ax=ax,
                color="green",
                figsize=figsize,
            )
    else:
        ax = areas.boundary.iloc[area_id:area_id + 1].plot(
            color="brown",
            figsize=figsize,
            ax=ax,
        )
        if len(polygons_in_area) > 0:
            ax = polygons_in_area.boundary.plot(
                ax=ax,
                color="green",
                figsize=figsize,
            )
    # Optionally, highlight a polygon.
    if polygon_id is not None:
        ax = polygons_in_area[polygon_id:polygon_id + 1].plot(
            ax=ax,
            color="orange",
            figsize=figsize,
        )
    if plot_centroids:
        ax = polygons_in_area["centroid"].plot(
            ax=ax,
            color="yellow",
            figsize=figsize,
            marker="x",
            markersize=2.0,
        )
    if verbose:
        print("Area {} has {} polygons.".format(
            area_id,
            len(polygons_in_area),
        ))
    return ax


def get_polygon_patches(polygons, *, edgecolor="red"):
    # See: https://gis.stackexchange.com/a/193695
    polygon_patches = []
    for _, polygon in polygons.iterrows():
        patch = PolygonPatch(
            polygon["geometry"],
            edgecolor=edgecolor,
            facecolor="none",
            linewidth=1,
        )
        polygon_patches.append(patch)
    return polygon_patches


def plot_frame(
    image: np.ndarray,
    *,
    title: Optional[str] = None,
    cmaps: List[str] = None,
    log: bool = False,
    bins: Union[int, str] = 20,
    show_hist: bool = True,
    output_path: Optional[PathSource] = None,
    polygons: Optional[Any] = None,
    polygon_color: str = "red",
    transform: Optional[Any] = None,
    nodata: Optional[Any] = None,
) -> None:
    """Plot bands and a histogram of a given raster array."""
    n_bands = image.shape[2]
    if cmaps is None:
        cmaps = ["gray" for _ in range(n_bands)]
    n_cols = n_bands
    if show_hist:
        n_cols += 1

    fig = plt.figure(figsize=(5.3 * n_cols, 5.0))
    gs = fig.add_gridspec(1, n_cols)

    if title is not None:
        fig.suptitle(title)

    if polygons is not None:
        polygon_patches = get_polygon_patches(
            polygons,
            edgecolor=polygon_color,
        )
    else:
        polygon_patches = None

    extent = None
    if transform is not None:
        extent = rasterio.plot.plotting_extent(image, transform=transform)

    image_to_plot = image
    if nodata is not None:
        image_to_plot = np.ma.masked_array(image, mask=(image == nodata))
    for i, band in enumerate(range(1, n_bands + 1)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(f"Band {band}")
        ax.imshow(image_to_plot[:, :, i], cmap=cmaps[i], origin="upper", extent=extent)
        if polygon_patches is not None:
            polygon_collection = PatchCollection(
                polygon_patches,
                match_original=True,
            )
            ax.add_collection(polygon_collection)

    if show_hist:
        ax = fig.add_subplot(gs[0, n_bands])
        raster_shape = (image.shape[2], image.shape[0], image.shape[1])
        rasterio.plot.show_hist(
            image_to_plot.reshape(raster_shape),
            bins=bins,
            stacked=False,
            alpha=0.3,
            log=log,
            histtype="stepfilled",
            title="Histogram",
            ax=ax,
        )
    fig.tight_layout()
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        plt.savefig(output_path)
        plt.close(fig)


def plot_frame_nodata(
    path: PathSource,
    *,
    title: Optional[str] = None,
    output_path: Optional[PathSource] = None,
):
    path = pathlib.Path(path)
    with rasterio.open(path, "r") as src:
        n_bands = src.count
        fig = plt.figure(figsize=(5.3 * n_bands, 4.5))
        gs = fig.add_gridspec(1, n_bands)
        if title is not None:
            fig.suptitle(title)
        cmap = matplotlib.colors.ListedColormap(["white", "red"], N=2)
        for band, nodataval in zip(range(1, n_bands + 1), src.nodatavals):
            img = src.read(band, masked=True)
            ax = fig.add_subplot(gs[0, band - 1])
            ax.set_title(f"Band {band} (nodata = {nodataval})")
            if nodataval is not None:
                nodata_img = np.where(img.mask, 1.0, 0.0)
                ax.imshow(nodata_img, cmap=cmap)
            else:
                ax.imshow(np.zeros_like(img), cmap=cmap)
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        plt.savefig(output_path)
        plt.close(fig)
