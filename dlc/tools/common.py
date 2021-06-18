#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""This module contains common functions used in this package."""
import pathlib
from typing import Optional

import geopandas as gpd

from .types import GeoDataFrameSource


def load_geodataframe(
    src: GeoDataFrameSource,
    *,
    copy: bool = True,
    driver: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """Load a GeoDataframe from a file path or another in-memory dataframe."""
    if not isinstance(src, gpd.GeoDataFrame):
        src = pathlib.Path(src)
        if driver is not None:
            return gpd.read_file(src, driver=driver)
        return gpd.read_file(src)
    return src.copy() if copy else src
