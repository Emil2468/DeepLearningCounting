#   Author: Luis Diego García Castro and Adolfo Enrique García Castro
"""This module contains common type definitions."""
from typing import Union, Any, Callable
from pathlib import Path
from geopandas import GeoDataFrame

GeoDataFrameSource = Union[Path, str, GeoDataFrame]

PathSource = Union[Path, str]

CountHeuristicFuntion = Callable[[Any], float]

__all__ = [
    "GeoDataFrameSource",
    "PathSource",
    "CountHeuristicFuntion",
]
