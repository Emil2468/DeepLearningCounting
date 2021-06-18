#   Author: Emil Møller Hansen

from typing import List, Optional


class Config():
    """
    Args
        dataset_path
            Absolute path to the dataset
        dataset_img_path
            Path to the images of all the tiles, relative to dataset_path
        polygons_filename
            Path, relative to dataset_path, to geopandas package from which to load the polygons describing the objects to count
        rectanlges_filename
            Path, relative to dataset_path, to geopandas package from which to load the frames describing what areas has been annotated
        dataset_img_search_pattern
            Search pattern to use when looking for files in dataset_img_path
        tiles_filename
            Name of the file to write a geopandas dataframe containing information about all the tiles to
        output_path
            Absolute path to write the resulting frame-images and geo-json file to
        output_filename 
            Name of the geo-json file to output
        frame_creators
            Names of the frame creators to use, separated by a space. For options, see dlc/tools/frames.py
        count_heuristic
            Name of the count heuristic to use. For options, see dlc/tools/frames.py
        overwrite
            Whether to overwrite existing frames
        sigma
            Sigma to use for FixedGaussianDensityFrameCreator. Only relevant if FixedGaussianDensityFrameCreator is added to the list of frame creators 
    """
    dataset_path: str = "/content/drive/MyDrive/datasets/sahel_subset/"
    dataset_img_path: str = "Images/StackedImages/"
    polygons_filename: str = "polygons.gpkg"
    rectanlges_filename: str = "rectangles.gpkg"
    dataset_img_search_pattern: str = "*.tif"
    tiles_filename: str = "tiles.gpkg"

    output_path: str = f"{dataset_path}/frames/"
    output_filename: str = "frames.geojson"

    frame_creators: List[str] = [
        "ImageFrameCreator", "UniformDensityFrameCreator",
        "SegmentationMaskFrameCreator"
    ]
    count_heuristic: Optional[str] = "sahel_count_heuristic"
    overwrite: bool = False
    sigma: float = 1.0

    def __init__(self) -> None:
        [setattr(self, k, v) for k, v in vars(Config).items() if not k.startswith("_")]
