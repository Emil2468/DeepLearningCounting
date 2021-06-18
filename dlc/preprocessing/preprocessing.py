#   Author: Emil Møller Hansen

from argparse import ArgumentError
import dlc.tools.db as db
from dlc.preprocessing.config import Config
from dlc.tools import frames
import dlc
import os


def preprocess_all(conf: Config):
    tiles = db.get_objects_from_images(conf.dataset_path + conf.dataset_img_path,
                                       conf.dataset_img_search_pattern)

    tiles.to_file(os.path.join(conf.dataset_path, conf.tiles_filename))

    tiles = dlc.tools.db.get_tiles_with_areas(
        os.path.join(conf.dataset_path, conf.tiles_filename),
        conf.dataset_path + conf.rectanlges_filename)

    polygons = db.get_polygons_with_area(conf.dataset_path + conf.polygons_filename,
                                         conf.dataset_path + conf.rectanlges_filename)
    areas = db.get_areas_with_tile(conf.dataset_path + conf.rectanlges_filename,
                                   conf.dataset_path + conf.tiles_filename)

    frame_factory = frames.FrameDataFactory()

    for frame_creator_name in conf.frame_creators:
        frame_creator = get_frame_creator(frame_creator_name, conf, areas, tiles,
                                          polygons)
        frame_factory.add_creator(frame_creator)

    frame_factory.run_jobs((tiles, areas),
                           dry_run=False,
                           overwrite=conf.overwrite,
                           output_path=conf.output_path + conf.output_filename,
                           save_keys=None)


def get_frame_creator(frame_creator_name: str, conf: Config, areas, tiles,
                      polygons) -> frames.FrameDataCreator:
    input_base_path = os.path.join(conf.dataset_path, conf.dataset_img_path)
    count_heuristic = get_count_heuristic(
        conf.count_heuristic) if conf.count_heuristic is not None else None

    if frame_creator_name == "SegmentationMaskFrameCreator":
        return frames.SegmentationMaskFrameCreator(input_base_path, conf.output_path,
                                                   areas, tiles, polygons)
    # elif frame_creator_name == "ImageFrameCreator": # commented out because it uses code from https://gitlab.com/rscph/planetunet which is a private repository
    #     return frames.ImageFrameCreator(input_base_path, conf.output_path, areas, tiles)
    elif frame_creator_name == "UniformDensityFrameCreator":
        return frames.UniformDensityFrameCreator(input_base_path,
                                                 conf.output_path,
                                                 areas,
                                                 tiles,
                                                 polygons,
                                                 count_heuristic=count_heuristic)
    elif frame_creator_name == "SegmentationBoundaryWeightsFrameCreator":
        return frames.SegmentationBoundaryWeightsFrameCreator(
            input_base_path, conf.output_path, areas, tiles, polygons)
    elif frame_creator_name == "AltImageFrameCreator":
        return frames.AltImageFrameCreator(input_base_path, conf.output_path, areas,
                                           tiles)
    elif frame_creator_name == "DistanceDensityFrameCreator":
        return frames.DistanceDensityFrameCreator(input_base_path,
                                                  conf.output_path,
                                                  areas,
                                                  tiles,
                                                  polygons,
                                                  count_heuristic=count_heuristic)
    elif frame_creator_name == "FixedGaussianDensityFrameCreator":
        return frames.FixedGaussianDensityFrameCreator(input_base_path,
                                                       conf.output_path,
                                                       areas,
                                                       tiles,
                                                       polygons,
                                                       count_heuristic=count_heuristic,
                                                       sigma=conf.sigma)
    elif frame_creator_name == "CentroidsFrameCreator":
        return frames.CentroidsFrameCreator(input_base_path, conf.output_path, areas,
                                            tiles, polygons)
    elif frame_creator_name == "ScalarFrameDataCreator":
        return frames.ScalarFrameDataCreator(input_base_path,
                                             areas,
                                             tiles,
                                             polygons,
                                             count_heuristic=count_heuristic)
    else:
        raise AttributeError(f"Frame creator {frame_creator_name} does not exist")


def get_count_heuristic(count_heuristic_name):
    return getattr(frames, count_heuristic_name)
