import os
import sys
from argparse import ArgumentParser
from loguru import logger
from yaml import FullLoader, load

import geopandas as gpd
import rasterio as rio
from rasterio.features import shapes
from shapely.geometry import shape

def format_logger(logger):
    """
    Format the logger from loguru.

    Args:
        logger (loguru.Logger): The logger object from loguru.

    Returns:
        loguru.Logger: The formatted logger object.
    """

    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")

    return logger


def get_config(config_key, desc=""):

    # Argument and parameter specification
    parser = ArgumentParser(description=desc)
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[config_key]

    return cfg


def get_bbox_origin(bbox_geom):
    """Get the lower xy coorodinates of a bounding box.

    Args:
        bbox_geom (geometry): bounding box

    Returns:
        tuple: lower xy coordinates of the passed geometry
    """
    coords = bbox_geom.exterior.coords.xy
    min_x = min(coords[0])
    min_y = min(coords[1])

    return (min_x, min_y)


def get_maximum_coordinates(bbox_geom):
    """
    Get the maximum easting and northing coordinates from a bounding box geometry.

    Args:
        bbox_geom (shapely.geometry.Polygon): The bounding box geometry.

    Returns:
        tuple: A tuple containing the maximum easting and northing coordinates.
    """
    coords = bbox_geom.exterior.coords.xy
    max_x = max(coords[0])
    max_y = max(coords[1])

    return (max_x, max_y)


def polygonize_binary_raster(binary_raster, crs=None, transform=None):

    if isinstance(binary_raster, str):
        with rio.open(binary_raster) as src:
            image=src.read(1)
            crs = src.crs
            transform = src.transform
    else:
        image = binary_raster

    mask= image==1
    geoms = ((shape(s), v) for s, v in shapes(image, mask, transform=transform))
    gdf=gpd.GeoDataFrame(geoms, columns=['geometry', 'class'])
    gdf.set_crs(crs=crs, inplace=True)

    return gdf