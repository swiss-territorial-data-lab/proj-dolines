import sys
from argparse import ArgumentParser
from loguru import logger
from tqdm import tqdm
from yaml import FullLoader, load

from geopandas import GeoDataFrame
from shapely.geometry import mapping
from shapely.validation import make_valid

import pygeohash as pgh
import visvalingamwyatt as vw


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


def geohash(row):
    """Geohash encoding (https://en.wikipedia.org/wiki/Geohash) of a location (point).
    If geometry type is a point then (x, y) coordinates of the point are considered. 
    If geometry type is a polygon then (x, y) coordinates of the polygon centroid are considered. 
    Other geometries are not handled at the moment    

    Args:
        row: geodaframe row

    Raises:
        Error: geometry error

    Returns:
        out (str): geohash code for a given geometry
    """
    
    if row.geometry.geom_type == 'Point':
        out = pgh.encode(latitude=row.geometry.y, longitude=row.geometry.x, precision=16)
    elif row.geometry.geom_type == 'Polygon':
        out = pgh.encode(latitude=row.geometry.centroid.y, longitude=row.geometry.centroid.x, precision=16)
    else:
        logger.error(f"{row.geometry.geom_type} type is not handled (only Point or Polygon geometry type)")
        sys.exit()

    return out


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


def simplify_with_vw(gdf, simplification_param):
    """
    Simplify a GeoDataFrame using the Visvalingam-Whyatt algorithm.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to simplify.
        simplification_param (float): The simplification parameter for the Visvalingam-Whyatt algorithm.

    Returns:
        GeoDataFrame: The simplified GeoDataFrame.

    Notes:
        Some features may not be simplified. This is indicated by a warning message.
        The function uses a fallback simplification parameter of half the original value
        if the original value results in a polygon with less than 3 vertices.
    """
    _gdf = gdf.copy()

    failed_transform = 0
    mapped_objects = mapping(_gdf)
    for feature in tqdm(mapped_objects['features'], "Simplifying features"):
        coords = feature['geometry']['coordinates'][0]
        simplified_coords = vw.Simplifier(coords).simplify(threshold=simplification_param)
        if len(simplified_coords) >= 3:
            feature['geometry']['coordinates'] = (tuple([tuple(arr) for arr in simplified_coords]),)
            continue
        else:
            simplified_coords = vw.Simplifier(coords).simplify(threshold=simplification_param/2)
            if len(simplified_coords) >= 3:
                feature['geometry']['coordinates'] = (tuple([tuple(arr) for arr in simplified_coords]),)
                continue
            
        failed_transform += 1

    logger.warning(f'Simplification failed for {failed_transform} out of {len(mapped_objects["features"])} features')

    simplified_gdf = GeoDataFrame.from_features(mapped_objects, crs='EPSG:2056')
    simplified_gdf.loc[simplified_gdf.is_valid==False, 'geometry'] = \
        simplified_gdf.loc[simplified_gdf.is_valid==False, 'geometry'].apply(make_valid)
    if (_gdf.geometry == simplified_gdf.geometry).all():
        logger.warning('no simplification happened')
    assert (_gdf.shape[0] == simplified_gdf.shape[0]), 'some elements disappeared during simplification'

    return simplified_gdf