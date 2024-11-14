import os
import sys
from glob import glob
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

import lidar

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, format_global_depressions, get_config, simplify_with_vw
from global_parameters import AOI_TYPE

logger = format_logger(logger)

def main(dem_list, min_size, min_depth, interval, bool_shp, simplification_param, save_extra=False, overwrite=False, output_dir='outputs'):

    written_files = []
    raw_potential_dolines_gdf = gpd.GeoDataFrame()
    for dem in dem_list:
        dem_name = os.path.basename(dem)
        dem_output_dir = os.path.join(output_dir, dem_name.rstrip('.tif'))

        if not os.path.exists(os.path.join(output_dir, dem_name.rstrip('.tif'), 'regions.shp')) or overwrite:
            os.makedirs(dem_output_dir, exist_ok=True)

            out_dem = os.path.join(dem_output_dir, "median_dem.tif")
            smoothed_dem = lidar.MedianFilter(dem, kernel_size=3, out_file=out_dem)
            sink_path = lidar.ExtractSinks(smoothed_dem, min_size, dem_output_dir)
            # dep_id_path, dep_level_path = lidar.DelineateDepressions(sink_path,
            #                                         min_size,
            #                                         min_depth,
            #                                         interval,
            #                                         dem_output_dir,
            #                                         bool_shp)

        local_dolines_gdf = gpd.read_file(os.path.join(dem_output_dir, 'regions.shp'))
        local_dolines_gdf['corresponding_dem'] = dem_name

        if local_dolines_gdf.empty:
            continue

        # Get depth
        dolines_stats = pd.read_csv(os.path.join(dem_output_dir, 'regions_info.csv'))
        local_dolines_gdf = pd.merge(local_dolines_gdf, dolines_stats[['region_id', 'max_depth']], left_on='id', right_on='region_id')
        local_dolines_gdf.rename(columns={'max_depth': 'depth'}, inplace=True)

        raw_potential_dolines_gdf = pd.concat([raw_potential_dolines_gdf, local_dolines_gdf[['geometry', 'corresponding_dem', 'depth']]], ignore_index=True)

    potential_dolines_gdf = format_global_depressions(raw_potential_dolines_gdf, simplification_param)

    if (potential_dolines_gdf.geometry.geom_type == 'MultiPolygon').any():
        potential_dolines_gdf = potential_dolines_gdf.explode(index_parts=False)
        potential_dolines_gdf = potential_dolines_gdf[potential_dolines_gdf.area > 1].copy()

    if save_extra:
        potential_dolines_gdf.to_file(os.path.join(output_dir, 'potential_dolines.gpkg'))

    return written_files


if __name__ == "__main__":
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs a stochastic depression detection with Whitebox Tools.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']

    MIN_SIZE = cfg['min_size']
    MIN_DEPTH = cfg['min_depth']
    INTERVAL = cfg['interval']
    BOOL_SHP = cfg['bool_shp']
    SIMPLIFICATION_PARAM = cfg['simplification_param']

    OVERWRITE = False

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    dem_list = glob(os.path.join(dem_dir, '*.tif'))
    if len(dem_list) == 0:
        logger.critical(f'No DEM found in {dem_dir}')
        sys.exit(1)

    written_files = main(dem_list, MIN_SIZE, MIN_DEPTH, INTERVAL, BOOL_SHP, SIMPLIFICATION_PARAM, save_extra=True, overwrite=OVERWRITE, output_dir=output_dir)

    logger.info('The following files were written:')
    for file in written_files:
        logger.info(file)

    logger.info('Done in {:.2f} seconds'.format(time() - tic))