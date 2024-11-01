import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import lidar

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config
from global_parameters import AOI_TYPE

logger = format_logger(logger)

def main(dem_list, min_size, min_depth, interval, bool_shp, output_dir='outputs'):

    written_files = []
    for dem in dem_list:
        out_dem = os.path.join(output_dir, "median_dem.tif")
        smoothed_dem = lidar.MedianFilter(dem, kernel_size=3, out_file=out_dem)
        sink_path = lidar.ExtractSinks(smoothed_dem, min_size, output_dir)
        dep_id_path, dep_level_path = lidar.DelineateDepressions(sink_path,
                                                   min_size,
                                                   min_depth,
                                                   interval,
                                                   output_dir,
                                                   bool_shp)

        written_files.append(dep_level_path)

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

    written_files = main(dem_list, MIN_SIZE, MIN_DEPTH, INTERVAL, BOOL_SHP, output_dir=output_dir)

    logger.info('The following files were written:')
    for file in written_files:
        logger.info(file)

    logger.info('Done in {:.2f} seconds'.format(time() - tic))