import os
import sys
from argparse import ArgumentParser
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import whitebox
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config
from global_parameters import AOI_TYPE

logger = format_logger(logger)

def main(dem_list, rmse, autocorr_range, iterations, working_dir='.', output_dir='outputs'):

    written_files = []
    for dem in dem_list[:2]:
        outpath = os.path.join(working_dir, output_dir, os.path.splitext(os.path.basename(dem))[0] + '_pdep.tif')
        wbt.stochastic_depression_analysis(
            dem = os.path.join(working_dir, dem),
            output = outpath,
            rmse = rmse,
            range = autocorr_range,
            iterations = iterations
        )

        written_files.append(outpath)

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

    RMSE = cfg['rmse']
    AUTOCORR_RANGE = cfg['autocorr_range']
    ITERATIONS = cfg['iterations']

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

    written_files = main(dem_list, RMSE, AUTOCORR_RANGE, ITERATIONS, WORKING_DIR, output_dir=output_dir)

    logger.info('The following files were written:')
    for file in written_files:
        logger.info(file)

    logger.info('Done in {:.2f} seconds'.format(time() - tic))