import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import numpy as np
import rasterio as rio

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config
from global_parameters import AOI_TYPE

logger = format_logger(logger)


def main(dem_dict, output_dir='outputs/slope'):
    """
    Calculate the slope of a DEM and save it to a new file.

    Parameters
    ----------
    dem_dict : dict
        Dictionary with keys as the name of the DEM tiles and values as a tuple containing the DEM array and its metadata.
    output_dir : str, optional
        Output directory for the results. Defaults to 'outputs/slope'.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dem_name, data in tqdm(dem_dict.items(), desc="Calculate slope"):
        dem_data, dem_meta = data

        # Calculate slope
        cell_size = dem_meta['transform'][0]
        px, py = np.gradient(dem_data, cell_size)
        slope = np.sqrt(px**2 + py**2)
        
        slope_dem_meta = dem_meta.copy()
        slope_dem_meta.update({'dtype': 'float32'})

        with rio.open(os.path.join(output_dir, 'slope_' + dem_name), 'w', **slope_dem_meta) as dst:
            dst.write(slope[np.newaxis, ...])

   
if __name__ == "__main__":
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    cfg = get_config(config_key=os.path.basename(__file__), desc=" This script calculates the slope of a DEM and save it to a new file")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info('Read data...')

    dem_list = glob(os.path.join(dem_dir, '*.tif'))
    if len(dem_list) == 0:
        logger.critical(f'No DEM found in {dem_dir}')
        sys.exit(1)
    dem_dict = {}
    for dem_path in tqdm(dem_list, desc="Read DEM"):
        with rio.open(dem_path) as src:
            dem_data = src.read(1)
            dem_meta = src.meta
        dem_dict[os.path.basename(dem_path)] = dem_data, dem_meta

    main(dem_dict, output_dir=output_dir)

    logger.success(f'Done! The files were written in the folder {output_dir}')

    toc = time()
    logger.info(f"Done in {toc - tic:0.4f} seconds")