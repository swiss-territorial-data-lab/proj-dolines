import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.windows import Window

from math import ceil, floor

from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

# ----- Get parameters -----

cfg = get_config(config_key=os.path.basename(__file__), desc="This script resamples the DEM files to 5 m.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
DEM_DIR = cfg['dem_dir']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info('Read data...')

dem_list = glob(os.path.join(DEM_DIR, '*.tif'))
if len(dem_list) == 0:
    logger.critical(f'No DEM found in {DEM_DIR}')
    sys.exit(1)

    
# ----- Data processing -----

for dem_path in tqdm(dem_list, desc="Smooth DEM"):

    with rio.open(dem_path) as src:
        dem_data = src.read()
        dem_meta = src.meta

    # Calculate slope
    cell_size = dem_meta['transform'][0]
    px, py = np.gradient(dem_data[0, :, :], cell_size)
    slope = np.sqrt(px**2 + py**2)
    
    slope_dem_meta = dem_meta.copy()
    slope_dem_meta.update({'dtype': 'float32'})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with rio.open(os.path.join(OUTPUT_DIR, 'slope_' + os.path.basename(dem_path)), 'w', **slope_dem_meta) as dst:
        dst.write(slope[np.newaxis, ...])


toc = time()
logger.info(f"Done in {toc - tic:0.4f} seconds")