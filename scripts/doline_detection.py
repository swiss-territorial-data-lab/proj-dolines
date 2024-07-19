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
from cv2 import GaussianBlur

from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

# ----- Get parameters -----

cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs the doline detection based on IGN's method.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
DEM_DIR = cfg['dem_dir']

AOI = cfg['aoi']

logger.info('Read data...')

dem_list = glob(os.path.join(DEM_DIR, '*.tif'))
aoi_gdf = gpd.read_file(AOI)

# ----- Generation of the DEM -----

for dem_path in tqdm(dem_list, desc="Smooth DEM"):

    with rio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta

    # Apply gaussian filter
    filtered_dem = GaussianBlur(dem_data, ksize=33, sigmaX=5)