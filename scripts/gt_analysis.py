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
from rasterstats import zonal_stats

from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# ----- Get config -----

tic = time()
logger.info('Starting...')

cfg = get_config(os.path.basename(__file__), desc="This script performs the ground truth analysis.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
SLOPE_DIR = cfg['slope_dir']
DEM_DIR = cfg['dem_dir']

AOI = cfg['aoi']
DOLINES = cfg['dolines']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

# ----- Data processing -----

logger.info('Read data ...')

aoi_gdf = gpd.read_file(AOI)
aoi_gdf.to_crs(2056, inplace=True)
dolines_gdf = gpd.read_file(DOLINES)
dolines_gdf.to_crs(2056, inplace=True)

dem_tile_list = glob(os.path.join(DEM_DIR, 'swissalti3d_*.tif'))

logger.info('Limit dolines to AOI...')

gt_gdf = gpd.sjoin(dolines_gdf[['objectid', 'uuid', 'geometry']], aoi_gdf[['name', 'geometry']], how='inner')
gt_gdf['area'] = gt_gdf.area
gt_gdf['perimeter'] = gt_gdf.length
new_attributes = ['min', 'max', 'median', 'slope']
for attr in new_attributes:
    gt_gdf[attr] = np.nan

for tile in tqdm(dem_tile_list, desc="Get zonal stats"):
    dem_name = os.path.basename(tile)
    slope_name = dem_name.replace('swissalti3d', 'slope')

    with rio.open(tile) as src:
        altitude = zonal_stats(gt_gdf, src.read(1), affine=src.transform, stats=['min', 'max', 'median'])

    with rio.open(os.path.join(SLOPE_DIR, slope_name)) as src:
        slope = zonal_stats(gt_gdf, src.read(1), affine=src.transform, stats=['median'])

    altitude_df = pd.DataFrame(altitude)
    if altitude_df['median'].notna().any():
        slope_df = pd.DataFrame(slope)

        altitude_df['slope'] = slope_df['median']
        altitude_df['doline_id'] = gt_gdf['objectid'].to_numpy()
        intersected_ids = altitude_df.loc[altitude_df.slope.notna(), 'doline_id']

        gt_gdf.loc[gt_gdf.objectid.isin(intersected_ids), new_attributes] = altitude_df.loc[altitude_df.slope.notna(), new_attributes].to_numpy()

gt_gdf.astype({'min': 'float32', 'max': 'float32', 'median': 'float32', 'slope': 'float32'}, copy=False)
gt_gdf['alti_diff'] = gt_gdf['max'] - gt_gdf['min']
gt_gdf.to_file(os.path.join(OUTPUT_DIR, 'gt_analysis.gpkg'))

logger.success(f'Done! The following file was written: {os.path.join(OUTPUT_DIR, "gt_analysis.gpkg")}.')
logger.info(f'Elapsed time: {time() - tic:0.2f} seconds')