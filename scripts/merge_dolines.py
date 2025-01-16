import os
import sys
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config, simplify_with_vw

logger = format_logger(logger)


# Start chronometer
tic = time()
logger.info('Starting...')

# ----- Get parameters -----

cfg = get_config(config_key=os.path.basename(__file__), desc="This script merge the files for the different geologic types.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

AOI = cfg['aoi']
DGM = cfg['dgm']
EVAPORITES = cfg['evaporites']
KARST_NU = cfg['karst_nu']
MARNES_SUR_KARST = cfg['marnes_sur_karst']
MOLASSE = cfg['molasse']
RSVMC = cfg['rsvmc']

EPSG = 2056

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Data processing -----

logger.info('Read data...')
aoi_gdf = gpd.read_file(AOI)
aoi_gdf_2056 = aoi_gdf.to_crs(EPSG)

dolines_on_dgm = gpd.read_file(DGM)
dolines_on_evaporites = gpd.read_file(EVAPORITES)
dolines_on_karst_nu = gpd.read_file(KARST_NU)
dolines_on_marnes_sur_karst = gpd.read_file(MARNES_SUR_KARST)
dolines_on_molasse = gpd.read_file(MOLASSE)
dolines_on_rsvmc = gpd.read_file(RSVMC)

logger.info('Format dolines...')
dolines = pd.concat([dolines_on_dgm, dolines_on_evaporites, dolines_on_karst_nu, dolines_on_marnes_sur_karst, dolines_on_molasse, dolines_on_rsvmc], ignore_index=True)
dolines = dolines[dolines['type']!='thalweg'].reset_index(drop=True)

simplified_dolines_gdf = simplify_with_vw(dolines, 3)

logger.info('Clip dolines to the AOI...')
simplified_dolines_in_aoi_gdf = simplified_dolines_gdf.sjoin(aoi_gdf_2056[['name', 'Type', 'geometry']].rename(columns={'Type': 'geological_type'}), how='inner')

filepath = os.path.join(OUTPUT_DIR, 'merged_dolines.gpkg')
simplified_dolines_in_aoi_gdf.to_file(filepath)

logger.info(f'Done! Elapsed time: {time() - tic:0.2f} seconds')
logger.info(f'Written file: {filepath}')