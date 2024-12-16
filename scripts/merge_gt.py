import os
import sys
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# ----- Get config -----

tic = time()
logger.info('Starting...')

cfg = get_config(os.path.basename(__file__), desc="This script merge the available ground truth.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

AOI = cfg['aoi']
REF_DATA = cfg['ref_data']
GT = cfg['gt']

GEOCOVER = REF_DATA['geocover']
TLM = REF_DATA['tlm']
NE = REF_DATA['ne']

CHASSERAL = GT['chasseral']
PT_GT = GT['point_gt']

POINTS = cfg['points']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Data processing -----

logger.info('Read data...')
aoi_gdf = gpd.read_file(AOI)
geocover_gdf = gpd.read_file(GEOCOVER)
geocover_gdf = geocover_gdf.explode(ignore_index=True)
tlm_gdf = gpd.read_file(TLM)
ne_gdf = gpd.read_file(NE)
ne_gdf = ne_gdf.explode(ignore_index=True)
chasseral_gdf = gpd.read_file(CHASSERAL)
point_gt_gdf = gpd.read_file(PT_GT)

for gdf in [geocover_gdf, tlm_gdf, ne_gdf, chasseral_gdf, point_gt_gdf]:
    gdf['id'] = gdf.index
    gdf.to_crs(2056, inplace=True)
    gdf.drop(columns=gdf.columns.difference(['id', 'geometry', 'OBJECTID', 'objectid', 'ID_DOL']), inplace=True)

logger.info('Merge reference data...')
geocover_vs_ne = gpd.sjoin(geocover_gdf, ne_gdf, how='left', lsuffix='geocover', rsuffix='ne')
first_ref_data_gdf = pd.concat([
    geocover_gdf[geocover_gdf.id.isin(geocover_vs_ne.loc[geocover_vs_ne['id_ne'].isnull(), 'id_geocover'])],
    ne_gdf
], ignore_index=True)
first_ref_data_gdf.loc[:, 'id'] = first_ref_data_gdf.index

first_ref_vs_tlm = gpd.sjoin(first_ref_data_gdf, tlm_gdf, how='left', lsuffix='ref1', rsuffix='tlm')
ref_data_gdf = pd.concat([
    first_ref_data_gdf[first_ref_data_gdf.id.isin(first_ref_vs_tlm.loc[first_ref_vs_tlm['id_tlm'].isnull(), 'id_ref1'])],
    tlm_gdf
], ignore_index=True)

ref_data_gdf.loc[:, 'id'] = ref_data_gdf.index
ref_data_gdf.rename(columns={'objectid': 'id_geocover'}, inplace=True)
filepath = os.path.join(OUTPUT_DIR, 'ref_data.gpkg')
ref_data_gdf.to_file(filepath)
written_files = [filepath]

if POINTS:
    logger.info('Merge available ground truth...')
    ref_data_gdf.loc[:, 'geometry'] = ref_data_gdf.centroid
    filtered_ref_data_gdf = gpd.overlay(ref_data_gdf, aoi_gdf.loc[aoi_gdf.name.isin(['Les Verrières (NE)', 'Sissach (BL)', 'Schwarzsee (FR)']), ['geometry']])

    ground_truth_gdf = pd.concat(
        [chasseral_gdf.rename(columns={'objectid': 'id_gt_chasseral'}), point_gt_gdf.rename(columns={'objectid': 'id_pt_gt'}), filtered_ref_data_gdf], 
        ignore_index=True
    )

    ground_truth_gdf.loc[:, 'id'] = ground_truth_gdf.index
    filepath = os.path.join(OUTPUT_DIR, 'ground_truth.gpkg')
    ground_truth_gdf.to_file(filepath)
    written_files.append(filepath)

logger.success(f'Done! The following files were written:')
for file in written_files:
    logger.success(file)

logger.info(f'Done in {time() - tic:0.4f} seconds')