import os
import sys
from loguru import logger
from time import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import mask
from rasterio.windows import Window

from math import floor, ceil
from skimage.metrics import structural_similarity
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from functions.fct_metrics import get_fractional_sets
from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# ----- Get config -----

tic = time()
logger.info('Starting...')

cfg = get_config(os.path.basename(__file__), desc="This script assesses the results of any doline detection method.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
DEM_DIR = cfg['dem_dir']

REF_DATA_TYPE = cfg['type']['ref_data']
REF_DATA = cfg[f'ref_data'][REF_DATA_TYPE.lower()]
DET_TYPE = cfg['type']['dets']
DETECTIONS = cfg[f'detections'][DET_TYPE.lower()]

PILOT_AREAS = cfg['pilot_areas']

RES = DEM_DIR.split('_')[2]

os.chdir(WORKING_DIR)
OUTPUT_DIR = OUTPUT_DIR if OUTPUT_DIR.endswith(DET_TYPE) else OUTPUT_DIR + '_' + DET_TYPE
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

# ----- Processing -----

logger.info('Read data...')

ref_data_gdf = gpd.read_file(REF_DATA)
ref_data_gdf.to_crs(2056, inplace=True)
ref_data_gdf['label_class'] = 'doline'
if 'OBJECTID' in ref_data_gdf.columns:
    ref_data_gdf.rename(columns={'OBJECTID': 'objectid'}, inplace=True)

detections_gdf = gpd.read_file(DETECTIONS)
detections_gdf['det_class'] = 'doline'
detections_gdf.rename(columns={'corresponding_dem': 'tile_id'}, inplace=True)

pilot_areas_gdf = gpd.read_file(PILOT_AREAS)
pilot_areas_gdf.to_crs(2056, inplace=True)

logger.info('Match detections with ground truth...')        # TODO: Associate dem name to ref data
ref_data_in_aoi_gdf = ref_data_gdf[ref_data_gdf.geometry.within(pilot_areas_gdf.geometry.union_all())].copy()
dets_in_aoi_gdf = detections_gdf[detections_gdf.geometry.within(pilot_areas_gdf.geometry.union_all())].copy()

tp_gdf, fp_gdf, fn_gdf, _ = get_fractional_sets(dets_in_aoi_gdf, ref_data_in_aoi_gdf)
tp_gdf['tag'] = 'TP'
fp_gdf['tag'] = 'FP'
fp_gdf['label_class'] = 'non-doline'
fn_gdf['tag'] = 'FN'
fn_gdf['det_class'] = 'non-doline'

tagged_detections_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf], ignore_index=True)

filepath = os.path.join(OUTPUT_DIR, f'{REF_DATA_TYPE}_tagged_detections.gpkg')
tagged_detections_gdf[detections_gdf.columns.tolist() + ['objectid', 'label_class', 'IOU', 'tag']].to_file(filepath)
written_files.append(filepath)

logger.info('Calculate metrics on all dets...')
metrics_dict = {
    'precision': precision_score(tagged_detections_gdf.label_class, tagged_detections_gdf.det_class, pos_label='doline'),
    'recall': recall_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline'),
    'f1': f1_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline'),
    'median IoU for TP': tp_gdf['IOU'].median()
}
metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['all']).transpose().round(3)


logger.info(f'Metrics:')
logger.info(f'- f1 score: {metrics_df.loc['all', "f1"]}')
logger.info(f'- median IoU for TP: {metrics_df.loc['all', "median IoU for TP"]}')

confusion_matrix_df = pd.DataFrame(confusion_matrix(
    tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class']
)).rename(columns={0: 'doline', 1: 'non-doline'}, index={0: 'doline', 1: 'non-doline'})

filepath = os.path.join(OUTPUT_DIR, f'{REF_DATA_TYPE}_confusion_matrix.csv')
confusion_matrix_df.to_csv(filepath)
written_files.append(filepath)

logger.info('Calculate the metrics by area...')
pilot_areas = tagged_detections_gdf.tile_id.unique().tolist()
metrics_per_area_dict = {
    'precision': [], 'recall': [], 'f1': [], 'median IoU for TP': []
}
for area_name in pilot_areas:
    results_in_area_gdf = tagged_detections_gdf[tagged_detections_gdf.tile_id == area_name].copy()

    metrics_per_area_dict['precision'].append(precision_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline'))
    metrics_per_area_dict['recall'].append(recall_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline'))
    metrics_per_area_dict['f1'].append(f1_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline'))
    metrics_per_area_dict['median IoU for TP'].append(results_in_area_gdf.loc[results_in_area_gdf.tag=='TP', 'IOU'].median())

    if recall_score(results_in_area_gdf['label_class'], results_in_area_gdf['det_class'], pos_label='doline') == 1:
        logger.warning(f'Area {area_name} is perfect!')

metrics_per_area_df = pd.DataFrame.from_dict(
    metrics_per_area_dict, orient='index', columns=pilot_areas
).transpose().round(3)
metrics_df = pd.concat([metrics_df, metrics_per_area_df])

# TODO: Make a graph P vs R on each area

logger.info('Compare image shapes...')

similarity_dict = {'structural similarity': []}
similarity_dir = os.path.join(OUTPUT_DIR, 'similarity_rasters')
os.makedirs(similarity_dir, exist_ok=True)
for area in pilot_areas_gdf.itertuples():
    area_dolines_gdf = dets_in_aoi_gdf[dets_in_aoi_gdf.tile_id == area.tile_id].copy()
    area_ref_data_gdf = ref_data_gdf[ref_data_gdf.tile_id == area.tile_id].copy()
    
    with rio.open(os.path.join(DEM_DIR, area.tile_id)) as src:
        meta = src.meta

        # We could crop the image before or after for cleaner results, but no impact on metrics
        ref_image = mask(src, area_ref_data_gdf.geometry)
        binary_ref_image = np.where(ref_image > 0, 1, 0)
        det_image = mask(src, area_dolines_gdf.geometry)
        binary_det_image = np.where(det_image > 0, 1, 0)

    similarity_dict['structural similarity'].append(structural_similarity(binary_ref_image, binary_det_image))

    filepath = os.path.join(similarity_dir, f'ref_{area.tile_id}.tif')
    with rio.open(filepath, 'w', **meta) as dest:
        dest.write(binary_ref_image)
    written_files.append(filepath)

    filepath =  os.path.join(similarity_dir, f'det_{area.tile_id}.tif')
    with rio.open(filepath, 'w', **meta) as dest:
        dest.write(binary_det_image)
    written_files.append(filepath)

similarity_df = pd.DataFrame.from_dict(similarity_dict, orient='index', columns=pilot_areas).transpose.round(3)
metrics_df = pd.concat([metrics_df, similarity_df], axis=1)

filepath = os.path.join(OUTPUT_DIR, f'{REF_DATA_TYPE}_metrics.csv')
metrics_df.to_csv(filepath)
written_files.append(filepath)

logger.success('Done! The following files were written:')
for file in written_files:
    logger.success(file)

logger.info(f'Done in {time() - tic:0.2f} seconds')
