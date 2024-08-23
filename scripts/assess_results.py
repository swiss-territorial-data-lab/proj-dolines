import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.mask import mask
from rasterio.windows import Window

from math import floor, ceil
from skimage.metrics import hausdorff_distance, structural_similarity
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from functions.fct_metrics import get_fractional_sets
from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)


def main(ref_data_type, detections_gdf, pilot_areas_gdf, save_extra=False, output_dir='outputs'):
    output_dir = output_dir if output_dir.endswith(DET_TYPE) else output_dir + '_' + DET_TYPE
    os.makedirs(output_dir, exist_ok=True)
    written_files = []

    if 'type' in detections_gdf:
        detections_gdf = detections_gdf[detections_gdf['type']!='thalweg'].copy()

    logger.info('Match detections with ground truth...')        # TODO: Associate dem name to ref data
    ref_data_in_aoi_gdf = ref_data_gdf.sjoin(pilot_areas_gdf[['tile_id', 'geometry']], how='inner')
    ref_data_in_aoi_gdf.loc[:, 'tile_id'] = [tile_id + '.tif' for tile_id in ref_data_in_aoi_gdf.tile_id]
    dets_in_aoi_gdf = detections_gdf[detections_gdf.geometry.within(pilot_areas_gdf.geometry.union_all())].copy()

    tp_gdf, fp_gdf, fn_gdf, _ = get_fractional_sets(dets_in_aoi_gdf, ref_data_in_aoi_gdf[['objectid', 'label_class', 'tile_id', 'geometry']])
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fp_gdf['label_class'] = 'non-doline'
    fn_gdf['tag'] = 'FN'
    fn_gdf['det_class'] = 'non-doline'

    tagged_detections_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf], ignore_index=True)

    logger.info('Calculate metrics on all dets...')
    metric_fct = f1_score if ref_data_type == 'geocover' else recall_score
    metric = metric_fct(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline')
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

    if save_extra:
        filepath = os.path.join(output_dir, f'{ref_data_type}_tagged_detections.gpkg')
        tagged_detections_gdf[detections_gdf.columns.tolist() + ['objectid', 'label_class', 'IOU', 'tag']].to_file(filepath)
        written_files.append(filepath)

        confusion_matrix_df = pd.DataFrame(confusion_matrix(
            tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class']
        )).rename(columns={0: 'doline', 1: 'non-doline'}, index={0: 'doline', 1: 'non-doline'})

        filepath = os.path.join(output_dir, f'{ref_data_type}_confusion_matrix.csv')
        confusion_matrix_df.to_csv(filepath)
        written_files.append(filepath)

        logger.info('Calculate the metrics for each pilot area...')
        pilot_areas = tagged_detections_gdf.tile_id.unique().tolist()
        metrics_per_area_dict = {
            'precision': [], 'recall': [], 'f1': [], 'median IoU for TP': []
        }
        for area_name in pilot_areas:
            results_in_area_gdf = tagged_detections_gdf[tagged_detections_gdf.tile_id == area_name].copy()
            if results_in_area_gdf[results_in_area_gdf.det_class == 'non-doline'].empty:
                logger.warning(f'Area {area_name} does not contain any label corresponding to a false negative!')

            metrics_per_area_dict['precision'].append(precision_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline'))
            metrics_per_area_dict['recall'].append(recall_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline', zero_division=0))
            metrics_per_area_dict['f1'].append(f1_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline'))
            metrics_per_area_dict['median IoU for TP'].append(results_in_area_gdf.loc[results_in_area_gdf.tag=='TP', 'IOU'].median())

        metrics_per_area_df = pd.DataFrame.from_dict(
            metrics_per_area_dict, orient='index', columns=pilot_areas
        ).transpose().round(3)
        metrics_df = pd.concat([metrics_df, metrics_per_area_df])

        # TODO: Make a graph P vs R on each area

        similarity_dict = {'structural similarity': []}
        similarity_dir = os.path.join(output_dir, 'similarity_rasters')
        os.makedirs(similarity_dir, exist_ok=True)
        for area in tqdm(pilot_areas_gdf.itertuples(), desc='Compare image shapes', total=pilot_areas_gdf.shape[0]):
            area_dolines_gdf = dets_in_aoi_gdf[dets_in_aoi_gdf.tile_id == area.tile_id+'.tif'].copy()
            area_ref_data_gdf = ref_data_in_aoi_gdf[ref_data_in_aoi_gdf.tile_id == area.tile_id+'.tif'].copy()

            if area_ref_data_gdf.empty or area_dolines_gdf.empty:
                logger.warning(f'No label or data for the area {area.name}.')
                similarity_dict['structural similarity'].append(0)
                continue
            
            with rio.open(os.path.join(DEM_DIR, area.tile_id + '.tif')) as src:
                meta = src.meta

                # Mask DEM with the dolines
                ref_image, _ = mask(src, area_ref_data_gdf.geometry)
                det_image, _ = mask(src, area_dolines_gdf.geometry)
                ref_cond = ref_image != meta['nodata']
                det_cond = det_image != meta['nodata']
                # Get the max and min coordinates of the dolines in the pilot area
                min_x = min(np.where(ref_cond)[1].min(), np.where(det_cond)[1].min())
                min_y = min(np.where(ref_cond)[2].min(), np.where(det_cond)[2].min())
                max_x = max(np.where(ref_cond)[1].max(), np.where(det_cond)[1].max())
                max_y = max(np.where(ref_cond)[2].max(), np.where(det_cond)[2].max())

                # Binary raster cropped to the dolines of the pilot area
                binary_ref_image = np.where(ref_cond, 1, 0)[:, min_x:max_x, min_y:max_y]
                binary_det_image = np.where(det_cond, 1, 0)[:, min_x:max_x, min_y:max_y]
                
                similarity_dict['structural similarity'].append(structural_similarity(binary_ref_image, binary_det_image, data_range=1, channel_axis=0))
                # TODO: do the hausdorff distance with the centroid of the points and not the binary image of the polygons
                # similarity_dict['hausdorff distance'].append(hausdorff_distance(binary_ref_image, binary_det_image, method='modified'))

        similarity_df = pd.DataFrame.from_dict(similarity_dict, orient='index', columns=pilot_areas).transpose().round(3)
        metrics_df = pd.concat([metrics_df, similarity_df], axis=1)

        filepath = os.path.join(output_dir, f'{ref_data_type}_metrics.csv')
        metrics_df.to_csv(filepath)
        written_files.append(filepath)

    return metric, written_files

if __name__ == '__main__':
    tic = time()
    logger.info('Starting...')

    # ----- Get config -----
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

    _, written_files = main(ref_data_gdf, detections_gdf, pilot_areas_gdf, save_extra=True, output_dir=OUTPUT_DIR)

    logger.success('Done! The following files were written:')
    for file in written_files:
        logger.success(file)

    logger.info(f'Done in {time() - tic:0.2f} seconds')
