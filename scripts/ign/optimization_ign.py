import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
from rasterio import open

import optuna
from functools import partial
from joblib import dump, load
from math import floor

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_optimization as opti
import assess_results, define_possible_areas, doline_detection, get_slope, merge_dem_over_aoi

logger = misc.format_logger(logger)


# ----- Define functions -----

def objective(trial, dem_dir, dem_correspondence_df, aoi_gdf, non_sedi_areas_gdf, ref_data_gdf, slope_dir='slope'):

    resolution = trial.suggest_float('resolution', 0.5, 5, step=0.5)
    max_slope = trial.suggest_float('max_slope', 0.7, 1.5, step=0.2)

    gaussian_kernel = trial.suggest_int('gaussian_kernel', 3, 33, step=2)
    gaussian_sigma = trial.suggest_float('gaussian_sigma', 0.5, 5, step=0.5)
    dem_diff_thrsld = trial.suggest_float('dem_diff_thrsld', 0.5, 5, step=0.5)
    min_area = trial.suggest_int('min_area', 30, 100, step=10)
    limit_compactness = trial.suggest_float('limit_compactness', 0.1, 0.5, step=0.1)
    min_voronoi_area = trial.suggest_int('min_voronoi_area', 20000, 300000, step=10000)
    max_long_area = trial.suggest_int('max_long_area', 2500, 8000, step=500)
    min_long_compactness = trial.suggest_float('min_long_compactness', 0.1, 0.5, step=0.05)
    min_round_compactness = trial.suggest_float('min_round_compactness', 0.1, 0.5, step=0.05)
    thalweg_buffer = trial.suggest_int('thalweg_buffer', 1, 15, step=2)
    thalweg_threshold = trial.suggest_float('thalweg_threshold', 0.1, 0.9, step=0.1)
    max_depth = trial.suggest_int('max_depth', 30, 100, step=5)

    dict_params = {
        'gaussian_kernel': gaussian_kernel,
        'gaussian_sigma': gaussian_sigma,
        'dem_diff_thrsld': dem_diff_thrsld,
        'min_area': min_area,
        'limit_compactness': limit_compactness,
        'min_voronoi_area': min_voronoi_area,
        'max_long_area': max_long_area,
        'min_long_compactness': min_long_compactness,
        'min_round_compactness': min_round_compactness,
        'thalweg_buffer': thalweg_buffer,
        'thalweg_threshold': thalweg_threshold,
        'max_depth': max_depth
    }

    merged_tiles = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, dem_dir, resolution)
    get_slope.main(merged_tiles, output_dir=slope_dir)
    possible_areas = define_possible_areas.main(slope_dir, non_sedi_areas_gdf, max_slope)

    detected_dolines_gdf, _ = doline_detection.main(merged_tiles, possible_areas, **dict_params)
    if detected_dolines_gdf.empty:
        return 0
    detected_dolines_gdf['det_class'] = 'doline'
    detected_dolines_gdf.rename(columns={'corresponding_dem': 'tile_id'}, inplace=True)

    del possible_areas, merged_tiles

    metric, _ =assess_results.main(detected_dolines_gdf, ref_data_gdf)

    return metric


# ----- Main -----

tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script optimizes the IGN's method for the detection of dolines.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
TILE_DIR = cfg['tile_dir']

REF_TYPE = cfg['ref_type']
REF_DATA = cfg[f'ref_data_{REF_TYPE.lower()}']
AOI = cfg['aoi']
DEM_CORRESPONDENCE = cfg['dem_correspondence']
NON_SEDIMENTARY_AREAS = cfg['non_sedimentary_areas']

logger.warning(f'The reference data of {REF_TYPE} will be used.')
logger.warning(f'Then the metric {"f1 score" if REF_TYPE.lower() == "ign" else "recall"} will be used.')

os.chdir(WORKING_DIR)
output_dir = os.path.join(OUTPUT_DIR) if REF_TYPE.lower() in OUTPUT_DIR.lower() else os.path.join(OUTPUT_DIR, REF_TYPE)
written_files = []

logger.info('Read data...')

# For the creation of the merged DEM
dem_correspondence_df, aoi_gdf = merge_dem_over_aoi.read_initial_data(AOI, DEM_CORRESPONDENCE)

# For the determination of possible areas
non_sedi_areas_gdf = gpd.read_file(NON_SEDIMENTARY_AREAS)

# For the assessment
ref_data_gdf = gpd.read_file(REF_DATA)
ref_data_gdf.to_crs(2056, inplace=True)
ref_data_gdf['label_class'] = 'doline'
if 'OBJECTID' in ref_data_gdf.columns:
    ref_data_gdf.rename(columns={'OBJECTID': 'objectid'}, inplace=True)

slope_dir = os.path.join(output_dir, 'slope')
study_path = os.path.join(output_dir, 'study.pkl')

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), study_name='Optimization of the IGN parameters')
# study = load(study_path, 'r')
objective = partial(
    objective, 
    dem_dir=TILE_DIR, dem_correspondence_df=dem_correspondence_df, aoi_gdf=aoi_gdf, non_sedi_areas_gdf=non_sedi_areas_gdf, ref_data_gdf=ref_data_gdf, slope_dir=slope_dir
)
study.optimize(objective, n_trials=500, callbacks=[opti.callback])

dump(study, study_path)
written_files.append(study_path)

if study.best_value !=0:
    logger.info('Save the best parameters')
    targets = {0: 'f1 score'}
    written_files.append(opti.save_best_parameters(study, targets, output_dir=output_dir))

    logger.info('Plot results...')
    output_plots = os.path.join(output_dir, 'plots')
    os.makedirs(output_plots, exist_ok=True)
    written_files.extend(opti.plot_optimization_results(study, targets, output_path=output_plots))

    logger.info('Produce results for the best parameters')
    merged_tiles = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, study.best_params['resolution'], save_extra=True, output_dir=OUTPUT_DIR)
    get_slope.main(merged_tiles, slope_dir)
    possible_areas = define_possible_areas.main(study.best_params['max_slope'])

    detected_dolines_gdf, doline_files = doline_detection.main(merged_tiles, possible_areas, **study.best_params)
    written_files.extend(doline_files)

    del possible_areas, merged_tiles

    metric, assessment_files =assess_results.main(detected_dolines_gdf, ref_data_gdf)
    written_files.extend(assessment_files)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer  
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")