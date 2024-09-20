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
from global_parameters import AOI_TYPE

logger = misc.format_logger(logger)


# ----- Define functions -----

def objective(trial, dem_dir, dem_correspondence_df, aoi_gdf, non_sedi_areas_gdf, ref_data_type, ref_data_gdf, output_dir='outputs'):

    resolution = trial.suggest_float('resolution', 0.5, 4, step=0.25)
    max_slope = trial.suggest_float('max_slope', 1, 4, step=0.2)

    gaussian_kernel = trial.suggest_int('gaussian_kernel', 15, 27, step=2)
    gaussian_sigma = trial.suggest_float('gaussian_sigma', 3, 7, step=0.5)
    dem_diff_thrsld = trial.suggest_float('dem_diff_thrsld', 0.2, 2, step=0.2)
    min_area = trial.suggest_int('min_area', 20, 90, step=10)
    limit_compactness = trial.suggest_float('limit_compactness', 0.025, 0.35, step=0.025)
    min_voronoi_area = trial.suggest_int('min_voronoi_area', 5000, 100000, step=5000)
    min_merged_area = trial.suggest_int('min_merged_area', 200000, 400000, step=50000)
    max_long_area = trial.suggest_int('max_long_area', 1500, 7000, step=500)
    min_long_compactness = trial.suggest_float('min_long_compactness', 0.025, 0.3, step=0.025)
    min_round_compactness = trial.suggest_float('min_round_compactness', 0.3, 0.75, step=0.03)
    thalweg_buffer = trial.suggest_int('thalweg_buffer', 1, 15, step=2)
    thalweg_threshold = trial.suggest_float('thalweg_threshold', 0.1, 0.9, step=0.1)
    max_depth = trial.suggest_int('max_depth', 50, 150, step=5)

    dict_params = {
        'gaussian_kernel': gaussian_kernel,
        'gaussian_sigma': gaussian_sigma,
        'dem_diff_thrsld': dem_diff_thrsld,
        'min_area': min_area,
        'limit_compactness': limit_compactness,
        'min_voronoi_area': min_voronoi_area,
        'min_merged_area': min_merged_area,
        'max_long_area': max_long_area,
        'min_long_compactness': min_long_compactness,
        'min_round_compactness': min_round_compactness,
        'thalweg_buffer': thalweg_buffer,
        'thalweg_threshold': thalweg_threshold,
        'max_depth': max_depth
    }

    merged_tiles = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, dem_dir, resolution)
    slope_dir = os.path.join(output_dir, 'slope')
    get_slope.main(merged_tiles, output_dir=slope_dir)
    possible_areas = define_possible_areas.main(slope_dir, non_sedi_areas_gdf, max_slope)

    detected_dolines_gdf, _ = doline_detection.main(merged_tiles, possible_areas, output_dir=output_dir, **dict_params)
    if detected_dolines_gdf.empty:
        return 0
    detected_dolines_gdf = assess_results.prepare_dolines_to_assessment(detected_dolines_gdf)

    del possible_areas, merged_tiles

    metric, _ = assess_results.main(ref_data_type, ref_data_gdf, detected_dolines_gdf, aoi_gdf, det_type='ign')

    return metric


def callback(study, trial):
   # cf. https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601
    if (trial.number%5) == 0:
        study_path=os.path.join(output_dir, 'study.pkl')
        dump(study, study_path)


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
# logger.warning(f'Then the {"f1 score" if REF_TYPE.lower() == "geocover" else "recall"} will be used as the metric.')

os.chdir(WORKING_DIR)
output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if REF_TYPE.lower() in OUTPUT_DIR.lower() else os.path.join(OUTPUT_DIR, REF_TYPE, AOI_TYPE)
written_files = []

if AOI_TYPE:
    logger.warning(f'Working only on the areas of type {AOI_TYPE}')

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
    dem_dir=TILE_DIR, dem_correspondence_df=dem_correspondence_df, aoi_gdf=aoi_gdf, non_sedi_areas_gdf=non_sedi_areas_gdf, ref_data_type=REF_TYPE, ref_data_gdf=ref_data_gdf, 
    output_dir=output_dir
)
study.optimize(objective, n_trials=100, callbacks=[callback])

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
    merged_tiles = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, TILE_DIR, resolution=study.best_params['resolution'],
                                           save_extra=True, output_dir=os.path.join(output_dir, 'merged_tiles'))
    get_slope.main(merged_tiles, slope_dir)
    possible_areas = define_possible_areas.main(slope_dir, non_sedi_areas_gdf, study.best_params['max_slope'])

    dict_params = {key: value for key, value in study.best_params.items() if key not in ['max_slope', 'resolution']}
    detected_dolines_gdf, doline_files = doline_detection.main(merged_tiles, possible_areas, save_extra=True, output_dir=output_dir, **dict_params)
    written_files.extend(doline_files)

    del possible_areas, merged_tiles

    detected_dolines_gdf = assess_results.prepare_dolines_to_assessment(detected_dolines_gdf)
    metric, assessment_files = assess_results.main(REF_TYPE, ref_data_gdf, detected_dolines_gdf, aoi_gdf, det_type='ign', 
                                                   dem_dir=os.path.join(output_dir, 'merged_tiles'), save_extra=True, output_dir=output_dir)
    written_files.extend(assessment_files)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer  
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")