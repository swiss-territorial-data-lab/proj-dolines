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
import assess_results, define_possible_areas, depression_detection, get_slope, merge_dem_over_aoi, post_processing

logger = misc.format_logger(logger)


# ----- Define functions -----

def objective(trial, dem_dir, dem_correspondence_df, aoi_gdf, water_bodies_gdf, rivers_gdf, ref_data_type, ref_data_gdf, working_dir, slope_dir='slope', output_dir='.'):

    resolution = trial.suggest_float('resolution', 0.5, 5, step=0.5)
    max_slope = trial.suggest_float('max_slope', 0.7, 1.5, step=0.2)

    simplification_param = trial.suggest_float('simplification_param', 0.01, 0.1, step=0.01)
    mean_filter_size = trial.suggest_int('mean_filter_size', 3, 9, step=2)
    fill_depth = trial.suggest_float('fill_depth', 0, 1.5, step=0.5)
    max_part_in_lake = trial.suggest_float('max_part_in_lake', 0.1, 0.3, step=0.05)
    max_part_in_river = trial.suggest_float('max_part_in_river', 0.1, 0.3, step=0.05)
    min_compactness = trial.suggest_float('min_compactness', 0.25, 0.75, step=0.05)
    max_area = trial.suggest_int('max_area', 2500, 8000, step=500)
    min_diameter = trial.suggest_float('min_diameter', 2, 10, step=0.5)
    min_depth = trial.suggest_float('min_depth', 0.1, 1, step=0.1)
    max_depth = trial.suggest_int('max_depth', 30, 100, step=5)

    post_process_params = {
        'max_part_in_lake': max_part_in_lake,
        'max_part_in_river': max_part_in_river,
        'min_compactness': min_compactness,
        'max_area': max_area,
        'min_diameter': min_diameter,
        'min_depth': min_depth,
        'max_depth': max_depth
    } 

    _ = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, dem_dir, resolution, save_extra=True, output_dir=os.path.join(output_dir, 'merged_dems'))
    # get_slope.main(merged_tiles, output_dir=slope_dir)
    # possible_areas = define_possible_areas.main(slope_dir, non_sedi_areas_gdf, max_slope)

    dem_list = glob(os.path.join(output_dir, 'merged_dems', '*.tif'))
    detected_depressions_gdf, _ = depression_detection.main(dem_list, simplification_param, mean_filter_size, fill_depth, working_dir=working_dir, output_dir=output_dir, overwrite=True)
    detected_dolines_gdf, _ = post_processing.main(detected_depressions_gdf, water_bodies_gdf, rivers_gdf, output_dir=output_dir, **post_process_params)
    if detected_dolines_gdf.empty:
        return 0
    detected_dolines_gdf['det_class'] = 'doline'
    detected_dolines_gdf.rename(columns={'corresponding_dem': 'tile_id'}, inplace=True)

    metric, _ = assess_results.main(ref_data_type, ref_data_gdf, detected_dolines_gdf, aoi_gdf, det_type='ign')

    return metric


# ----- Main -----

tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script optimizes the watershed method for the detection of dolines.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
TILE_DIR = cfg['tile_dir']

REF_TYPE = cfg['ref_type']
REF_DATA = cfg[f'ref_data_{REF_TYPE.lower()}']
AOI = cfg['aoi']
DEM_CORRESPONDENCE = cfg['dem_correspondence']
NON_SEDIMENTARY_AREAS = cfg['non_sedimentary_areas']
TLM_DATA = cfg['tlm_data']
GROUND_COVER = cfg['ground_cover_layer']
RIVERS = cfg['rivers']

logger.warning(f'The reference data of {REF_TYPE} will be used.')
logger.warning(f'Then the {"f1 score" if REF_TYPE.lower() == "geocover" else "recall"} will be used as the metric.')

os.chdir(WORKING_DIR)
output_dir = os.path.join(OUTPUT_DIR) if REF_TYPE.lower() in OUTPUT_DIR.lower() else os.path.join(OUTPUT_DIR, REF_TYPE)
written_files = []

logger.info('Read data...')

# For the creation of the merged DEM
dem_correspondence_df, aoi_gdf = merge_dem_over_aoi.read_initial_data(AOI, DEM_CORRESPONDENCE)

# # For the determination of possible areas
# non_sedi_areas_gdf = gpd.read_file(NON_SEDIMENTARY_AREAS)

# For the post-processing
ground_cover_gdf = gpd.read_file(TLM_DATA, layer=GROUND_COVER)
rivers_gdf = gpd.read_file(RIVERS)
dissolved_rivers_gdf, water_bodies_gdf = post_processing.prepare_filters(ground_cover_gdf, rivers_gdf)

# For the assessment
ref_data_gdf = gpd.read_file(REF_DATA)
ref_data_gdf.to_crs(2056, inplace=True)
ref_data_gdf['label_class'] = 'doline'
if 'OBJECTID' in ref_data_gdf.columns:
    ref_data_gdf.rename(columns={'OBJECTID': 'objectid'}, inplace=True)

slope_dir = os.path.join(output_dir, 'slope')
study_path = os.path.join(output_dir, 'study.pkl')

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), study_name='Optimization of the watershed parameters')
# study = load(study_path, 'r')
objective = partial(
    objective, 
    dem_dir=TILE_DIR, dem_correspondence_df=dem_correspondence_df, aoi_gdf=aoi_gdf, water_bodies_gdf=water_bodies_gdf, rivers_gdf=dissolved_rivers_gdf, ref_data_type=REF_TYPE, ref_data_gdf=ref_data_gdf,
    working_dir=WORKING_DIR, slope_dir=slope_dir, output_dir=output_dir
)
study.optimize(objective, n_trials=100, callbacks=[opti.callback])

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
    _ = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, study.best_params['resolution'], save_extra=True, output_dir=OUTPUT_DIR)
    # get_slope.main(merged_tiles, slope_dir)
    # possible_areas = define_possible_areas.main(study.best_params['max_slope'])

    detected_depressions_gdf, depression_files = depression_detection.main(
        TILE_DIR, study.best_params['simplification_param'], study.best_params['mean_filter_size'], study.best_params['fill_depth'], working_dir=output_dir, overwrite=True
    )
    written_files.extend(depression_files)
    best_pp_param = {key: value for key, value in study.best_params.items() if key in [
        'max_part_in_lake', 'max_part_in_river', 'min_compactness', 'max_area', 'min_diameter', 'min_depth', 'max_depth'
    ]}
    detected_dolines_gdf, _ = post_processing.main(detected_depressions_gdf, water_bodies_gdf, rivers_gdf, **best_pp_param)
    # del possible_areas, merged_tiles

    metric, assessment_files = assess_results.main(REF_TYPE, ref_data_gdf, detected_dolines_gdf, ref_data_gdf, det_type='ign')
    written_files.extend(assessment_files)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer  
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")