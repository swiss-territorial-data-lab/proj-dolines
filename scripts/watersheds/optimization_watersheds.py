import os
import sys
from glob import glob
from loguru import logger
from time import time

import geopandas as gpd

import optuna
from functools import partial
from joblib import dump

sys.path.insert(1, 'scripts')
import functions.fct_misc as misc
import functions.fct_optimization as opti
import assess_results, depression_detection, merge_dem_over_aoi, post_processing
from global_parameters import AOI_TYPE

logger = misc.format_logger(logger)


# ----- Define functions -----

def objective(trial, dem_dir, dem_correspondence_df, aoi_gdf, ref_data_type, ref_data_gdf,
              non_sedimentary_areas_gdf, builtup_areas_gdf, water_bodies_gdf, rivers_gdf, working_dir, slope_dir='slope', output_dir='.'):
    """
    Objective function to optimize for doline detection with the watershed method.

    Parameters
    ----------
    trial : optuna.trial
        The trial object from the Optuna optimization process.
    dem_dir : str
        The directory where the DEMs are stored.
    dem_correspondence_df : pandas.DataFrame
        The DataFrame containing the correspondence between the DEMs and the AOIs.
    aoi_gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the AOIs.
    water_bodies_gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the water bodies.
    rivers_gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the rivers.
    ref_data_type : str
        The type of reference data, possible values are 'geocover' and 'tlm'.
    ref_data_gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the formatted reference data.
    working_dir : str
        The directory where the intermediate files will be saved.
    slope_dir : str, optional
        The directory where the slope files will be saved. Defaults to 'slope'.
    output_dir : str, optional
        The directory where the output files will be saved. Defaults to '.'.

    Returns
    -------
    float
        The value of the objective function, i.e. the f1 score.
    """

    resolution = trial.suggest_float('resolution', 0.5, 2.5, step=0.5)
    # max_slope = trial.suggest_float('max_slope', 0.7, 1.5, step=0.2)

    mean_filter_size = trial.suggest_int('mean_filter_size', 1, 5, step=2)
    fill_depth = trial.suggest_float('fill_depth', 0.5, 5, step=0.25)
    max_part_in_lake = trial.suggest_float('max_part_in_lake', 0.05, 0.35, step=0.05)
    max_part_in_river = trial.suggest_float('max_part_in_river', 0.1, 0.4, step=0.05)
    min_compactness = trial.suggest_float('min_compactness', 0.25, 0.75, step=0.05)
    min_area = trial.suggest_int('min_area', 5, 75, step=5)
    max_area = trial.suggest_int('max_area', 1500, 7500, step=500)
    min_diameter = trial.suggest_float('min_diameter', 5, 15, step=0.5)
    min_depth = trial.suggest_float('min_depth', 0.5, 1.5, step=0.1)
    max_depth = trial.suggest_int('max_depth', 60, 180, step=5)
    max_std_elev = trial.suggest_float('max_std_elev', 0.015, 10, log=True)

    post_process_params = {
        'max_part_in_lake': max_part_in_lake,
        'max_part_in_river': max_part_in_river,
        'min_compactness': min_compactness,
        'min_area': min_area,
        'max_area': max_area,
        'min_diameter': min_diameter,
        'min_depth': min_depth,
        'max_depth': max_depth,
        'max_std_elev': max_std_elev
    } 

    _ = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, dem_dir, resolution, save_extra=True, output_dir=os.path.join(output_dir, 'merged_dems'))
    # get_slope.main(merged_tiles, output_dir=slope_dir)
    # possible_areas = define_possible_areas.main(slope_dir, non_sedi_areas_gdf, max_slope)

    dem_list = glob(os.path.join(output_dir, 'merged_dems', '*.tif'))
    detected_depressions_gdf, _ = depression_detection.main(dem_list, non_sedimentary_areas_gdf, builtup_areas_gdf, mean_filter_size, fill_depth, working_dir=working_dir, output_dir=output_dir, overwrite=True)
    detected_dolines_gdf, _ = post_processing.main(detected_depressions_gdf, water_bodies_gdf, rivers_gdf, output_dir=output_dir, **post_process_params)
    if detected_dolines_gdf.empty:
        logger.info(f'Metrics:')
        logger.info(f'- f1 score: 0')
        return 0
    detected_dolines_gdf = assess_results.prepare_dolines_to_assessment(detected_dolines_gdf)

    metric, _ = assess_results.main(ref_data_type, ref_data_gdf, detected_dolines_gdf, aoi_gdf, det_type='watersheds')

    return metric


def callback(study, trial):
    """
    Save the study to disk every 5 trials.
    cf. https://stackoverflow.com/questions/62144904/python-how-to-retrive-the-best-model-from-optuna-lightgbm-study/62164601#62164601

    Parameters
    ----------
    study : optuna.study.Study
        The study object from the Optuna optimization process.
    trial : optuna.trial.Trial
        The trial object from the Optuna optimization process.
    """
    
    if (trial.number%5) == 0:
        study_path=os.path.join(output_dir, 'study.pkl')
        dump(study, study_path)


# ----- Main -----

tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script optimizes the watershed method for the detection of dolines.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
TILE_DIR = cfg['tile_dir']

REF_TYPE = cfg['ref_type']
REF_DATA = cfg[f'ref_data'][REF_TYPE.lower()]
AOI = cfg['aoi']
DEM_CORRESPONDENCE = cfg['dem_correspondence']
NON_SEDIMENTARY_AREAS = cfg['non_sedimentary_areas']
BUILTUP_AREAS = cfg['builtup_areas']
TLM_DATA = cfg['tlm_data']
GROUND_COVER = cfg['ground_cover_layer']
RIVERS = cfg['rivers']

logger.warning(f'The reference data of {REF_TYPE} will be used.')
# logger.warning(f'Then the {"f1 score" if REF_TYPE.lower() == "geocover" else "recall"} will be used as the metric.')

os.chdir(WORKING_DIR)
output_dir = OUTPUT_DIR if REF_TYPE.lower() in OUTPUT_DIR.lower() else os.path.join(OUTPUT_DIR, REF_TYPE)
written_files = []

if AOI_TYPE:
    logger.warning(f'Working only on the areas of type {AOI_TYPE}')
    output_dir = os.path.join(output_dir, AOI_TYPE) if AOI_TYPE else output_dir

logger.info('Read data...')

# For the creation of the merged DEM
dem_correspondence_df, aoi_gdf = merge_dem_over_aoi.read_initial_data(AOI, DEM_CORRESPONDENCE)

# # For the determination of possible areas
non_sedi_areas_gdf = gpd.read_file(NON_SEDIMENTARY_AREAS)
builtup_areas_gdf = gpd.read_file(BUILTUP_AREAS)

# For the post-processing
ground_cover_gdf = gpd.read_file(TLM_DATA, layer=GROUND_COVER)
rivers_gdf = gpd.read_file(RIVERS)
dissolved_rivers_gdf, water_bodies_gdf = post_processing.prepare_filters(ground_cover_gdf, rivers_gdf)

# For the assessment
ref_data_gdf = assess_results.prepare_reference_data_to_assessment(REF_DATA)

slope_dir = os.path.join(output_dir, 'slope')
study_path = os.path.join(output_dir, 'study.pkl')

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(), study_name='Optimization of the watershed parameters')
# study = load(study_path, 'r')
objective = partial(
    objective, 
    dem_dir=TILE_DIR, dem_correspondence_df=dem_correspondence_df, aoi_gdf=aoi_gdf, ref_data_type=REF_TYPE, ref_data_gdf=ref_data_gdf,
    non_sedimentary_areas_gdf=non_sedi_areas_gdf, builtup_areas_gdf=builtup_areas_gdf, water_bodies_gdf=water_bodies_gdf, rivers_gdf=dissolved_rivers_gdf, 
    working_dir=WORKING_DIR, slope_dir=slope_dir, output_dir=output_dir
)
study.optimize(objective, n_trials=100, callbacks=[callback])

dump(study, study_path)
written_files.append(study_path)

if study.best_value !=0:
    logger.info('Save the best parameters')
    targets = {0: "f1 score"}
    written_files.append(opti.save_best_parameters(study, targets, output_dir=output_dir))

    logger.info('Plot results...')
    output_plots = os.path.join(output_dir, 'plots')
    os.makedirs(output_plots, exist_ok=True)
    written_files.extend(opti.plot_optimization_results(study, targets, output_path=output_plots))

    logger.info('Produce results for the best parameters')
    _ = merge_dem_over_aoi.main(dem_correspondence_df, aoi_gdf, TILE_DIR, study.best_params['resolution'], save_extra=True, output_dir=os.path.join(output_dir, 'merged_dems'))
    # get_slope.main(merged_tiles, slope_dir)
    # possible_areas = define_possible_areas.main(study.best_params['max_slope'])

    dem_list = glob(os.path.join(output_dir, 'merged_dems', '*.tif'))
    detected_depressions_gdf, depression_files = depression_detection.main(
        dem_list, study.best_params['simplification_param'], study.best_params['mean_filter_size'], study.best_params['fill_depth'], 
        working_dir=WORKING_DIR, output_dir=output_dir, overwrite=True
    )
    written_files.extend(depression_files)
    best_pp_param = {key: value for key, value in study.best_params.items() if key in [
        'max_part_in_lake', 'max_part_in_river', 'min_compactness', 'min_area', 'max_area', 'min_diameter', 'min_depth', 'max_depth'
    ]}
    detected_dolines_gdf, _ = post_processing.main(detected_depressions_gdf, water_bodies_gdf, dissolved_rivers_gdf, output_dir=output_dir, **best_pp_param)
    # del possible_areas, merged_tiles

    detected_dolines_gdf = assess_results.prepare_dolines_to_assessment(detected_dolines_gdf)
    metric, assessment_files = assess_results.main(REF_TYPE, ref_data_gdf, detected_dolines_gdf, aoi_gdf, det_type='watersheds', 
                                                   dem_dir=os.path.join(output_dir, 'merged_dems'), save_extra=True, output_dir=output_dir)
    written_files.extend(assessment_files)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer  
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")