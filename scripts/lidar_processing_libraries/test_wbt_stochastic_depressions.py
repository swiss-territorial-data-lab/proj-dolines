import os
import sys
from glob import glob
from loguru import logger
from time import time

import geopandas as gpd
import numpy as np
import rasterio as rio
from skimage.morphology import disk, opening, closing


import whitebox
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_local_depressions, format_logger, format_global_depressions, get_config
from global_parameters import AOI_TYPE

logger = format_logger(logger)

def main(dem_list, autocorr_range, iterations, threshold, simplification_param, non_sedimentary_gdf, builtup_areas_gdf, 
         save_extra=False, overwrite=False, working_dir='.', output_dir='outputs'):

    if not save_extra:
        wbt.set_verbose_mode(False)

    rmse = {
        '2019_2523_1199.tif': 0.3,
        '2019_2568_1126.tif': 0.3,
        '2020_2697_1157.tif': 1,
        '2023_2783_1163.tif': 1,
        '2023_2795_1170.tif': 0.3,
        '2023_2822_1184.tif': 1,
        '2019_2709_1204.tif': 2,
        '2019_2573_1224.tif': 0.3,
        '2019_2724_1234.tif': 0.3,
        '2021_2590_1170.tif': 0.3,
        '2020_2622_1267.tif': 0.3
    }

    written_files = []
    potential_dolines_gdf = gpd.GeoDataFrame()
    for dem_path in dem_list:
        dem_name = os.path.basename(dem_path)
        outpath = os.path.join(working_dir, output_dir, dem_name.rstrip('.tif') + '_pdep.tif')
        
        if not os.path.exists(outpath) or overwrite:
            wbt.stochastic_depression_analysis(
                dem = os.path.join(working_dir, dem_path),
                output = outpath,
                rmse = rmse[dem_name],
                range = autocorr_range,
                iterations = iterations
            )
            written_files.append(outpath)

        with rio.open(outpath) as src:
            proba_depression = src.read(1)
            im_meta = src.meta

        logger.info('Remove resiudal pixels...')
        binary_image = np.where(proba_depression > threshold, 1, 0)
        closed_binary_image = closing(binary_image, disk(5))
        opened_binary_image = opening(closed_binary_image, disk(3))

        potential_dolines_gdf = format_local_depressions(opened_binary_image, dem_name, dem_path, im_meta, potential_dolines_gdf, non_sedimentary_gdf, builtup_areas_gdf)

    
    potential_dolines_gdf = format_global_depressions(potential_dolines_gdf, simplification_param)

    if save_extra:
        filepath = os.path.join(working_dir, output_dir, 'potential_dolines.gpkg')
        potential_dolines_gdf.to_file(filepath)
        written_files.append(filepath)

    return potential_dolines_gdf, written_files


if __name__ == "__main__":
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs a stochastic depression detection with Whitebox Tools.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']

    AUTOCORR_RANGE = cfg['autocorr_range']
    ITERATIONS = cfg['iterations']
    THRESHOLD = cfg['threshold']
    SIMPLIFICATION_PARAM = cfg['simplification_param']
    NON_SEDIMENTARY_AREAS = cfg['non_sedimentary_areas']
    BUILTUP_AREAS = cfg['builtup_areas']

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    logger.info('Read data...')
    dem_list = glob(os.path.join(dem_dir, '*.tif'))
    if len(dem_list) == 0:
        logger.critical(f'No DEM found in {dem_dir}')
        sys.exit(1)

    non_sedimentary_gdf = gpd.read_parquet(NON_SEDIMENTARY_AREAS)
    builtup_areas_gdf = gpd.read_file(BUILTUP_AREAS)

    _, written_files = main(
        dem_list, AUTOCORR_RANGE, ITERATIONS, THRESHOLD, SIMPLIFICATION_PARAM, non_sedimentary_gdf, builtup_areas_gdf, save_extra=True, working_dir=WORKING_DIR, output_dir=output_dir
    )

    logger.info('The following files were written:')
    for file in written_files:
        logger.info(file)

    logger.info('Done in {:.2f} seconds'.format(time() - tic))