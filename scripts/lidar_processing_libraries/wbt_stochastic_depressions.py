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
from functions.fct_misc import format_local_depressions, format_logger, get_config
from functions.fct_rasters import polygonize_binary_raster_w_dem_name
from global_parameters import AOI_TYPE

logger = format_logger(logger)

def main(dem_list, autocorr_range, iterations, threshold, non_sedimentary_gdf, builtup_areas_gdf, 
         save_extra=False, overwrite=False, working_dir='.', output_dir='outputs'):

    if not save_extra:
        wbt.set_verbose_mode(False)

    written_files = []
    potential_dolines_gdf = gpd.GeoDataFrame()
    for dem_path in dem_list:
        logger.info('Compute depression probability...')
        dem_name = os.path.basename(dem_path)
        outpath = os.path.join(working_dir, output_dir, dem_name.rstrip('.tif') + '_pdep.tif')
        rmse = 0.3 if dem_name != '2019_2573_1224.tif' else 0.5     # Warning: to adapt for production, only ok with the POC data
        
        if not os.path.exists(outpath) or overwrite:
            wbt.stochastic_depression_analysis(
                dem = os.path.join(working_dir, dem_path),
                output = outpath,
                rmse = rmse,
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

        logger.info('Polygonize depressions...')
        depressions_gdf = polygonize_binary_raster_w_dem_name(opened_binary_image, im_meta, dem_name)
        if depressions_gdf.empty:
            continue

        potential_dolines_gdf = format_local_depressions(
            dem_path, depressions_gdf, non_sedimentary_gdf, builtup_areas_gdf, potential_dolines_gdf, simplification_param=1.5, simplified_dem_meta=im_meta
        )

    
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
        dem_list, AUTOCORR_RANGE, ITERATIONS, THRESHOLD, non_sedimentary_gdf, builtup_areas_gdf, save_extra=True, working_dir=WORKING_DIR, output_dir=output_dir
    )

    logger.info('The following files were written:')
    for file in written_files:
        logger.info(file)

    logger.info('Done in {:.2f} seconds'.format(time() - tic))