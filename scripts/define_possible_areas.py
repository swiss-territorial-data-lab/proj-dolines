import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.mask import mask
from shapely.geometry import box
from skimage.morphology import disk, opening

from functions.fct_misc import format_logger, get_config
from global_parameters import ALL_PARAMS_IGN, AOI_TYPE

logger = format_logger(logger)

def main(slope_dir, non_sedi_areas_gdf, max_slope=1.1, save_extra=False, output_dir='outputs'):
    """
    Main function to produce a binary raster of possible doline areas.

    Parameters
    ----------
    slope_dir : str
        Directory containing the slope tiles.
    non_sedi_areas_gdf : geopandas.GeoDataFrame
        GeoDataFrame with the non-sedimentary areas.
    max_slope : float, optional
        Maximum slope value to consider an area as a possible doline. Defaults to 1.1.
    save_extra : bool, optional
        Whether to save the results. Defaults to False.
    output_dir : str, optional
        Output directory for the results. Defaults to 'outputs'.

    Returns
    -------
    possible_area_dict : dict
        Dictionary with keys as the name of the DEM tiles and values as a tuple containing the binary raster of the possible doline areas and its metadata.
    """

    if save_extra:
        os.makedirs(output_dir, exist_ok=True)

    slope_tiles_list = glob(os.path.join(slope_dir, '*.tif'))
    if len(slope_tiles_list) == 0:
        logger.critical('No tile in the slope directory')
        sys.exit(1)

    possible_area_dict = {}
    for tile_path in tqdm(slope_tiles_list, desc='Produce binary raster of possible doline areas'):
        tile_name = os.path.basename(tile_path)
        new_tile_name = tile_name.replace('slope', 'possible_area')

        # Mask non-sedimentary areas
        with rio.open(tile_path) as src:
            meta = src.meta
            if not non_sedi_areas_gdf.geometry.intersects(box(*src.bounds)).any():
                sedimentary_slopes = src.read()
            else:
                sedimentary_slopes, _ = mask(src, non_sedi_areas_gdf.geometry, invert=True)
        # Remove areas with a high slope
        doline_areas = np.where((sedimentary_slopes!=meta['nodata']) & (sedimentary_slopes<max_slope), 1, 0)

        # Perform opening, opposit to the original paper with half disk size
        opened_doline_areas = opening(doline_areas[0, :, :], disk(6))

        meta.update({'dtype': 'int16'})
        if save_extra:
            with rio.open(os.path.join(output_dir, new_tile_name), 'w', **meta) as dst:
                dst.write(opened_doline_areas[np.newaxis, ...])
            
            non_sedi_areas_gdf.to_file(os.path.join(output_dir, 'non_sedi_areas.gpkg'))

        possible_area_dict[tile_name.lstrip('slope_')] = (opened_doline_areas[np.newaxis, ...], meta)

    return possible_area_dict

if __name__ == '__main__':
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----
    cfg = get_config(config_key=os.path.basename(__file__), desc="This script produces a binary raster of possible doline areas.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    SLOPE_DIR = cfg['slope_dir']

    NON_SEDIMENTARY_AREAS = cfg['non_sedimentary_areas']
    AOI = cfg['aoi']
    
    EPSG = 2056

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
        aoi_type_key = AOI_TYPE
    else:
        aoi_type_key = 'All types'
    slope_dir = os.path.join(SLOPE_DIR, AOI_TYPE) if AOI_TYPE else SLOPE_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR

    MAX_SLOPE = ALL_PARAMS_IGN[aoi_type_key]['max_slope']

    # ----- Data processing -----

    logger.info('Read data...')

    non_sedi_areas_gdf = gpd.read_parquet(NON_SEDIMENTARY_AREAS)
    aoi_gdf = gpd.read_file(AOI)
    aoi_gdf.to_crs(EPSG, inplace=True)

    logger.info('Limit non-sedimentary info to AOI...')
    aoi_gdf.loc[:, 'geometry'] = aoi_gdf.geometry.buffer(1000)
    non_sedi_areas_gdf = gpd.overlay(non_sedi_areas_gdf, aoi_gdf, keep_geom_type=True)

    _ = main(slope_dir, non_sedi_areas_gdf, max_slope=MAX_SLOPE, save_extra=True, output_dir=output_dir)

    logger.success(f'Done! The files were written in {output_dir}.')

    logger.info(f'Done in {time() - tic:0.2f} seconds')

