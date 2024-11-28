import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.features import rasterize
from rasterstats import zonal_stats

import whitebox 
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_global_depressions, format_local_depressions, format_logger, get_config, simplify_with_vw
from functions.fct_rasters import polygonize_raster
from global_parameters import ALL_PARAMS_WATERSHEDS, AOI_TYPE

logger = format_logger(logger)

def main(dem_list, simplification_param, non_sedimentary_gdf, builtup_areas_gdf, mean_filter_size=7, fill_depth=0.5, working_dir='.', output_dir='outputs', overwrite=False, save_extra=False):
    """
    Main function to detect depressions in a DEM.

    Parameters
    ----------
    dem_list : list
        List of paths to the DEM files.
    simplification_param : float
        Simplification parameter for the potential dolines with the Visvalingam-Whyatt algorithm.
    mean_filter_size : int, optional
        Size of the mean filter for noise removal. Defaults to 7.
    fill_depth : float, optional
        Maximum depth for the depression filling. Defaults to 0.5.
    working_dir : str, optional
        Working directory. Defaults to '.'.
    output_dir : str, optional
        Output directory for the results. Defaults to 'outputs'.
    overwrite : bool, optional
        Whether to overwrite existing files. Defaults to False.
    save_extra : bool, optional
        Whether to save intermediate results. Defaults to False.

    Returns
    -------
    simplified_pot_dolines_gdf : geopandas.GeoDataFrame
        GeoDataFrame with the potential dolines.
    written_files : list
        List of paths to the written files.
    """
    os.makedirs(output_dir, exist_ok=True)
    dem_processing_dir = os.path.join(working_dir, output_dir, 'dem_processing') # WBT works with absolute paths
    os.makedirs(dem_processing_dir, exist_ok=True)
    pour_points_dir = os.path.join(working_dir, output_dir, 'pour_points')
    os.makedirs(pour_points_dir, exist_ok=True)
    written_files = []

    if (len(dem_list) == 0) or isinstance(dem_list, str):
        logger.critical('No DEM files found.')
        sys.exit(1)

    if not save_extra:
        wbt.set_verbose_mode(False)

    potential_dolines_gdf = gpd.GeoDataFrame()
    for dem_path in dem_list:
        dem_name = os.path.basename(dem_path)
        smoothed_dem_path = os.path.join(dem_processing_dir, 'smoothed_dem_' + dem_name)
        simplified_dem_path = os.path.join(dem_processing_dir, 'simplified_dem_' + dem_name)
        flow_path = os.path.join(dem_processing_dir, 'flow_direction_' + dem_name)
        sink_path = os.path.join(dem_processing_dir, 'sink_' + dem_name)
        watershed_path = os.path.join(dem_processing_dir, 'watersheds_' + dem_name)
        zonal_fill_path = os.path.join(dem_processing_dir, 'zonal_fill_' + dem_name)

        logger.info(f'Perform hydrological processing for area {dem_name.rstrip(".tif")}...')
        if os.path.exists(watershed_path) and not overwrite:
            with rio.open(watershed_path) as src:
                wtshd_band = src.read(1)
                wtshd_meta = src.meta

            with rio.open(simplified_dem_path) as src:
                simplified_dem_arr = src.read(1)

        else:
            # Noise removal
            if mean_filter_size > 1:
                wbt.mean_filter(
                    i=os.path.join(working_dir, dem_path),
                    output=smoothed_dem_path,
                    filterx=mean_filter_size,
                    filtery=mean_filter_size,
                )
            else:
                smoothed_dem_path = os.path.join(working_dir, dem_path)

            wbt.fill_depressions(
                dem=smoothed_dem_path,
                output=simplified_dem_path,
                max_depth=fill_depth,
            )

            # Watershed calculation
            wbt.d8_pointer(
                dem=simplified_dem_path,
                output=flow_path,
            )

            wbt.find_no_flow_cells(
                dem=simplified_dem_path,
                output=sink_path,
            )
            # Transform pour points to polygons
            pour_points_gdf = polygonize_raster(sink_path, dtype=np.int16)
            pour_points_gdf.loc[:, 'geometry'] = pour_points_gdf.geometry.centroid
            pour_points_gdf.loc[:, 'number'] = np.arange(1, len(pour_points_gdf) + 1)

            filepath = os.path.join(pour_points_dir, f'pour_points_{dem_name.rstrip('.tif')}.shp')
            pour_points_gdf.to_file(filepath)
            written_files.append(filepath)

            wbt.watershed(
                d8_pntr=flow_path,
                pour_pts=filepath,
                output=watershed_path,
            )

        logger.info(f'Perform zonal fill for area {dem_name.rstrip('.tif')}...')
        # Implement zonal fill
        logger.info('Step 1: polygonize the watershed...')
        # Step 1: polygonize the watershed
        with rio.open(watershed_path) as src:
            wtshd_band = src.read(1)
            wtshd_meta = src.meta
        watersheds_gdf = polygonize_raster(wtshd_band, meta=wtshd_meta)
        
        logger.info('Step 2: get the minimal altitude on each polygon boundary...')
        # Step 2: get the minimal altitude on each polygon boundary
        min_alti_list = zonal_stats(watersheds_gdf.geometry.boundary, simplified_dem_path, affine=wtshd_meta['transform'], stats='min')

        zonal_fill_gdf =  watersheds_gdf.copy()
        zonal_fill_gdf['min_alti'] = [x['min'] for x in min_alti_list]

        logger.info('Step 3: transform the geodataframe back to raster...')
        # Step 3: transform the geodataframe back to raster
        out_arr = wtshd_band.astype(np.float64)
        shapes_w_new_value = ((geom, value) for geom, value in zip(zonal_fill_gdf.geometry, zonal_fill_gdf.min_alti))
        zonal_fill_arr = rasterize(shapes=shapes_w_new_value, fill=wtshd_meta['nodata'], out=out_arr, transform=wtshd_meta['transform'], dtype=np.float64)
        wtshd_meta.update(dtype=np.float64)

        logger.info('Compare zonal fill with simplified dem...')
        # Compare zonal fill will with simplified dem
        with rio.open(simplified_dem_path) as src:
            simplified_dem_arr = src.read(1)
            simplified_dem_meta = src.meta

        nodata_value_dem = simplified_dem_meta['nodata']
        difference_arr = np.where(
            (simplified_dem_arr==nodata_value_dem) | (zonal_fill_arr==wtshd_meta['nodata']),
            nodata_value_dem,
            zonal_fill_arr - simplified_dem_arr
        )
        potential_dolines_arr = np.where(difference_arr > 0, 1, 0)

        potential_dolines_gdf = format_local_depressions(potential_dolines_arr, dem_name, dem_path, simplified_dem_meta, potential_dolines_gdf, non_sedimentary_gdf, builtup_areas_gdf)

    simplified_pot_dolines_gdf = format_global_depressions(potential_dolines_gdf, simplification_param)

    filepath = os.path.join(output_dir, 'potential_dolines.gpkg')
    simplified_pot_dolines_gdf.to_file(filepath)
    written_files.append(filepath)

    if save_extra:
        with rio.open(zonal_fill_path, 'w+', **wtshd_meta) as out:
            out.write_band(1, zonal_fill_arr)
        
        with rio.open(os.path.join(dem_processing_dir, f'difference_{dem_name}'), 'w+', **simplified_dem_meta) as out:
            out.write_band(1, difference_arr)

    return simplified_pot_dolines_gdf, written_files


if __name__ == '__main__':
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs the doline detection based on Obu J. & Podobnikar T. (2013).")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']

    NON_SEDIMENTARY_AREAS = cfg['non_sedimentary_areas']
    BUILTUP_AREAS = cfg['builtup_areas']

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
        aoi_type_key = AOI_TYPE
    else:
        aoi_type_key = 'All types'
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR

    VW_THRESHOLD = ALL_PARAMS_WATERSHEDS[aoi_type_key]['simplification_param']
    MEAN_FILTER = ALL_PARAMS_WATERSHEDS[aoi_type_key]['mean_filter_size']
    FILL_DEPTH = ALL_PARAMS_WATERSHEDS[aoi_type_key]['fill_depth']

    # ----- Data processing -----

    logger.info('Read data...')
    dem_list = glob(os.path.join(dem_dir, '*.tif'))
    non_sedimentary_areas_gdf = gpd.read_file(NON_SEDIMENTARY_AREAS)
    builtup_areas_gdf = gpd.read_file(BUILTUP_AREAS)

    potential_dolines_gdf, written_files = main(
        dem_list, VW_THRESHOLD, non_sedimentary_areas_gdf, builtup_areas_gdf, MEAN_FILTER, FILL_DEPTH, working_dir=WORKING_DIR, output_dir=output_dir, save_extra=True, overwrite=True
    )

    logger.success('Done! The following files were written:')
    for file in written_files:
        logger.success(file)

    logger.success(f'In addition, the rasters for the different steps were saved in the folder {os.path.join(output_dir, 'dem_processing')}')

    logger.info(f'Done in {time() - tic:0.2f} seconds')
