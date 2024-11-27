import os
import sys
from glob import glob
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats

import lidar

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, format_global_depressions, get_config
from global_parameters import AOI_TYPE

logger = format_logger(logger)

def main(dem_list, min_size, min_depth, interval, bool_shp, simplification_param, save_extra=False, overwrite=False, output_dir='outputs'):

    written_files = []
    raw_potential_dolines_gdf = gpd.GeoDataFrame()
    for dem in dem_list:
        dem_name = os.path.basename(dem)
        dem_output_dir = os.path.join(output_dir, dem_name.rstrip('.tif'))
        logger.info(f'Processing {dem_name}...')

        if not os.path.exists(os.path.join(output_dir, dem_name.rstrip('.tif'), 'depressions.shp')) or overwrite:
            os.makedirs(dem_output_dir, exist_ok=True)

            out_dem = os.path.join(dem_output_dir, "median_dem.tif")
            smoothed_dem = lidar.MedianFilter(dem, kernel_size=3, out_file=out_dem)
            sink_path = lidar.ExtractSinks(smoothed_dem, min_size, dem_output_dir)
            dep_id_path, dep_level_path = lidar.DelineateDepressions(sink_path,
                                                    min_size,
                                                    min_depth,
                                                    interval,
                                                    dem_output_dir,
                                                    bool_shp)

        local_dolines_gdf = gpd.read_file(os.path.join(dem_output_dir, 'depressions.shp'))
        dolines_info_df = pd.read_csv(os.path.join(dem_output_dir, 'depressions_info.csv'))
        local_dolines_gdf = pd.merge(local_dolines_gdf[['id', 'geometry']], dolines_info_df[['id', 'level', 'region_id']], on='id')
        local_dolines_gdf['corresponding_dem'] = dem_name

        logger.info('Keep depressions of level 1 or 2 depending on the size...')
        # Check id for 1st level that are large enough
        level_one_dolines_gdf = local_dolines_gdf[local_dolines_gdf['level'] == 1].copy()
        large_dolines_gdf = level_one_dolines_gdf[level_one_dolines_gdf.area > 100].copy()
        
        # Check if all 1st level are eliminated for a region id
        large_dolines_count = large_dolines_gdf.groupby('region_id').count().reset_index()
        all_dolines_count = level_one_dolines_gdf.groupby('region_id').count().reset_index()
        comp_dolines_count = pd.merge(
            large_dolines_count[['region_id', 'id']], all_dolines_count['region_id'], 
            how='right', on='region_id', suffixes=('', '_all')
        )
        lost_dolines = comp_dolines_count.loc[comp_dolines_count['id'].isna(), 'region_id'].to_list()

        # Keep 1st and 2nd level for those regions
        level_one_two_dolines_gdf = local_dolines_gdf[(local_dolines_gdf['level'].isin([1, 2])) & local_dolines_gdf['region_id'].isin(lost_dolines)].copy()
        # Merge the two levels
        level_one_two_dolines_gdf = level_one_two_dolines_gdf.dissolve(
            by='region_id', aggfunc={'id': 'first','level': 'max', 'corresponding_dem': 'first'}, as_index=False
        )

        # Split separate dolines inside the same region
        filtered_nested_dolines_gdf = level_one_two_dolines_gdf.explode()
        filtered_nested_dolines_gdf.loc[filtered_nested_dolines_gdf.id.duplicated(keep=False), 'id'] = [
            i + filtered_nested_dolines_gdf.shape[0] for i in range(filtered_nested_dolines_gdf.id.duplicated(keep=False).sum())
        ]

        filtered_local_dolines_gdf = pd.concat([
            local_dolines_gdf[~local_dolines_gdf.region_id.isin(lost_dolines) & (local_dolines_gdf['level']==1)], 
            filtered_nested_dolines_gdf
        ], ignore_index=True)

        if filtered_local_dolines_gdf.empty:
            continue

        logger.info('Get zonal stats of depth and elevation...')
        depression_stats = zonal_stats(filtered_local_dolines_gdf.geometry, dem, stats=['min', 'max', 'std'])
        filtered_local_dolines_gdf['depth'] = [x['max'] - x['min'] for x in depression_stats]
        filtered_local_dolines_gdf['std'] = [x['std'] for x in depression_stats]

        raw_potential_dolines_gdf = pd.concat([raw_potential_dolines_gdf, filtered_local_dolines_gdf[['geometry', 'corresponding_dem', 'depth', 'std']]], ignore_index=True)

    potential_dolines_gdf = format_global_depressions(raw_potential_dolines_gdf, simplification_param)

    if (potential_dolines_gdf.geometry.geom_type == 'MultiPolygon').any():
        potential_dolines_gdf = potential_dolines_gdf.explode(index_parts=False)
        potential_dolines_gdf = potential_dolines_gdf[potential_dolines_gdf.area > 1].copy()

    if save_extra:
        potential_dolines_gdf.to_file(os.path.join(output_dir, 'potential_dolines.gpkg'))

    return written_files


if __name__ == "__main__":
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs a stochastic depression detection with Whitebox Tools.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']

    MIN_SIZE = cfg['min_size']
    MIN_DEPTH = cfg['min_depth']
    INTERVAL = cfg['interval']
    BOOL_SHP = cfg['bool_shp']
    SIMPLIFICATION_PARAM = cfg['simplification_param']

    OVERWRITE = False

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    dem_list = glob(os.path.join(dem_dir, '*.tif'))
    if len(dem_list) == 0:
        logger.critical(f'No DEM found in {dem_dir}')
        sys.exit(1)

    written_files = main(dem_list, MIN_SIZE, MIN_DEPTH, INTERVAL, BOOL_SHP, SIMPLIFICATION_PARAM, save_extra=True, overwrite=OVERWRITE, output_dir=output_dir)

    logger.info('The following files were written:')
    for file in written_files:
        logger.info(file)

    logger.info('Done in {:.2f} seconds'.format(time() - tic))