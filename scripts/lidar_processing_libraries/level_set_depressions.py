import os
import sys
os.environ['GDAL_DATA'] = os.path.join(f'{os.sep}'.join(sys.executable.split(os.sep)[:-1]), 'Library', 'share', 'gdal')     # Avoid a warning

from glob import glob
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

import lidar

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_local_depressions, format_logger, get_config
from global_parameters import ALL_PARAMS_LEVEL_SET, AOI_TYPE

logger = format_logger(logger)


def main(dem_list, min_size, min_depth_dep, interval, bool_shp, area_limit, non_sedimentary_gdf, builtup_areas_gdf, save_extra=False, overwrite=False, output_dir='outputs'):

    written_file = []
    potential_dolines_gdf = gpd.GeoDataFrame()
    for dem in dem_list:
        dem_name = os.path.basename(dem)
        dem_output_dir = os.path.join(output_dir, dem_name.rstrip('.tif'))
        logger.info(f'Processing {dem_name}...')

        if not os.path.exists(os.path.join(output_dir, dem_name.rstrip('.tif'), 'depressions.shp')) or overwrite:
            os.makedirs(dem_output_dir, exist_ok=True)

            out_dem = os.path.join(dem_output_dir, "median_dem.tif")
            smoothed_dem = lidar.MedianFilter(dem, kernel_size=3, out_file=out_dem)
            sink_path = lidar.ExtractSinks(smoothed_dem, min_size, dem_output_dir)
            _, _ = lidar.DelineateDepressions(sink_path,
                                                    min_size,
                                                    min_depth_dep,
                                                    interval,
                                                    dem_output_dir,
                                                    bool_shp)

        local_dolines_gdf = gpd.read_file(os.path.join(dem_output_dir, 'depressions.shp'))
        # Dissolve dolines made of multiple detections because of pixels connected through their corner only
        local_dolines_gdf = local_dolines_gdf.dissolve('id', as_index=False)

        dolines_info_df = pd.read_csv(os.path.join(dem_output_dir, 'depressions_info.csv'))
        local_dolines_gdf = pd.merge(local_dolines_gdf[['id', 'geometry']], dolines_info_df[['id', 'level', 'region_id', 'children_id']], on='id')
        local_dolines_gdf['corresponding_dem'] = dem_name

        print('\n')
        logger.info('Keep depressions of level 1 or 2 depending on the size...')
        small_level_one_gdf = local_dolines_gdf[(local_dolines_gdf.level==1) & (local_dolines_gdf.area < area_limit)].copy()
        corr_level_two_gdf = gpd.sjoin(
            local_dolines_gdf.loc[local_dolines_gdf.level==2, ['id', 'geometry', 'children_id']],
            small_level_one_gdf[['id', 'geometry']],
            lsuffix='l2', rsuffix='l1'
        )

        # Transform children list from string to list
        children_dict = {
            id_l2: [
                float(number.lstrip(' ')) for number in corr_level_two_gdf.loc[corr_level_two_gdf.id_l2==id_l2, 'children_id'].iloc[0].lstrip("'[").rstrip("]'").split(':')
            ] 
            for id_l2 in corr_level_two_gdf.id_l2.unique().tolist()
        }
        # Filter ids of level 2 based on children
        id_l2_to_keep = [
            id_l2 for id_l2 in corr_level_two_gdf.id_l2.unique().tolist() 
            if corr_level_two_gdf.loc[corr_level_two_gdf.id_l2==id_l2, 'id_l1'].sort_values().tolist() == children_dict[id_l2] # add id only if all children present among small dets
        ]
        id_l1_to_keep = corr_level_two_gdf.loc[corr_level_two_gdf.id_l2.isin(id_l2_to_keep), 'id_l1'].unique().tolist()
        ids_to_merge = id_l1_to_keep + id_l2_to_keep

        level_one_two_dolines_gdf = local_dolines_gdf[local_dolines_gdf['id'].isin(ids_to_merge)].copy()

        # Merge the two levels
        level_one_two_dolines_gdf = level_one_two_dolines_gdf.dissolve(
            by='region_id', aggfunc={'id': 'first', 'level': 'max', 'corresponding_dem': 'first'}, as_index=False
        )

        # Split separate dolines inside the same region
        filtered_nested_dolines_gdf = level_one_two_dolines_gdf.explode(ignore_index=True)
        filtered_nested_dolines_gdf.loc[filtered_nested_dolines_gdf.id.duplicated(keep=False), 'id'] = [
            i + local_dolines_gdf.shape[0] for i in range(filtered_nested_dolines_gdf.id.duplicated(keep=False).sum())
        ]

        filtered_local_dolines_gdf = pd.concat([
            local_dolines_gdf[~local_dolines_gdf.id.isin(ids_to_merge) & (local_dolines_gdf['level']==1)], 
            filtered_nested_dolines_gdf
        ], ignore_index=True)

        if filtered_local_dolines_gdf.empty:
            continue

        potential_dolines_gdf = format_local_depressions(dem, filtered_local_dolines_gdf, non_sedimentary_gdf, builtup_areas_gdf, potential_dolines_gdf, simplification_param=1.5)

    if (potential_dolines_gdf.geometry.geom_type == 'MultiPolygon').any():
        potential_dolines_gdf = potential_dolines_gdf.explode(index_parts=False)
        potential_dolines_gdf = potential_dolines_gdf[potential_dolines_gdf.area > 1].copy()

    if save_extra:
        written_file = os.path.join(output_dir, 'potential_dolines.gpkg')
        potential_dolines_gdf.to_file(written_file)

    return potential_dolines_gdf, [written_file]


if __name__ == "__main__":
    # Start chronometer
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs a stochastic depression detection with Whitebox Tools.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']

    BOOL_SHP = False
    
    NON_SEDIMENTARY_AREAS = cfg['non_sedimentary_areas']
    BUILTUP_AREAS = cfg['builtup_areas']

    OVERWRITE = True

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
        aoi_type_key = AOI_TYPE
    else:
        aoi_type_key='All types'
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    param_dict = {param_name: ALL_PARAMS_LEVEL_SET[aoi_type_key][param_name] for param_name in ['min_size', 'min_depth_dep', 'interval', 'area_limit']}

    logger.info('Read data...')
    dem_list = glob(os.path.join(dem_dir, '*.tif'))
    if len(dem_list) == 0:
        logger.critical(f'No DEM found in {dem_dir}')
        sys.exit(1)
        
    non_sedimentary_gdf = gpd.read_parquet(NON_SEDIMENTARY_AREAS)
    builtup_areas_gdf = gpd.read_file(BUILTUP_AREAS)

    _, written_file = main(
        dem_list, bool_shp=BOOL_SHP, non_sedimentary_gdf=non_sedimentary_gdf, builtup_areas_gdf=builtup_areas_gdf, save_extra=True, overwrite=OVERWRITE, output_dir=output_dir,
        **param_dict
    )

    logger.info(f'The following file was written: {written_file[0]}')
    logger.info('Done in {:.2f} seconds'.format(time() - tic))