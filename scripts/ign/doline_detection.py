import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from cv2 import GaussianBlur
from rasterstats import zonal_stats

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config
from functions.fct_rasters import polygonize_binary_raster
from global_parameters import ALL_PARAMS_IGN, AOI_TYPE

logger = format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

def main(dem_dict, fitted_area_dict,
         gaussian_kernel, gaussian_sigma, dem_diff_thrsld, min_area, limit_compactness,
         max_voronoi_area, min_merged_area, min_long_area, max_long_area, min_long_compactness, min_round_area, min_round_compactness,
         thalweg_buffer, thalweg_threshold, max_depth,
         epsg=2056, save_extra=False, output_dir='outputs'):
    """
    Main function to detect dolines.

    Parameters
    ----------
    dem_dict : dict
        Dictionary with keys as the name of the DEM tiles and values as a tuple containing the DEM array and its metadata.
    fitted_area_dict : dict
        Dictionary with keys as the name of the DEM tiles and values as a tuple containing the polygon of the binary raster of possible doline areas and its metadata.
    gaussian_kernel : int
        Kernel size [px] for the Gaussian filter.
    gaussian_sigma : int
        Sigma value [px] for the Gaussian filter.
    dem_diff_thrsld : float
        Threshold value [m] for the difference between the original and smoothed DEM.
    min_area : float
        Minimum area [m2] for the detected dolines.
    limit_compactness : float
        Limit value [-] between long and round dolines.
    max_voronoi_area : float
        Maximum area [m2] for the voronoi polygons.
    min_merged_area : float
        Minimum area [m2] for the merged voronoi polygons.
    min_long_area : float
        Minimum area [m2] for the long depressions.
    max_long_area : float
        Maximum area [m2] for the long depressions.
    min_long_compactness : float
        Minimum compactness [-] for the long depressions.
    min_round_area : float
        Minimum area [m2] for the round depressions outside dense zones.
    min_round_compactness : float
        Minimum compactness [-] for the round depressions outside dense zones.
    thalweg_buffer : float
        Buffer size [m] for the thalwegs.
    thalweg_threshold : float
        Maximum difference [m] between the buffer outline and the depression outline for it to be considered a thalwegs and not a doline.
    max_depth : float
        Maximum depth [m] for the dolines.
    save_extra : bool, optional
        Whether to save the intermediate results. Defaults to False.
    output_dir : str, optional
        Output directory for the results. Defaults to 'outputs'.

    Returns
    -------
    cleaned_sinkholes_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the detected doline data.
    written_files : list
        List of written file paths.
    """

    os.makedirs(output_dir, exist_ok=True)
    # diff_dir = os.path.join(output_dir, 'diff_smoothed_dem')
    # os.makedirs(diff_dir, exist_ok=True)
    written_files = []

    # ----- Generation of the DEM -----

    all_potential_sinkholes_gdf = gpd.GeoDataFrame()
    for dem_name, data in tqdm(dem_dict.items(), desc="Detect dolines"):

        dem_data, dem_meta = data
        dem_data = dem_data[0, :, :] if len(dem_data.shape) > 2 else dem_data

        # Apply gaussian filter
        filtered_dem = GaussianBlur(dem_data, ksize=(gaussian_kernel, gaussian_kernel), sigmaX=gaussian_sigma)

        dem_diff = filtered_dem - dem_data

        potential_sinkholes_arr = np.where((dem_diff>dem_diff_thrsld) & (dem_data!=dem_meta['nodata']), 1, 0)
        potential_sinkholes_arr = potential_sinkholes_arr.astype('int16')

        potential_area, _ = fitted_area_dict[dem_name]

        # Apply mask of flat non-sedimentary areas
        potential_sinkholes_arr = np.where(potential_area==1, potential_sinkholes_arr, 0)

        # # Output difference between original and smoothed dem on AOI for visualization
        # filtered_dem_diff = np.where(potential_area==1, dem_diff, 0)
        # with rio.open(os.path.join(diff_dir, 'diff_' + dem_name), 'w', **dem_meta) as dst:
        #     dst.write(filtered_dem_diff[np.newaxis, ...])
        
        potential_sinkholes_gdf = polygonize_binary_raster(potential_sinkholes_arr, crs=dem_meta['crs'], transform=dem_meta['transform'])
        potential_sinkholes_gdf['corresponding_dem'] = dem_name

        all_potential_sinkholes_gdf = pd.concat([all_potential_sinkholes_gdf, potential_sinkholes_gdf]) if not potential_sinkholes_gdf.empty else all_potential_sinkholes_gdf

    if all_potential_sinkholes_gdf.empty:
        logger.info('No sinkhole detected')
        return pd.DataFrame(), []

    logger.info('Filter depressions based on area and compactness...')

    # Filter the depressions based on area
    # Inital code only keep depressions over 300 m2. Based on the GT, we keep those above 35 m2.
    filtered_sinkholes_gdf = all_potential_sinkholes_gdf[all_potential_sinkholes_gdf.area > min_area].copy()

    # Determine compactness
    filtered_sinkholes_gdf['compactness'] = 4 * np.pi * filtered_sinkholes_gdf.area / filtered_sinkholes_gdf.length**2
    filtered_sinkholes_gdf['type'] = ['round' if compactness > limit_compactness else 'long' for compactness in filtered_sinkholes_gdf.compactness]

    logger.info('Estimate depression density with voronoi polygons...')

    # Make the Voronoi diagram of the round depressions
    center_geoms = filtered_sinkholes_gdf[filtered_sinkholes_gdf['type']=='round'].geometry.centroid
    voronoi_result_polys = center_geoms.voronoi_polygons()

    # Find area with a high density of round depressions based on the area of vornoi polygons
    dissolved_vornoi_polys = voronoi_result_polys[voronoi_result_polys.area < max_voronoi_area].union_all('coverage')
    if dissolved_vornoi_polys.is_empty:
        sinkholes_in_dense_area_gdf = gpd.GeoDataFrame(columns=filtered_sinkholes_gdf.columns)
    else:
        dissolved_vornoi_polys_gs = gpd.GeoSeries(
            [geom for geom in dissolved_vornoi_polys.geoms] if dissolved_vornoi_polys.geom_type == 'MultiPolygon' else dissolved_vornoi_polys, 
            crs=('epsg:'+str(epsg))
        )
        assert dissolved_vornoi_polys_gs.is_valid.all(), 'Dissolved voronoi polygons are not all valid.'

        # Only keep large dense area and a buffer around
        large_dense_areas = dissolved_vornoi_polys_gs[dissolved_vornoi_polys_gs.area > min_merged_area].copy()
        large_dense_areas = large_dense_areas.geometry.buffer(100)
        large_dense_areas_union = large_dense_areas.union_all()

        logger.info('Filter depressions depending on the local density...')
        # Perform the time-consuming spatial operation separatly to avoid checking separatly for different types.
        sinkholes_in_dense_area_gdf = filtered_sinkholes_gdf[filtered_sinkholes_gdf.geometry.within(large_dense_areas_union)].copy()

    # Only keep long depression that are within the large dense area
    long_sinkholes_gdf = sinkholes_in_dense_area_gdf[
        (sinkholes_in_dense_area_gdf['type']=='long')
        & (sinkholes_in_dense_area_gdf.area > min_long_area)
        & (sinkholes_in_dense_area_gdf.area < max_long_area)
        & (sinkholes_in_dense_area_gdf.compactness > min_long_compactness)
        # & filtered_sinkholes_gdf.geometry.within(large_dense_areas_union)
    ].copy()
    
    round_sinkholes_gdf = filtered_sinkholes_gdf[
        (filtered_sinkholes_gdf['type']=='round')
        & (
            (
                filtered_sinkholes_gdf.index.isin(sinkholes_in_dense_area_gdf.index.tolist())
            )
            |(
                ~filtered_sinkholes_gdf.index.isin(sinkholes_in_dense_area_gdf.index.tolist())
                & (filtered_sinkholes_gdf.area > min_round_area)
                & (filtered_sinkholes_gdf.compactness > min_round_compactness)
            )
        )
    ].copy()
    round_sinkholes_gdf.loc[:, 'type'] = [
        'round in dense area' if identifier in sinkholes_in_dense_area_gdf.index.tolist() else 'very round outside dense area' 
        for identifier in round_sinkholes_gdf.index
    ]

    sinkholes_gdf = pd.concat([round_sinkholes_gdf, long_sinkholes_gdf], ignore_index=True)
    sinkholes_gdf['doline_id'] = sinkholes_gdf.index

    if all_potential_sinkholes_gdf.empty:
        logger.info('No sinkhole detected')
        return pd.DataFrame(), []

    logger.info('Deal with thalwegs producing false positives and too deep dolines...')

    # Get an outline on the surrounding terrain
    sinkholes_gdf['buffered_outline'] = sinkholes_gdf.geometry.buffer(thalweg_buffer).boundary

    sinkholes_gdf['alti_diff'] = np.nan
    for dem_tile in tqdm(sinkholes_gdf.corresponding_dem.unique(), desc="Get lowest point of sinkhole and its buffer outline"):
        sinkholes_on_tiles_gdf = sinkholes_gdf[sinkholes_gdf.corresponding_dem == dem_tile].copy()
        if sinkholes_on_tiles_gdf.empty:
            continue

        dem_data, dem_meta = dem_dict[dem_tile]

        # Get lowest point of sinkhole and its buffer outline
        sinkholes_lowest_alti = zonal_stats(sinkholes_on_tiles_gdf.geometry, dem_data, affine=dem_meta['transform'], nodata=dem_meta['nodata'], stats=['min', 'max'])
        buffer_lowest_alti = zonal_stats(sinkholes_on_tiles_gdf.buffered_outline, dem_data, affine=dem_meta['transform'], nodata=dem_meta['nodata'], stats='min')

        # Keep results in a dataframe
        alti_dict={
            'doline_id': sinkholes_on_tiles_gdf.doline_id,
            'min_sinkhole': [x['min'] for x in sinkholes_lowest_alti],
            'max_sinkhole': [x['max'] for x in sinkholes_lowest_alti],
            'min_buffer_line': [x['min'] for x in buffer_lowest_alti],
            'geometry': sinkholes_on_tiles_gdf.geometry
        }
        alti_gdf = gpd.GeoDataFrame(alti_dict, crs=epsg)

        # Detect and mark thalwegs
        alti_gdf['alti_diff'] = alti_gdf['min_buffer_line'] - alti_gdf['min_sinkhole']
        thalweg_ids = alti_gdf.loc[alti_gdf['alti_diff'] < thalweg_threshold, 'doline_id']

        sinkholes_gdf.loc[sinkholes_gdf.doline_id.isin(thalweg_ids), 'type'] = 'thalweg'
        sinkholes_gdf.loc[sinkholes_gdf.doline_id.isin(alti_gdf.doline_id), 'alti_diff'] = alti_gdf.alti_diff

        # Remove sinkholes that are too deep or in steep areas
        alti_gdf['depth'] = alti_gdf['max_sinkhole'] - alti_gdf['min_sinkhole']
        too_deep_steep_ids = alti_gdf.loc[alti_gdf['depth'] > max_depth, 'doline_id']
        sinkholes_gdf = sinkholes_gdf[~sinkholes_gdf.doline_id.isin(too_deep_steep_ids)].copy()

    cleaned_sinkholes_gdf = sinkholes_gdf[['doline_id', 'type', 'compactness', 'alti_diff', 'corresponding_dem', 'geometry']]

    filepath = os.path.join(output_dir, 'sinkholes.gpkg')
    cleaned_sinkholes_gdf.to_file(filepath)
    written_files.append(filepath)

    if save_extra:
        filepath = os.path.join(output_dir, 'all_potential_sinkholes.gpkg')
        all_potential_sinkholes_gdf.to_file(filepath)
        written_files.append(filepath)

        filepath = os.path.join(output_dir, 'potential_sinkholes.gpkg')
        filtered_sinkholes_gdf.to_file(filepath)
        written_files.append(filepath)

        try:
            filepath = os.path.join(output_dir, 'large_dense_area.gpkg')
            large_dense_areas.to_file(filepath)
            written_files.append(filepath)
        except UnboundLocalError:
            pass

    return cleaned_sinkholes_gdf, written_files


if __name__ == '__main__':
    # ----- Get parameters -----

    cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs the doline detection based on IGN's method.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']
    EPSG = cfg['epsg']

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
        aoi_type_key = AOI_TYPE
    else:
        aoi_type_key = 'All types'
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE) if AOI_TYPE else OUTPUT_DIR

    param_dict = {k: v for k, v in ALL_PARAMS_IGN[aoi_type_key].items() if k not in ['resolution', 'max_slope']}

    logger.info('Read data...')

    dem_list = glob(os.path.join(dem_dir, '*.tif'))
    dem_dict = {}
    potential_area_dict = {}
    for dem_path in dem_list:
        with rio.open(dem_path) as src:
            dem_data = src.read(1)
            dem_meta = src.meta

        dem_dict[os.path.basename(dem_path)] = (dem_data, dem_meta)

        potential_areas_path = os.path.join(DEM_DIR, 'possible_areas', AOI_TYPE, 'possible_area_' + os.path.basename(dem_path)) if AOI_TYPE \
            else os.path.join(DEM_DIR, 'possible_areas', 'possible_area_' + os.path.basename(dem_path))
        with rio.open(potential_areas_path) as src:
            potential_area = src.read(1)
            potential_area_meta = src.meta

        potential_area_dict[os.path.basename(dem_path)] = (potential_area, potential_area_meta)

    if len(dem_list) == 0:
        logger.critical('No DEM files found.')
        sys.exit(1)

    dolines_gdf, written_files = main(dem_dict, potential_area_dict, epsg=EPSG, save_extra=True, output_dir=output_dir, **param_dict)

    logger.success('Done! The following files were written:')
    for file in written_files:
        logger.success(file)

    logger.info(f'Done in {time() - tic:0.4f} seconds')