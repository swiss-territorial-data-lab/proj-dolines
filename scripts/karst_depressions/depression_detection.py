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
from shapely.geometry import mapping
from rasterio.features import rasterize
from rasterstats import zonal_stats

import visvalingamwyatt as vw
import whitebox 
wbt = whitebox.WhiteboxTools()
from rdp import rdp

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config
from functions.fct_rasters import polygonize_binary_raster, polygonize_raster

logger = format_logger(logger)

def main(dem_list, simplification_param, mean_filter_size=5, fill_depth=0.5, working_dir='.', output_dir='outputs', overwrite=False, save_extra=False):
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
        basin_path = os.path.join(dem_processing_dir, 'basins_' + dem_name)
        zonal_fill_path = os.path.join(dem_processing_dir, 'zonal_fill_' + dem_name)

        if os.path.exists(watershed_path) and not overwrite:
            with rio.open(watershed_path) as src:
                wtshd_band = src.read(1)
                wtshd_meta = src.meta

            with rio.open(simplified_dem_path) as src:
                simplified_dem_arr = src.read(1)

        else:
            # Noise removal
            wbt.mean_filter(
                i=os.path.join(working_dir, dem_path),
                output=smoothed_dem_path,
                filterx=mean_filter_size,
                filtery=mean_filter_size,
            )

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
        # Step 1: polygonize the watershed boundaries
        with rio.open(watershed_path) as src:
            wtshd_band = src.read(1)
            wtshd_meta = src.meta
        watersheds_gdf = polygonize_raster(wtshd_band, meta=wtshd_meta)
        
        # Step 2: get the minimal altitude on each polygon
        min_alti_list = zonal_stats(watersheds_gdf.geometry.boundary, simplified_dem_path, affine=wtshd_meta['transform'], stats='min')

        zonal_fill_gdf =  watersheds_gdf.copy()
        zonal_fill_gdf['min_alti'] = [x['min'] for x in min_alti_list]

        # Step 3: transform the geodataframe back to raster
        out_arr = wtshd_band.astype(np.float64)
        shapes_w_new_value = ((geom, value) for geom, value in zip(zonal_fill_gdf.geometry, zonal_fill_gdf.min_alti))
        zonal_fill_arr = rasterize(shapes=shapes_w_new_value, fill=wtshd_meta['nodata'], out=out_arr, transform=wtshd_meta['transform'], dtype=np.float64)
        wtshd_meta.update(dtype=np.float64)

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

        local_depression_gdf = polygonize_binary_raster(potential_dolines_arr.astype(np.int16), crs=simplified_dem_meta['crs'], transform=simplified_dem_meta['transform'])
        local_depression_gdf['corresponding_dem'] = dem_name

        if local_depression_gdf.empty:
            continue

        # Get depth
        depression_stats = zonal_stats(local_depression_gdf.geometry, dem_path, affine=simplified_dem_meta['transform'], stats=['min', 'max'])
        local_depression_gdf['depth'] = [x['max'] - x['min'] for x in depression_stats]

        potential_dolines_gdf = pd.concat([potential_dolines_gdf, local_depression_gdf[['geometry', 'corresponding_dem', 'depth']]], ignore_index=True)

    mapped_objects = mapping(potential_dolines_gdf)
    for feature in tqdm(mapped_objects['features'], "Simplifying features"):
        coords = feature['geometry']['coordinates'][0]
        coords_after_rdp = [rdp(x, epsilon=simplification_param) for x in coords]
        # coords_post_vw = vw.Simplifier(coords).simplify(threshold=simplification_param)
        feature['geometry']['coordinates'] = (tuple([tuple(arr) for arr in coords_after_rdp]),)

    simplified_pot_dolines_gdf = gpd.GeoDataFrame.from_features(mapped_objects, crs='EPSG:2056')
    # assert (potential_dolines_gdf.geometry == simplified_pot_dolines_gdf.geometry).all(), 'no simplification happened'

    simplified_pot_dolines_gdf['diameter'] = simplified_pot_dolines_gdf.minimum_bounding_radius()*2
    # compute Schwartzberg compactness, the ratio of the perimeter to the circumference of the circle whose area is equal to the polygon area
    simplified_pot_dolines_gdf['compactness'] = 2*np.pi*np.sqrt(simplified_pot_dolines_gdf.area/np.pi)/simplified_pot_dolines_gdf.length

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

    RDP_EPS = cfg['rdp_eps']
    overwrite = False

    os.chdir(WORKING_DIR)

    # ----- Data processing -----

    logger.info('Read data...')

    dem_list = glob(os.path.join(DEM_DIR, '*.tif'))
    potential_dolines_gdf, written_files = main(dem_list, RDP_EPS, working_dir=WORKING_DIR, output_dir=OUTPUT_DIR, save_extra=True)

    logger.success('Done! The following files were written:')
    for file in written_files:
        logger.success(file)

    logger.success(f'In addition, the rasters for the different steps were saved in the folder {os.path.join(OUTPUT_DIR, 'dem_processing')}')

    logger.info(f'Done in {time() - tic:0.2f} seconds')
