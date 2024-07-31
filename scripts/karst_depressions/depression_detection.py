import os
import sys
from glob import glob
from loguru import logger
from time import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import shape
from rasterio.features import shapes, rasterize
from rasterstats import zonal_stats

import whitebox 
wbt = whitebox.WhiteboxTools()

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config, polygonize_binary_raster

logger = format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

# ----- Get parameters -----

cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs the doline detection based on Obu J. & Podobnikar T. (2013).")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
DEM_DIR = cfg['dem_dir']

OVERWRITE = False
NODATA_VALUE = -32768
NODATA_VALUE_DEM = -9999

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
dem_processing_dir = os.path.join(WORKING_DIR, OUTPUT_DIR, 'dem_processing') # WBT works with absolute paths
os.makedirs(dem_processing_dir, exist_ok=True)
written_files = []

# ----- Data processing -----

logger.info('Read data...')

dem_list = glob(os.path.join(DEM_DIR, '*.tif'))
dolines_gdf = gpd.GeoDataFrame()
if len(dem_list) == 0:
    logger.critical('No DEM files found.')
    sys.exit(1)

for dem_path in dem_list:
    dem_name = os.path.basename(dem_path)
    smoothed_dem_path = os.path.join(dem_processing_dir, 'smoothed_dem_' + dem_name)
    simplified_dem_path = os.path.join(dem_processing_dir, 'simplified_dem_' + dem_name)
    flow_path = os.path.join(dem_processing_dir, 'flow_direction_' + dem_name)
    sink_path = os.path.join(dem_processing_dir, 'sink_' + dem_name)
    watershed_path = os.path.join(dem_processing_dir, 'watersheds_' + dem_name)
    basin_path = os.path.join(dem_processing_dir, 'basins_' + dem_name)
    zonal_fill_path = os.path.join(dem_processing_dir, 'zonal_fill_' + dem_name)

    if os.path.exists(watershed_path) and not OVERWRITE:
        with rio.open(watershed_path) as src:
            wtshd_band = src.read(1)
            wtshd_meta = src.meta

        with rio.open(simplified_dem_path) as src:
            simplified_dem_arr = src.read(1)

    else:
        # Noise removal
        wbt.mean_filter(
            i=os.path.join(WORKING_DIR, dem_path),
            output=smoothed_dem_path,
            filterx=5,
            filtery=5,
        )

        wbt.fill_depressions(
            dem=smoothed_dem_path,
            output=simplified_dem_path,
            max_depth=0.5,
        )

        # Watershed calculation
        wbt.d8_pointer(
            dem=simplified_dem_path,
            output=flow_path,
        )

        wbt.sink(
            i=simplified_dem_path,
            output=sink_path,
        )

        wbt.watershed(
            d8_pntr=flow_path,
            pour_pts=sink_path,
            output=watershed_path,
        )

        # Test the function "basins" from WhiteboxTools
        wbt.basins(
            d8_pntr=flow_path,
            output=basin_path
        )

    logger.info(f'Perform zonal fill for area {dem_name.rstrip('.tif')}...')
    # Implement zonal fill
    # Step 1: polygonize the watershed boundaries
    with rio.open(watershed_path) as src:
        wtshd_band = src.read(1)
        wtshd_meta = src.meta

    unique_values = np.unique(wtshd_band)
    wtshd_shapes = list(shapes(wtshd_band, transform=wtshd_meta['transform']))
    polygons = [(shape(geom), value) for geom, value in wtshd_shapes if value != NODATA_VALUE]

    watersheds_gdf=gpd.GeoDataFrame(polygons, columns=['geometry', 'number'])
    watersheds_gdf.set_crs(crs=wtshd_meta['crs'], inplace=True)
    
    # Step 2: get the minimal altitude on each polygon
    min_alti_list = zonal_stats(watersheds_gdf.geometry.boundary, simplified_dem_path, affine=wtshd_meta['transform'], stats='min')

    zonal_fill_gdf =  watersheds_gdf.copy()
    zonal_fill_gdf['min_alti'] = [x['min'] for x in min_alti_list]

    # Step 3: transform the geodataframe back to raster
    out_arr = wtshd_band.astype(np.float64)
    shapes_w_new_value = ((geom, value) for geom, value in zip(zonal_fill_gdf.geometry, zonal_fill_gdf.min_alti))
    zonal_fill_arr = rasterize(shapes=shapes_w_new_value, fill=NODATA_VALUE, out=out_arr, transform=wtshd_meta['transform'], dtype=np.float64)
    wtshd_meta.update(dtype=np.float64)
    with rio.open(zonal_fill_path, 'w+', **wtshd_meta) as out:
        out.write_band(1, zonal_fill_arr)

    # Compare zonal fill will with simplified dem
    with rio.open(simplified_dem_path) as src:
        simplified_dem_arr = src.read(1)
        simplified_dem_meta = src.meta

    difference_arr = np.where(
        (simplified_dem_arr==NODATA_VALUE_DEM) | (zonal_fill_arr==NODATA_VALUE),
        NODATA_VALUE_DEM,
        zonal_fill_arr - simplified_dem_arr
    )
    with rio.open(os.path.join(dem_processing_dir, f'difference_{dem_name}'), 'w+', **simplified_dem_meta) as out:
        out.write_band(1, difference_arr)
    potential_dolines_arr = np.where(difference_arr > 0, 1, 0)

    local_dolines_gdf = polygonize_binary_raster(potential_dolines_arr.astype(np.int16), crs=simplified_dem_meta['crs'], transform=simplified_dem_meta['transform'])
    local_dolines_gdf['corresponding_dem'] = dem_name
    dolines_gdf = pd.concat([dolines_gdf, local_dolines_gdf[['geometry', 'corresponding_dem']]], ignore_index=True)

filepath = os.path.join(OUTPUT_DIR, 'dolines.gpkg')
dolines_gdf.to_file(filepath)
written_files.append(filepath)

logger.success('Done! The following files were written:')
for file in written_files:
    logger.success(file)

logger.success(f'In addition, the rasters for the different steps were saved in the folder {dem_processing_dir}')

logger.info(f'Done in {time() - tic:0.2f} seconds')

