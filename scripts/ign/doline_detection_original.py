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

logger = format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

# ----- Get parameters -----

cfg = get_config(config_key=os.path.basename(__file__), desc="This script performs the doline detection based on IGN's method.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']
DEM_DIR = cfg['dem_dir']
MASK_DIR = cfg['mask_dir']

EPSG = 2056

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
diff_dir = os.path.join(DEM_DIR, 'diff_smoothed_dem')
os.makedirs(diff_dir, exist_ok=True)
written_files = []

logger.info('Read data...')

dem_list = glob(os.path.join(DEM_DIR, '*.tif'))

if len(dem_list) == 0:
    logger.critical('No DEM files found.')
    sys.exit(1)

# ----- Generation of the DEM -----

all_potential_sinkhols_gdf = gpd.GeoDataFrame()
for dem_path in tqdm(dem_list, desc="Detect dolines"):
    dem_name = os.path.basename(dem_path)

    with rio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta

    # Apply gaussian filter
    filtered_dem = GaussianBlur(dem_data, ksize=(33, 33), sigmaX=5)

    dem_diff = filtered_dem - dem_data

    potential_sinkholes_arr = np.where((dem_diff>1.5) & (dem_data!=dem_meta['nodata']), 1, 0)
    potential_sinkholes_arr = potential_sinkholes_arr.astype('int16')

    potential_areas_path = os.path.join(MASK_DIR, 'ouv_ferm_disk_12_' + dem_name)
    with rio.open(potential_areas_path) as src:
        potential_area = src.read(1)

    # Apply mask of flat non-sedimentary areas
    potential_sinkholes_arr = np.where(potential_area==0, potential_sinkholes_arr, 0)

    # # Output difference between original and smoothed dem on AOI for visualization
    # filtered_dem_diff = np.where(potential_area==1, dem_diff, 0)
    # with rio.open(os.path.join(diff_dir, 'diff_' + dem_name), 'w', **dem_meta) as dst:
    #     dst.write(filtered_dem_diff[np.newaxis, ...])
    
    potential_sinkholes_gdf = polygonize_binary_raster(potential_sinkholes_arr, crs=dem_meta['crs'], transform=dem_meta['transform'])
    potential_sinkholes_gdf['corresponding_dem'] = dem_name

    all_potential_sinkhols_gdf = pd.concat([all_potential_sinkhols_gdf, potential_sinkholes_gdf]) \
                                        if not potential_sinkholes_gdf.empty else all_potential_sinkhols_gdf


logger.info('Filter depressions based on area, compactness and density...')

# Filter the depressions based on area
filtered_sinkholes_gdf = all_potential_sinkhols_gdf[all_potential_sinkhols_gdf.area > 300].copy()

# Determine compactness
filtered_sinkholes_gdf['compactness'] = 4 * np.pi * filtered_sinkholes_gdf.area / filtered_sinkholes_gdf.length**2
filtered_sinkholes_gdf['type'] = ['round' if compactness > 0.33 else 'long' for compactness in filtered_sinkholes_gdf.compactness]

filepath = os.path.join(OUTPUT_DIR, 'all_potential_sinkholes.gpkg')
all_potential_sinkhols_gdf.to_file(filepath)
written_files.append(filepath)

filepath = os.path.join(OUTPUT_DIR, 'potential_sinkholes.gpkg')
filtered_sinkholes_gdf.to_file(filepath)
written_files.append(filepath)

# Make the Voronoi diagram of the round depressions
center_geoms = filtered_sinkholes_gdf[filtered_sinkholes_gdf['type']=='round'].geometry.centroid
voronoi_result_polys = center_geoms.voronoi_polygons()

# Find area with a high density of round depressions based on the area of vornoi polygons
dissolved_vornoi_polys = voronoi_result_polys[voronoi_result_polys.area < 60000].union_all('coverage')
dissolved_vornoi_polys_gs = gpd.GeoSeries([geom for geom in dissolved_vornoi_polys.geoms], crs=('epsg:'+str(EPSG)))
assert dissolved_vornoi_polys_gs.is_valid.all(), 'Dissolved voronoi polygons are not all valid.'

# Only keep large dense area and a buffer around
large_dense_areas = dissolved_vornoi_polys_gs[dissolved_vornoi_polys_gs.area > 200000].copy()
large_dense_areas = large_dense_areas.geometry.buffer(100)
large_dense_areas_union = large_dense_areas.union_all()

filepath = os.path.join(OUTPUT_DIR, 'large_dense_area.gpkg')
large_dense_areas.to_file(filepath)
written_files.append(filepath)

# Only keep long depression that are within the large dense area
long_sinkholes_gdf = filtered_sinkholes_gdf[
    (filtered_sinkholes_gdf['type']=='long')
    & (filtered_sinkholes_gdf.area < 9500)
    & (filtered_sinkholes_gdf.area > 700)
    & (filtered_sinkholes_gdf.compactness > 0.11)
    & filtered_sinkholes_gdf.geometry.within(large_dense_areas_union)
].copy()

# Outside the dense areas, only keep very round depressions
round_sinkholes_gdf = filtered_sinkholes_gdf[
    (filtered_sinkholes_gdf['type']=='round')
    & (
        (
            filtered_sinkholes_gdf.geometry.within(large_dense_areas_union)
        )
        |(
            ~filtered_sinkholes_gdf.geometry.within(large_dense_areas_union)
            & (filtered_sinkholes_gdf.compactness > 0.38)
            & (filtered_sinkholes_gdf.area > 1000)
        )
    )
]
round_sinkholes_gdf.loc[:, 'type'] = [
    'round in dense area' if geometry.within(large_dense_areas_union) else 'very round outside dense area' 
    for geometry in round_sinkholes_gdf.geometry
]

sinkholes_gdf = pd.concat([round_sinkholes_gdf, long_sinkholes_gdf], ignore_index=True)
sinkholes_gdf['doline_id'] = sinkholes_gdf.index

logger.info('Deal with thalwegs producing false positives and too deep dolines...')

# Get an outline on the surrounding terrain
sinkholes_gdf['buffered_outline'] = sinkholes_gdf.geometry.buffer(15).boundary

sinkholes_gdf['alti_diff'] = np.nan
for dem_tile in tqdm(sinkholes_gdf.corresponding_dem.unique(), desc="Get lowest point of sinkhole and its buffer outline"):
    dem_path = os.path.join(DEM_DIR, dem_tile)
    sinkholes_on_tiles_gdf = sinkholes_gdf[sinkholes_gdf.corresponding_dem == dem_tile].copy()

    # Get lowest point of sinkhole and its buffer outline
    with rio.open(dem_path) as src:
        sinkholes_lowest_alti = zonal_stats(sinkholes_on_tiles_gdf.geometry, src.read(1), affine=src.transform, stats='min')
        buffer_lowest_alti = zonal_stats(sinkholes_on_tiles_gdf.buffered_outline, src.read(1), affine=src.transform, stats='min')

    # Keep results in a dataframe
    alti_dict={
        'doline_id': sinkholes_on_tiles_gdf.doline_id,
        'min_sinkhole': [x['min'] for x in sinkholes_lowest_alti],
        'min_buffer_line': [x['min'] for x in buffer_lowest_alti],
        'geometry': sinkholes_on_tiles_gdf.geometry
    }
    alti_gdf = gpd.GeoDataFrame(alti_dict, crs=EPSG)

    # Detect and mark thalwegs
    alti_gdf['alti_diff'] = alti_gdf['min_buffer_line'] - alti_gdf['min_sinkhole']
    thalweg_ids = alti_gdf.loc[alti_gdf['alti_diff'] < 0.8, 'doline_id']

    sinkholes_gdf.loc[sinkholes_gdf.doline_id.isin(thalweg_ids), 'type'] = 'thalweg'
    sinkholes_gdf.loc[sinkholes_gdf.doline_id.isin(alti_gdf.doline_id), 'alti_diff'] = alti_gdf.alti_diff

filepath = os.path.join(OUTPUT_DIR, 'sinkholes.gpkg')
sinkholes_gdf[['doline_id', 'type', 'compactness', 'alti_diff', 'corresponding_dem', 'geometry']].to_file(filepath)
written_files.append(filepath)

logger.success('Done! The following files were written:')
for file in written_files:
    logger.success(file)

logger.info(f'Done in {time() - tic:0.4f} seconds')