import os
import sys
from loguru import logger
from time import time

import geopandas as gpd

sys.path.insert(1, 'scripts')
from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

# Start chronometer
tic = time()
logger.info('Starting...')

# ----- Get parameters -----

cfg = get_config(config_key=os.path.basename(__file__), desc="This script processes the detected depressions to improve the doline detection.")

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

POTENTIAL_DOLINES = cfg['potential_dolines']

TLM_DATA = cfg['tlm_data']
GROUND_COVER_LAYER = cfg['ground_cover_layer']
RIVERS = cfg['rivers']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Processing -----

logger.info('Read data...')
potential_dolines_gdf = gpd.read_file(POTENTIAL_DOLINES)

rivers_gdf = gpd.read_file(RIVERS)
ground_cover_gdf = gpd.read_file(TLM_DATA, layer=GROUND_COVER_LAYER)

logger.info('Filter additional data...')

rivers_gdf = rivers_gdf[
    ~rivers_gdf.CODE.isin([12111, 12121, 12131, 33111, 33121, 33131, 43111, 43121, 43131, 53131])
    & (rivers_gdf.BIOGEO!='Mittelland')
].copy()
rivers_gdf.loc[:, 'geometry'] = rivers_gdf.geometry.buffer(1)
dissolved_rivers_gdf = rivers_gdf.dissolve(by='BIOGEO').explode()
dissolved_rivers_gdf['buff_river_geom'] = dissolved_rivers_gdf.geometry

water_bodies_gdf = ground_cover_gdf[ground_cover_gdf.objektart=='Stehende Gewaesser'].copy()
water_bodies_gdf['wb_geom'] = water_bodies_gdf.geometry

logger.info('Filter potential dolines in lakes...')

potential_dolines_gdf['doline_id'] = potential_dolines_gdf.index

depressions_in_lakes_gdf = gpd.sjoin(potential_dolines_gdf, water_bodies_gdf, how='left', predicate='intersects')
depressions_in_lakes_gdf['part_in_lake'] =  round(
    depressions_in_lakes_gdf.geometry.intersection(depressions_in_lakes_gdf.wb_geom).area / depressions_in_lakes_gdf.geometry.area, 3
)

potential_dolines_gdf = depressions_in_lakes_gdf.loc[
    (depressions_in_lakes_gdf.part_in_lake < 0.15) | (depressions_in_lakes_gdf.part_in_lake.isna()), potential_dolines_gdf.columns.tolist() + ['part_in_lake']
]

logger.info('Filter potential dolines in rivers...')

depressions_near_rivers_gdf = gpd.sjoin(potential_dolines_gdf, dissolved_rivers_gdf, how='left', predicate='intersects')
depressions_near_rivers_gdf['part_in_river'] =  round(
    depressions_near_rivers_gdf.geometry.intersection(depressions_near_rivers_gdf.buff_river_geom).area / depressions_near_rivers_gdf.geometry.area, 3
)

potential_dolines_gdf = depressions_near_rivers_gdf.loc[
    (depressions_near_rivers_gdf.part_in_river < 0.3) | (depressions_near_rivers_gdf.part_in_river.isna()), 
    potential_dolines_gdf.columns.tolist() + ['part_in_river']
]

logger.info('Filter based on attributes...')

dolines_gdf = potential_dolines_gdf[
    (potential_dolines_gdf.compactness > 0.5)
    & (potential_dolines_gdf.area < 4000)
    & (potential_dolines_gdf.diameter > 2.5)
    & (potential_dolines_gdf.depth > 0.6)
    & (potential_dolines_gdf.depth < 100)
].copy()

logger.info('Save file...')
filepath = os.path.join(OUTPUT_DIR, 'dolines.gpkg')
dolines_gdf.to_file(filepath)

logger.success(f'Done! The file {filepath} was written.')
logger.info(f'Done in {time() - tic:0.2f} seconds')