import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd

from urllib.error import HTTPError
from urllib.request import urlretrieve

import functions.fct_misc as misc
logger = misc.format_logger(logger)


# ----- Define functions -----

def number_to_first_four_digits(number):
    return int(str(number)[:4])

# ----- Get config -----
tic = time()
logger.info('Starting...')

cfg = misc.get_config(config_key=os.path.basename(__file__), desc="This script downloads the tiles from swissALTI3D.")

# Load input parameters

WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

AOI = cfg['aoi']
OVERWRITE = cfg['overwrite']

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
written_files = []

# ----- Data processing -----

logger.info('Read AOI data')
aoi_gdf = gpd.read_file(AOI)
aoi_gdf = aoi_gdf.to_crs(2056)

aoi_gdf['origin'] = [misc.get_bbox_origin(bbox_geom) for bbox_geom in aoi_gdf.geometry]
aoi_gdf['max'] = [misc.get_maximum_coordinates(bbox_geom) for bbox_geom in aoi_gdf.geometry]

for aoi in tqdm(aoi_gdf.itertuples(), desc="Download tiles", total=aoi_gdf.shape[0]):

    year = aoi.year
    min_x, min_y = aoi.origin
    max_x, max_y = aoi.max

    min_x = number_to_first_four_digits(min_x)
    min_y = number_to_first_four_digits(min_y)
    max_x = number_to_first_four_digits(max_x) + 1
    max_y = number_to_first_four_digits(max_y) + 1

    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            tile_location = str(x)[:4] + '-' + str(y)[:4]
            outpath = os.path.join(OUTPUT_DIR, f'swissalti3d_{year}_' + tile_location + '.tif')
            if os.path.isfile(outpath) and not OVERWRITE:
                continue
            
            url = 'https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_' + str(year) + '_' + tile_location \
                + '/swissalti3d_' + str(year) + '_' + tile_location + '_0.5_2056_5728.tif'
            try:
                path, header = urlretrieve(url, outpath)
                assert os.path.isfile(outpath), f'File {outpath} not found after its download.'
                written_files.append(outpath)

            except HTTPError as e:
                logger.error(f'Tile {tile_location} in area {aoi.name} not found for year {year}.')


logger.success(f'Done!{"The following files were written:" if len(written_files) < 25 else ""}')
if len(written_files) < 25:
    logger.success(written_files)
else:
    logger.success(f"Files were written in the folder {OUTPUT_DIR}")

toc = time()
logger.info(f"Done in {toc - tic:0.4f} seconds")