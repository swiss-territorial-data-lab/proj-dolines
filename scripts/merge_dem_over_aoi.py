import os
import sys
from glob import glob
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge

from functions.fct_misc import format_logger, get_config

logger = format_logger(logger)

def main(dem_correspondence_pd, aoi_gdf, dem_dir, resolution, save_extra=False, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    dem_dict = {}
    for aoi in tqdm(aoi_gdf.itertuples(), desc="Merge DEMs", total=aoi_gdf.shape[0]):
        dem_list = [
            os.path.join(dem_dir, dem) for dem in dem_correspondence_pd[aoi.name].tolist()
            if isinstance(dem, str) and os.path.exists(os.path.join(dem_dir, dem))
        ]

        if len(dem_list) == 0:
            logger.warning(f'No DEMs found for AOI {aoi.name}')
            continue
        else:
            with rio.open(dem_list[0]) as src:
                meta = src.meta

        merged_dem, out_transform = merge(dem_list, res=resolution, resampling=Resampling.bilinear)

        meta.update({'height': merged_dem.shape[1], 'width': merged_dem.shape[2], 'transform': out_transform})
        tile_name = str(aoi.year) + '_' + str(round(out_transform[2]))[:4] + '_' + str(round(out_transform[5]))[:4] + '.tif'

        if save_extra:
            with rio.open(os.path.join(output_dir, tile_name), 'w', **meta) as dst:
                dst.write(merged_dem)

        dem_dict[tile_name] = (merged_dem, meta)

    return dem_dict


def read_initial_data(aoi_path, dem_correspondence_csv):
    dem_correspondence_pd = pd.read_csv(dem_correspondence_csv)

    aoi_gdf = gpd.read_file(aoi_path)
    aoi_gdf = aoi_gdf.to_crs(2056)
    aoi_gdf.loc[:, 'geometry'] = aoi_gdf.geometry.buffer(2000)

    return dem_correspondence_pd, aoi_gdf


if __name__ == '__main__':
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    cfg = get_config(os.path.basename(__file__), desc="This script merges the DEM files over the AOI.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']
    DEM_CORRESPONDENCE = cfg['dem_correspondence']
    AOI = cfg['aoi']

    RES = cfg['res'] # in meters

    os.chdir(WORKING_DIR)

     # ----- Data processing -----

    logger.info('Read AOI data')
    dem_correspondence_pd, aoi_gdf = read_initial_data(AOI, DEM_CORRESPONDENCE)

    _ = main(dem_correspondence_pd, aoi_gdf, DEM_DIR, RES, save_extra=True, output_dir=OUTPUT_DIR)
    

    logger.success(f'Done! The files were written in {OUTPUT_DIR}.')
    logger.info(f'Elapsed time: {time() - tic:0.2f} seconds')