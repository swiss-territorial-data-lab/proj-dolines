import os
from argparse import ArgumentParser
from loguru import logger
from time import time
from tqdm import tqdm
from yaml import FullLoader, load

import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.merge import merge

from functions.fct_misc import format_logger
from global_parameters import ALL_PARAMS_IGN, ALL_PARAMS_LEVEL_SET, ALL_PARAMS_WATERSHEDS, AOI_TYPE, GDAL_DATA

logger = format_logger(logger)

def main(dem_correspondence_pd, aoi_gdf, dem_dir, resolution, save_extra=False, output_dir='outputs'):
    """
    Merge DEMs for each AOI

    Parameters
    ----------
    dem_correspondence_pd : pandas.DataFrame
        Dataframe with the correspondence between the AOIs and the DEMs
    aoi_gdf : geopandas.GeoDataFrame
        GeoDataFrame with the AOIs
    dem_dir : str
        Directory where the DEMs are stored
    resolution : float
        Resolution of the merged DEMs
    save_extra : bool, optional
        Whether to save the merged DEMs. Defaults to False
    output_dir : str, optional
        Directory where to save the merged DEMs. Defaults to 'outputs'

    Returns
    -------
    dict
        Dictionary with the merged DEMs and their associated metadata
    """
    
    if save_extra:
        os.makedirs(output_dir, exist_ok=True)

    buffered_aoi_gdf = aoi_gdf.copy()
    buffered_aoi_gdf.loc[:, 'geometry'] = aoi_gdf.geometry.buffer(2000)
    dem_dict = {}
    for aoi in tqdm(buffered_aoi_gdf.itertuples(), desc=f"Merge DEMs to a resolution of {resolution} m", total=buffered_aoi_gdf.shape[0]):
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

        merged_dem, out_transform = merge(dem_list, res=resolution, resampling=Resampling.bilinear, nodata=meta['nodata'])

        meta.update({'height': merged_dem.shape[1], 'width': merged_dem.shape[2], 'transform': out_transform})
        tile_name = str(aoi.year) + '_' + str(round(out_transform[2]))[:4] + '_' + str(round(out_transform[5]))[:4] + '.tif'

        if save_extra:
            with rio.open(os.path.join(output_dir, tile_name), 'w', **meta) as dst:
                dst.write(merged_dem)

        dem_dict[tile_name] = (merged_dem[0, :, :], meta)

    return dem_dict


def read_initial_data(aoi_path, dem_correspondence_csv, EPSG):
    dem_correspondence_pd = pd.read_csv(dem_correspondence_csv)

    aoi_gdf = gpd.read_file(aoi_path)
    aoi_gdf = aoi_gdf.to_crs(EPSG)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
        aoi_gdf = aoi_gdf[aoi_gdf['Type'] == AOI_TYPE].copy()

    return dem_correspondence_pd, aoi_gdf


if __name__ == '__main__':
    tic = time()
    logger.info('Starting...')

    # ----- Get parameters -----

    # Argument and parameter specification
    parser = ArgumentParser(description="This script merges the DEM files over the AOI.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

    METHOD_TYPE = args.config_file.split('_')[-1].split('.')[0].upper()

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']
    DEM_CORRESPONDENCE = cfg['dem_correspondence']
    AOI = cfg['aoi']
    
    EPSG = 2056

    if AOI_TYPE:
        output_dir = os.path.join(OUTPUT_DIR, AOI_TYPE)
        aoi_type_key = AOI_TYPE
    else:
        aoi_type_key = 'All types'
        output_dir = OUTPUT_DIR

    if METHOD_TYPE == 'IGN':
        RES = ALL_PARAMS_IGN[aoi_type_key]['resolution']
    elif METHOD_TYPE == 'WATERSHEDS':
        RES = ALL_PARAMS_WATERSHEDS[aoi_type_key]['resolution']
    elif METHOD_TYPE == 'LEVEL-SET':
        RES = ALL_PARAMS_LEVEL_SET[aoi_type_key]['resolution']
    else:
        logger.warning('No default resolution found for this method. Using data from the config file.')
        RES = cfg['res']

    os.chdir(WORKING_DIR)

     # ----- Data processing -----

    logger.info('Read AOI data')
    dem_correspondence_pd, aoi_gdf = read_initial_data(AOI, DEM_CORRESPONDENCE, EPSG)


    _ = main(dem_correspondence_pd, aoi_gdf, DEM_DIR, RES, save_extra=True, output_dir=output_dir)
    

    logger.success(f'Done! The files were written in {OUTPUT_DIR}.')
    logger.info(f'Elapsed time: {time() - tic:0.2f} seconds')