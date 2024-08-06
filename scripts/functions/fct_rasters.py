import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import shape
from rasterio.features import shapes


def polygonize_binary_raster(binary_raster, crs=None, transform=None):

    if isinstance(binary_raster, str):
        with rio.open(binary_raster) as src:
            image=src.read(1)
            crs = src.crs
            transform = src.transform
    else:
        image = binary_raster

    mask= image==1
    geoms = ((shape(s), v) for s, v in shapes(image, mask, transform=transform))
    gdf = gpd.GeoDataFrame(geoms, columns=['geometry', 'number'])
    gdf.set_crs(crs=crs, inplace=True)

    return gdf


def polygonize_raster(raster, meta=None, dtype=None):
    
    if isinstance(raster, str):
        with rio.open(raster) as src:
            band = src.read(1)
            meta = src.meta
    else:
        band = raster

    if dtype:
        band = band.astype(dtype)

    shapes_on_band = list(shapes(band, transform=meta['transform']))
    polygons = [(shape(geom), value) for geom, value in shapes_on_band if value != meta['nodata']]

    gdf = gpd.GeoDataFrame(polygons, columns=['geometry', 'number'])
    gdf.set_crs(crs=meta['crs'], inplace=True)

    return gdf