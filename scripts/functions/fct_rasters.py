import geopandas as gpd
import rasterio as rio
from shapely.geometry import shape
from rasterio.features import shapes


def polygonize_binary_raster(binary_raster, crs=None, transform=None):
    """
    Transform a binary raster into a geopandas GeoDataFrame.

    Parameters
    ----------
    binary_raster : numpy array or str
        The binary raster to be polygonized. If a str, it is interpreted as a
        path to a raster file.
    crs : str, optional
        The coordinate reference system of the raster. If not specified, it is
        inferred from the raster file.
    transform : Affine, optional
        The transformation of the raster. If not specified, it is inferred from
        the raster file.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame of the areas equal to one in the binary raster.

    """
    
    if isinstance(binary_raster, str):
        with rio.open(binary_raster) as src:
            image=src.read(1)
            crs = src.crs
            transform = src.transform
    else:
        image = binary_raster

    assert image.min() == 0 and image.max() == 1, 'Raster must be binary'
    image = image.astype('int16')

    mask= image==1
    geoms = ((shape(s), v) for s, v in shapes(image, mask, transform=transform))
    gdf = gpd.GeoDataFrame(geoms, columns=['geometry', 'number'])
    gdf.set_crs(crs=crs, inplace=True)

    return gdf


def polygonize_raster(raster, meta=None, dtype=None):
    """
    Transform a raster into a geopandas GeoDataFrame.

    Parameters
    ----------
    raster : numpy array or str
        The raster to be polygonized. If a str, it is interpreted as a
        path to a raster file.
    meta : dict, optional
        The metadata of the raster. If not specified, it is inferred from
        the raster file.
    dtype : numpy.dtype, optional
        The data type of the raster. If not specified, it is inferred from
        the raster file.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame of the different areas and their value in the raster.

    """

    if isinstance(raster, str):
        with rio.open(raster) as src:
            band = src.read(1)
            meta = src.meta
    else:
        band = raster

    if dtype:
        band = band.astype(dtype)

    shapes_on_band = list(shapes(band, mask=band!=meta['nodata'], transform=meta['transform']))
    polygons = [(shape(geom), value) for geom, value in shapes_on_band]

    gdf = gpd.GeoDataFrame(polygons, columns=['geometry', 'number'])
    gdf.set_crs(crs=meta['crs'], inplace=True)

    return gdf