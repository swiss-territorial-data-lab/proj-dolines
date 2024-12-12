import os
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import rasterio as rio
from rasterio.mask import mask

import matplotlib.pyplot as plt
from skimage.metrics import hausdorff_distance
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from functions.fct_metrics import get_fractional_sets, median_group_distance
from functions.fct_misc import format_logger, get_config
from global_parameters import AOI_TYPE

logger = format_logger(logger)


def median_distance_between_datasets(reference_gdf, detections_gdf, rounding_digits=2):
    """
    Calculate the median distance between all pairs of closest elements in the reference data and detections.

    Parameters
    ----------
    reference_gdf : GeoDataFrame
        GeoDataFrame of the reference data
    detections_gdf : GeoDataFrame
        GeoDataFrame of the detections
    rounding_digits : int, optional
        Number of decimal places to round the output to (default: 2)

    Returns
    -------
    median_distance : float
        Median distance between all pairs of closest elements in the reference data and detections
    """
    _ref_gdf = reference_gdf.copy()
    _dets_gdf = detections_gdf.copy()
    
    # Get the median distance between elements of the reference data and detection dataset
    nearest_left_join_gdf = _ref_gdf[['id', 'geometry']].sjoin_nearest(_dets_gdf[['doline_id', 'geometry']], how='left', distance_col='distance')
    nearest_left_join_gdf.drop_duplicates(subset=['id', 'distance'], inplace=True)
    nearest_right_join_gdf = _ref_gdf[['id', 'geometry']].sjoin_nearest(_dets_gdf[['doline_id', 'geometry']], how='right', distance_col='distance')
    nearest_right_join_gdf.drop_duplicates(subset=['doline_id', 'distance'], inplace=True)
    nearest_join_gdf = pd.concat([nearest_left_join_gdf, nearest_right_join_gdf], ignore_index=True)
    nearest_join_gdf.drop_duplicates(subset=['id', 'doline_id'], inplace=True)

    return nearest_join_gdf['distance'].median().round(rounding_digits)


def prepare_dolines_to_assessment(dolines_gdf):
    """
    Prepare the dolines GeoDataFrame for the assessment by adding a 'det_class' column and renaming 'corresponding_dem' to 'tile_id'.
    """
    prepared_dolines_gdf = dolines_gdf.copy()
    prepared_dolines_gdf['det_class'] = 'doline'
    prepared_dolines_gdf.rename(columns={'corresponding_dem': 'tile_id'}, inplace=True)

    return prepared_dolines_gdf


def prepare_reference_data_to_assessment(ref_path):
    """
    Ensure the stability of the reference data format and add a 'label_class' column.
    """
    ini_ref_data_gdf = gpd.read_file(ref_path)
    ini_ref_data_gdf.to_crs(2056, inplace=True)
    ini_ref_data_gdf['label_class'] = 'doline'
    if (ini_ref_data_gdf.geometry.geom_type == 'MultiPolygon').any():
        ref_data_gdf = ini_ref_data_gdf.explode(index_parts=False)
        assert ini_ref_data_gdf.shape[0] == ref_data_gdf.shape[0], 'Some multipart geometries were present in the reference data.'
    else:
        ref_data_gdf = ini_ref_data_gdf.copy()

    return ref_data_gdf


def main(ref_data_type, ref_data_gdf, detections_gdf, pilot_areas_gdf, det_type, dem_dir='outputs', save_extra=False, output_dir='outputs'):
    """
    Main function to assess the quality of the doline detection model.

    Parameters
    ----------
    ref_data_type : str
        Type of reference data, either 'geocover' or 'tlm'.
    ref_data_gdf : GeoDataFrame
        GeoDataFrame of the reference data.
    detections_gdf : GeoDataFrame
        GeoDataFrame of the detections.
    pilot_areas_gdf : GeoDataFrame
        GeoDataFrame of the pilot areas.
    det_type : str
        Type of detection method, either 'watersheds' or 'ign'.
    dem_dir : str, optional
        Directory of the DEM files (default: 'outputs').
    save_extra : bool, optional
        Whether to save global and fine-grained metrics (default: False).
    output_dir : str, optional
        Directory of the output files (default: 'outputs').

    Returns
    -------
    metric : float
        F1 score of the detection model.
    written_files : list of str
        List of paths to the written files.
    """
    _ref_gdf = ref_data_gdf.copy()
    _dets_gdf = detections_gdf.copy()
    _pilot_areas_gdf = pilot_areas_gdf.copy()

    logger.info(f'Source for the reference data: {ref_data_type}')
    logger.info(f'Detection method: {det_type}')
    name_suffix = f'{AOI_TYPE + "_" if AOI_TYPE else ""}{ref_data_type}_'

    assert (ref_data_type.lower() in ['merged_reference', 'ground_truth']), 'Reference data type must be geocover, tlm or ground_truth.'
    assert (det_type.lower() in ['watersheds', 'ign', 'lidar_lib', 'stochastic_depressions']), 'Detection method must be watersheds or ign.'

    if _dets_gdf.loc[0, 'tile_id'].endswith('.tif'):
        _dets_gdf.loc[:, 'tile_id'] = _dets_gdf.loc[:, 'tile_id'].str.rstrip('.tif')

    if 'type' in _dets_gdf:
        _dets_gdf = _dets_gdf[_dets_gdf['type']!='thalweg'].copy()

    logger.info('Match detections with reference data...')        # TODO: Associate dem name to ref data
    tile_id = f'tile_id_{det_type}' if det_type in ['watersheds', 'ign'] else 'tile_id'
    _pilot_areas_gdf['tile_id'] = _pilot_areas_gdf[tile_id]
    ref_data_in_aoi_gdf = _ref_gdf.sjoin(_pilot_areas_gdf[['tile_id', 'geometry']], how='inner')
    dets_in_aoi_gdf = _dets_gdf[_dets_gdf.geometry.within(_pilot_areas_gdf.geometry.union_all())].copy()

    tp_gdf, fp_gdf, fn_gdf, _ = get_fractional_sets(dets_in_aoi_gdf, ref_data_in_aoi_gdf[['id', 'label_class', 'tile_id', 'geometry']], iou_threshold=0.1)
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fp_gdf['label_class'] = 'non-doline'
    fn_gdf['tag'] = 'FN'
    fn_gdf['det_class'] = 'non-doline'

    tagged_detections_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf], ignore_index=True)

    logger.info('Calculate metrics on all dets...')
    metric_fct = f1_score # if ref_data_type == 'geocover' else recall_score
    resemblance_column = 'dist_centroid' if (ref_data_in_aoi_gdf.geometry.geom_type == 'Point').any() else 'IoU'
    metric = metric_fct(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline')
    metrics_dict = {
        'nbr labels': tagged_detections_gdf[tagged_detections_gdf.label_class == 'doline'].shape[0],
        'nbr detections': tagged_detections_gdf[tagged_detections_gdf.det_class == 'doline'].shape[0],
        'precision': precision_score(tagged_detections_gdf.label_class, tagged_detections_gdf.det_class, pos_label='doline'),
        'recall': recall_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline'),
        'f1': f1_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline'),
    }

    metrics_dict['median_distance'] = median_distance_between_datasets(ref_data_in_aoi_gdf, dets_in_aoi_gdf)
    metrics_dict['median_group_distance'], group_med_dist_gdf = median_group_distance(ref_data_in_aoi_gdf, dets_in_aoi_gdf)

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['all']).transpose().round(3)

    logger.info(f'Metrics:')
    logger.info(f'- f1 score: {metrics_df.loc['all', "f1"]}')

    if resemblance_column == 'dist_centroid':
        resemblance_result = 'median dist for TP'
        metrics_df[resemblance_result] = 0 if tp_gdf.empty else tp_gdf['dist_centroid'].median()
        logger.info(f'- median distance between centroids for TP: {metrics_df.loc['all', "median dist for TP"]}')
    else:
        resemblance_result = 'median IoU for TP'
        metrics_df[resemblance_result] =  0 if tp_gdf.empty else tp_gdf['IoU'].median()
        logger.info(f'- median IoU for TP: {metrics_df.loc['all', "median IoU for TP"]}')
    
    written_files = []
    if save_extra:
        output_dir = output_dir if (det_type.lower() in output_dir) | (det_type.lower() in os.getcwd()) else os.path.join(output_dir, det_type)
        os.makedirs(output_dir, exist_ok=True)

        tagged_detections_gdf = tagged_detections_gdf[detections_gdf.columns.tolist() + ['id', 'label_class', resemblance_column, 'tag']].copy()
        if resemblance_column == 'dist_centroid':
            # Save polygons to separate file
            not_a_point = tagged_detections_gdf.geometry.geom_type!='Point'
            filepath = os.path.join(output_dir, f'{name_suffix}tagged_polys.gpkg')
            tagged_detections_gdf[not_a_point].to_file(filepath)
            written_files.append(filepath)

            # Transform TP and FP polygons into points
            tagged_detections_gdf.loc[not_a_point, 'geometry'] = tagged_detections_gdf.loc[not_a_point, 'geometry'].centroid

        filepath = os.path.join(output_dir, f'{name_suffix}tagged_detections.gpkg')
        tagged_detections_gdf.to_file(filepath)
        written_files.append(filepath)

        group_med_dist_gdf.sort_values('distance', inplace=True)
        dist_ref_gdf = pd.merge(
            ref_data_in_aoi_gdf[['id', 'tile_id', 'geometry']], group_med_dist_gdf[['id', 'doline_id', 'distance', 'group_id', 'group_distance']].drop_duplicates('id'), 
            on='id'
        )
        dist_det_gdf = pd.merge(dets_in_aoi_gdf, group_med_dist_gdf[['id', 'doline_id', 'distance', 'group_id', 'group_distance']].drop_duplicates('doline_id'), on='doline_id')
        all_med_dist_gdf = pd.concat([dist_ref_gdf, dist_det_gdf], ignore_index=True)
        filepath = os.path.join(output_dir, f'{name_suffix}grouped_results.gpkg')
        all_med_dist_gdf.to_file(filepath)
        written_files.append(filepath)

        confusion_matrix_df = pd.DataFrame(confusion_matrix(
            tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class']
        )).rename(columns={0: 'doline', 1: 'non-doline'}, index={0: 'doline', 1: 'non-doline'})

        filepath = os.path.join(output_dir, f'{name_suffix}confusion_matrix.csv')
        confusion_matrix_df.to_csv(filepath)
        written_files.append(filepath)

        logger.info('Calculate the metrics for each pilot area...')
        pilot_areas = _dets_gdf.tile_id.sort_values().unique().tolist()
        if pilot_areas != tagged_detections_gdf.tile_id.sort_values().unique().tolist():
            logger.error('Tile ids not corresponding between labels and detections')
        metrics_per_area_dict = {
            'nbr labels': [], 'nbr detections': [],
            'precision': [], 'recall': [], 'f1': [], resemblance_result: [], 'median_distance': [], 'median_group_distance': []
        }
        for area_name in pilot_areas:
            results_in_area_gdf = tagged_detections_gdf[tagged_detections_gdf.tile_id == area_name].copy()
            if results_in_area_gdf[results_in_area_gdf.det_class == 'non-doline'].empty:
                logger.warning(f'Area {area_name} does not contain any label corresponding to a false negative!')

            metrics_per_area_dict['nbr labels'].append(results_in_area_gdf[results_in_area_gdf.label_class == 'doline'].shape[0])
            metrics_per_area_dict['nbr detections'].append(results_in_area_gdf[results_in_area_gdf.det_class == 'doline'].shape[0])

            metrics_per_area_dict['precision'].append(precision_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline', zero_division=0))
            metrics_per_area_dict['recall'].append(recall_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline', zero_division=0))
            metrics_per_area_dict['f1'].append(f1_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, pos_label='doline'))
            metrics_per_area_dict[resemblance_result].append(results_in_area_gdf.loc[results_in_area_gdf.tag=='TP', resemblance_column].median())
            metrics_per_area_dict['median_distance'].append(median_distance_between_datasets(
                ref_data_in_aoi_gdf[ref_data_in_aoi_gdf.tile_id == area_name],
                results_in_area_gdf[~results_in_area_gdf.doline_id.isna()]
            ))

            med_group_dist, _ = median_group_distance(ref_data_in_aoi_gdf[ref_data_in_aoi_gdf.tile_id == area_name],results_in_area_gdf[~results_in_area_gdf.doline_id.isna()])
            metrics_per_area_dict['median_group_distance'].append(med_group_dist)

        metrics_per_area_df = pd.DataFrame.from_dict(
            metrics_per_area_dict, orient='index', columns=pilot_areas
        ).transpose().round(3)
        metrics_df = pd.concat([metrics_df, metrics_per_area_df])

        if False:
            similarity_dict = {'hausdorff distance': []}
            # similarity_dir = os.path.join(output_dir, 'similarity_rasters')
            # os.makedirs(similarity_dir, exist_ok=True)
            for area in tqdm(_pilot_areas_gdf.itertuples(), desc='Compare image shapes', total=_pilot_areas_gdf.shape[0]):
                area_dolines_gdf = dets_in_aoi_gdf[dets_in_aoi_gdf.tile_id == area.tile_id].copy()
                area_ref_data_gdf = ref_data_in_aoi_gdf[ref_data_in_aoi_gdf.tile_id == area.tile_id].copy()

                if area_ref_data_gdf.empty or area_dolines_gdf.empty:
                    logger.warning(f'No label or data for the area {area.name}.')
                    similarity_dict['hausdorff distance'].append(0)
                    continue
                
                with rio.open(os.path.join(dem_dir, area.tile_id)) as src:
                    meta = src.meta

                    # Mask DEM with the dolines
                    ref_image, _ = mask(src, area_ref_data_gdf.geometry)
                    det_image, _ = mask(src, area_dolines_gdf.geometry)

                    # Determine hausdorff distance
                    area_pixel_doline_gdf = area_dolines_gdf.copy()
                    area_pixel_doline_gdf.loc[:, 'geometry'] = area_pixel_doline_gdf.geometry.centroid.buffer(meta['transform'][0], cap_style=3)
                    area_pixel_ref_gdf = _ref_gdf.copy()
                    area_pixel_ref_gdf.loc[:, 'geometry'] = area_pixel_ref_gdf.geometry.centroid.buffer(meta['transform'][0], cap_style=3)

                    ref_image, _ = mask(src, area_pixel_ref_gdf.geometry, nodata=0)  # Set nodata to 0, because the hausdorff distance works with non-zero pixels
                    ref_image[ref_image>0] = 1
                    det_image, _ = mask(src, area_pixel_doline_gdf.geometry, nodata=0)
                    det_image[det_image>0] = 1

                    similarity_dict['hausdorff distance'].append(hausdorff_distance(ref_image, det_image, method='modified'))
                
                # meta.update({'height': binary_ref_image.shape[1], 'width': binary_ref_image.shape[2], 'transform': meta['transform']})
                # with rio.open(os.path.join(similarity_dir, 'ref_' + area_tile_id), 'w', **meta) as dst:
                #     dst.write(binary_ref_image)
                
                # with rio.open(os.path.join(similarity_dir, 'det_' + area_tile_id), 'w', **meta) as dst:
                #     dst.write(binary_det_image)

            similarity_df = pd.DataFrame.from_dict(similarity_dict, orient='index', columns=pilot_areas).transpose().round(3)
            metrics_df = pd.concat([metrics_df, similarity_df], axis=1)

        metrics_df = metrics_df.reset_index().rename(columns={'index': 'tile_id'})
        metrics_df = pd.merge(
            _pilot_areas_gdf[['name', tile_id]], metrics_df, 
            left_on=tile_id, right_on='tile_id', how='outer'
        )
        if tile_id != 'tile_id':
            metrics_df.drop(columns=f'tile_id_{det_type}', inplace=True)

        filepath = os.path.join(output_dir, f'{name_suffix}metrics.csv')
        metrics_df.to_csv(filepath)
        written_files.append(filepath)

        logger.info(f'Make some graphs...')
        sub_metrics_df = metrics_df[~metrics_df.name.isna()].sort_values(by='name')
        graphs_dir = os.path.join(output_dir, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)

        fixed_params = {'grid': True, 'legend': True, 'ylim': (0, None)}

        # Make a graph of the metrics between 0 and 1 for each zone
        fig, ax = plt.subplots()
        if resemblance_column == 'IoU':
            df_plot = sub_metrics_df.plot(
                x='name', y=['precision', 'recall', 'f1', 'median IoU for TP'], kind='line', style='o', ax=ax, grid=True, legend=True,
            )
        else:
            df_plot = sub_metrics_df.plot(
                x='name', y=['precision', 'recall', 'f1'], kind='line', style='o', ax=ax,  grid=True, legend=True,
            )
        ax.set(
            title='Metrics per zone', xlabel='Zone name', ylabel='Metric value', xticks=range(sub_metrics_df.shape[0]), ylim=(0,1),
        )
        fig.set_size_inches(1.5*sub_metrics_df.shape[0], 5)

        filepath = os.path.join(graphs_dir, f'{name_suffix}metrics_per_zone.jpg')
        fig.savefig(filepath, bbox_inches='tight')
        written_files.append(filepath)

        if resemblance_column == 'dist_centroid':
            # Plot the centroid distance between labels and detections for TP
            fig, ax = plt.subplots()
            df_plot = sub_metrics_df.plot(
                x='name', y='median dist for TP', kind='line', style='o', ax=ax,
                title='Median distance between labels and detections for TP', xlabel='Zone name', ylabel='Median distance',
                figsize=(1.5*sub_metrics_df.shape[0], 5), **fixed_params
            )

            filepath = os.path.join(graphs_dir, f'{name_suffix}median_dist_for_TP.jpg')
            fig.savefig(filepath, bbox_inches='tight')
            written_files.append(filepath)

        # Make a graph f1 vs median distance
        fig, ax = plt.subplots()
        df_plot = sub_metrics_df.plot(
            x='f1', y='median_distance', kind='scatter', ax=ax,
            title='F1 vs median distance between labels and detections', xlabel='F1', ylabel='Median distance', **fixed_params
        )

        for row in sub_metrics_df.itertuples():
            ax.annotate(row.name, (row.f1, row.median_distance))

        filepath = os.path.join(graphs_dir, f'{name_suffix}f1_vs_median_distance.jpg')
        fig.savefig(filepath, bbox_inches='tight')
        written_files.append(filepath)

        # Make a graph f1 vs median distance
        fig, ax = plt.subplots()
        df_plot = sub_metrics_df.plot(
            x='f1', y='median_group_distance', kind='scatter', ax=ax,
            title='F1 vs median distance between labels and detections', xlabel='F1', ylabel='Median group distance', **fixed_params
        )
        ax.set_ylim(ymin=0)

        for row in sub_metrics_df.itertuples():
            ax.annotate(row.name, (row.f1, row.median_group_distance))

        filepath = os.path.join(graphs_dir, f'{name_suffix}f1_vs_median_group_distance.jpg')
        fig.savefig(filepath, bbox_inches='tight')
        written_files.append(filepath)

    return metric, written_files


if __name__ == '__main__':
    tic = time()
    logger.info('Starting...')

    # ----- Get config -----
    cfg = get_config(os.path.basename(__file__), desc="This script assesses the results of any doline detection method.")

    WORKING_DIR = cfg['working_dir']
    OUTPUT_DIR = cfg['output_dir']
    DEM_DIR = cfg['dem_dir']

    REF_DATA_TYPE = cfg['type']['ref_data']
    REF_DATA = cfg[f'ref_data'][REF_DATA_TYPE.lower()]
    DET_TYPE = cfg['type']['dets']
    DETECTIONS = cfg[f'detections'][DET_TYPE.lower()]

    PILOT_AREAS = cfg['pilot_areas']

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
    dem_dir = os.path.join(DEM_DIR, AOI_TYPE) if AOI_TYPE else DEM_DIR
    det_path = os.path.join(os.path.dirname(DETECTIONS), AOI_TYPE, os.path.basename(DETECTIONS)) if AOI_TYPE else DETECTIONS

     # ----- Processing -----

    logger.info('Read data...')

    ref_data_gdf = prepare_reference_data_to_assessment(REF_DATA)

    detections_gdf = gpd.read_file(det_path)
    detections_gdf = prepare_dolines_to_assessment(detections_gdf)

    pilot_areas_gdf = gpd.read_file(PILOT_AREAS)
    pilot_areas_gdf.to_crs(2056, inplace=True)
    if AOI_TYPE:
        pilot_areas_gdf = pilot_areas_gdf[pilot_areas_gdf['Type'] == AOI_TYPE]

    _, written_files = main(REF_DATA_TYPE, ref_data_gdf, detections_gdf, pilot_areas_gdf, det_type=DET_TYPE, dem_dir=dem_dir, save_extra=True, output_dir=OUTPUT_DIR)

    logger.success('Done! The following files were written:')
    for file in written_files:
        logger.success(file)

    logger.info(f'Done in {time() - tic:0.2f} seconds')
