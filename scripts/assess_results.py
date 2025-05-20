import os
from loguru import logger
from time import time

import geopandas as gpd
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, precision_score, recall_score

from functions.fct_metrics import get_fractional_sets
from functions.fct_misc import format_logger, get_config
from global_parameters import AOI_TYPE, GDAL_DATA

logger = format_logger(logger)


def prepare_dolines_to_assessment(dolines_gdf):
    """
    Prepare the dolines GeoDataFrame for the assessment by adding a 'det_class' column and renaming 'corresponding_dem' to 'tile_id'.
    """
    prepared_dolines_gdf = dolines_gdf.copy()
    prepared_dolines_gdf['det_class'] = 'doline'
    prepared_dolines_gdf.rename(columns={'corresponding_dem': 'tile_id'}, inplace=True)

    return prepared_dolines_gdf


def prepare_reference_data_to_assessment(ref_path, epsg=2056):
    """
    Ensure the stability of the reference data format and add a 'label_class' column.
    """
    ini_ref_data_gdf = gpd.read_file(ref_path)
    ini_ref_data_gdf.to_crs(epsg, inplace=True)
    ini_ref_data_gdf['label_class'] = 'doline'
    if (ini_ref_data_gdf.geometry.geom_type == 'MultiPolygon').any():
        ref_data_gdf = ini_ref_data_gdf.explode(index_parts=False)
        assert ini_ref_data_gdf.shape[0] == ref_data_gdf.shape[0], 'Some multipart geometries were present in the reference data.'
    else:
        ref_data_gdf = ini_ref_data_gdf.copy()

    return ref_data_gdf


def main(ref_data_type, ref_data_gdf, detections_gdf, pilot_areas_gdf, det_type, save_extra=False, output_dir='outputs'):
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
        Type of detection method, either 'watersheds', 'ign', 'leve-set' or 'stochastic depressions'.
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
    assert (det_type.lower() in ['watersheds', 'ign', 'level-set', 'stochastic_depressions']), 'Detection method must be watersheds, level-set, stochastic depressions, or ign.'

    if _dets_gdf.loc[0, 'tile_id'].endswith('.tif'):
        _dets_gdf.loc[:, 'tile_id'] = _dets_gdf.loc[:, 'tile_id'].str.rstrip('.tif')

    if 'type' in _dets_gdf:
        _dets_gdf = _dets_gdf[_dets_gdf['type']!='thalweg'].copy()

    logger.info('Match detections with reference data...')        # TODO: Associate dem name to ref data
    tile_id = f'tile_id_{det_type}' if det_type in ['watersheds', 'ign'] else 'tile_id'
    _pilot_areas_gdf.loc[:, 'tile_id'] = _pilot_areas_gdf[tile_id]
    ref_data_in_aoi_gdf = _ref_gdf[['id', 'label_class', 'geometry']].sjoin(_pilot_areas_gdf[['tile_id', 'geometry']], how='inner')
    dets_in_aoi_gdf = _dets_gdf[_dets_gdf.geometry.within(_pilot_areas_gdf.geometry.union_all())].copy()

    logger.info('Get fractional sets...')
    tp_gdf, fp_gdf, fn_gdf, _ = get_fractional_sets(dets_in_aoi_gdf, ref_data_in_aoi_gdf[['id', 'label_class', 'tile_id', 'geometry']], iou_threshold=0.1)
    tp_gdf['tag'] = 'TP'
    fp_gdf['tag'] = 'FP'
    fp_gdf['label_class'] = 'non-doline'
    fn_gdf['tag'] = 'FN'
    fn_gdf['det_class'] = 'non-doline'

    tagged_detections_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf], ignore_index=True)

    logger.info('Calculate metrics on all dets...')
    metric_fct = fbeta_score # if ref_data_type == 'geocover' else recall_score
    resemblance_column = 'dist_centroid' if (ref_data_in_aoi_gdf.geometry.geom_type == 'Point').any() else 'IoU'
    metric = metric_fct(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], beta=2, pos_label='doline')
    metrics_dict = {
        'nbr labels': tagged_detections_gdf[tagged_detections_gdf['label_class'] == 'doline'].shape[0],
        'nbr detections': tagged_detections_gdf[tagged_detections_gdf['det_class'] == 'doline'].shape[0],
        'precision': precision_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline'),
        'recall': recall_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline'),
        'f1': f1_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], pos_label='doline'),
        'f2': fbeta_score(tagged_detections_gdf['label_class'], tagged_detections_gdf['det_class'], beta=2, pos_label='doline'),
    }

    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['all']).transpose().round(3)

    logger.info(f'Metrics:')
    logger.info(f"- f1 score: {metrics_df.loc['all', 'f1']}")

    if resemblance_column == 'dist_centroid':
        resemblance_result = 'median dist for TP'
        metrics_df[resemblance_result] = 0 if tp_gdf.empty else tp_gdf['dist_centroid'].median()
        logger.info(f"- median distance between centroids for TP: {metrics_df.loc['all', 'median dist for TP']}")
    else:
        resemblance_result = 'median IoU for TP'
        metrics_df[resemblance_result] = 0 if tp_gdf.empty else tp_gdf['IOU'].median()
        logger.info(f"- median IoU for TP: {metrics_df.loc['all', 'median IoU for TP']}")
    
    written_files = []
    if save_extra:
        output_dir = output_dir if (det_type.lower().replace('-', '_') in output_dir) | (det_type.lower().replace('-', '_') in os.getcwd())\
              else os.path.join(output_dir, det_type.replace('-', '_'))
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
            'precision': [], 'recall': [], 'f1': [], 'f2': [], resemblance_result: []
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
            metrics_per_area_dict['f2'].append(fbeta_score(results_in_area_gdf.label_class, results_in_area_gdf.det_class, beta=2, pos_label='doline'))
            metrics_per_area_dict[resemblance_result].append(results_in_area_gdf.loc[results_in_area_gdf.tag=='TP', resemblance_column].median())

        metrics_per_area_df = pd.DataFrame.from_dict(
            metrics_per_area_dict, orient='index', columns=pilot_areas
        ).transpose().round(3)
        metrics_df = pd.concat([metrics_df, metrics_per_area_df])

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
                x='name', y=['precision', 'recall', 'f2', 'median IoU for TP'], kind='line', style='o', ax=ax, grid=True, legend=True,
            )
        else:
            df_plot = sub_metrics_df.plot(
                x='name', y=['precision', 'recall', 'f2'], kind='line', style='o', ax=ax,  grid=True, legend=True,
            )
        ax.set(
            title='Metrics per zone', xlabel='Zone name', ylabel='Metric value', xticks=range(sub_metrics_df.shape[0]), xticklabels=sub_metrics_df.name, ylim=(0,1),
        )
        ax.tick_params(labelrotation=45)
        fig.set_size_inches(1*sub_metrics_df.shape[0], 5)

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
            ax.set(
            title='Metrics per zone', xlabel='Zone name', ylabel='Metric value', xticks=range(sub_metrics_df.shape[0]), xticklabels=sub_metrics_df.name,
            )
            ax.tick_params(labelrotation=90)
            filepath = os.path.join(graphs_dir, f'{name_suffix}median_dist_for_TP.jpg')
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

    REF_DATA_TYPE = cfg['type']['ref_data']
    REF_DATA = cfg[f'ref_data'][REF_DATA_TYPE.lower()]
    DET_TYPE = cfg['type']['dets']
    DETECTIONS = cfg[f'detections'][DET_TYPE.lower()]

    PILOT_AREAS = cfg['pilot_areas']
    
    EPSG = 2056

    os.chdir(WORKING_DIR)

    if AOI_TYPE:
        logger.warning(f'Working only on the areas of type {AOI_TYPE}')
        det_path = DETECTIONS if AOI_TYPE.lower() in DETECTIONS.lower() else os.path.join(os.path.dirname(DETECTIONS), AOI_TYPE, os.path.basename(DETECTIONS))
    else:
        det_path = DETECTIONS

     # ----- Processing -----

    logger.info('Read data...')

    ref_data_gdf = prepare_reference_data_to_assessment(REF_DATA, EPSG)

    detections_gdf = gpd.read_file(det_path)
    detections_gdf = prepare_dolines_to_assessment(detections_gdf)

    pilot_areas_gdf = gpd.read_file(PILOT_AREAS)
    pilot_areas_gdf.to_crs(EPSG, inplace=True)
    if AOI_TYPE:
        pilot_areas_gdf = pilot_areas_gdf[pilot_areas_gdf['Type'] == AOI_TYPE]

    _, written_files = main(REF_DATA_TYPE, ref_data_gdf, detections_gdf, pilot_areas_gdf, det_type=DET_TYPE, save_extra=True, output_dir=OUTPUT_DIR)

    logger.success('Done! The following files were written:')
    for file in written_files:
        logger.success(file)

    logger.info(f'Done in {time() - tic:0.2f} seconds')
