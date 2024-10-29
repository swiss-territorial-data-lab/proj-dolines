import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import distance

import networkx as nx
from itertools import product

from functions.fct_misc import geohash


def get_fractional_sets(dets_gdf, labels_gdf, iou_threshold=0.25):
    """
    Find the intersecting detections and labels.
    If the labels are polygons, control the IoU and class to get the TP.
    Tag detections and labels not intersecting or not intersecting enough as FP and FN respectively.
    Save the intersections with mismatched class ids in a separate geodataframe.

    Source: STDL object detector, v2.1.0
    https://github.com/swiss-territorial-data-lab/object-detector/blob/master/helpers/metrics.py

    Args:
        dets_gdf (geodataframe): geodataframe of the detections.
        labels_gdf (geodataframe): geodataframe of the labels.
        iou_threshold (float): threshold to apply on the IoU to determine if detections and labels can be matched. Defaults to 0.25.

    Raises:
        Exception: CRS mismatch

    Returns:
        tuple:
        - geodataframe: true positive intersections between a detection and a label;
        - geodataframe: false postive detection;
        - geodataframe: false negative labels;
        - geodataframe: intersections between a detection and a label with a mismatched class id.
    """

    _dets_gdf = dets_gdf.reset_index(drop=True)
    _labels_gdf = labels_gdf.reset_index(drop=True)
    
    if len(_labels_gdf) == 0:
        fp_gdf = _dets_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()
        mismatched_classes_gdf = gpd.GeoDataFrame()
        return tp_gdf, fp_gdf, fn_gdf, mismatched_classes_gdf
    
    assert(_dets_gdf.crs == _labels_gdf.crs), f"CRS Mismatch: detections' CRS = {_dets_gdf.crs}, labels' CRS = {_labels_gdf.crs}"

    # we add a id column to the labels dataset, which should not exist in detections too;
    # this allows us to distinguish matching from non-matching detections
    _labels_gdf['label_id'] = _labels_gdf.index
    _dets_gdf['det_id'] = _dets_gdf.index
    # We need to keep both geometries after sjoin to check the best intersection over union
    _labels_gdf['label_geom'] = _labels_gdf.geometry
    
    # TRUE POSITIVES
    left_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='left', predicate='intersects', lsuffix='det', rsuffix='label')
    
    # Test that something is detect
    candidates_tp_gdf = left_join[left_join.label_id.notnull()].copy()

    # IoU computation between labels and detections
    geom1 = candidates_tp_gdf['geometry'].to_numpy().tolist()
    geom2 = candidates_tp_gdf['label_geom'].to_numpy().tolist()
    if (labels_gdf.geometry.geom_type == 'Point').any():
        candidates_tp_gdf['dist_centroid'] = [distance(i.centroid, ii) for (i, ii) in zip(geom1, geom2)]
        resemblance_column = 'dist_centroid'
        # Filter detections based on the distance to the centroid
        best_matches_gdf = candidates_tp_gdf.groupby(['det_id'], group_keys=False).apply(lambda g:g[g.dist_centroid==g.dist_centroid.min()])
    else: 
        candidates_tp_gdf['IOU'] = [intersection_over_union(i, ii) for (i, ii) in zip(geom1, geom2)]
        resemblance_column = 'IOU'
        # Filter detections based on IoU value
        best_matches_gdf = candidates_tp_gdf.groupby(['det_id'], group_keys=False).apply(lambda g:g[g.IOU==g.IOU.max()])
        
    best_matches_gdf.drop_duplicates(subset=['det_id'], inplace=True) # <- could change the results depending on which line is dropped (but rarely effective)

    if 'tile_id_det' in best_matches_gdf.columns:
        best_matches_gdf.rename(columns={'tile_id_label': 'tile_id'}, inplace=True)
        best_matches_gdf.drop(columns=['tile_id_det'], inplace=True)

    # Detection, resp labels, with IOU lower than threshold value are considered as FP, resp FN, and saved as such
    actual_matches_gdf = best_matches_gdf[best_matches_gdf['IOU'] >= iou_threshold].copy() if resemblance_column == 'IOU' else best_matches_gdf.copy()

    # Duplicate detections of the same labels are removed too
    ascendance = False if resemblance_column == 'IOU' else True
    actual_matches_gdf = actual_matches_gdf.sort_values(by=[resemblance_column], ascending=ascendance).drop_duplicates(subset=['label_id', 'tile_id'])
    actual_matches_gdf[resemblance_column] = actual_matches_gdf[resemblance_column].round(3)

    matched_det_ids = actual_matches_gdf['det_id'].unique().tolist()
    matched_label_ids = actual_matches_gdf['label_id'].unique().tolist()
    fp_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.det_id.isin(matched_det_ids)].drop_duplicates(subset=['det_id'], ignore_index=True)
    fn_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.label_id.isin(matched_label_ids)].drop_duplicates(subset=['label_id'], ignore_index=True)
    fn_gdf_temp.loc[:, 'geometry'] = fn_gdf_temp.label_geom

    # Test that labels and detections share the same class (id starting at 1 for labels and at 0 for detections) -> !! not from source fct!!
    condition = actual_matches_gdf.label_class == actual_matches_gdf.det_class
    tp_gdf = actual_matches_gdf[condition].reset_index(drop=True)
    mismatched_classes_gdf = actual_matches_gdf[~condition].reset_index(drop=True)
    mismatched_classes_gdf.drop(columns=['x', 'y', 'z', 'dataset_label', 'label_geom'], errors='ignore', inplace=True)
    mismatched_classes_gdf.rename(columns={'dataset_det': 'dataset'}, inplace=True)


    # FALSE POSITIVES
    fp_gdf = left_join[left_join.label_id.isna()].copy()
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf = pd.concat([fp_gdf_temp, fp_gdf], ignore_index=True)
    fp_gdf.drop(
        columns=_labels_gdf.drop(columns='geometry').columns.to_list() + ['index_label', 'dataset_label', 'label_geom', resemblance_column, 'tile_id_label'], 
        errors='ignore', 
        inplace=True
    )
    fp_gdf.rename(columns={'dataset_det': 'dataset', 'tile_id_det': 'tile_id'}, inplace=True)
    
    # FALSE NEGATIVES
    right_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='right', predicate='intersects', lsuffix='det', rsuffix='label')
    fn_gdf = right_join[right_join.det_id.isna()].copy()
    try:
        fn_gdf.drop_duplicates(subset=['label_id', 'tile_id'], inplace=True)
    except KeyError:
        fn_gdf.drop_duplicates(subset=['label_id', 'tile_id_label'], inplace=True)
    fn_gdf = pd.concat([fn_gdf_temp, fn_gdf], ignore_index=True)
    fn_gdf.drop(
        columns=_dets_gdf.drop(columns='geometry').columns.to_list() + ['dataset_det', 'index_label', 'x', 'y', 'z', 'label_geom', resemblance_column, 'index_det', 'tile_id_det'], 
        errors='ignore', 
        inplace=True
    )
    fn_gdf.rename(columns={'dataset_label': 'dataset', 'tile_id_label': 'tile_id'}, inplace=True)
    
    return tp_gdf, fp_gdf, fn_gdf, mismatched_classes_gdf


def intersection_over_union(polygon1_shape, polygon2_shape):
    """Determine the intersection area over union area (IOU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection

    return polygon_intersection / polygon_union


def median_group_distance(gt, dets):
    """
    Compute the median distance between groups of a geodataframe, where a group is a set of connected
    geohashes. The function takes a geodataframe of the GT and a geodataframe of the detections as input.

    The function first adds a geohash column to the two input geodataframes, then performs a spatial join
    to get the nearest detection and GT for each detection and GT. The resulting geodataframe is then
    used to make groups of connected geohashes using NetworkX. The median distance between each group is
    then computed.

    Parameters
    ----------
    gt : geodaframe
        geodaframe of the GT
    dets : geodaframe
        geodaframe of the detections

    Returns
    -------
    median_group_dist : float
        median distance between groups
    group_med_dist_gdf : geodaframe
        geodaframe with the median distance between each group

    """

    ### --- helper functions --- ###

    def add_geohash(gdf, prefix=None, suffix=None):
        """Add geohash column to a geodaframe.

        Args:
            gdf: geodaframe
            prefix (string): custom geohash string with a chosen prefix 
            suffix (string): custom geohash string with a chosen suffix

        Returns:
            out (gdf): geodataframe with geohash column
        """

        out_gdf = gdf.copy()
        out_gdf['geohash'] = gdf.to_crs(epsg=4326).apply(geohash, axis=1)

        if prefix is not None:
            out_gdf['geohash'] = prefix + out_gdf['geohash'].astype(str)

        if suffix is not None:
            out_gdf['geohash'] = out_gdf['geohash'].astype(str) + suffix

        return out_gdf


    def make_groups():
        """Identify groups based on pairing nodes with NetworkX. The Graph is a collection of nodes.
        Nodes are hashable objects (geohash (str)).

        Returns:
            groups (list): list of connected geohash groups
        """

        g = nx.Graph()
        for row in nearest_join_gdf[nearest_join_gdf.geohash_gt.notnull()].itertuples():
            g.add_edge(row.geohash_dt, row.geohash_gt)

        groups = list(nx.connected_components(g))

        return groups


    def assign_groups(row):
        """Assign a group number to GT and detection of a geodataframe

        Args:
            row (row): geodataframe row

        Returns:
            row (row): row with a new 'group_id' column
        """

        group_index = {node: i for i, group in enumerate(groups) for node in group}
    
        row['group_id'] = group_index[row['geohash_gt']]
        # try:
        #     row['group_id'] = group_index[row['geohash']]
        # except: 
        #     row['group_id'] = None
        
        return row
    
    _gt_gdf = gt.copy()
    _dets_gdf = dets.copy()

    _gt_gdf = add_geohash(_gt_gdf, suffix='_gt')
    _gt_gdf.drop_duplicates(subset='geohash', inplace=True)
    
    _dets_gdf = add_geohash(_dets_gdf, suffix='_dt')
    _dets_gdf.drop_duplicates(subset='geohash', inplace=True)

    # spatial joins
    nearest_l_join_gdf = _gt_gdf[['objectid', 'geometry', 'geohash']].sjoin_nearest(
        _dets_gdf[['doline_id', 'geometry', 'geohash']], 
        how='left', lsuffix='gt', rsuffix='dt', distance_col='distance'
    )
    nearest_r_join_gdf = _gt_gdf[['objectid', 'geometry', 'geohash']].sjoin_nearest(
        _dets_gdf[['doline_id', 'geometry', 'geohash']], 
        how='right', lsuffix='gt', rsuffix='dt', distance_col='distance'
    )
    nearest_join_gdf = pd.concat([nearest_l_join_gdf, nearest_r_join_gdf], ignore_index=True)
    nearest_join_gdf.drop_duplicates(subset=['objectid', 'doline_id'], inplace=True)

    # Make groups with numbers
    groups = make_groups()
    nearest_join_gdf = nearest_join_gdf.apply(lambda row: assign_groups(row), axis=1)

    group_med_dist_gdf = nearest_join_gdf.copy()
    group_med_dist_gdf['group_distance'] = group_med_dist_gdf['distance']

    # Deal with larger pairs
    pair_number_df = nearest_join_gdf.groupby('group_id').size().reset_index().rename(columns={0: 'size'})
    large_group_ids = pair_number_df.loc[pair_number_df['size']>1, 'group_id'].tolist()
    for group_id in large_group_ids:
        all_dists = []
        group_gdf = nearest_join_gdf[nearest_join_gdf.group_id==group_id].copy()
        all_gt_ids = group_gdf.loc[group_gdf.group_id==group_id, 'objectid'].tolist()
        all_dt_ids = group_gdf.loc[group_gdf.group_id==group_id, 'doline_id'].tolist()

        for gt_id, dt_id in product(all_gt_ids, all_dt_ids):
            all_dists.append(distance(
                group_gdf.loc[group_gdf.objectid==gt_id, 'geometry'].iloc[0], 
                group_gdf.loc[group_gdf.doline_id==dt_id, 'geometry'].iloc[0]
            ))

        group_med_dist_gdf.loc[group_med_dist_gdf.group_id==group_id, 'group_distance'] = np.median(all_dists)

    median_group_dist = group_med_dist_gdf['group_distance'].median()

    return median_group_dist, group_med_dist_gdf.round({'distance': 2, 'group_distance': 0})