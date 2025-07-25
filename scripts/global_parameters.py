import os
from sys import executable

AOI_TYPE = 'DRM'    # Possible types: None (i.e all types), DRM (Dépressions sur roche meuble), Evaporites, Karst nu, Marnes sur karst, Molasse, 
                            # RSVMC (Roches sédimentaires variées, à matrice calcaire)
os.environ['GDAL_DATA'] = os.path.join(f'{os.sep}'.join(executable.split(os.sep)[:-1]), 'Library', 'share', 'gdal')     # Avoid a warning
GDAL_DATA = os.environ['GDAL_DATA']

ALL_PARAMS_IGN = {
    'All types': {
        'resolution': 1, 'max_slope': 3,
        'gaussian_kernel': 25, 'gaussian_sigma': 9.5, 'dem_diff_thrsld': 0.6,
        'min_area': 25, 'limit_compactness': 0.25, 'max_voronoi_area': 35000, 'min_merged_area': 200000, 'min_long_area': 80, 'max_long_area': 2750, 'min_long_compactness': 0.19,
        'min_round_area': 480, 'min_round_compactness': 0.29, 'thalweg_buffer': 3, 'thalweg_threshold': 0.5, 'max_depth': 75
    },
    # Optimum
    'DRM': {
        'resolution': 1.25, 'max_slope': 2.4,
        'gaussian_kernel': 25, 'gaussian_sigma': 9.5, 'dem_diff_thrsld': 1,
        'min_area': 15, 'limit_compactness': 0.3, 'max_voronoi_area': 90000, 'min_merged_area': 180000, 'min_long_area': 500, 'max_long_area': 2000, 'min_long_compactness': 0.31,
        'min_round_area': 320, 'min_round_compactness': 0.38, 'thalweg_buffer': 2, 'thalweg_threshold': 1.5, 'max_depth': 150
    },
    # Optimum
    'Evaporites': {
        'resolution': 2, 'max_slope': 2,
        'gaussian_kernel': 19, 'gaussian_sigma': 9.5, 'dem_diff_thrsld': 1.1,
        'min_area': 40, 'limit_compactness': 0.35, 'max_voronoi_area': 60000, 'min_merged_area': 180000, 'min_long_area': 700, 'max_long_area': 2000, 'min_long_compactness': 0.25,
        'min_round_area': 280, 'min_round_compactness': 0.53, 'thalweg_buffer': 2.5, 'thalweg_threshold': 0.9, 'max_depth': 90,
    },
    'Karst nu': {
        'resolution': 1.5, 'max_slope': 3,
        'gaussian_kernel': 35, 'gaussian_sigma': 7, 'dem_diff_thrsld': 1.6,
        'min_area': 20, 'limit_compactness': 0.3, 'max_voronoi_area': 50000, 'min_merged_area': 80000, 'min_long_area': 340, 'max_long_area': 2250, 'min_long_compactness': 0.34,
        'min_round_area': 600, 'min_round_compactness': 0.38, 'thalweg_buffer': 1, 'thalweg_threshold': 1.2, 'max_depth': 100,
    },
    'Marnes sur karst': {
        'resolution': 2, 'max_slope': 3,
        'gaussian_kernel': 21, 'gaussian_sigma': 7, 'dem_diff_thrsld': 0.7,
        'min_area': 15, 'limit_compactness': 0.2, 'max_voronoi_area': 120000, 'min_merged_area': 200000, 'min_long_area': 240, 'max_long_area': 3000, 'min_long_compactness': 0.19,
        'min_round_area': 400, 'min_round_compactness': 0.56, 'thalweg_buffer': 4, 'thalweg_threshold': 0.5, 'max_depth': 60,
    },
    'Molasse': {
        'resolution': 0.75, 'max_slope': 1.8,
        'gaussian_kernel': 41, 'gaussian_sigma': 9, 'dem_diff_thrsld': 0.5,
        'min_area': 15, 'limit_compactness': 0.25, 'max_voronoi_area': 75000, 'min_merged_area': 160000, 'min_long_area': 120, 'max_long_area': 1000, 'min_long_compactness': 0.4,
        'min_round_area': 240, 'min_round_compactness': 0.41, 'thalweg_buffer': 1, 'thalweg_threshold': 0.5, 'max_depth': 130,
    },
    'RSVMC': {
        'resolution': 1.75, 'max_slope': 2,
        'gaussian_kernel': 33, 'gaussian_sigma': 9.5, 'dem_diff_thrsld': 1.4,
        'min_area': 15, 'limit_compactness': 0.2, 'max_voronoi_area': 125000, 'min_merged_area': 100000, 'min_long_area': 580, 'max_long_area': 1500, 'min_long_compactness': 0.19,
        'min_round_area': 560, 'min_round_compactness': 0.5, 'thalweg_buffer': 2.5, 'thalweg_threshold': 1.1, 'max_depth': 175,
    },
}
ALL_PARAMS_LEVEL_SET = {
    'All types': {
        'resolution': 1, 'min_size': 30, 'min_depth_dep': 3, 'interval': 0.85, 'area_limit': 15,
        'max_part_in_lake': 0.15, 'max_part_in_river' : 0.25, 'min_compactness': 0.25, 'min_area': 15, 'max_area': 3250, 'min_diameter': 1.5, 'min_depth': 1, 'max_depth': 35,
        'max_std_elev': 10
    },
    'DRM': {
        'resolution': 0.5, 'min_size': 16, 'min_depth_dep': 19, 'interval': 0.85, 'area_limit': 15,
        'max_part_in_lake': 0.25, 'max_part_in_river' : 0.2, 'min_compactness': 0.05, 'min_area': 45, 'max_area': 2000, 'min_diameter': 1.5, 'min_depth': 2.2, 'max_depth': 50,
        'max_std_elev': 8
    },
    'Evaporites': {
        'resolution': 0.5, 'min_size': 20, 'min_depth_dep': 11, 'interval': 1.1, 'area_limit': 60,
        'max_part_in_lake': 0.3, 'max_part_in_river' : 0.4, 'min_compactness': 0.6, 'min_area': 45, 'max_area': 3250, 'min_diameter': 5.5, 'min_depth': 2.2, 'max_depth': 70,
        'max_std_elev': 25
    },
    # Optimum
    'Karst nu': {
        'resolution': 1.5, 'min_size': 10, 'min_depth_dep': 10, 'interval': 1, 'area_limit': 50,
        'max_part_in_lake': 0.15, 'max_part_in_river' : 0.05, 'min_compactness': 0.55, 'min_area': 20, 'max_area': 3250, 'min_diameter': 4, 'min_depth': 2.2, 'max_depth': 50,
        'max_std_elev': 11
    },
    'Marnes sur karst': {
        'resolution': 0.5, 'min_size': 24, 'min_depth_dep': 18, 'interval': 0.75, 'area_limit': 125,
        'max_part_in_lake': 0.05, 'max_part_in_river' : 0.05, 'min_compactness': 0.3, 'min_area': 20, 'max_area': 2500, 'min_diameter': 6, 'min_depth': 0.60, 'max_depth': 60,
        'max_std_elev': 7
    },
    'Molasse': {
        'resolution': 0.5, 'min_size': 16, 'min_depth_dep': 17, 'interval': 0.95, 'area_limit': 80,
        'max_part_in_lake': 0.35, 'max_part_in_river' : 0.05, 'min_compactness': 0.6, 'min_area': 15, 'max_area': 2750, 'min_diameter': 1.5, 'min_depth': 0.8, 'max_depth': 140,
        'max_std_elev': 23
    },
    # Optimum
    'RSVMC': {
        'resolution': 0.5, 'min_size': 18, 'min_depth_dep': 17, 'interval': 1.15, 'area_limit': 90,
        'max_part_in_lake': 0.35, 'max_part_in_river' : 0.4, 'min_compactness': 0.6, 'min_area': 45, 'max_area': 3250, 'min_diameter': 6.5, 'min_depth': 1.6, 'max_depth': 45,
        'max_std_elev': 7
    }
}
ALL_PARAMS_WATERSHEDS = {
    'All types': {
        'resolution': 1, 'mean_filter_size': 4, 'fill_depth': 0.5,
        'max_part_in_lake': 0.35, 'max_part_in_river': 0.05, 'min_compactness': 0.2, 'min_area': 15, 'max_area': 3000, 'min_diameter': 7, 'min_depth': 0.6, 'max_depth': 170,
        'max_std_elev': 25,
    },
    'DRM': {
        'resolution': 1, 'mean_filter_size': 3, 'fill_depth': 0.5,
        'max_part_in_lake': 0.3, 'max_part_in_river': 0.3, 'min_compactness': 0.45, 'min_area': 45, 'max_area': 3250, 'min_diameter': 6, 'min_depth': 2.2, 'max_depth': 190,
        'max_std_elev': 16,
    },
    'Evaporites': {
        'resolution': 0.5, 'mean_filter_size': 0.5, 'fill_depth': 0.5,
        'max_part_in_lake': 0.35, 'max_part_in_river': 0.3, 'min_compactness': 0.4, 'min_area': 55, 'max_area': 3250, 'min_diameter': 7, 'min_depth': 2.6, 'max_depth': 120,
        'max_std_elev': 23,
    },
    'Karst nu': {
        'resolution': 1.5, 'mean_filter_size': 2, 'fill_depth': 1,
        'max_part_in_lake': 0.25, 'max_part_in_river': 0.3, 'min_compactness': 0.55, 'min_area': 25, 'max_area': 1750, 'min_diameter': 4.5, 'min_depth': 2.4, 'max_depth': 150,
        'max_std_elev': 10,
    },
    # Optimum
    'Marnes sur karst': {
        'resolution': 0.5, 'mean_filter_size': 3.75, 'fill_depth': 0.5,
        'max_part_in_lake': 0.25, 'max_part_in_river': 0.05, 'min_compactness': 0.25, 'min_area': 20, 'max_area': 3000, 'min_diameter': 5, 'min_depth': 0.6, 'max_depth': 170,
        'max_std_elev': 21,
    },
    # Optimum
    'Molasse': {
        'resolution': 0.5, 'mean_filter_size': 4.75, 'fill_depth': 0.5,
        'max_part_in_lake': 0.35, 'max_part_in_river': 0.05, 'min_compactness': 0.25, 'min_area': 10, 'max_area': 3250, 'min_diameter': 3, 'min_depth': 0.6, 'max_depth': 170,
        'max_std_elev': 22,
    },
    'RSVMC': {
        'resolution': 0.5, 'mean_filter_size': 4, 'fill_depth': 1.5,
        'max_part_in_lake': 0.05, 'max_part_in_river': 0.15, 'min_compactness': 0.3, 'min_area': 40, 'max_area': 3250, 'min_diameter': 9.5, 'min_depth': 1.2, 'max_depth': 170,
        'max_std_elev': 26,
    },
}
ALL_PARAMS_STOCHASTIC_DEPS = {
    'DRM': {
        'resolution': 0.5, 'autocorr_range': 6, 'iterations': 200, 'threshold': 0.95,
        'max_part_in_lake': 0.15, 'max_part_in_river' : 0.2, 'min_compactness': 0.4, 'min_area': 30, 'max_area': 2000, 'min_diameter': 1, 'min_depth': 1.6, 'max_depth': 125, 
        'max_std_elev': 11
    },
    'Evaporites': {
        'resolution': 1, 'autocorr_range': 10, 'iterations': 200, 'threshold': 0.8,
        'max_part_in_lake': 0.2, 'max_part_in_river' : 0.2, 'min_compactness': 0.35, 'min_area': 10, 'max_area': 1500, 'min_diameter': 1, 'min_depth': 2.2, 'max_depth': 55,
        'max_std_elev': 17
    },
    'Karst nu': {
        'resolution': 0.5, 'autocorr_range': 3, 'iterations': 150, 'threshold': 0.95,
        'max_part_in_lake': 0.15, 'max_part_in_river': 0.35, 'min_compactness': 0.5, 'min_area': 25, 'max_area': 1750, 'min_diameter': 2, 'min_depth': 2, 'max_depth': 75,
        'max_std_elev': 22
    },
    'Molasse': {
        'resolution': 0.5, 'autocorr_range': 2.5, 'iterations': 250, 'threshold': 0.75,
        'max_part_in_lake': 0.35, 'max_part_in_river': 0.1, 'min_compactness': 0.3, 'min_area': 15, 'max_area': 3250, 'min_diameter': 2.5, 'min_depth': 1, 'max_depth': 80,
        'max_std_elev': 23
    },
    'RSVMC': {
        'resolution': 1.5, 'autocorr_range': 5, 'iterations': 100, 'threshold': 0.9,
        'max_part_in_lake': 0.1, 'max_part_in_river': 0.15, 'min_compactness': 0.45, 'min_area': 20, 'max_area': 2500, 'min_diameter': 5, 'min_depth': 1.8, 'max_depth': 80,
        'max_std_elev': 5
    }
}