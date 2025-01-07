echo 'Run doline detection with the watershed method'
echo '--- Merge DEMs over AOI ---'
python scripts/merge_dem_over_aoi.py config/config_level-set.yaml
echo '--- Detect depressions ---'
python scripts/lidar_processing_libraries/level_set_depressions.py config/config_level-set.yaml
echo '--- Filter for dolines ---'
python scripts/post_processing.py config/config_level-set.yaml
echo '--- Assess results ---'
python scripts/assess_results.py config/config_level-set.yaml
echo '--- DONE! ---'
pause