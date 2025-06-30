echo 'Run doline detection with the stochastic method'
echo '--- Merge DEMs over AOI ---'
python scripts/merge_dem_over_aoi.py config/config_stochastic-deps.yaml
echo '--- Detect depressions ---'
python scripts/lidar_processing_libraries/wbt_stochastic_depressions.py config/config_stochastic-deps.yaml
echo '--- Filter for dolines ---'
python scripts/post_processing.py config/config_stochastic-deps.yaml
echo '--- Assess results ---'
python scripts/assess_results.py config/config_stochastic-deps.yaml
echo '--- DONE! ---'
pause