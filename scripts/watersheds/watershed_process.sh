echo 'Run doline detection with the watershed method'
echo '--- Merge DEMs over AOI ---'
python scripts/merge_dem_over_aoi.py config/config_watersheds.yaml
echo '--- Detect depressions ---'
python scripts/watersheds/depression_detection.py config/config_watersheds.yaml
echo '--- Filter for dolines ---'
python scripts/post_processing.py config/config_watersheds.yaml
echo '--- Assess results ---'
python scripts/assess_results.py config/config_watersheds.yaml
echo '--- DONE! ---'
pause