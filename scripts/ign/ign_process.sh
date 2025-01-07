echo 'Run doline detection with the IGN method'
echo '--- Merge DEMs over AOI ---'
python scripts/merge_dem_over_aoi.py config/config_ign.yaml
echo '--- Determine slope ---'
python scripts/ign/determine_slope.py config/config_ign.yaml
echo '--- Define suitable areas for dolines ---'
python scripts/ign/define_possible_areas.py config/config_ign.yaml
echo '--- Detect dolines ---'
python scripts/ign/doline_detection.py config/config_ign.yaml
echo '--- Assess results ---'
python scripts/assess_results.py config/config_ign.yaml
echo '--- DONE!---'
pause