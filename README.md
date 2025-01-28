# Standardization of doline mapping in Switzerland using automatic detection methods

This repository aims to detect dolines based on the digital elevation model (DEM) of Switzerland.

Several methods were tested, they are presented here with some evaluation metrics.

**Table of content**

- [Setup](#setup)
- [Data](#data)
- [Metrics](#metrics)
- [Methods](#methods)
    - [IGN](#ign)
    - [Watersheds](#watersheds)
    - [Level-set](#level-set)
    - [Stochastic depressions](#stochastic-depressions)
- [References](#references)

## Setup

Tested on Windows 10 with python 3.11. No specific hardware requirement was identified.

Create conda environment and then, install the `lidar` library with conda as indicated below. After this first step, the other libaries can be installed with the requirement file.

```
conda create -n <env name> python=3.11
conda activate <env name>
conda install -c conda-forge mamba
mamba install -c conda-forge lidar
pip install -r requirements.txt
```

If you encounter difficulties with the `lidar` libary, please refer to the [official documentation](https://lidar.gishub.org/installation/).

## Data

All the methods were tested and optimized with the [Swiss DEM](https://www.swisstopo.admin.ch/en/height-model-swissalti3d). It has a spatial resolution of 0.5 m. The tiles necessary to cover the area of interest are downloaded and merged with the following command:

```
python scripts/download_tiles.py config/<config file>
python scripts/merge_dem_over_aoi.py <config file>
```

The results optimized for the ground truth established by an expert. They were additionally tested on the reference data, produced from polygons provided by the expert and existing datasets, namely swisstopo's products [GeoCover](https://www.swisstopo.admin.ch/en/geological-model-2d-geocover) and [swissTLM3D](https://www.swisstopo.admin.ch/en/landscape-model-swisstlm3d), as well as doline data provided by the Canton of Neuchatel.

## Metrics

The following metrics were used to assess the results:

* precision: part of the detections that are correct;
* recall: part of the reference data that is detected;
* F1 score: harmonic mean between the precision and the recall;
* F2 score: weighted harmonic mean between the precision and the recall giving more importance to the recall.

The F2 score was used to optimize the parameters of all methods.

## Methods

### IGN method

This method was used by the IGN to generalize the contour lines generation in karstic plateaus for topographic maps. It is described in Touya et al. (2019). Here we perform Step 2 of the procedure, which consists of delimiting the plateau zones and detecting the dolines within them.

<!-- Ajouter la description des scripts -->

The workflow is run with the following commands:


```
python scripts/get_slope.py config/config_ign.yaml
python scripts/define_possible_areas.py config/config_ign.yaml
python scripts/ign/doline_detection.py config/config_ign.yaml
```

A bash script can be used to run the full workflow:

```
scripts/ign/ign_process.sh
```

To determine the best parameters for the Swiss topography, the algorithm is optimized with the following command:

```
python scripts/ign/optimization_ign.py config/config_ign.yaml
```

After the optimization, the following metrics were obtained:


| **Data** | **F1 score** | **IoU for TP** | **mdist**           |
|--------------------|:------------:|:--------------:|:-------------------:|
| Reference data         |              |                |                     |
| Ground truth              |              |                |                     |

_Table 1: metrics for each type of reference data with the IGN's method._

### Watershed method

The detection of dolines through the detections of sinks in a watershed was first proposed by Obu & Podobnikar (2013). We use here the version with pre-processed DEM as presented by Telbisz et al. (2016) and used by Čonč et al. (2022).

<!-- Ajouter la description des scripts -->

The workflow is run with the following commands:

```
python scripts/watersheds/depression_detection.py config/config_watersheds.yaml
```

A bash script can be used to run the full workflow:

```
scripts/watersheds/watersheds_process.sh
```

To determine the best parameters for the Swiss topography, the algorithm is optimized with the following command:

```
python scripts/ign/optimization_watersheds.py config/config_watersheds.yaml
```

After the optimization, the following metrics were obtained:


| **Reference data** | **F1 score** | **IoU for TP** | **mdist**           |
|--------------------|:------------:|:--------------:|:-------------------:|
| _GeoCover_         |              |                |                     |
| _TLM_              |              |                |                     |

_Table 2: metrics for each type of reference data with the watershed method._

### Level-set method

<!-- to be completed-->

```
python scripts/lidar_processing_libraries/level_set_depressions.py config/config_level-set.yaml
```

A bash script can be used to run the full workflow:

```
scripts/lidar_processing_libraries/level_set_process.sh
```

To determine the best parameters for the Swiss topography, the algorithm is optimized with the following command:

``` 
python scripts/lidar_processing_libraries/optimization_level-set.py config/config_level-set.yaml
```

### Stochastic method

<!-- to be completed-->

```
python scripts/lidar_processing_libraries/wbt_stochastic_depressions.py config/config_stochastic_deps.yaml
```

A bash script can be used to run the full workflow:

```
scripts/lidar_processing_libraries/stochastic_process.sh
```

To determine the best parameters for the Swiss topography, the algorithm is optimized with the following command:

```
python scripts/lidar_processing_libraries/optimization_stochastic_deps.py config/config_stochastic_deps.yaml
```

### Post-processing

<!-- to be completed-->

Applied to all workflows except the IGN one.

```
python scripts/watersheds/post_processing.py config/<config file>
python scripts/assess_results.py config/<config file>
```

## References

Čonč, Špela, Teresa Oliveira, Ruben Portas, Rok Černe, Mateja Breg Valjavec, and Miha Krofel. ‘Dolines and Cats: Remote Detection of Karst Depressions and Their Application to Study Wild Felid Ecology’. Remote Sensing 14, no. 3 (29 January 2022): 656. https://doi.org/10.3390/rs14030656.

Obu, Jaroslav, and Tomaž Podobnikar. ‘ALGORITEM ZA PREPOZNAVANJE KRAŠKIH KOTANJ NA PODLAGI DIGITALNEGA MODELA RELIEFA = Algorithm for Karst Depression Recognition Using Digital Terrain Model’. Geodetski Vestnik 57, no. 2 (2013): 260–70.

Telbisz, Tamás, Tamás Látos, Márton Deák, Balázs Székely, Zsófia Koma, and Tibor Standovár. ‘The Advantage of Lidar Digital Terrain Models in Doline Morphometry Compared to Topographic Map Based Datasets – Aggtelek Karst (Hungary) as an Example’. Acta Carsologica 45, no. 1 (7 July 2016). https://doi.org/10.3986/ac.v45i1.4138.

Touya, Guillaume, Hugo Boulze, Anouk Schleich, and Hervé Quinquenel. ‘Contour Lines Generation in Karstic Plateaus for Topographic Maps’. Proceedings of the ICA 2 (10 July 2019): 1–8. https://doi.org/10.5194/ica-proc-2-133-2019.

Wu, Qiusheng. ‘Lidar: A Python Package for Delineating Nested Surface Depressions from Digital Elevation Data’. Journal of Open Source Software 6, no. 59 (2 March 2021): 2965. https://doi.org/10.21105/joss.02965.
