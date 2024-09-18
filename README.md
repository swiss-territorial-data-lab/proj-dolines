# Standardization of doline mapping in Switzerland using automatic detection methods

This repository aims to detect dolines based on the digital elevation model (DEM) of Switzerland.

Several methods were tested, they are presented here with some evaluation metrics.

**Table of content**

- [Setup](#setup)
- [Data](#data)
- [Metrics](#metrics)
- [IGN](#ign)
- [Watersheds](#watersheds)
- [References](#references)

## Setup

Tested on Windows 10 with python 3.12.4. No specific requirement was identified.

Create an environment and then, install the dependencies with the following command:

```
pip install -r requirements.txt
```

## Data

All the methods were tested and optimized with the [Swiss DEM](https://www.swisstopo.admin.ch/en/height-model-swissalti3d). It has a spatial resolution of 0.5 m. The tiles necessary to cover the area of interest are downloaded with the following command:

```
python scripts/download_tiles.py config/<config file of the workflow>
```

The results were compared alternatively with the dolines in the products [GeoCover](https://www.swisstopo.admin.ch/en/geological-model-2d-geocover) and [swissTLM3D](https://www.swisstopo.admin.ch/en/landscape-model-swisstlm3d).

## Metrics

The following metrics were used to assess the results:

* f1 score: harmonic mean between the precision and the recall.
    * precision: part of the detections that are correct
    * recall: part of the reference data that is detected
* IoU of the TP: intersection over union (IoU) of the true positive (TP), _i.e._ of the matching pairs of detections and reference objects.
* mdist: median of the shortest distance between each reference object and a detection.

## IGN

This method was used by the IGN to generalize the contour lines generation in karstic plateaus for topographic maps. It is described in Touya et al. (2019). Here we perform step 2 of the procedure, which consists of delimiting the plateau zones and detecting the dolines within them.

<!-- Ajouter la description des scripts -->

The workflow is run with the following commands:

´´´
python scripts/merge_dem_over_aoi.py config/config_ign.yaml
python scripts/get_slope.py config/config_ign.yaml
python scripts/define_possible_areas.py config/config_ign.yaml
python scripts/ign/doline_detection.py config/config_ign.yaml
python scripts/assess_results.py config/config_ign.yaml
´´´

To determine the best parameters for the Swiss topography, the algorithm is optimized with the following command:

```
python scripts/ign/optimization_ign.py config/config_ign.yaml
```

After the optimization, the following metrics were obtained:


| **Reference data** | **f1 score** | **IoU for TP** | **mdist**           |
|--------------------|:------------:|:--------------:|:-------------------:|
| _GeoCover_         |              |                |                     |
| _TLM_              |              |                |                     |

_Table 1: metrics for each type of reference data with the IGN's method._

## Watersheds

The detection of dolines through the detections of sinks in a watershed was first proposed by Obu & Podobnikar (2013). We use here the version with pre-processed DEM as presented by Telbisz et al. (2016) and used by Čonč et al. (2022).

<!-- Ajouter la description des scripts -->

The workflow is run with the following commands:

```
python scripts/merge_dem_over_aoi.py config/config_watersheds.yaml
python scripts/watersheds/depression_detection.py config/config_watersheds.yaml
python scripts/watersheds/post_processing.py config/config_watersheds.yaml
python scripts/assess_results.py config/config_watersheds.yaml
```

To determine the best parameters for the Swiss topography, the algorithm is optimized with the following command:

```
python scripts/ign/optimization_ign.py config/config_ign.yaml
```

After the optimization, the following metrics were obtained:


| **Reference data** | **f1 score** | **IoU for TP** | **mdist**           |
|--------------------|:------------:|:--------------:|:-------------------:|
| _GeoCover_         |              |                |                     |
| _TLM_              |              |                |                     |

_Table 2: metrics for each type of reference data with the watershed method._

## References

Čonč, Špela, Teresa Oliveira, Ruben Portas, Rok Černe, Mateja Breg Valjavec, and Miha Krofel. ‘Dolines and Cats: Remote Detection of Karst Depressions and Their Application to Study Wild Felid Ecology’. Remote Sensing 14, no. 3 (29 January 2022): 656. https://doi.org/10.3390/rs14030656.

Obu, Jaroslav, and Tomaž Podobnikar. ‘ALGORITEM ZA PREPOZNAVANJE KRAŠKIH KOTANJ NA PODLAGI DIGITALNEGA MODELA RELIEFA = Algorithm for Karst Depression Recognition Using Digital Terrain Model’. Geodetski Vestnik 57, no. 2 (2013): 260–70.

Telbisz, Tamás, Tamás Látos, Márton Deák, Balázs Székely, Zsófia Koma, and Tibor Standovár. ‘The Advantage of Lidar Digital Terrain Models in Doline Morphometry Compared to Topographic Map Based Datasets – Aggtelek Karst (Hungary) as an Example’. Acta Carsologica 45, no. 1 (7 July 2016). https://doi.org/10.3986/ac.v45i1.4138.

Touya, Guillaume, Hugo Boulze, Anouk Schleich, and Hervé Quinquenel. ‘Contour Lines Generation in Karstic Plateaus for Topographic Maps’. Proceedings of the ICA 2 (10 July 2019): 1–8. https://doi.org/10.5194/ica-proc-2-133-2019.
