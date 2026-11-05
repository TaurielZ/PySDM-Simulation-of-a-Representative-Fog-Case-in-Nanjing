# PySDM Simulation of a Representative Fog Case in Nanjing

[![DOI](https://zenodo.org/badge/1445431368.svg)](https://doi.org/10.5281/zenodo.23967680)

## Overview

This repository contains the Python scripts used to simulate a representative fog event in Nanjing, eastern China, using the Python Super-Droplet Model (PySDM).

The scripts configure and run a parcel-model simulation and produce time-dependent fog droplet size distributions (DSDs) and related microphysical diagnostics. The model setup includes condensational growth and, when enabled, collision-coalescence. This repository contains the main simulation scripts for the Nanjing case. It does not contain the separate sensitivity-experiment scripts.



### Python scripts

* `Run.py`: Main entry point. Reads the aerosol size distribution, sets the run options, launches the simulation, saves CSV output, and generates figures.
* `Settings.py`: Defines the parcel initial conditions, aerosol properties, thermodynamic parameters, updraft profile, and optional seeding settings.
* `Simulation.py`: Builds and runs the PySDM parcel model and calculates droplet spectra and associated microphysical diagnostics.



## Default model configuration

The default settings in `Run.py` represent the Nanjing fog case:

|Parameter|Default value|
|-|-:|
|Time step|0.1 s|
|Simulation duration|4.70 h|
|Initial pressure|1019.40 hPa|
|Initial temperature|273.54 K|
|Initial relative humidity|0.985|
|Parcel volume|1.0 m³|
|Minimum updraft velocity|1.0 × 10⁻⁴ m s⁻¹|
|Maximum updraft velocity|0.075 m s⁻¹|
|Updraft period|170 min|
|Initial aerosol hygroscopicity parameter (κ)|0.30|
|Seeding hygroscopicity parameter (κ)|0.8|

By default, collision-coalescence is enabled, the control case is run. These options can be changed in the user-configuration section of `Run.py`.



## Model processes and diagnostics

The model setup includes:

* Parcel thermodynamic evolution.
* Condensational growth and evaporation.
* Collision-coalescence, when enabled.

The scripts calculate and save microphysical quantities including:

* Fog droplet number concentration.
* Liquid water content (LWC).
* Volume-mean radius (Rv).
* Effective radius (Reff).
* Time-dependent droplet size distributions.



## Software requirements and installation

The scripts use PySDM and the following directly imported Python packages:

* NumPy
* pandas
* Matplotlib
* pystrict

Install the dependencies with:

```bash
python -m pip install -r requirements.txt
```

The PySDM package and its runtime dependencies are installed through the PySDM package installer. For the current installation instructions and dependency information, see the [official PySDM documentation](https://open-atmos.github.io/PySDM/).



## Citation

If you use this code, please cite the specific software release:

Zhang, J. (2026). PySDM simulation code for a representative fog case in Nanjing (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23967681

The associated manuscript is currently under review. Its full bibliographic information will be added upon publication.

Please also cite PySDM according to the citation guidance in the [official PySDM documentation](https://open-atmos.github.io/PySDM/).

## 

