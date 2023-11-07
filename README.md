# Code for working with the Kinodata3D-docked dataset and training binding affinity prediction models
## Setup
### 1) Clone this repo
### 2) Set up Python environment
Use [mamba](https://mamba.readthedocs.io/en/latest/micromamba-installation.html#umamba-install) to set up a Python environment
```
mamba env create -f environment.yml
mamba activate kinodata
```
and install this package in editable/develop mode.
```
pip install -e .
```
### Obtain raw data
The raw data, docked poses and kinase pdb files, can be obtained [from Zenodo](todo). After downloading the archiven extract it in the
root directory of this repository
```
cd PATH_TO_REPO
unzip ...
```

