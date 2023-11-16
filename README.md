## Kinodata-3D dataset and models
This repository contains a [pyg](https://pytorch-geometric.readthedocs.io/en/latest/)-based implementation
of the [Kinodata-3D dataset](add_repo_link) and the code used to train and evaluate models
presented in the [Kinodata-3D publication](add_paper_link)
![](_static/dataset_generation.png)
##
## Installation
We currently only support installation from source.
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
### 3) Obtain raw data
The raw data, docked poses and kinase pdb files, can be obtained [from Zenodo](add_zenodo_link). 
After downloading the archives extract it in the root directory of this repository.
```
cd PATH_TO_REPO
unzip ...
```
See [Kinodata-3D repo](add_repo_link) for more information and the code used to generate the raw data.

## Usage instructions
- [Kinodata-3D Pytorch Geometric Dataset](examples/pyg_dataset.ipynb)
- [Kinodata-3D data splits](examples/data_splits.ipynb)
- [Kinodata-3D affinity prediction models](examples/models.ipynb)
