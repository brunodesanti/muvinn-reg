# Automated three-dimensional image registration for longitudinal photoacoustic imaging

## Table of Contents
- [About](#about)
- [Context](#context)
- [Installation](#installation)
- [Screenshots](#screenshots)
- [Repository](#repository)
- [Reference](#reference)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## About

This repository includes data and codes used for the paper "Automated three-dimensional image registration for longitudinal photoacoustic imaging", *Journal of Biomedical Optics*, **Special Issue Pioneer in Biomedical Optics, Lihong V. Wang, 2023**

## Context

Photoacoustic tomography (PAT) has great potential in monitoring disease progression and treatment response in breast cancer. However, due to variations in breast repositioning there is a chance of geometric misalignment between images. The proposed framework involves the use of a coordinate-based neural network to represent the displacement field between pairs of PAT volumetric images. A loss function based on normalized cross correlation and Frangi vesselness feature extraction at multiple scales was implemented. We refer to our image registration framework as MUVINN-reg, which stands for Multiscale Vesselness-based Image registration using Neural Networks.

![Algorithm description](https://github.com/brunodesanti/test/blob/main/description.png?raw=true)

## Installation

It is recommended to install a virtual envinroment on Anaconda:
```console
conda create -n muvinn-reg python=3.8
conda activate muvinn-reg
```

Then clone the repository:
```console
git clone git@github.com:brunodesanti/muvinn-reg.git
```

Install all the required packages listed in the file *requirements.txt*

Try to run the notebook ```apply_muvinn.ipynb``` which will apply MUVINN-reg to co-register a pair of PAT repeated scans.

In case of problems with the installation, please report an issue!

## Repository

### Data
Data is stored in ```notebooks/data``` as Python dictionaries with the following fields:
- `dict["rec"]`: reconstruction as numpy array
- `dict["cup_size"]`: size of the cup
- `dict["cup_mask"]`: binary mask of the cup
- `dict["depth_map"]`: depth map
- `dict["metadata"]`: metadata
    - `dict["metadata"]["id_session"]`: ID of the imaging session
    - `dict["metadata"]["id_scan"]`: ID of the imaging scan
    - `dict["metadata"]["wl"]`: illumination wavelength

### Source Python codes
Python codes are included in the following folders:
- ```models```:  scripts for training models 
- ```networks```: scripts for initializing networks (SIREN and ReLU)
- ```objectives```:  scripts for interpolation, normalized cross correlation and regularizers
- ```utils```:  scripts for processing, visualizing and evaluating

### Notebooks
In the notebooks folder, you will find all the notebooks used for the paper.

- ```notebooks/experiments```: contains notebooks to run all the experiments shown in the paper
- ```notebooks/mevislab```: contains notebook to export data for MeVisLab and MeVisLab projects to visualize and annotate landmarks on PAT volumes
- ```notebooks/evaluation```: contains notebooks to analyze performance metrics and to show registration process

## Reference
If you use data, codes or part of codes in this repository please cite:

@article{desanti2023,
title={Automated three-dimensional image registration for longitudinal photoacoustic imaging},
author={De Santi, Bruno and Kim, Lucia and Bulthuis, Rianne F. G. and Lucka, Felix and Manohar, Srirang},
journal={Journal of Biomedical Optics}
volume = {},
journal = {Journal of Biomedical Optics},
number = {},
publisher = {SPIE},
pages = {},
year={2023},
doi = {},
URL = {}
}


## Acknowledgements
This work was supported by:
- REACT-EU project “Foto-akoestische mammografie naar de kliniek met de PAM3+”
- 4TU Precision Medicine program (4tu.nl/precision-medicine)
- Pioneers in Healthcare Innovation (PIHC) fund 2020 for project “Photoacoustic breast tomography: Towards monitoring of neoadjuvant chemotherapy response.”

Authors are grateful to Dr. Jelmer Wolterink for his help during the development of the algorithm. Part of the codes in this repository were adapted from his GitHub repository: https://github.com/MIAGroupUT/IDIR. 

## License

This project is licensed under the [License Name](LICENSE.md) - see the [LICENSE.md](LICENSE.md) file for details.
