# Automated three-dimensional image registration for longitudinal photoacoustic imaging
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/pytorch-semseg/blob/master/LICENSE)

## Contents
- [About](#about)
- [Context](#context)
- [Installation](#installation)
- [Data](#data)
- [Repository](#repository)
- [Reference](#reference)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## About

This repository includes codes and notebooks used for the paper "Automated three-dimensional image registration for longitudinal photoacoustic imaging", *Journal of Biomedical Optics*, **Special Issue Pioneer in Biomedical Optics, Lihong V. Wang, 2023**

## Context

Photoacoustic tomography (PAT) has great potential in monitoring disease progression and treatment response in breast cancer. However, due to variations in breast repositioning there is a chance of geometric misalignment between images. The proposed framework involves the use of a coordinate-based neural network to represent the displacement field between pairs of PAT volumetric images. A loss function based on normalized cross correlation and Frangi vesselness feature extraction at multiple scales was implemented. We refer to our image registration framework as MUVINN-reg, which stands for Multiscale Vesselness-based Image registration using Neural Networks.

![Algorithm description](https://github.com/brunodesanti/muvinn-reg/blob/main/description.png?raw=true)

<video src="https://youtu.be/ztyOAqt72UA?si=nmm-jmKwfmRnfaqG"></video>

## Installation
First of all, make sure your environments is compatible with the requirements listed in the ```requirements.txt```.

If not, we recommend to create a new envinronment from scratch, for example, using Anaconda:
```console
conda create -n muvinn-reg-env python=3.8.0 ipython

conda activate muvinn-reg-env

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

conda install -c conda-forge monai=1.2.0

conda install matplotlib

conda install -c conda-forge tqdm

conda install -c anaconda scikit-image
```

## Data

Data is stored in an external online storage: 
[Data link](https://data.4tu.nl/)

Data is organized as Python dictionaries with the following fields:
- `dict["rec"]`: reconstruction as numpy array
- `dict["cup_size"]`: size of the cup
- `dict["cup_mask"]`: binary mask of the cup
- `dict["depth_map"]`: depth map
- `dict["metadata"]`: metadata
    - `dict["metadata"]["id_session"]`: ID of the imaging session
    - `dict["metadata"]["id_scan"]`: ID of the imaging scan
    - `dict["metadata"]["wl"]`: illumination wavelength

For data sharing regulations, we can not share PAM3 data on this repository directly. It is recommended to download PAM3 data from [Data link](https://data.4tu.nl/).

## Repository

If you want to run an example of MUVINN-reg co-registration, run ```apply_muvinn.ipynb``` 

But first remember to install the Ipython kernel to run notebooks on the muvinn-reg environment:
```console
python -m ipykernel install --user --name muvinn-reg-env --display-name "muvinn-reg-env"
```

Select muvinn-reg-env as kernel on your notebook.

Change the fixed and moving image paths in the notebook and run.

In case of errors/problems, please contact me!

### Source codes
Source codes are included in the following folders:
- ```models```:  scripts for training models 
- ```networks```: scripts for initializing networks (SIREN and ReLU)
- ```objectives```:  scripts for interpolation, normalized cross correlation and regularizers
- ```utils```:  scripts for processing, visualizing and evaluating

### Notebooks
In the notebooks folder, you will find all the notebooks used for the paper.
- ```notebooks/data```: contains notebooks to prepare dataset
- ```notebooks/evaluation```: contains notebooks to analyze performance metrics and to show registration process
- ```notebooks/experiments```: contains notebooks to run all the experiments shown in the paper
- ```notebooks/mevislab```: contains notebook to export data for MeVisLab and MeVisLab projects to visualize and annotate landmarks on PAT volumes

## Reference
If you use data, codes or part of codes of this repository please cite:

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

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
