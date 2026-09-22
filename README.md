# Tomato-Drought-Impact-Assessment
# Drought-Spec-Net

Drought-Spec-Net is a hybrid one-dimensional convolutional neural network for early tomato drought-stress detection using visible–near-infrared (Vis–NIR) spectral data. This repository contains code for spectral-data analysis, drought classification, literature-informed potential yield-impact mapping, and RGB-based visible canopy stress estimation. The LLM-based reporting component is not included in this repository.

## Spectral Dataset

The public Vis–NIR spectral dataset used in this study is not redistributed in this repository. It can be downloaded from:

**Dataset link:** [Insert the original dataset URL here]

After downloading the dataset, place the CSV file in:

`data/spectral/`

The dataset should contain 211 spectral features covering wavelengths from 348 to 1052 nm and a label column named `y`, where:

* `0` represents normal plants.
* `1` represents drought-stressed plants.

Please cite the original dataset publications when using these data.

## WVSU RGB Dataset

This repository includes 44 RGB images of individual greenhouse tomato plants and their corresponding three-class ilastik masks. The mask values represent:

* `0`: Background
* `128`: Healthy-green canopy tissue
* `255`: Visibly stressed canopy tissue

The RGB images and masks are used to calculate the healthy fraction and RGB-derived visible canopy stress index.

## Installation

Clone the repository:

`git clone https://github.com/[username]/Drought-Spec-Net.git`

Move into the repository:

`cd Drought-Spec-Net`

Install the required Python packages:

`pip install -r requirements.txt`

## Running the Code

Run the spectral-data analysis:

`python spectral_model/spectral_analysis.py`

Train and evaluate Drought-Spec-Net:

`python spectral_model/train_drought_spec_net.py`

Generate the potential yield-impact results:

`python yield_impact_mapping/potential_yield_impact.py`

Calculate the RGB-derived visible canopy stress index:

`python rgb_canopy_analysis/calculate_dsi.py`

Update the dataset and image paths in the configuration file before running the scripts.

## Important Note

The potential yield-impact values are exploratory, literature-informed indicators derived from drought-classification probabilities. They are not validated predictions of actual yield loss because the spectral dataset does not contain measured plant-level yield.

