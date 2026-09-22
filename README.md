# Tomato-Drought-Impact-Assessment

This repository contains the implementation of Drought-Spec-Net for early tomato drought-stress detection using visible–near-infrared (Vis–NIR) spectral data. It also includes literature-informed potential yield-impact mapping and RGB mask-based visible canopy stress estimation.

The LLM-based agronomic reporting component is not included in this repository.

## Repository Contents

### Drought-Spec-Net

[`ProposedDroughtSpecNetV1.py`](https://github.com/PRISM-Research-Lab/Tomato-Drought-Impact-Assessment/blob/main/ProposedDroughtSpecNetV1.py) contains the implementation of the proposed Drought-Spec-Net model.

The script performs:

* Vis–NIR spectral-data preprocessing
* Training and evaluation of Drought-Spec-Net
* Comparison with baseline classification models
* Calculation of classification-performance metrics
* Literature-informed potential yield-impact mapping
* Generation of figures and summary results

### RGB Mask Images

The [`RGB_Mask_images`](https://github.com/PRISM-Research-Lab/Tomato-Drought-Impact-Assessment/tree/main/RGB_Mask_images) folder contains 44 three-class masks generated from individual greenhouse tomato plant images using ilastik.

The grayscale mask values represent:

* `0`: Background
* `128`: Healthy-green canopy tissue
* `255`: Visibly stressed canopy tissue

Each mask represents a different individual tomato plant from the WVSU greenhouse dataset.

### DSI Estimation

[`DSI_Estimation_from-RGM-Mask.py`](https://github.com/PRISM-Research-Lab/Tomato-Drought-Impact-Assessment/blob/main/DSI_Estimation_from-RGM-Mask.py) calculates plant-level canopy composition and the RGB-derived visible canopy stress index from the three-class masks.

For each plant image, the script calculates:

* Healthy pixel count
* Visibly stressed pixel count
* Total canopy pixel count
* Healthy fraction
* Visible canopy stress index
* Healthy and visibly stressed canopy percentages
* Summary statistics and graphical results

The visible canopy stress index is calculated as:

```text
Visible Canopy Stress Index =
Visibly Stressed Pixels / Total Canopy Pixels
```

The index ranges from 0 to 1, where values closer to 0 indicate predominantly healthy-green canopy tissue and values closer to 1 indicate a greater proportion of visibly stressed canopy tissue.

## Spectral Dataset

The public Vis–NIR spectral dataset used to train and evaluate Drought-Spec-Net is not redistributed in this repository. It can be downloaded from:

**Dataset link:** [Insert the original dataset URL here]

After downloading the dataset, update the dataset path in `ProposedDroughtSpecNetV1.py`.

The spectral dataset contains:

* 378 spectral samples
* 246 normal samples
* 132 drought-stressed samples
* 211 spectral bands
* Wavelength range of 348–1052 nm
* Label `0` for normal samples
* Label `1` for drought-stressed samples

Users should cite the original dataset publications when using these data.

## Installation

Clone the repository:

```bash
git clone https://github.com/PRISM-Research-Lab/Tomato-Drought-Impact-Assessment.git
cd Tomato-Drought-Impact-Assessment
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running Drought-Spec-Net

Run the spectral drought-detection and potential yield-impact analysis:

```bash
python ProposedDroughtSpecNetV1.py
```

Before running the script, confirm that the spectral dataset path is configured correctly.

## Running the RGB Mask Analysis

Run the visible canopy stress analysis:

```bash
python DSI_Estimation_from-RGM-Mask.py
```

Before running the script, confirm that its input path points to the `RGB_Mask_images` folder.

## Important Notes

The spectral and RGB datasets are independent and unpaired. The RGB masks were not used as inputs to Drought-Spec-Net or to validate its spectral predictions.

The potential yield-impact values are exploratory, literature-informed indicators derived from drought-classification probabilities. They are not validated predictions of actual yield loss because the spectral dataset does not contain measured plant-level yield.

The RGB-derived index represents visible canopy condition and should not be interpreted as a direct physiological measurement of drought severity.
