# Smoking-Prediction
An implementation of "[Enhancing Deep Learning Models for Predicting Smoking Status Using Clinical Data in Patients with COPD.](https://doi.org/10.1177/20552076251393380)", *DIGITAL HEALTH*. 2025; 11.

## Requirements
- shap
- torch
- optuna
- xgboost
- imbalanced-learn
- tab-transformer-pytorch
- numpy==1.24.4
- pandas==2.2.3
- missingpy==0.2.0
- statsmodels==0.14.4
- scikit-learn==1.1.2

## Install
```bash
# w. python 3.10.x
pip install -r requirements.txt
```

## Datasets
The experiment was conducted using survey data collected from COPD patients at the C University Hospital.

## Directories
- `models/` model code
- `tools/` Code for parameter tuning and checking the shap values

## Run
```bash
# impute
python impute.py

# parameter tuning
python tuning.py

# check the shap values
python shap.py
```

## Citation
```bash
@article{doi:10.1177/20552076251393380,
author = {Sehyun Cho and Hyeonseok Jin and Kyungbaek Kim and Sola Cho and Ja Yun Choi},
title ={Enhancing deep learning models for predicting smoking Status using clinical data in patients with chronic obstructive pulmonary disease},
journal = {DIGITAL HEALTH},
volume = {11},
number = {},
pages = {20552076251393380},
year = {2025},
doi = {10.1177/20552076251393380},
URL = {https://doi.org/10.1177/20552076251393380},
eprint = {https://doi.org/10.1177/20552076251393380},
}
```
