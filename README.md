# Smoking-Prediction
An implementation of "Enhancing Deep Learning Models for Predicting Smoking Status Using Clinical Data in Patients with COPD.", [*Digital Health*](https://journals.sagepub.com/home/DHJ) [accepted]

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
TBA
