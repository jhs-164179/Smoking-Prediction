import os
import torch
import random
import argparse
import numpy as np


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def makeparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./survey.csv', help="Path for csv file")
    parser.add_argument('--impute', type=str, default='fill0', help="Select impute methods (fill0, fillstat1, fillstat2, mice, iterative, knn, missforest)")
    parser.add_argument('--impute_save_path', type=str, default='./imputed_file.csv', help="Path for imputed csv file")
    parser.add_argument('--model', type=str, default='xgboost', help="Select models (xgboost, resnn, tabtransformer, fttransformer)")
    args = parser.parse_args()
    return args