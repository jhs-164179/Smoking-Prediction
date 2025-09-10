import optuna
import pandas as pd
from .tools import xgb_tune, resnn_tune, tabtransformer_tune, fttransformer_tune
from .utils import seed_everything, makeparser


def main(args):
    df = pd.read_csv(args.impute_save_path)
    study = optuna.create_study(direction='maximize')
    if args.model == 'xgboost':
        study.optimize(lambda trial:xgb_tune.objective(trial, df), n_trials=100)
    elif args.model == 'resnn':
        study.optimize(lambda trial:resnn_tune.objective(trial, df), n_trials=100)
    elif args.model == 'tabtransformer':
        study.optimize(lambda trial:tabtransformer_tune.objective(trial, df), n_trials=100)
    elif args.model == 'fttransformer':
        study.optimize(lambda trial:fttransformer_tune.objective(trial, df), n_trials=100)
    else:
        print('plz select valid model')
    print(f'Best trial: {study.best_trial}')
    print(f'Best value: {study.best_value}')
    print(f'Best params: {study.best_params}')

if __name__ == '__main__':
    seed_everything()
    args = makeparser()
    main(args)