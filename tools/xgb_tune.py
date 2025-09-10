import optuna
import numpy as np
import pandas as pd
import scipy.stats as st
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler, OneHotEncoder


def preprocessing(df, num, cat, numeric, category, train=True):
    df_num, df_cat = df[num], df[cat]
    if train:
        df_num = pd.DataFrame(numeric.fit_transform(df_num), columns=df_num.columns)
        df_cat = pd.DataFrame(category.fit_transform(df_cat), columns=df_cat.columns)
    else:
        df_num = pd.DataFrame(numeric.transform(df_num), columns=df_num.columns)
        df_cat = pd.DataFrame(category.transform(df_cat), columns=df_cat.columns)
    res = pd.concat([df_num, df_cat], axis=1)
    return res


def test_with_5fold(df, numeric='minmax', category='label', shuffle=True):
    f1s = []
    X = df.drop('BS3_1', axis=1)
    y = df[['BS3_1']]
    numeric_col = [
        'FEV1', 'FEV1FVC', 'age', 'BS6_3', 'BS6_2_1', 'BD1',
        '건강문해력', 'Total_slp_wk', 'EQ_5D', 'BE3_31', 'BE5_1', '질환유병기간'
    ]
    cat_col = []
    for col in X.columns:
        if col not in numeric_col:
            cat_col.append(col)

    df_num, df_cat = X[numeric_col], X[cat_col]
    if numeric == 'minmax':
        n_pre = MinMaxScaler()
    else:
        n_pre = StandardScaler()
    df_num = pd.DataFrame(n_pre.fit_transform(df_num), columns=df_num.columns)

    if category == 'label':
        c_pre = OrdinalEncoder()
        df_cat = pd.DataFrame(c_pre.fit_transform(df_cat), columns=df_cat.columns)
    else:
        c_pre = OneHotEncoder(sparse_output=False)
        df_cat = pd.DataFrame(c_pre.fit_transform(df_cat))
        # df_cat = df_cat.astype(float)

    X = pd.concat([df_num, df_cat], axis=1)

    if shuffle:
        skf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=shuffle)
    # X.reset_index().drop('index', axis=1, inplace=True)
    # y.reset_index().drop('index', axis=1, inplace=True)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        # print(fold+1)
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]
        pos = train_y.value_counts()[0] / train_y.value_counts()[1]
        print(pos)

        model = XGBClassifier(random_state=42, n_estimators=20, max_depth=4, scale_pos_weight=pos)
        model.fit(train_X, train_y)
        preds = model.predict(test_X)
        f1_ = f1_score(test_y, preds, average='macro')
        f1s.append(f1_)
    return f1s


def get_cv_results(f1s:list):
    f1s = np.array(f1s)
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    ci95 = st.t.interval(.95, df=len(f1s)-1, loc=mean_f1, scale=std_f1/np.sqrt(len(f1s)))
    return mean_f1, std_f1, ci95


def objective(trial:optuna.Trial, df):
    X = df.drop('BS3_1', axis=1)
    y = df[['BS3_1']]
    numeric_col = [
        'FEV1', 'FEV1FVC', 'age', 'BS6_3', 'BS6_2_1', 'BD1',
        '건강문해력', 'Total_slp_wk', 'EQ_5D', 'BE3_31', 'BE5_1', '질환유병기간'
    ]
    cat_col = []
    for col in X.columns:
        if col not in numeric_col:
            cat_col.append(col)

    df_num, df_cat = X[numeric_col], X[cat_col]    
    df_num = pd.DataFrame(StandardScaler().fit_transform(df_num), columns=df_num.columns)
    df_cat = pd.DataFrame(OrdinalEncoder().fit_transform(df_cat), columns=df_cat.columns)
    X = pd.concat([df_num, df_cat], axis=1)
    pos = y.value_counts()[0] / y.value_counts()[1]

    scores = []
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 20, 300),
        'max_depth' : trial.suggest_int('max_depth', 4, 15),
        'learning_rate' : trial.suggest_float('learning_rate', .001, .01),
        'gamma' : trial.suggest_float('gamma', 0, 20),
        'alpha' : trial.suggest_float('alpha', 0, 10),
        'lambda' : trial.suggest_float('lambda', 1, 20),
        'min_child_weight' : trial.suggest_float('min_child_weight', 0, 10),
        'max_delta_step' : trial.suggest_int('max_delta_step', 10, 50),
        'subsample' : trial.suggest_float('subsample', .1, 1),
        'sampling_method' : trial.suggest_categorical('sampling_method', ['uniform']),
        'tree_method' : trial.suggest_categorical('tree_method', ['hist', 'approx']),
        'grow_policy' : trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'max_bin' : trial.suggest_int('max_bin', 256, 512),
        # 'scale_pos_weight' : trial.suggest_categorical('scale_pos_weight', [1, pos]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

        model = XGBClassifier(objective='binary:logistic', device='cuda', random_state=42, scale_pos_weight=pos, **param)    
        model.fit(train_X, train_y, verbose=False, eval_set=[(train_X, train_y),(test_X, test_y)])
        pred = model.predict(test_X)
        scores.append(f1_score(test_y, pred, average='macro'))
        
    print(f'Std: {np.std(scores):.6f}')
    return np.mean(scores)