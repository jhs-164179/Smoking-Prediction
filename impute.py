import sys
import sklearn
import warnings
import numpy as np
import pandas as pd
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from missingpy import MissForest
from sklearn.impute import IterativeImputer, KNNImputer
from .utils import seed_everything, makeparser
from statsmodels.imputation.mice import MICEData
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings('ignore')


def main(args):
    df = pd.read_csv(args.path)
    cat_col = []
    num_col = [
        'FEV1', 'FEV1FVC', 'age', 'BS6_3', 'BS6_2_1', 'BD1',
        '건강문해력', 'Total_slp_wk', 'EQ_5D', 'BE3_31', 'BE5_1', '질환유병기간'
    ]
    for col in df.columns:
        if col not in num_col and col != 'BS3_1':
            cat_col.append(col)

    na_values = df.isna().sum()
    na_values = na_values[na_values!=0].sort_values(ascending=False)
    print('Na Values (Top-8)\n', na_values[:8])
    
    na_ratio = na_values/len(df)*100
    print('Na Ratio (Top-8)\n', na_ratio[:8])

    save_path = args.impute_save_path
    if args.impute=='fill0':    
        # fill 0
        imp0 = df.copy()
        imp0.fillna(0, inplace=True)
        imp0.to_csv(save_path, index=False)
    elif args.impute=='fillstat1':
        # fill median and mode for numeric and categorical, respectively.
        imp1 = df.copy()
        for col in imp1.columns:
            if col in num_col:
                imp1[col] = imp1[col].fillna(imp1[col].median())
            elif col in cat_col:
                imp1[col] = imp1[col].fillna(imp1[col].mode()[0])
        imp1.to_csv(save_path, index=False)
    elif args.impute=='fillstat2':
        # fill median and mode for numeric and categorical per cage, respectively.
        imp2 = df.copy()
        imp2['cage'] = imp2['age']//10
        age_groups = {age:imp2['cage']==age for age in range(4, 10)}
        for col in imp2.columns:
            if col in num_col:
                for age, mask in age_groups.items():
                    median_value = imp2.loc[mask, col].median()
                    imp2.loc[mask, col] = imp2.loc[mask, col].fillna(median_value)
            elif col in cat_col:
                for age, mask in age_groups.items():
                    mode_value = imp2.loc[mask, col].mode()
                    if not mode_value.empty:
                        imp2.loc[mask, col] = imp2.loc[mask, col].fillna(mode_value[0])
        imp2.drop('cage', axis=1, inplace=True)
        imp2.to_csv(save_path, index=False)
    elif args.impute=='mice':
        # MICE impute
        imp3 = df.copy()
        mice = MICEData(imp3)
        for i in range(10):
            mice.update_all()
        imp3 = mice.data
        imp3.to_csv(save_path, index=False)
    elif args.impute=='iterative':
        # Iterative impute
        imp4 = df.copy()
        imp4_num = imp4[num_col]
        imp4_cat = imp4[cat_col]

        num_iter = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42)
        cat_iter = IterativeImputer(estimator=RandomForestClassifier(random_state=42), random_state=42)

        imp4_num = pd.DataFrame(num_iter.fit_transform(imp4_num), columns=imp4_num.columns)
        imp4_cat = pd.DataFrame(cat_iter.fit_transform(imp4_cat), columns=imp4_cat.columns)

        imp4[num_col] = imp4_num
        imp4[cat_col] = imp4_cat

        imp4.to_csv(save_path, index=False)
    elif args.impute=='knn':
        # KNN impute
        imp5 = df.copy()

        imputer = KNNImputer(weights='distance')
        imp5 = pd.DataFrame(imputer.fit_transform(imp5), columns=imp5.columns)

        for col in imp5.columns:
            if col in cat_col:
                imp5[col] = np.round(imp5[col])

        imp5.to_csv(save_path, index=False)
    elif args.impute=='missforest':
        # Missforest impute
        imp6 = df.copy()

        imputer = MissForest(random_state=42)
        imp6 = pd.DataFrame(imputer.fit_transform(imp6), columns=imp6.columns)

        for col in imp6.columns:
            if col in cat_col:
                imp6[col] = np.round(imp6[col])

        imp6.to_csv(save_path, index=False)       
    else:
        print('plz select valid impute method')


if __name__ == "__main__":
    seed_everything()
    args = makeparser()
    main(args)