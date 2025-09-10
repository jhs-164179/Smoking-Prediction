import torch
import optuna
import numpy as np
import pandas as pd
from torch import nn
import scipy.stats as st
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler, OneHotEncoder
from ..models import TabTransformer


class CustomDataset(Dataset):
    def __init__(self, x, y, cat_col, numeric_col):
        self.X_cat = np.array(x[cat_col]).astype(np.int32)
        self.X_num = np.array(x[numeric_col]).astype(np.float32)
        self.y = np.array(y).astype(np.float32)
        
    def __len__(self):
        return len(self.X_cat)
    
    def __getitem__(self, idx):
        X_cat = torch.from_numpy(self.X_cat[idx])
        X_num = torch.from_numpy(self.X_num[idx])
        y = torch.from_numpy(self.y[idx])
        return X_cat, X_num, y


def preprocessing(df, numeric='minmax', category='label'):
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

    X = pd.concat([df_num, df_cat], axis=1)
    uniques = []
    for col in cat_col:
        uniques.append(len(X[col].unique()))

    return X, y, uniques, numeric_col, cat_col


# def test_with_imputations(model, train_loader, test_loader, train_X, test_y, numeric_col, uniques):
def test_with_imputations(model, train_loader, test_loader, test_y):
    # combined_array = np.column_stack((train_X[numeric_col].mean().values, train_X[numeric_col].std().values))

    # class_counts = torch.tensor([test_y.value_counts()[0], test_y.value_counts()[1]])
    # class_weights = 1.0 / class_counts
    # class_weights /= class_weights.sum()

    # device = torch.device('cuda')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_negatives = (test_y == 0).sum()
    num_positives = (test_y == 1).sum()
    class_weights = torch.tensor(num_negatives / num_positives, dtype=torch.float32).to(device)    
    # model = TabTransformer(
    #     categories=tuple(uniques),
    #     num_continuous=len(numeric_col),
    #     dim=64,dim_out=1,depth=6,heads=8,attn_dropout=.1,ff_dropout=.1,mlp_hidden_mults=(4,2),
    #     mlp_act=nn.ReLU(inplace=True),
    #     continuous_mean_std=torch.tensor(combined_array, dtype=torch.float32)
    # )
    model = model.to(device)
    optim = Adam(model.parameters(), lr=.0001)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    best_f1 = 0.0
    best_epoch = 0
    epochs = 50
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for x_cat, x_num, yy in train_loader:
            optim.zero_grad()
            x_cat, x_num, yy = x_cat.to(device), x_num.to(device), yy.to(device)
            preds = model(x_cat, x_num)
            loss = criterion(preds.squeeze(), yy.squeeze())
            loss.backward()
            optim.step()
            running_loss += loss.item()
        # print(f'{epoch+1} Epoch | Loss: {running_loss/len(train_loader):.4f}')

        model.eval()
        val_loss = 0.0
        correct = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_cat, x_num, yy in test_loader:
                x_cat, x_num, yy = x_cat.to(device), x_num.to(device), yy.to(device)
                preds = model(x_cat, x_num)
                val_loss += criterion(preds.squeeze(), yy.squeeze()).item()
                yy = yy.detach().cpu().numpy().squeeze()
                preds = torch.sigmoid(preds).detach().cpu().numpy().squeeze()
                pred_labels = np.where(preds>=.5, 1, 0)
                correct += (pred_labels == yy).sum().item()
                
                val_preds.extend(pred_labels.tolist())
                val_targets.extend(yy.tolist())
        val_loss /= len(test_loader)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch+1
            torch.save(model.state_dict(), 'bestTabTransformer.pth')
            
    print(f'Best Epoch: {best_epoch} | Best F1 : {best_f1:.4f}')  
    return best_f1  


def test_with_5fold(df, numeric, category, shuffle=True):
    f1s = []
    X, y, uniques, numeric_col, cat_col = preprocessing(df, numeric, category)
    if shuffle:
        skf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=shuffle)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

        train_set = CustomDataset(train_X, train_y, cat_col, numeric_col)
        test_set = CustomDataset(test_X, test_y, cat_col, numeric_col)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True, pin_memory=True)

        f1_value = test_with_imputations(
            train_loader, test_loader, train_X, test_y, numeric_col, uniques
        )
        f1s.append(f1_value)

    return f1s


def get_cv_results(f1s:list):
    f1s = np.array(f1s)
    mean_f1 = np.mean(f1s)
    std_f1 = np.std(f1s)
    ci95 = st.t.interval(.95, df=len(f1s)-1, loc=mean_f1, scale=std_f1/np.sqrt(len(f1s)))
    return mean_f1, std_f1, ci95


def objective(trial:optuna.Trial, df):
    X, y, uniques, numeric_col, cat_col = preprocessing(df, 'standard', 'label')

    scores = []
    param = {
        'dim' : trial.suggest_int('dim', 32, 128),
        'depth' : trial.suggest_int('depth', 2, 8),
        'heads' : trial.suggest_int('heads', 2, 8),
        'dropout' : trial.suggest_float('dropout', 0, 0.5),
        'mlp_hidden_mults' : trial.suggest_categorical('mlp_hidden_mults', [(4,2),(2,1),(4,1),(2,2)]),
        'mlp_act' : trial.suggest_categorical('mlp_act', [nn.ReLU, nn.GELU, nn.LeakyReLU, nn.SiLU]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

        train_set = CustomDataset(train_X, train_y, cat_col, numeric_col)
        test_set = CustomDataset(test_X, test_y, cat_col, numeric_col)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True, pin_memory=True)

        combined_array = np.column_stack((train_X[numeric_col].mean().values, train_X[numeric_col].std().values))

        model = TabTransformer(
            categories=tuple(uniques),
            num_continuous=len(numeric_col),
            dim=param['dim'],dim_out=1,depth=param['depth'],heads=param['heads'],attn_dropout=param['dropout'],ff_dropout=param['dropout'],mlp_hidden_mults=param['mlp_hidden_mults'],
            mlp_act=param['mlp_act'](),
            continuous_mean_std=torch.tensor(combined_array, dtype=torch.float32)
        )

        f1_value = test_with_imputations(
            model, train_loader, test_loader, test_y
        )
        scores.append(f1_value)

    return np.mean(scores)