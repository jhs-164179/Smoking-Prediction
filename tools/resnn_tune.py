import torch
import optuna
import numpy as np
import pandas as pd
from torch import nn
import scipy.stats as st
from torch.optim import Adam
from imblearn.over_sampling import SMOTE
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler, OneHotEncoder
from ..models import FocalLoss, ResNN

class CustomDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.values).long()

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def preprocessing(df, numeric='minmax', category='label'):
    X = df.drop('BS3_1', axis=1)
    y = df['BS3_1']
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
    # uniques = []
    # for col in cat_col:
    #     uniques.append(len(X[col].unique()))
    return X, y


# def test_with_imputations(train_loader, test_loader, test_y, input_dim):
def test_with_imputations(model, train_loader, test_loader, test_y):
    class_counts = torch.tensor([test_y.value_counts()[0], test_y.value_counts()[1]])
    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()

    device = torch.device('cuda')
    # model = ResNN(input_dim=input_dim, hidden_dim=512, num_classes=2)
    model = model.to(device)
    optim = Adam(model.parameters(), lr=.0001)
    # class_counts = torch.tensor([test_y.value_counts()[0], test_y.value_counts()[1]])
    # class_weights = 1.0 / class_counts
    # class_weights /= class_weights.sum()
    
    criterion = FocalLoss(weight=class_weights.to(device))
    best_f1 = 0.0
    best_epoch = 0
    epochs=500
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xx, yy in train_loader:
            xx, yy = xx.to(device), yy.to(device)
            optim.zero_grad()
            preds = model(xx)
            loss = criterion(preds, yy)
            loss.backward()
            optim.step()
            running_loss += loss.item()
        # print(f'{epoch+1} Epoch | Loss: {running_loss/len(train_loader):.4f}')

        model.eval()
        val_loss = 0
        correct = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xx, yy in test_loader:
                xx, yy = xx.to(device), yy.to(device)
                preds = model(xx)
                val_loss = criterion(preds, yy).item()
                yy = yy.detach().cpu().numpy().squeeze()
                preds = preds.detach().cpu().numpy().squeeze()
                preds_labels = preds.argmax(axis=1)
                correct += (preds_labels == yy).sum().item()
                val_preds.extend(preds_labels.tolist())
                val_targets.extend(yy.tolist())

        val_loss /= len(test_loader)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        # print(f'{epoch+1} Epoch | TestLoss: {val_loss:.4f} | TestF1: {val_f1:.4f}')
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch+1
            torch.save(model.state_dict(), 'bestResNN.pth')

    print(f'Best Epoch: {best_epoch} | Best F1 : {best_f1:.4f}')
    return best_f1


def test_with_5fold(df, numeric, category, shuffle=True):
    f1s = []
    X, y = preprocessing(df, numeric, category)
    if shuffle:
        skf = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=shuffle)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

        train_set = CustomDataset(train_X, train_y)
        test_set = CustomDataset(test_X, test_y)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True, pin_memory=True)

        f1_value = test_with_imputations(
            train_loader, test_loader, test_y, input_dim=train_X.shape[-1]
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
    X, y = preprocessing(df, 'standard', 'label')

    scores = []
    param = {
        'hidden_dim' : trial.suggest_int('hidden_dim', 32, 1024),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        train_X, train_y = X.iloc[train_idx], y.iloc[train_idx]
        test_X, test_y = X.iloc[test_idx], y.iloc[test_idx]

        train_set = CustomDataset(train_X, train_y)
        test_set = CustomDataset(test_X, test_y)
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True, pin_memory=True)

        model = ResNN(input_dim=train_X.shape[-1], hidden_dim=param['hidden_dim'], num_classes=2)

        f1_value = test_with_imputations(
            model, train_loader, test_loader, test_y
        )
        scores.append(f1_value)

    return np.mean(scores)