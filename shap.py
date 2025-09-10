import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .tools import xgb_shap, resnn_shap, tabtransformer_shap, fttransformer_shap
from .utils import seed_everything, makeparser


def main(args):
    df = pd.read_csv(args.impute_save_path)
    if args.model == 'xgboost':
        model, test_X = xgb_shap.return_model(df)
        explainer = shap.Explainer(model)
        shap_values = explainer(test_X)
        col_dict={
            # categorical (HE_DM | DE1_dg: 당뇨병 의사진단 여부 --> 당뇨병 유병률(Diabetes prevalence))
            'sex':'Sex', 'occp':'Occupation', 'Edu':'Education level', 'EC1_1':'EEA', '금연지지가족친구여부':'FFSQ', 
            'marri_1':'Marital status', '손자손녀동거여부':'LWG', '건강문해력':'PHS', 'HE_PFThs':'CRH', 'HE_cough1':'CL3M',
            'HE_sput1':'SP3M', '가래양상':'Sputum characteristics', 'BH9_11':'Influenza vaccination', 'DI1_dg':'PDG', 
            'DE1_dg':'Diabetes prevalence', 'DC6_dg':'PDLC', 'BP1':'PSL', 'BP5':'DS2W', 'D_1_1':'Health literacy', 'BS5_1':'PQS-1M', 
            '자기효능감':'Self-efficacy', 'N_DUSUAL':'CUFI', 'BE3_31':'WDW', 'BE5_1':'STDW', 'LQ4_05':'Activity limitation', 
            'BH1_1':'Health check-up', 'BO2_1':'BWCY', 'BO1_1':'BWChY', '질환유병기간':'COPD duration', 'DI1_pt':'Hypertension treatment',
            'DI1_2':'AMU', 'DE1_pt':'Diabetes treatment', 'DE1_3':'DMU', '심한금단증상경험여부':'SWS-E', '전문가의금연권유':'PAQ',
            # numeric
            #  'FEV1':'FEV1', 'HE_fev1fvc':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
            'FEV1':'FEV1', 'FEV1FVC':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
            'BS6_2_1':'Smoking duration', 'BD1':'Alcohol consumption', 'Total_slp_wk':'AST', 'EQ_5D':'QoL'
        }
        f_names = [col_dict[fcol] for fcol in test_X.columns]
        plt.suptitle('Shap Values (XGBoost)')
        shap.summary_plot(shap_values, test_X, feature_names=[col_dict[fcol] for fcol in test_X.columns])
        plt.show()
    elif args.model == 'resnn':
        model, test_set, test_loader = resnn_shap.forfinal(df)
        model.load_state_dict(torch.load('./bestResNN.pth'))
        def model_func(data):
            data = torch.from_numpy(data).float()
            device = torch.device('cuda')
            data = data.to(device)
            return model(data).detach().cpu().numpy()
        explainer = shap.Explainer(model_func, test_set.X.detach().cpu().numpy())
        shap_values = explainer.shap_values(test_set.X.detach().cpu().numpy())
        cols = df.drop('BS3_1', axis=1).columns
        col_dict={
            # categorical (HE_DM | DE1_dg: 당뇨병 의사진단 여부 --> 당뇨병 유병률(Diabetes prevalence))
            'sex':'Sex', 'occp':'Occupation', 'Edu':'Education level', 'EC1_1':'EEA', '금연지지가족친구여부':'FFSQ', 
            'marri_1':'Marital status', '손자손녀동거여부':'LWG', '건강문해력':'PHS', 'HE_PFThs':'CRH', 'HE_cough1':'CL3M',
            'HE_sput1':'SP3M', '가래양상':'Sputum characteristics', 'BH9_11':'Influenza vaccination', 'DI1_dg':'PDG', 
            'DE1_dg':'Diabetes prevalence', 'DC6_dg':'PDLC', 'BP1':'PSL', 'BP5':'DS2W', 'D_1_1':'Health literacy', 'BS5_1':'PQS-1M', 
            '자기효능감':'Self-efficacy', 'N_DUSUAL':'CUFI', 'BE3_31':'WDW', 'BE5_1':'STDW', 'LQ4_05':'Activity limitation', 
            'BH1_1':'Health check-up', 'BO2_1':'BWCY', 'BO1_1':'BWChY', '질환유병기간':'COPD duration', 'DI1_pt':'Hypertension treatment',
            'DI1_2':'AMU', 'DE1_pt':'Diabetes treatment', 'DE1_3':'DMU', '심한금단증상경험여부':'SWS-E', '전문가의금연권유':'PAQ',
            # numeric
            #  'FEV1':'FEV1', 'HE_fev1fvc':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
            'FEV1':'FEV1', 'FEV1FVC':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
            'BS6_2_1':'Smoking duration', 'BD1':'Alcohol consumption', 'Total_slp_wk':'AST', 'EQ_5D':'QoL'
        }

        feature_names=[col_dict[fcol] for fcol in cols]
        plt.suptitle('Shap Values (ResNN)')
        shap.summary_plot(shap_values[:,:,0], test_set.X.detach().cpu().numpy(), feature_names=feature_names)
        plt.show()
    elif args.model == 'tabtransformer':
        model, test_set, cat_col, numeric_col = tabtransformer_shap.forfinal(df)
        dd_cat = test_set.X_cat
        dd_num = test_set.X_num
        def model_func(data):
            data = torch.from_numpy(data)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x_cat, x_num = data[:, :dd_cat.shape[1]].to(device).int(), data[:, dd_cat.shape[1]:].to(device).float()
            return model(x_cat, x_num).detach().cpu().numpy()
            
        data = np.concatenate((dd_cat, dd_num), axis=1)
        explainer = shap.Explainer(model_func, data)
        shap_values = explainer.shap_values(data)
        cols = cat_col + numeric_col
        col_dict={
        # categorical (HE_DM | DE1_dg: 당뇨병 의사진단 여부 --> 당뇨병 유병률(Diabetes prevalence))
        'sex':'Sex', 'occp':'Occupation', 'Edu':'Education level', 'EC1_1':'EEA', '금연지지가족친구여부':'FFSQ', 
        'marri_1':'Marital status', '손자손녀동거여부':'LWG', '건강문해력':'PHS', 'HE_PFThs':'CRH', 'HE_cough1':'CL3M',
        'HE_sput1':'SP3M', '가래양상':'Sputum characteristics', 'BH9_11':'Influenza vaccination', 'DI1_dg':'PDG', 
        'DE1_dg':'Diabetes prevalence', 'DC6_dg':'PDLC', 'BP1':'PSL', 'BP5':'DS2W', 'D_1_1':'Health literacy', 'BS5_1':'PQS-1M', 
        '자기효능감':'Self-efficacy', 'N_DUSUAL':'CUFI', 'BE3_31':'WDW', 'BE5_1':'STDW', 'LQ4_05':'Activity limitation', 
        'BH1_1':'Health check-up', 'BO2_1':'BWCY', 'BO1_1':'BWChY', '질환유병기간':'COPD duration', 'DI1_pt':'Hypertension treatment',
        'DI1_2':'AMU', 'DE1_pt':'Diabetes treatment', 'DE1_3':'DMU', '심한금단증상경험여부':'SWS-E', '전문가의금연권유':'PAQ',
        # numeric
        #  'FEV1':'FEV1', 'HE_fev1fvc':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
        'FEV1':'FEV1', 'FEV1FVC':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
        'BS6_2_1':'Smoking duration', 'BD1':'Alcohol consumption', 'Total_slp_wk':'AST', 'EQ_5D':'QoL'
        }

        feature_names=[col_dict[fcol] for fcol in cols]
        plt.suptitle('Shap Values (TabTransformer)')
        shap.summary_plot(shap_values, data, feature_names=feature_names)
        plt.show()
    elif args.model == 'fttransformer':
        model, test_set, test_loader = fttransformer_shap.forfinal(df)
        model.load_state_dict(torch.load('./bestFT.pth'))
        dd_cat = test_set.X_cat
        dd_num = test_set.X_num
        def model_func(data):
            data = torch.from_numpy(data)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            x_cat, x_num = data[:, :dd_cat.shape[1]].to(device).int(), data[:, dd_cat.shape[1]:].to(device).float()
            return model(x_cat, x_num).detach().cpu().numpy()
            

        data = np.concatenate((dd_cat, dd_num), axis=1)
        explainer = shap.Explainer(model_func, data)
        shap_values = explainer.shap_values(data)
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

        cols = cat_col + numeric_col
        col_dict={
            # categorical (HE_DM | DE1_dg: 당뇨병 의사진단 여부 --> 당뇨병 유병률(Diabetes prevalence))
            'sex':'Sex', 'occp':'Occupation', 'Edu':'Education level', 'EC1_1':'EEA', '금연지지가족친구여부':'FFSQ', 
            'marri_1':'Marital status', '손자손녀동거여부':'LWG', '건강문해력':'PHS', 'HE_PFThs':'CRH', 'HE_cough1':'CL3M',
            'HE_sput1':'SP3M', '가래양상':'Sputum characteristics', 'BH9_11':'Influenza vaccination', 'DI1_dg':'PDG', 
            'DE1_dg':'Diabetes prevalence', 'DC6_dg':'PDLC', 'BP1':'PSL', 'BP5':'DS2W', 'D_1_1':'Health literacy', 'BS5_1':'PQS-1M', 
            '자기효능감':'Self-efficacy', 'N_DUSUAL':'CUFI', 'BE3_31':'WDW', 'BE5_1':'STDW', 'LQ4_05':'Activity limitation', 
            'BH1_1':'Health check-up', 'BO2_1':'BWCY', 'BO1_1':'BWChY', '질환유병기간':'COPD duration', 'DI1_pt':'Hypertension treatment',
            'DI1_2':'AMU', 'DE1_pt':'Diabetes treatment', 'DE1_3':'DMU', '심한금단증상경험여부':'SWS-E', '전문가의금연권유':'PAQ',
            # numeric
            #  'FEV1':'FEV1', 'HE_fev1fvc':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
            'FEV1':'FEV1', 'FEV1FVC':'FEV1/FVC', 'age':'Age', 'BS6_3':'Smoking amount',
            'BS6_2_1':'Smoking duration', 'BD1':'Alcohol consumption', 'Total_slp_wk':'AST', 'EQ_5D':'QoL'
        }

        feature_names=[col_dict[fcol] for fcol in cols]                
        plt.suptitle('Shap Values (FT-Transformer)')
        shap.summary_plot(shap_values, data, feature_names=feature_names)
        plt.show()
    else:
        print('plz select valid model')


if __name__ == '__main__':
    plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
    seed_everything()
    args = makeparser()
    main(args)