from model import model1
import pandas as pd

def get_dataframes():
    feat_imp = model1.model_train.feature_importances_
    feat_names = model1.model_train.feature_names_in_
    feat_df = pd.DataFrame(feat_imp, index=feat_names, columns=['Importance'])
    feat_df = feat_df.sort_values(by='Importance', ascending=True)
    feat_df['Importance'] = feat_df['Importance'].multiply(100)

    prob_df = pd.read_csv('repository/prob_df')
    prob_df.drop(columns='Unnamed: 0', inplace=True)

    return prob_df, feat_df

