import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import xgboost as xgb
from imblearn.combine import SMOTEENN


model_df = pd.read_csv('repository/model_df.csv')
model_df.drop(columns='Unnamed: 0', inplace=True)
X, y = model_df.drop(['Customer ID', 'Customer Status', 'Churn Category'], axis=1), model_df[['Customer Status','Churn Category']]

class CustomPipeline:

    def __init__(self, X, y):

        self.col = X.columns
        self.cat_cols = [col for col in X.columns if X[col].dtype == 'object']

        self.OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        trans_data = self.OH_encoder.fit_transform(X[self.cat_cols])

        encode_col = self.OH_encoder.get_feature_names_out(input_features=self.cat_cols)
        self.passthrough_col = [col for col in X.columns if col not in self.cat_cols]
        self.new_cols = list(encode_col) + self.passthrough_col

        data = np.hstack((trans_data, X[self.passthrough_col].values))
        X_enc = pd.DataFrame(data, columns=self.new_cols)

        sm = SMOTEENN(random_state=42)
        X_tgt, y_tgt = sm.fit_resample(X_enc, y)

        self.cols = X_enc.columns  # keeping the names of all columns

        self.scaler = MinMaxScaler()
        Xtgt_scaled = self.scaler.fit_transform(X_tgt)
        Xtgt_scaled = pd.DataFrame(Xtgt_scaled, columns=self.cols)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(Xtgt_scaled, y_tgt, test_size=0.2)

        model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                  colsample_bynode=1, colsample_bytree=0.5, gamma=0.2, gpu_id=-1,
                                  importance_type='gain', interaction_constraints='',
                                  learning_rate=0.1, max_delta_step=0, max_depth=8,
                                  min_child_weight=1, missing=np.nan, monotone_constraints='()',
                                  n_estimators=165, n_jobs=4, num_parallel_tree=1, random_state=0,
                                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                                  tree_method='exact', use_label_encoder=False,
                                  validate_parameters=1, verbosity=0, eval_metric='logloss')

        self.model_train = model.fit(self.X_train, self.y_train)

    def predict(self, val=None):

        if val is None:
            result = self.model_train.predict(self.X_test)
            report = classification_report(self.y_test, result)
            return print(report)

        elif isinstance(val, np.ndarray):
            print(val.shape)
            val = pd.DataFrame(val, columns=self.col)
            val_enc = pd.DataFrame(self.OH_encoder.transform(val[self.cat_cols]),
                                   columns=self.OH_encoder.get_feature_names_out(input_features=self.cat_cols))
            val_enc[self.passthrough_col] = val[self.passthrough_col]

            val_scaled = self.scaler.transform(val_enc)
            val_scaled = pd.DataFrame(val_scaled, columns=self.cols)

            result = self.model_train.predict(val_scaled)
            return result

        elif isinstance(val, pd.DataFrame):
            val_enc = pd.DataFrame(self.OH_encoder.transform(val[self.cat_cols]),
                                   columns=self.OH_encoder.get_feature_names_out(input_features=self.cat_cols))
            val_enc[self.passthrough_col] = val[self.passthrough_col]

            val_scaled = self.scaler.transform(val_enc)
            val_scaled = pd.DataFrame(val_scaled, columns=self.cols)

            result = self.model_train.predict(val_scaled)

            return result


model1 = CustomPipeline(X, y.iloc[:,0])
model2 = CustomPipeline(X, y.iloc[:,1])

