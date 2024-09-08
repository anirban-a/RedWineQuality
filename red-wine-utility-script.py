import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import power_transform


data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.rename(columns=dict(map(lambda name:(name,'_'.join(name.split(' '))), data.columns.to_list())), inplace=True)
X = data.drop(columns=['quality'], axis=1)
y = data.loc[:,'quality']


power_transform_cols = [
    'fixed_acidity'
]

class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            X[col] = power_transform(X[[col]], method='box-cox')
        return X

preprocessor = ColumnTransformer(transformers=[
    ('boxcox', BoxCoxTransformer(), power_transform_cols),
], remainder='passthrough')

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor)
])

pipeline.fit_transform(X,y)
