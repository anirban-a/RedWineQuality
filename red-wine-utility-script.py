# %% [code]
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import power_transform, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping


data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.rename(columns=dict(map(lambda name:(name,'_'.join(name.split(' '))), data.columns.to_list())), inplace=True)
X = data.drop(columns=['quality'], axis=1)
y = data.loc[:,'quality']


def split_outliers(df: pd.DataFrame, col:str, viz=True):
    q1,q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3-q1
    upper_tail = q3 + 1.5*iqr
    lower_tail = q1 - 1.5*iqr
    upper = df[col]>upper_tail
    lower = df[col]<lower_tail
    outliers_mask = upper|lower
    
    outliers = df.loc[outliers_mask, col]
    non_outliers = df.loc[~outliers_mask, col]
    
    if viz:
        ax=sns.scatterplot(x=outliers, y=y[outliers.index], color='red', alpha=0.65, label='Outliers')
        sns.scatterplot(x=non_outliers, y=y[non_outliers.index], color='green', alpha=0.35, ax=ax, label='Non-outliers')
        plt.legend()
        plt.show()

        fig,ax=plt.subplots(figsize=(4,5))
        oc = len(outliers.index)
        noc = len(non_outliers.index)
        sns.barplot(pd.DataFrame(
            {'outlier':[oc*100/(oc+noc)],'non_outlier':[noc*100/(oc+noc)]}, index=np.arange(1)
        ), ax=ax, palette=['#7AB','#EDA'])
        ax.bar_label(ax.containers[0],fmt='%.2f', label_type='center')
        ax.set(ylabel='percentage')
        plt.show()
    
    return (outliers, non_outliers)

class PowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='box-cox'):
        self.method = method
    def fit(self, X, y=None):
        return self
    
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            method = self.method
            if sum(X[col]<=0)>0:
                print(f'PowerTransformer: The column {col} contains non-positive values. Falling back to Yeo-Johnson Transform')
                method='yeo-johnson'
            X[col] = power_transform(X[[col]], method=method)
        return X

class MeanTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, trim=0.1):
        self.trim = trim

    def trimmed_mean(self, seq:np.array, trim=0.1):
        ss = sorted(seq)
        n = len(ss)
        trim_cnt = int(trim*n)
        return np.mean(ss[trim_cnt:-trim_cnt])
    
    def fit(self, X:pd.DataFrame, y=None):
        self.impute_col = dict()
        for col in X.columns:
            outliers,non_outliers=split_outliers(X,col,viz=False)
            self.impute_col[col]=self.trimmed_mean(non_outliers)
        return self
    def transform(self, X:pd.DataFrame):
        X = X.copy()
        for col in X.columns:
            if col in self.impute_col:
                outliers,_ = split_outliers(X, col,viz=False)
                X.loc[outliers.index, col]=self.impute_col[col]
        return X

    
    
power_transform_cols = [
    'fixed_acidity',
    'volatile_acidity',
    'citric_acid',
    'residual_sugar',
    'alcohol'
]

mean_transform_cols = [
    'chlorides',
    'free_sulfur_dioxide',
    'total_sulfur_dioxide',
    'density',
    'pH',
    'sulphates'
]


early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, mode='max')

model = keras.Sequential([
    layers.Dense(input_shape=[11], units = 121, activation='relu'),
    layers.BatchNormalization(),
#     layers.Dropout(0.3),
    layers.Dense(input_shape=[11], units = 256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(input_shape=[11], units = 256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(units = 512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(units = 512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(units=6, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

preprocessor = ColumnTransformer(transformers=[
    ('boxcox', PowerTransformer(), power_transform_cols),
    ('mean', MeanTransformer(), mean_transform_cols)
], remainder='passthrough')


pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('scaling',StandardScaler())
])

encoder = OneHotEncoder(sparse_output=False)
def one_hot_encode(y):
    return encoder.fit_transform(np.reshape(y, (-1,1)))


y = data.loc[:,'quality']
y = one_hot_encode(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)
X_train_t = pipeline.fit_transform(X_train)
X_test_t = pipeline.transform(X_test)

history = model.fit(
    X_train_t, y_train,
    batch_size=512,
    epochs=300,
    callbacks=[early_stopping],
    validation_split=0.2,
    shuffle=True,
    verbose=0)
y_pred = model.predict(X_test_t)

def transform_to_class(row):
    class_=None
    max_=-1
    for i,j in zip(row, np.arange(0,6)):
        if i>max_:
            class_=j
            max_=i
    return class_+3

y_pred_class = [transform_to_class(i) for i in y_pred]
y_test_class = [transform_to_class(i) for i in y_test]

print(f"Prediction Accuracy Score: {accuracy_score(y_test_class, y_pred_class)}")