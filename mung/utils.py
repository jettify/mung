import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.compose import make_column_transformer
from sklearn.datasets import load_boston as sk_load_boston
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import shuffle


def make_advesarial_dataset(X_train, X_train_new, seed=None):
    # add warning
    # assert X_train.shape == X_train_new.shape
    y_train_new = np.zeros(X_train_new.shape[0])
    y_train = np.ones(X_train.shape[0])
    X = np.concatenate((X_train, X_train_new), axis=0)
    y = np.concatenate((y_train, y_train_new), axis=0)
    X, y = shuffle(X, y, random_state=seed)
    return X, y


def advesarial_validator(
        X_train, X_train_new, categorical_features=None, seed=None):
    X, y = make_advesarial_dataset(X_train, X_train_new, seed=seed)

    params = {
        'n_estimators': [100, 200, 300],
        'random_state': [seed],
        'learning_rate': [0.1, 0.01],
        'categorical_feature': [categorical_features],
    }
    clf = lgb.LGBMClassifier(objective='binary', metric='auc')
    grid = GridSearchCV(clf, params, scoring='roc_auc', cv=3)
    grid_result = grid.fit(X, y)
    return grid_result.best_score_


def load_boston(seed=None):
    boston = sk_load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=seed)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.6)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


def load_adult(seed=None):
    dtypes = [
        ('Age', 'float32'),
        ('Workclass', 'category'),
        ('fnlwgt', 'float32'),
        ('Education', 'category'),
        ('Education-Num', 'float32'),
        ('Marital Status', 'category'),
        ('Occupation', 'category'),
        ('Relationship', 'category'),
        ('Race', 'category'),
        ('Sex', 'category'),
        ('Capital Gain', 'float32'),
        ('Capital Loss', 'float32'),
        ('Hours per week', 'float32'),
        ('Country', 'category'),
        ('Target', 'category'),
    ]
    raw_data = pd.read_csv(
        'tests/data/adult.data',
        names=[d[0] for d in dtypes],
        na_values=['?', '  ?', ' ?'],
        dtype=dict(dtypes),
        skipinitialspace=True,
    )
    # redundant with Education-Num
    exclude = ['Education', 'fnlwgt', 'Target']
    X = raw_data.drop(exclude, axis=1)
    y = (raw_data['Target'] == '>50K').astype(int)

    cats = [d[0] for d in dtypes if d[1] == 'category' and d[0] not in exclude]
    nums = [d[0] for d in dtypes if d[1] != 'category' and d[0] not in exclude]
    pipeline = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='na'),
        OrdinalEncoder(),
    )
    X = X[nums + cats]

    transformer = make_column_transformer(
        (MinMaxScaler(), nums),
        (pipeline, cats),
        remainder='drop',
    )
    X = transformer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.6, random_state=seed)
    return X_train, X_test, y_train, y_test
