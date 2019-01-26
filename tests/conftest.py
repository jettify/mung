import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import pytest


@pytest.fixture(scope='session')
def seed():
    return 0


@pytest.fixture(scope='session')
def boston(seed):
    boston = load_boston()
    X, y = shuffle(boston.data, boston.target, random_state=seed)
    X = X.astype(np.float32)
    offset = int(X.shape[0] * 0.6)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, y_train, X_test, y_test


pytest_plugins = []
