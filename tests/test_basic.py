from mung import Munge, KerasRegressionApprox
from mung.utils import advesarial_validator
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import pytest


def test_ctor(seed):
    p = 0.85
    s = 1.2
    m = Munge(p=p, s=s, seed=seed)
    assert m.p == p
    assert m.s == s


@pytest.fixture(scope='session')
def iris():
    iris = datasets.load_iris()

    scaler = MinMaxScaler()
    features = scaler.fit_transform(iris.data)

    X = features[:, [0, 2]]
    y = iris.target
    return X, y


def test_basic_fit_sample(seed, iris):
    p = 0.85
    s = 0.1
    m = Munge(p=p, s=s, seed=seed)
    x, _ = iris
    m.fit(x)

    n_samples = 10
    new_x = m.sample(n_samples)
    assert new_x.shape == (n_samples, x.shape[1])

    n_samples = 5000
    new_x = m.sample(n_samples)
    assert new_x.shape == (n_samples, x.shape[1])


def test_iris_with_advesarial_validator(seed, iris):
    p = 0.5
    s = 0.05
    m = Munge(p=p, s=s, seed=seed)
    X, _ = iris
    m.fit(X)

    n_samples = X.shape[0]
    X_new = m.sample(n_samples)
    score = advesarial_validator(X, X_new, seed=seed)
    assert score > 0.4 and score < 0.6, score
    n_samples = 10
    new_X = m.sample(n_samples)
    assert new_X.shape == (n_samples, X.shape[1])

    n_samples = 5000
    new_X = m.sample(n_samples)
    assert new_X.shape == (n_samples, X.shape[1])


def test_keras_regressor(boston, seed):
    X_train, y_train, X_test, y_test = boston
    gbr = GradientBoostingRegressor(
        n_estimators=500, max_depth=3, random_state=seed)
    gbr.fit(X_train, y_train)

    kr = KerasRegressionApprox(
        gbr, sample_multiplier=2, epochs=16, batch_size=8)
    kr.fit(X_train, y_train)
    kr.predict(X_test)

    keras_approx_mse = mean_squared_error(y_test, kr.predict(X_test))
    gbr_mse = mean_squared_error(y_test, gbr.predict(X_test))
    assert gbr_mse < keras_approx_mse
