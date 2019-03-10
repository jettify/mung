import pytest

from sklearn import datasets
from mung.utils import load_boston, load_adult


@pytest.fixture(scope='session')
def seed():
    return 0


@pytest.fixture(scope='session')
def boston(seed):
    return load_boston(seed)


@pytest.fixture(scope='session')
def adult(seed):
    return load_adult(seed)


@pytest.fixture(scope='session')
def iris():
    iris = datasets.load_iris()
    X = iris.data[:, [0, 2]]
    y = iris.target
    return X, y


pytest_plugins = []
