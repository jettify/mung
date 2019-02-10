import pytest

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


pytest_plugins = []
