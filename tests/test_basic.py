import numpy as np

from mung import Munge, EncodingType
from mung.helpers import ScaledFreqEncoder
from mung.utils import advesarial_validator


def test_ctor(seed):
    p = 0.85
    s = 1.2
    m = Munge(p=p, s=s, seed=seed)
    assert m.p == p
    assert m.s == s


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


def test_dataset_generation_constensy(iris):
    x, _ = iris
    p = 0.90
    s = 0.15
    n_samples = 5000
    for seed in [1, 42, 128]:
        m1 = Munge(p=p, s=s, seed=seed)
        m1.fit(x)
        new_x1 = m1.sample(n_samples)

        m2 = Munge(p=p, s=s, seed=seed)
        m2.fit(x)
        new_x2 = m2.sample(n_samples)
        assert np.allclose(new_x1, new_x2)


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


def test_boston_with_advesarial_validator(seed, boston):
    p = 0.6
    s = 3
    m = Munge(p=p, s=s, seed=seed)
    X = boston[0]
    m.fit(X)

    n_samples = X.shape[0]
    X_new = m.sample(n_samples)
    score = advesarial_validator(X, X_new, seed=seed)
    assert score > 0.4 and score < 0.6, score


def test_adult_with_advesarial_validator(seed, adult):
    p = 0.4
    s = 2
    m = Munge(
        p=p,
        s=s,
        binary_categoricals_ecoding=EncodingType.FREQUENCY,
        seed=seed)
    X = adult[0]
    categorical_features = list(range(5, X.shape[1]))
    m.fit(X, categorical_features=categorical_features)

    n_samples = X.shape[0]
    X_new = m.sample(n_samples)
    score = advesarial_validator(
        X, X_new, categorical_features=categorical_features,
        seed=seed)
    assert score > 0.4 and score < 0.7, score


def test_fit_data_with_categoricals(seed, adult):
    p = 0.85
    s = 0.1
    m = Munge(p=p, s=s, seed=seed)
    x = adult[0]
    # ['Workclass', 'Marital Status', 'Occupation',
    #  'Relationship', 'Race', 'Sex', 'Country']
    categorical_features = [1, 3, 4, 5, 6, 7, 11]
    m.fit(x, categorical_features=categorical_features)

    n_samples = 10
    new_x = m.sample(n_samples)
    assert new_x.shape == (n_samples, x.shape[1])

    n_samples = 5000
    new_x = m.sample(n_samples)
    assert new_x.shape == (n_samples, x.shape[1])


def test_freq(adult, seed):
    # ['Workclass', 'Marital Status', 'Occupation',
    #  'Relationship', 'Race', 'Sex', 'Country']
    X = adult[0]
    categorical_features = [5, 6, 7, 8, 9, 10, 11]
    freq_encoder = ScaledFreqEncoder()
    freq_encoder.fit(X, categorical_features=categorical_features)
    X_encoded = freq_encoder.transform(X)
    X_reverted = freq_encoder.inverse_transform(X_encoded)
    assert np.allclose(X, X_reverted)


def test_scaled_freq_encocer(seed):
    X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [3.0], [.0]])
    encoder = ScaledFreqEncoder()
    encoder.fit(X, categorical_features=[0])
    encoded = encoder.transform(X)
    X_reverted = encoder.inverse_transform(encoded)
    assert np.allclose(X, X_reverted)
