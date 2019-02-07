from enum import Enum

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import check_pairwise_arrays

from .utils import ScaledFreqEncoder


class EncodingType(str, Enum):
    ONE_HOT = 'one_hot'
    FREQUENCY = 'frequency'


def argmink(arr, k=1):
    return np.argpartition(arr, k, axis=0)[k]


def _argmin_min_reduce(dist, start):
    indices = np.apply_along_axis(argmink, 1, dist)
    values = dist[np.arange(dist.shape[0]), indices]
    return indices, values


def pairwise_distances(X, Y, axis=1, metric="euclidean", metric_kwargs=None):
    X, Y = check_pairwise_arrays(X, Y)

    if metric_kwargs is None:
        metric_kwargs = {}

    if axis == 0:
        X, Y = Y, X

    indices, values = zip(*pairwise_distances_chunked(
        X, Y, reduce_func=_argmin_min_reduce, metric=metric,
        **metric_kwargs))
    indices = np.concatenate(indices)
    values = np.concatenate(values)
    return indices, values


class Munge:

    def __init__(self, *, p=0.8, s=1, seed=None):
        self.p: float = p
        self.s: float = s

        self.X = None
        self.categorical_features = []
        self.binary_categoricals = []
        self.seed = seed
        self.freq_enc = None

    def fit(self, X, categorical_features=None):
        """Estimate model parameters with the Munge algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        self
        """
        categorical_features = categorical_features or []
        for cat_id in categorical_features:
            if len(np.unique(X[:, cat_id])) <= 2:
                self.binary_categoricals.append(cat_id)
            else:
                self.categorical_features.append(cat_id)

        self.X = X
        if self.categorical_features:
            self.freq_enc = ScaledFreqEncoder(self.categorical_features)
            self.freq_enc.fit(X)
        return self

    def sample(self, n_samples=1):
        """Generate random samples from the fitted Gaussian distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.
        Returns
        -------
        X : array, shape (n_samples, n_features)
            Randomly generated sample
        return self.X
        """

        if self.categorical_features:
            X = self.freq_enc.transform(self.X)
        else:
            X = self.X

        binary_cats = set(self.binary_categoricals)

        rows = X.shape[0]
        replace = rows < n_samples
        num_features = X.shape[1]

        sample_rng = np.random.RandomState(new_seed(self.seed, 0))
        feature_rng = np.random.RandomState(new_seed(self.seed, 1))
        value_rng = np.random.RandomState(new_seed(self.seed, 2))

        row_ids = sample_rng.choice(rows, n_samples, replace=replace)

        sampled_data = X[row_ids, :]
        argmin, _ = pairwise_distances(
            sampled_data, X, metric='euclidean', axis=1)
        for i in range(n_samples):
            nearest_idx = argmin[i]
            nearest = X[nearest_idx]
            for feature_idx in range(num_features):
                if feature_rng.rand() < self.p:
                    if feature_idx not in binary_cats:
                        old = sampled_data[i, feature_idx]
                        new = nearest[feature_idx]
                        sd = np.abs(old - new) / self.s
                        sampled_data[i, feature_idx] = value_rng.normal(
                            new, sd)
                    else:
                        sampled_data[i, feature_idx] = new

        if self.categorical_features:
            sampled_data = self.freq_enc.inverse_transform(sampled_data)
        return sampled_data


def new_seed(seed, increment):
    if seed is None:
        return None
    return seed + increment
