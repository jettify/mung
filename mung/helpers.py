import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances_chunked
from sklearn.metrics.pairwise import check_pairwise_arrays


class ScaledFreqEncoder(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.categories = None
        self._cat_freq = {}
        self._freq_cat = {}

    def fit(self, X, y=None, categorical_features=None):
        self.categories = categorical_features or []
        for cat_id in self.categories:
            unique, counts = np.unique(X[:, cat_id], return_counts=True)
            self._cat_freq[cat_id] = {}
            self._freq_cat[cat_id] = {}
            for u, c in zip(unique, counts):
                if c in self._freq_cat[cat_id]:
                    while c in self._freq_cat[cat_id]:
                        c += 1

                self._cat_freq[cat_id][u] = c
                self._freq_cat[cat_id][c] = u

    def transform(self, X):
        X_new = X.copy()
        for cat_id, table in self._cat_freq.items():
            max_ = sum(table.values())
            for val, freq in table.items():
                X_new[X[:, cat_id] == val, cat_id] = freq / max_

        return X_new

    def inverse_transform(self, X):
        X_new = X.copy()
        for cat_id, table in self._cat_freq.items():
            freqs = np.array(list(self._cat_freq[cat_id].values()))

            max_ = sum(table.values())
            X[:, cat_id] = X[:, cat_id] * max_
            r = pairwise_distances_argmin(
                X_new[:, cat_id].reshape(-1, 1) * max_,
                freqs.reshape(-1, 1)
            )
            for freq in table.values():
                X_new[freqs[r] == freq, cat_id] = self._freq_cat[cat_id][freq]
        return X_new


def argmink(arr, k=1):
    return np.argpartition(arr, k, axis=0)[k]


def _argmin_min_reduce(dist, start):
    indices = np.apply_along_axis(argmink, 1, dist)
    values = dist[np.arange(dist.shape[0]), indices]
    return indices, values


def pairwise_distances(X, Y, axis=1, metric='euclidean', metric_kwargs=None):
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


def logit(x):
    return - np.log(1 / x - 1)
