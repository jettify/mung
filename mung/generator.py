from enum import Enum

import numpy as np

from .helpers import ScaledFreqEncoder, pairwise_distances


class EncodingType(str, Enum):
    ONE_HOT = 'one_hot'
    FREQUENCY = 'frequency'


class Munge:

    def __init__(
            self, *,
            p=0.8,
            s=1,
            binary_categoricals_ecoding=EncodingType.ONE_HOT,
            seed=None):
        self._binary_cats_ecoding = EncodingType(binary_categoricals_ecoding)
        self._validate_params(s, p)

        self.p: float = p
        self.s: float = s

        self.X = None
        self.categorical_features = []
        self.binary_categoricals = []
        self.seed = seed
        self._freq_enc = None

    def _validate_params(self, s, p):
        if p < 0.0 or p > 1.0:
            raise ValueError('Tunning parameter p should be from 0.0 to 1.0')

        if s <= 0:
            raise ValueError('Tunning parameter s should be greater then 0')

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
        is_binary = self._binary_cats_ecoding == EncodingType.ONE_HOT
        for cat_id in categorical_features:
            if is_binary and len(np.unique(X[:, cat_id])) <= 2:
                self.binary_categoricals.append(cat_id)
            else:
                self.categorical_features.append(cat_id)

        self.X = X
        if self.categorical_features:
            self._freq_enc = ScaledFreqEncoder()
            self._freq_enc.fit(
                X,
                categorical_features=self.categorical_features)
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
            X = self._freq_enc.transform(self.X)
        else:
            X = self.X

        binary_cats = set(self.binary_categoricals)

        rows = X.shape[0]
        replace = rows < n_samples
        num_features = X.shape[1]

        sample_rng = np.random.RandomState(_new_seed(self.seed, 0))
        feature_rng = np.random.RandomState(_new_seed(self.seed, 1))
        value_rng = np.random.RandomState(_new_seed(self.seed, 2))

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
            sampled_data = self._freq_enc.inverse_transform(sampled_data)
        return sampled_data


def _new_seed(seed, increment):
    if seed is None:
        return None
    return seed + increment
