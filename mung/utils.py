import numpy as np
import lightgbm as lgb

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances_argmin


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
    categorical_features = categorical_features or 'auto'
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


class ScaledFreqEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, categories='auto', dtype=np.float64):
        self.categories = categories
        self.dtype = dtype
        self._cat_freq = {}
        self._freq_cat = {}

    def fit(self, X, y=None):
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
            for val, freq in table.items():
                X_new[freqs[r] == freq, cat_id] = self._freq_cat[cat_id][freq]

        return X_new
