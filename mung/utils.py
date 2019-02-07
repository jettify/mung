import numpy as np
import lightgbm as lgb

from sklearn.model_selection import GridSearchCV
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
