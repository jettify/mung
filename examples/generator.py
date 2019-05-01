import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mung import Munge

from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV


def load_data(seed=42):
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



def new_data(X_train, y_train):
    return gmm.sample(X_train.shape[0])[0]



def advesarial_validator(X_train, seed=42):
    p = 0.5
    s = 10
    m = Munge(p=p, s=s, seed=seed)
    m.fit(X_train)

    X_train_new = m.sample(X_train.shape[0] * 1)
    y_train_new = np.zeros(X_train_new.shape[0])

    y_train = np.ones(X_train.shape[0])

    X = np.concatenate((X_train, X_train_new), axis=0)
    y = np.concatenate((y_train, y_train_new), axis=0)
    X, y = shuffle(X, y, random_state=seed)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=2, random_state=0)
    print(p, s, cross_val_score(clf, X, y, cv=3, scoring='roc_auc'))


def advesarial_validator2(X_train, seed=42):
    for i in range(1, 10):
        for j in range(1, 10):
            n_components = i
            max_iter = j * 100
            gmm = BayesianGaussianMixture(n_components=n_components, max_iter=max_iter)
            gmm.fit(X_train)

            X_train_new = gmm.sample(X_train.shape[0] * 1)[0]
            y_train_new = np.zeros(X_train_new.shape[0])

            y_train = np.ones(X_train.shape[0])

            X = np.concatenate((X_train, X_train_new), axis=0)
            y = np.concatenate((y_train, y_train_new), axis=0)
            X, y = shuffle(X, y, random_state=seed)

            clf = RandomForestClassifier(
                n_estimators=200, max_depth=2, random_state=0)
            print(n_components, max_iter, cross_val_score(clf, X, y, cv=3, scoring='roc_auc'))




def main():
    seed = 42
    X_train, y_train, X_test, y_test = load_data(seed=seed)
    advesarial_validator2(X_train)

if __name__ == '__main__':
    main()
