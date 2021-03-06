import numpy as np
np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import random as rn
rn.seed(12345)

import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

from mung import Munge
from mung.approx import KerasRegressionApprox
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization
import lightgbm as lgb



def main():
    seed = 42
    X_train, y_train, X_test, y_test = load_data()

    gbr = fit_gbr_model(X_train, y_train, seed=seed)
    gbr_mse = mean_squared_error(y_test, gbr.predict(X_test))
    print(f'GBR: {gbr_mse}')

    lgbm = fit_lgbm_model(X_train, y_train, seed=seed)
    lgbm_mse = mean_squared_error(y_test, lgbm.predict(X_test))
    print(f'LGBM: {lgbm_mse}')

    rf = fit_rf_model(X_train, y_train, seed=seed)
    lgbm_mse = mean_squared_error(y_test, rf.predict(X_test))
    print(f'RF: {lgbm_mse}')

    # nn_keras = fit_keras_model(X_train, y_train, seed=seed)
    # nn_keras_mse = mean_squared_error(y_test, nn_keras.predict(X_test))
    # print(f'NN Keras: {nn_keras_mse}')
    # print(f'GBR: {gbr_mse}')
    nn_keras_approx = fit_nn_keras_approx(lgbm, X_train, y_train, seed=seed)
    nn_keras_approx_mse = mean_squared_error(y_test, nn_keras_approx.predict(X_test))
    print(f'NN Keras Approx: {nn_keras_approx_mse}')

    # nn = fit_mlp_model(X_train, y_train, seed=seed)
    # nn_a = nn_approx(gbr, X_train, y_train, seed=seed)
    # nn_mse = mean_squared_error(y_test, nn.predict(X_test))
    # nn_a_mse = mean_squared_error(y_test, nn_a.predict(X_test))
    # print(f'NN: {nn_mse}')
    # print(f'NN Approx: {nn_a_mse}')



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


def fit_gbr_model(X_train, y_train, seed=42):
    tuned_parameters = [{
        'n_estimators': [100, 500],
        'max_depth': [2, 3],
        'min_samples_split': [2, 3],
        'learning_rate': [0.01, 0.005],
        'loss': ['ls'],
        'random_state': [seed]
    }]
    clf = GridSearchCV(
        GradientBoostingRegressor(), tuned_parameters, cv=5)
    grid_result = clf.fit(X_train, y_train)
    print("GBR Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return clf


def fit_rf_model(X_train, y_train, seed=42):
    clf = RandomForestRegressor()
    params = {
        'n_estimators': [100],
        'min_samples_leaf': [1, 2, 4, 8],
        'min_samples_split': [2, 5, 10],
        'random_state': [seed],
        'bootstrap': [True, False],
    }
    clf = GridSearchCV(clf, params, cv=5)
    grid_result = clf.fit(X_train, y_train)
    print("RF Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return clf


def fit_lgbm_model(X_train, y_train, seed=42):
    clf = lgb.LGBMRegressor()
    params = {
        'n_estimators': [200],
        'learning_rate': [0.1],
        'n_iter_no_change': [500],
        'max_depth': [3],
        'min_samples_split': [2],
        'random_state': [seed],
    }
    clf = GridSearchCV(
        GradientBoostingRegressor(), params, cv=5)
    grid_result = clf.fit(X_train, y_train)
    print("LGBM Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return clf


def nn_approx(clf, X_train, y_train, seed=42):
    p = 0.6
    s = 20
    m = Munge(p=p, s=s, seed=seed)
    m.fit(X_train)

    X_train_new = m.sample(X_train.shape[0] * 2)
    y_train_new = clf.predict(X_train_new)
    X = np.concatenate((X_train, X_train_new), axis=0)
    y = np.concatenate((y_train, y_train_new), axis=0)
    X, y = shuffle(X, y, random_state=seed)
    nn = fit_mlp_model(X, y)
    return nn


def fit_nn_keras_approx(clf, X_train, y_train, seed=42):
    tuned_parameters = [{
        'sample_multiplier': [40],
        'epochs': [256],
        'batch_size': [32],
        'random_state': [seed],
        'clf': [clf],
        'hidden_layer_size': [10],
    }]

    # nn = KerasRegressionApprox(clf)
    # rgr = GridSearchCV(nn, tuned_parameters, cv=2)
    # grid_result = rgr.fit(X_train, y_train)
    nn = KerasRegressionApprox(
        clf,
        p=0.9,
        s=2,
        sample_multiplier=100,
        epochs=int(256),
        batch_size=256*4,
        random_state=seed,
        hidden_layer_size=10,
    )
    nn.fit(X_train, y_train, shuffle=False)
    # print("Keras Approx Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return nn


def fit_mlp_model(X_train, y_train, seed=42):
    tuned_parameters = [
        {
            'hidden_layer_sizes': [8, 10, (10, 10)],
            'activation': ['relu'],
            'solver': ['lbfgs'],
            'alpha': [0.001],
            'batch_size': ['auto'],
            'learning_rate': ['adaptive'],
            'learning_rate_init': [0.001],
            'max_iter': [1000],
            'random_state': [seed]
        }
    ]
    rgr = GridSearchCV(MLPRegressor(), tuned_parameters, cv=2)
    grid_result = rgr.fit(X_train, y_train)
    print("MLP Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return rgr


def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    adam = keras.optimizers.Adam()
    model.compile(loss='mean_squared_error', optimizer=adam)
    # model.summary()
    return model


def fit_keras_model(X_train, y_train, seed=42):
    # estimator = KerasRegressor(build_fn=baseline_model)
    tuned_parameters = [{
        'epochs': [256, 512],
        'batch_size': [8, 16, 32],
    }]
    nn = KerasRegressor(build_fn=baseline_model, verbose=1)
    rgr = GridSearchCV(nn, tuned_parameters, cv=2)

    grid_result = rgr.fit(X_train, y_train)
    print("Keras Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return rgr



if __name__ == '__main__':
    main()
