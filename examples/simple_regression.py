import numpy as np
import lightgbm as lgb

from keras.layers import Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor
from mung import Munge
from mung.utils import load_boston

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def fit_lgbm_model(X_train, y_train, X_test, y_test, seed=None):
    params = {
        'num_boost_round': [800],
        'n_estimators': [200, 300],
        'learning_rate': [0.05],
        'max_depth': [2, 3],
        'random_state': [seed],
        'metric': ['l2'],
    }
    regressor = lgb.LGBMRegressor()
    clf = GridSearchCV(regressor, params, cv=5)
    grid_result = clf.fit(X_train, y_train)
    score_ = grid_result.best_score_
    prams_ = grid_result.best_params_
    print("LGBM Best: {} using {}".format(score_, prams_))
    lgbm_mse = mean_squared_error(y_test, clf.predict(X_test))
    print(f'LGBM MSE: {lgbm_mse}')
    return clf.best_estimator_


def make_model(
        input_size=None,
        activation='relu',
        loss='mean_squared_error',
        optimizer_params=None,
        hidden_layer_size=None,
        seed=42):

    hidden_size = hidden_layer_size or int(input_size * 0.75)
    model = Sequential()
    model.add(Dense(
        hidden_size,
        input_dim=input_size,
        kernel_initializer='uniform',
        activation=activation,
        use_bias=False,
    ))
    model.add(BatchNormalization())

    model.add(Dense(
        hidden_size,
        kernel_initializer='uniform',
        activation=activation,
        use_bias=False,
    ))
    model.add(BatchNormalization())

    model.add(Dense(1, kernel_initializer='uniform'))

    opt = SGD(lr=0.0005, momentum=0.0, decay=0.0, nesterov=True)
    model.compile(loss=loss, optimizer=opt)
    model.summary()
    return model


def fit_approx_keras_model(X_train, y_train, X_test, y_test, clf, seed=None):
    p = 0.9
    s = 2.0
    m = Munge(p=p, s=s, seed=seed)
    m.fit(X_train)

    X_train_new = m.sample(X_train.shape[0] * 50)
    y_train_new = clf.predict(X_train_new)

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate((X_train, X_train_new)))

    X_train = scaler.transform(X_train)
    X_train_new = scaler.transform(X_train_new)
    X_test = scaler.transform(X_test)

    kr = KerasRegressor(
        build_fn=make_model,
        input_size=X_train_new.shape[1],
        epochs=256,
        batch_size=200,
        hidden_layer_size=12,
    )
    kr.fit(X_train_new, y_train_new)
    kr.fit(X_train, y_train)

    keras_mse = mean_squared_error(y_test, kr.predict(X_test))
    print(f'Keras MSE: {keras_mse}')
    return kr


def main():
    seed = 42
    X_train, y_train, X_test, y_test = load_boston(seed=seed)
    clf = fit_lgbm_model(X_train, y_train, X_test, y_test, seed=seed)
    kra = fit_approx_keras_model(X_train, y_train, X_test, y_test, clf, seed=seed)
    assert (clf, kr)


if __name__ == '__main__':
    main()
