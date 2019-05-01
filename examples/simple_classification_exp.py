import random as rn
rn.seed(12345)

import numpy as np
np.random.seed(42)
from tensorflow import set_random_seed
set_random_seed(42)

import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

from keras import backend as K
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)




from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.layers.normalization import BatchNormalization
import lightgbm as lgb

from sklearn.metrics import log_loss, roc_auc_score

from mung.utils import load_adult
from mung.approx import KerasClassifictionApprox



def main():
    seed = 42
    adult = load_adult(seed)
    X_train, X_test, y_train, y_test = adult
    categorical_features = list(range(5, X_train.shape[1]))

    lgb_clf = fit_lgbm_model(X_train, y_train, categorical_features, seed=seed)
    lgbm_preds = lgb_clf.predict_proba(X_test)[:, 1]
    lgbm_loglos = log_loss(y_test, lgbm_preds)
    lgmb_auc = roc_auc_score(y_test, lgbm_preds)
    print(f'LGBM: logloss: {lgbm_loglos} auc: {lgmb_auc}')

    fit_keras_approx_model(lgb_clf,
        X_train, y_train, categorical_features,
        X_test, y_test, seed=seed)

    return
    kc = fit_keras_model(
        X_train, y_train, categorical_features,
        X_test, y_test, seed=seed)


def fit_lgbm_model(X_train, y_train, categorical_features=None, seed=42):
    # clf = lgb.LGBMClassifier()
    params = {
        'n_estimators': [100, 200, 300],
        'random_state': [seed],
        'learning_rate': [0.1, 0.05],
        'max_bin': [256, 512],
    }
    clf = GridSearchCV(lgb.LGBMClassifier(), params, cv=5)
    grid_result = clf.fit(X_train, y_train, categorical_feature=categorical_features)
    print("LGBM Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return clf


def baseline_model():
    hidden_size = 20
    model = Sequential()
    model.add(Dense(int(0.5 * hidden_size), input_dim=91, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(int(0.5 * hidden_size), kernel_initializer='normal', activation='relu'))

    model.add(BatchNormalization())
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    adam = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_crossentropy', 'accuracy'])
    model.summary()
    return model


def fit_keras_model(X_train, y_train, categorical_features, X_test, y_test, seed=42):
    from sklearn.compose import make_column_transformer

    nums = [i for i in range(X_train.shape[1]) if i not in categorical_features]
    cats = [i for i in range(X_train.shape[1]) if i in categorical_features]
    transformer = make_column_transformer(
        (MinMaxScaler(), nums),
        (OneHotEncoder(sparse=False), cats),
        remainder='drop',
    )

    X = transformer.fit_transform(X_train)

    tuned_parameters = [{
        'epochs': [256],
        'batch_size': [512],
    }]
    rgr = KerasClassifier(
        build_fn=baseline_model,
        epochs=256,
        batch_size= 1 * 2048,
        verbose=1
    )

    # rgr = GridSearchCV(nn, tuned_parameters, cv=2)
    grid_result = rgr.fit(X, y_train)
    # print("Keras Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    x = transformer.transform(X_test)
    kc_preds = rgr.predict_proba(x)[:, 1]
    kc_loglos = log_loss(y_test, kc_preds.astype(np.float64))
    kc_auc = roc_auc_score(y_test, kc_preds)
    print(f'Keras: logloss: {kc_loglos} auc: {kc_auc}')
    return rgr


def fit_keras_approx_model(clf, X_train, y_train, categorical_features, X_test, y_test, seed=42):
    import ipdb
    ipdb.set_trace()
    from sklearn.compose import make_column_transformer

    nums = [i for i in range(X_train.shape[1]) if i not in categorical_features]
    cats = [i for i in range(X_train.shape[1]) if i in categorical_features]
    transformer = make_column_transformer(
        (MinMaxScaler(), nums),
        (OneHotEncoder(sparse=False), cats),
        remainder='drop',
    )

    X = transformer.fit_transform(X_train)

    kc = KerasClassifictionApprox(
        clf,
        sample_multiplier=20,
        epochs=128,
        batch_size=512)

    import ipdb
    ipdb.set_trace()

    kc.fit(X, y_train)
    x = transformer.transform(X_test)
    kc_preds = kc.predict_proba(x)[:, 1]
    kc_loglos = log_loss(y_test, kc_preds.astype(np.float64))
    kc_auc = roc_auc_score(y_test, kc_preds)
    print(f'Keras Approx: logloss: {kc_loglos} auc: {kc_auc}')
    return kc




if __name__ == '__main__':
    main()
