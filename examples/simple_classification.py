import lightgbm as lgb
import numpy as np
import tensorflow as tf

from hyperopt import hp, tpe
from hyperopt.fmin import fmin
from keras import backend as K
from mung.utils import load_adult
from sklearn.metrics import log_loss, roc_auc_score


def set_global_seeds(seed=42):
    # make results reproducible, Keras does not support seeds
    # explicitly properly
    np.random.seed(seed)
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)


def fit_lgbm_model(X_train, y_train, X_test, y_test, categorical_features=None,
                   seed=None):
    num_iterations = 200
    learning_rate = 0.1
    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'num_leaves': int(params['num_leaves']),
            'max_bin': int(params['max_bin']),
            #'num_iterations': int(params['num_iterations']),
            'colsample_bytree': '{:.3f}'.format(params['colsample_bytree']),
        }

        clf = lgb.LGBMClassifier(
            boosting_type='gbdt',
            metrics=['binary_logloss'],
            num_iterations=num_iterations,
            learning_rate=learning_rate,
            #num_leaves=8,
            random_state=seed,
            **params
        )
        clf.fit(X_train, y_train, verbose=0)

        preds = clf.predict(X_test)
        lgbm_loglos = log_loss(y_test, preds)
        lgmb_auc = roc_auc_score(y_test, preds)
        print(f'LGBM: logloss: {lgbm_loglos} auc: {lgmb_auc}')
        print(params)
        return lgbm_loglos

    space = {
        'n_estimators': hp.uniform('n_estimators', 100, 2000),
        #'num_iterations': hp.uniform('num_iterations', 1000, 2000),
        'num_leaves': hp.quniform('num_leaves', 2, 256, 2),
        'max_bin': hp.quniform('max_bin', 8, 512, 2),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.1, 1.0),
    }

    #params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=40)
    #print(params)
    params = {
        'colsample_bytree': 0.5,
        'max_bin': 256,
        'n_estimators': 1444,
        'num_leaves': 16.0}
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt',
        random_state=seed,
        metrics=['binary_logloss'],
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        n_estimators=int(params['n_estimators']),
        num_leaves=int(params['num_leaves']),
        max_bin=int(params['max_bin']),
        colsample_bytree=params['colsample_bytree'],
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    lgbm_logloss = log_loss(y_test, preds)
    lgmb_auc = roc_auc_score(y_test, preds)
    print(f'LGBM: logloss: {lgbm_logloss} auc: {lgmb_auc}')
    return clf


def main():
    seed = 42
    set_global_seeds(seed=seed)

    adult = load_adult(seed, test_size=0.8)
    X_train, X_test, y_train, y_test = adult
    categorical_features = list(range(5, X_train.shape[1]))
    lgb_clf = fit_lgbm_model(
        X_train, y_train, X_test, y_test, categorical_features, seed=seed)



if __name__ == '__main__':
    main()
