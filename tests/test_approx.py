import lightgbm as lgb

from sklearn.metrics import mean_squared_error, log_loss
from sklearn.ensemble import GradientBoostingRegressor

from mung.approx import KerasRegressionApprox, KerasClassifictionApprox


def test_keras_regressor(boston, seed):
    X_train, y_train, X_test, y_test = boston
    gbr = GradientBoostingRegressor(
        n_estimators=500, max_depth=3, random_state=seed)
    gbr.fit(X_train, y_train)
    kr = KerasRegressionApprox(
        gbr, sample_multiplier=2, epochs=16, batch_size=8)

    categorical_features = None
    kr.fit(X_train, y_train, categorical_features)
    kr.predict(X_test)

    keras_approx_mse = mean_squared_error(y_test, kr.predict(X_test))
    gbr_mse = mean_squared_error(y_test, gbr.predict(X_test))
    assert gbr_mse < keras_approx_mse


def test_keras_classificator(adult, seed):
    X_train, X_test, y_train, y_test = adult
    categorical_features = list(range(5, X_train.shape[1]))

    clf = lgb.LGBMClassifier(
        objective='binary',
        learning_rate=0.01,
        metric='logloss',
        categorical_feature=categorical_features)
    clf.fit(X_train, y_train)

    kc = KerasClassifictionApprox(
        clf, sample_multiplier=2, epochs=20, batch_size=4000)
    kc.fit(X_train, y_train)

    keras_approx_logloss = log_loss(y_test, kc.predict(X_test))
    clf_logloss = log_loss(y_test, clf.predict_proba(X_test)[:, 1])
    assert clf_logloss < keras_approx_logloss
