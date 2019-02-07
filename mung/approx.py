import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import shuffle
from keras.layers import Dense
from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasRegressor

from .generator import Munge


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


class KerasRegressionApprox(BaseEstimator, RegressorMixin):

    def __init__(
            self,
            clf=None,
            epochs=128,
            batch_size=8,
            random_state=None,
            activation='relu',
            loss='mean_squared_error',
            optimizer_params=None,
            sample_multiplier=10,
            hidden_layer_size=None,
            p=0.5,
            s=2):

        self.clf = clf
        self.model = None
        self.random_state = random_state
        self.sample_multiplier = sample_multiplier
        self.p = p
        self.s = s
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size

    def _new_data(self, X_train, y_train):
        m = Munge(p=self.p, s=self.s, seed=self.random_state)
        m.fit(X_train)

        X_train_new = m.sample(X_train.shape[0] * self.sample_multiplier)
        y_train_new = self.clf.predict(X_train_new)
        X = np.concatenate((X_train_new, X_train), axis=0)
        y = np.concatenate((y_train_new, y_train), axis=0)
        X, y = shuffle(X, y, random_state=self.random_state)
        return X, y

    def fit(self, X, y, **kw):
        X_train, y_train = self._new_data(X, y)
        input_size = X.shape[1]

        self.model = KerasRegressor(
            build_fn=make_model,
            input_size=input_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            hidden_layer_size=self.hidden_layer_size,
            verbose=1)
        self.model.fit(X_train, y_train, **kw)
        return self

    def predict(self, y, **kw):
        return self.model.predict(y, **kw)
