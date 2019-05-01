from __future__ import print_function
print(__doc__)

import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.metrics import mean_squared_error


def main():
    cal_housing = fetch_california_housing()

    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                        cal_housing.target,
                                                        test_size=0.2,
                                                        random_state=1)

    print("Training GBRT...")
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                    learning_rate=0.1, loss='huber',
                                    random_state=1)
    clf.fit(X_train, y_train)
    gbr_mse = mean_squared_error(y_test, clf.predict(X_test))
    import ipdb
    ipdb.set_trace()
    print(gbr_mse)


# Needed on Windows because plot_partial_dependence uses multiprocessing
if __name__ == '__main__':
    main()
