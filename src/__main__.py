import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from preprocessing import load_and_process


def main() -> None:
    X_train, y_train, X_test, y_test, _, _ = load_and_process()
    # Quick test with linear regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_test_pred = reg.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(rmse_test)

    # Quick test with extremely randomized trees
    reg = ExtraTreesRegressor(n_estimators=100).fit(X_train, y_train)
    reg.fit(X_train, y_train)
    y_test_pred = reg.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(rmse_test)


if __name__ == "__main__":
    main()
