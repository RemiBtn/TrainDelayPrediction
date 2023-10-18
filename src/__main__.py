import numpy as np
import xgboost as xgb
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from preprocessing import load_and_process


def main() -> None:
    x_train, y_train, x_test, y_test, _, _ = load_and_process()
    # Quick test with linear regression
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(f"Linear regression: {rmse_test:.3f}")

    # Quick test with KNN
    reg = KNeighborsRegressor()
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(f"KNN: {rmse_test:.3f}")

    # Quick test with decision tree
    reg = DecisionTreeRegressor()
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(f"decision tree: {rmse_test:.3f}")

    # Quick test with random forest
    reg = RandomForestRegressor()
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(f"Random forest: {rmse_test:.3f}")

    # Quick test with extremely randomized trees
    reg = ExtraTreesRegressor().fit(x_train, y_train)
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(f"Extremely randomized trees: {rmse_test:.3f}")

    # Quick test with AdaBoost
    reg = MultiOutputRegressor(AdaBoostRegressor())
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(f"AdaBoost: {rmse_test:.3f}")

    # Quick test with XGBoost
    reg = xgb.XGBRegressor()
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(f"XGBoost: {rmse_test:.3f}")


if __name__ == "__main__":
    main()
