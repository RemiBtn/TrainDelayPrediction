import numpy as np
import pandas as pd
import xgboost as xgb
from codecarbon import EmissionsTracker
from low_dimensionnal_embedding import LowDimDataset
from preprocessing import load_and_process
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

tracker = EmissionsTracker(project_name="ml_trains", measure_power_secs=10, gpu_ids=None)
emissions = dict()
rmse = dict()


def simple_model_test(x_train, y_train, x_test, y_test):
    # Quick test with linear regression
    tracker.start_task("Linear regression")
    reg = LinearRegression(fit_intercept=True)
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    rmse["linear_regression"] = rmse_test
    emissions["linear_regression"] = tracker.stop_task()

    # Quick test with KNN
    tracker.start_task("kNN")
    reg = KNeighborsRegressor(n_neighbors=75, weights="distance")
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    rmse["kNN"] = rmse_test
    emissions["kNN"] = tracker.stop_task()

    # Quick test with random forest
    tracker.start_task("Random Forest")
    reg = RandomForestRegressor(n_estimators=200, max_depth=10)
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    rmse["random_forest"] = rmse_test
    emissions["random_forest"] = tracker.stop_task()

    # Quick test with extremely randomized trees
    tracker.start_task("extremely randomized trees")
    reg = ExtraTreesRegressor(n_estimators=200, max_depth=10)
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    rmse["extremely_randomized_trees"] = rmse_test
    emissions["extremely_randomized_trees"] = tracker.stop_task()

    # Quick test with AdaBoost
    tracker.start_task("AdaBoost")
    reg = MultiOutputRegressor(AdaBoostRegressor(n_estimators=10, learning_rate=0.01))
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    rmse["adaboost"] = rmse_test
    emissions["adaboost"] = tracker.stop_task()

    # Quick test with XGBoost
    tracker.start_task("XGBoost")
    reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    reg.fit(x_train, y_train)
    y_test_pred = reg.predict(x_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    rmse["xgboost"] = rmse_test
    emissions["xgboost"] = tracker.stop_task()


def main() -> None:
    tracker.start_task("Load dataset")
    # x_train, y_train, x_test, y_test, _, _ = load_and_process()
    x_train, y_train, x_test, y_test, _, _ = LowDimDataset().get_train_test_split()
    emissions["preprocessing"] = tracker.stop_task()
    simple_model_test(x_train, y_train, x_test, y_test)

    results = []
    for model, emission in emissions.items():
        if model == "preprocessing":
            continue
        results.append({"model": model, "emission": emission.emissions, "rmse": rmse[model]})
    results = pd.DataFrame(results)
    results.to_csv("best_models_co2_emissions.csv")


if __name__ == "__main__":
    main()
