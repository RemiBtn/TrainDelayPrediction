import numpy as np
import xgboost as xgb
from analyses import visualize_regression_weights, visualize_tree_nodes
from models import compare_models_with_grid_search_cv
from preprocessing import load_and_process
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def main() -> None:
    x_train, y_train, x_test, y_test, _, _ = load_and_process()

    # Define the parameter grids
    linear_reg_params = {"fit_intercept": [True, False]}
    ridge_params = {"alpha": [0.1, 1.0, 10.0]}
    lasso_params = {"alpha": [0.1, 1.0, 10.0]}
    knn_params = {
        "n_neighbors": [3, 5, 7, int(np.sqrt(x_train.shape[0]))],
        "weights": ["uniform", "distance"],
    }
    svr_params = {
        "estimator__C": [0.1, 1, 10],
        "estimator__epsilon": [0.1, 0.2, 0.3],
        "estimator__kernel": ["linear", "poly", "rbf"],
    }
    decision_tree_params = {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
    random_forest_params = {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30]}
    extra_trees_params = {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30]}
    ada_boost_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}
    xgb_params = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7, 10, 20, 30],
    }

    # Perform grid search for each model
    model_params = [
        linear_reg_params,
        ridge_params,
        lasso_params,
        knn_params,
        svr_params,
        decision_tree_params,
        random_forest_params,
        extra_trees_params,
        ada_boost_params,
        xgb_params,
    ]
    models = [
        LinearRegression(),
        Ridge(),
        Lasso(),
        KNeighborsRegressor(),
        MultiOutputRegressor(SVR()),
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        ExtraTreesRegressor(),
        AdaBoostRegressor(),
        xgb.XGBRegressor(),
    ]
    model_names = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "KNeighborsRegressor",
        "MultiOutputRegressor_SVR",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "AdaBoostRegressor",
        "XGBRegressor",
    ]

    grid_search_results = compare_models_with_grid_search_cv(
        x_train, y_train, x_test, y_test, model_params[:4], models[:4], model_names[:4]
    )
    print(grid_search_results)

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
    transformed_dataset, _, feature_names = load_and_process(
        return_transformers_and_feature_names=True
    )

    # Quick visualization
    X_train, y_train, X_test, y_test, _, _ = transformed_dataset
    visualize_regression_weights(X_train, y_train[:, 0], feature_names)
    visualize_tree_nodes(X_train, y_train[:, 0], feature_names)


if __name__ == "__main__":
    main()
