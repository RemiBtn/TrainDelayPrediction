import numpy as np
from analyses import visualize_regression_weights, visualize_tree_nodes
from preprocessing import load_and_process
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def main() -> None:
    X_train, y_train, X_test, y_test, _, _ = load_and_process()
    # Quick test with linear regression
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_test_pred = reg.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=y_test_pred))
    print(rmse_test)

    transformed_dataset, _, feature_names = load_and_process(
        return_transformers_and_feature_names=True
    )
    X_train, y_train, X_test, y_test, _, _ = transformed_dataset
    visualize_regression_weights(X_train, y_train[:, 0], feature_names)
    visualize_tree_nodes(X_train, y_train[:, 0], feature_names)


if __name__ == "__main__":
    main()
