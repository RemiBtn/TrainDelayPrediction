import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocessing import load_and_process
from scipy.sparse import csr_matrix
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree


def visualize_regression_weights(
    X_train: csr_matrix, y_train: csr_matrix, feature_names: list(str), n_coef: int = 10
) -> None:
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    coefficients = pd.DataFrame({"feature_name": feature_names, "coef": np.transpose(reg.coef_)})
    coefficients = coefficients.sort_values(by="coef", key=abs, ascending=False)
    plot_coef(coefficients, feature_names, n_coef)


def plot_coef(coefficients: list[float], features_names: list[str], n_coef: int = 10) -> None:
    top_coefficients = coefficients.head(n_coef)
    plt.figure(figsize=(8, 6))
    plt.barh(features_names, top_coefficients)
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title(f"Top {n_coef} Features by Coefficient Magnitude")
    plt.show()


def visualize_tree_nodes(
    X_train: csr_matrix, y_train: csr_matrix, feature_names: list[str], n_nodes: int = 3
) -> None:
    tree = DecisionTreeRegressor()
    tree.fit(X_train, y_train)

    plt.figure(figsize=(15, 10))
    plot_tree(tree, feature_names=feature_names, filled=True, max_depth=n_nodes)
    plt.show()
