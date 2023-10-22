import matplotlib.pyplot as plt
import numpy as np
import tqdm
from sklearn.neighbors import KNeighborsRegressor

from preprocessing import load_and_process


def scaled_knn_error(data, departure_scaling, arrival_scaling):
    total_error = 0
    for x_train, y_train, x_test, y_test in data:
        x_train = x_train.copy()
        x_test = x_test.copy()
        x_train[:, 2:60] *= departure_scaling
        x_test[:, 2:60] *= departure_scaling
        x_train[:, 60:118] *= arrival_scaling
        x_test[:, 60:118] *= arrival_scaling
        dataset_error = []
        for n_neighbors in range(19, 76, 7):
            knn = KNeighborsRegressor(n_neighbors, n_jobs=-1)
            knn.fit(x_train, y_train)
            y_test_pred = knn.predict(x_test)
            error = np.sum((y_test_pred - y_test) ** 2)
            dataset_error.append(error)
        total_error += min(dataset_error) / x_test.shape[0]
    return total_error / len(data)


def same_scale():
    data = [load_and_process()[:-2] for _ in range(10)]
    scaling_factors = np.logspace(-0.8, 1, 26, base=10)
    errors = []
    for scaling in tqdm.tqdm(scaling_factors):
        error = scaled_knn_error(data, scaling, scaling)
        errors.append(error)
    plt.plot(scaling_factors, errors)
    plt.xscale("log")
    plt.show()


def main() -> None:
    same_scale()


if __name__ == "__main__":
    main()
