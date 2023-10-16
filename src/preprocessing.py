import os
from typing import TypeAlias

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Fixing randomness to get reproducible results
random_seed = 42
np.random.seed(random_seed)


TransformedDataset: TypeAlias = tuple[
    csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix, csr_matrix
]


def load_data(
    add_month: bool = False, split_2023: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if os.path.exists("../data/regularite-mensuelle-tgv-aqst.csv"):
        filepath = "../data/regularite-mensuelle-tgv-aqst.csv"
    elif os.path.exists("./data/regularite-mensuelle-tgv-aqst.csv"):
        filepath = "./data/regularite-mensuelle-tgv-aqst.csv"
    else:
        raise FileNotFoundError("Could not find `regularite-mensuelle-tgv-aqst.csv`.")
    data = pd.read_csv(filepath, delimiter=";")
    if add_month:
        data["month"] = data.apply(lambda row: row["date"][-2:], axis=1)
    if split_2023:
        data_before_2023 = data[data["date"] < "2023-01"]
        data_2023 = data[data["date"] >= "2023-01"]
        return data_before_2023, data_2023
    return data


def split_explicative_target(
    data: pd.DataFrame, add_month: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    explicative_columns = [
        "date",
        "service",
        "gare_depart",
        "gare_arrivee",
        "duree_moyenne",
        "nb_train_prevu",
    ]
    target_columns = [
        "retard_moyen_arrivee",
        "prct_cause_externe",
        "prct_cause_infra",
        "prct_cause_gestion_trafic",
        "prct_cause_materiel_roulant",
        "prct_cause_gestion_gare",
        "prct_cause_prise_en_charge_voyageurs",
    ]
    if add_month:
        explicative_columns.append("month")
    explicative_variables = data[explicative_columns]
    target_variables = data[target_columns]
    return explicative_variables, target_variables


@np.vectorize
def date_to_number(date: str) -> int:
    year, month = map(int, date.split("-"))
    return 12 * (year - 2000) + month - 1


@np.vectorize
def number_to_date(x: int | float) -> str:
    year, month = divmod(round(x), 12)
    return f"{year + 2000}-{month + 1}"


def get_data_transformers(
    use_month: bool = True,
) -> tuple[ColumnTransformer, ColumnTransformer]:
    x_categorical_features = ["service", "gare_depart", "gare_arrivee"]
    if use_month:
        x_categorical_features.append("month")
    x_standard_scaled_features = ["duree_moyenne", "nb_train_prevu"]
    y_standard_scaled_features = ["retard_moyen_arrivee"]
    y_percentage_features = [
        "prct_cause_externe",
        "prct_cause_infra",
        "prct_cause_gestion_trafic",
        "prct_cause_materiel_roulant",
        "prct_cause_gestion_gare",
        "prct_cause_prise_en_charge_voyageurs",
    ]
    date_encoder = FunctionTransformer(
        date_to_number, number_to_date, check_inverse=False
    )
    date_encoder = Pipeline(
        [("date_encoder", date_encoder), ("standard_scaler", StandardScaler())]
    )
    percentage_scaler = FunctionTransformer(
        lambda x: x / 100, lambda x: 100 * x, validate=True, accept_sparse=True
    )
    x_transformer = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(), x_categorical_features),
            ("standard_scaled", StandardScaler(), x_standard_scaled_features),
            ("date", date_encoder, ["date"]),
        ]
    )
    y_transformer = ColumnTransformer(
        [
            ("standard_scaled", StandardScaler(), y_standard_scaled_features),
            ("percentages", percentage_scaler, y_percentage_features),
        ]
    )
    return x_transformer, y_transformer


def embedding(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    x_2023: pd.DataFrame,
    y_2023: pd.DataFrame,
    use_month: bool = True,
    return_transformers: bool = False,
) -> (
    TransformedDataset
    | tuple[TransformedDataset, tuple[ColumnTransformer, ColumnTransformer]]
):
    x_transformer, y_transformer = get_data_transformers(use_month)
    x_train = x_transformer.fit_transform(x_train)
    y_train = y_transformer.fit_transform(y_train)
    x_test = x_transformer.transform(x_test)
    y_test = y_transformer.transform(y_test)
    x_2023 = x_transformer.transform(x_2023)
    y_2023 = y_transformer.transform(y_2023)
    transformed_dataset = x_train, y_train, x_test, y_test, x_2023, y_2023
    if return_transformers:
        return transformed_dataset, (x_transformer, y_transformer)
    return transformed_dataset


def load_and_split_train_test(
    one_hot_month: bool = True, test_size: float | int | None = 0.2
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    data_before_2023, data_2023 = load_data(one_hot_month, True)
    x_before_2023, y_before_2023 = split_explicative_target(
        data_before_2023, one_hot_month
    )
    x_2023, y_2023 = split_explicative_target(data_2023, one_hot_month)
    x_train, x_test, y_train, y_test = train_test_split(
        x_before_2023, y_before_2023, test_size=test_size
    )
    return x_train, y_train, x_test, y_test, x_2023, y_2023


def load_and_process(
    one_hot_month: bool = True,
    test_size: float | int | None = 0.2,
    return_transformers: bool = False,
) -> (
    TransformedDataset
    | tuple[TransformedDataset, tuple[ColumnTransformer, ColumnTransformer]]
):
    x_train, y_train, x_test, y_test, x_2023, y_2023 = load_and_split_train_test(
        one_hot_month, test_size
    )
    return embedding(
        x_train,
        y_train,
        x_test,
        y_test,
        x_2023,
        y_2023,
        use_month=one_hot_month,
        return_transformers=return_transformers,
    )
