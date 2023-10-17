import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

# Fixing randomness to get reproducible results
random_seed = 0
np.random.seed(random_seed)

data = pd.read_csv("./regularite-mensuelle-tgv-aqst.csv", delimiter=";")
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

X = data[explicative_columns]
y_delay = data[target_columns[0]]
y_causes = data[target_columns[1:]]

# Focus on delay for now
X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.2)

# Numeric features
numeric_features = ["duree_moyenne", "nb_train_prevu"]
num_pipeline = Pipeline([("std_scaler", StandardScaler())])

# Categorical features
categorical_features = ["service", "gare_depart", "gare_arrivee"]

# Create and apply the preprocessing pipeline

data_transformer = ColumnTransformer(
    [
        ("numerical", num_pipeline, numeric_features),
        ("date", OrdinalEncoder(), ["date"]),
        ("categorical", OneHotEncoder(), categorical_features),
    ]
)

X_train = data_transformer.fit_transform(X_train)
X_test = data_transformer.transform(X_test)


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


# Get the feature names and concatenate them
one_hot_encoder = data_transformer.named_transformers_["categorical"]
one_hot_feature_names = one_hot_encoder.get_feature_names_out(categorical_features)
numeric_feature_names = numeric_features
date_feature_name = ["date"]
all_feature_names = np.concatenate(
    [numeric_feature_names, date_feature_name, one_hot_feature_names]
)

# Get the greatest coefficients (using absolute values)
coefficients = pd.DataFrame({"feature_name": all_feature_names, "coef": np.transpose(reg.coef_)})

# Sort the DataFrame by the absolute value of coefficients in descending order
coefficients = coefficients.sort_values(by="coef", key=abs, ascending=False)

# Get the top 10 coefficients
top_10_coefficients = coefficients.head(10)

# Plot the coefficients
plt.figure(figsize=(8, 6))
plt.barh(top_10_coefficients["feature_name"], top_10_coefficients["coef"])
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title("Top 10 Features by Coefficient Magnitude")
plt.show()


# Train and plot a decision tree
tree = DecisionTreeRegressor(random_state=random_seed)
tree.fit(X_train, y_train)

plt.figure(figsize=(15, 10))
plot_tree(tree, feature_names=all_feature_names, filled=True, max_depth=3)
plt.show()
