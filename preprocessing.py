import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

# Fixing randomness to get reproducible results
random_seed = 0
np.random.seed(random_seed)

data = pd.read_csv('./regularite-mensuelle-tgv-aqst.csv', delimiter=';')
explicative_columns = [
    'date',
    'service',
    'gare_depart',
    'gare_arrivee',
    'duree_moyenne',
    'nb_train_prevu'
]

target_columns = [
    'retard_moyen_arrivee',
    'prct_cause_externe',
    'prct_cause_infra',
    'prct_cause_gestion_trafic',
    'prct_cause_materiel_roulant',
    'prct_cause_gestion_gare',
    'prct_cause_prise_en_charge_voyageurs'
]

X = data[explicative_columns]
y_delay = data[target_columns[0]]
y_causes = data[target_columns[1:]]

# Focus on delay for now
X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.2)

# Quick test with extremely randomized trees (no need for preprocessing)
reg = ExtraTreesRegressor(n_estimators=100).fit(X_train, y_train)
reg.score(X_test, y_test)