import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, category_col, proxy_target_col, m=10):
        self.category_col = category_col       # Ex: 'PrimaryPropertyType'
        self.proxy_target_col = proxy_target_col # Ex: 'SiteEUI(kBtu/sf)' (La colonne pour calculer la moyenne)
        self.m = m
        self.mapping = {}
        self.global_mean = 0

    def fit(self, X, y=None):
        
        # Copie de travail
        temp_df = X.copy()
        
        # On vérifie que la colonne proxy est bien là
        if self.proxy_target_col not in temp_df.columns:
            raise ValueError(f"La colonne {self.proxy_target_col} manque dans X pour le fit !")

        # Calcul de la moyenne globale (sur le SiteEUI et non le Total)
        self.global_mean = temp_df[self.proxy_target_col].mean()
        
        # Calcul des stats par catégorie
        stats = temp_df.groupby(self.category_col)[self.proxy_target_col].agg(['mean', 'count'])
        
        # Lissage
        smoothed = (stats['count'] * stats['mean'] + self.m * self.global_mean) / (stats['count'] + self.m)
        self.mapping = smoothed.to_dict()
        
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Application du mapping sur la catégorie
        new_col_name = 'Mean_EUI_by_Type'
        X_out[new_col_name] = X_out[self.category_col].map(self.mapping).fillna(self.global_mean)
        
        # --- IMPORTANT ---
        # On ne renvoie QUE la nouvelle colonne encodée.
        # On ne renvoie PAS 'SiteEUI' ni 'PrimaryPropertyType'.
        # Ainsi, SiteEUI disparaît proprement et le modèle final ne le verra jamais (zéro fuite).
        return X_out[[new_col_name]]

# --- B. Préparation des données (Plus de calculs préalables !) ---

X = df_clean.drop(columns=['SiteEnergyUse(kBtu)', 'TotalGHGEmissions']) # On garde PrimaryPropertyType ici !
y = df_clean['SiteEnergyUse(kBtu)']
y_log = np.log1p(y)

# Split (toujours utile pour le test final, même avec CV)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42, stratify=X['PrimaryPropertyType']
)

# --- C. Définition du Pipeline Avancé ---

# Etape 1 : On calcule la moyenne (Target Encoding) -> Sortie : [150.5, 80.2, ...]
# Etape 2 : On scale le résultat immédiatement -> Sortie : [1.2, -0.5, ...]
target_encoding_branch = Pipeline([
    ('encoder', SmoothedTargetEncoder(category_col='PrimaryPropertyType', proxy_target_col='SiteEUI(kBtu/sf)', m=10)),
    ('scaler', StandardScaler()) 
])


# Le ColumnTransformer va envoyer ce petit bloc de 2 colonnes à notre classe
target_enc_cols = ['PrimaryPropertyType', 'SiteEUI(kBtu/sf)']

# 2. Les autres colonnes numériques (déjà existantes)
numeric_features = ['YearBuilt', 'NumberofBuildings', 'NumberofFloors', 'PropertyGFAParking', 'PropertyGFABuilding(s)', 'DistanceToCenter', 'Use_Steam', 'Use_Gas']
numeric_transformer = StandardScaler()

# 3. Les colonnes catégorielles (Neighborhood)
categorical_features = ['Neighborhood']
categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

# 4. Le ColumnTransformer qui assemble tout ça
preprocessor = ColumnTransformer(
    transformers=[
        # Branche spéciale Target Encoding qui s'applique sur PrimaryPropertyType
        ('target_enc', target_encoding_branch, target_enc_cols),
        
        # Branche classique numérique
        ('num', numeric_transformer, numeric_features),
        
        # Branche OneHot
        ('cat', categorical_transformer, categorical_features)
    ]
)

# --- D. Comparaison avec Cross-Validation ---

models_to_test = {
    'Linear Regression': LinearRegression(),
     # /!\ SVM est sensible aux outliers et à l'échelle 
    # C=1.0 est la valeur par défaut, augmenter C peut réduire le biais mais augmenter la variance
    'SVR (Support Vector)': SVR(kernel='rbf', C=1, epsilon=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
}

results = []

print(f"{'Modèle':<20} | {'RMSE Moyen (CV)':<15} | {'Écart-type':<10}")
print("-" * 50)

# On définit la stratégie de CV (5 plis, mélangés)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models_to_test.items():
    # Pipeline complet : Preprocessing + Modèle
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # CROSS VALIDATION
    # scoring='neg_root_mean_squared_error' car sklearn cherche à maximiser le score,
    # donc il utilise l'erreur négative. On prendra l'opposé.
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
    
    rmse_scores = -cv_scores # On repasse en positif
    mean_rmse = rmse_scores.mean()
    std_rmse = rmse_scores.std()
    
    results.append({'Model': name, 'Mean RMSE': mean_rmse, 'Std': std_rmse})
    print(f"{name:<20} | {mean_rmse:<15.4f} | {std_rmse:<10.4f}")