"""
preprocess_tabular.py
=====================

Exemples de prétraitement de données tabulaires :
- Chargement d'un CSV avec pandas
- Nettoyage basique (valeurs manquantes, doublons)
- Séparation features / cible
- Différentes méthodes de preprocessing :
    * Imputation (moyenne / médiane / plus fréquent)
    * Standardisation (StandardScaler)
    * Normalisation Min-Max (MinMaxScaler)
    * Encodage One-Hot pour les variables catégorielles
- Split train / validation / test

Exécution de démonstration dans le main.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Chargement et nettoyage basique
# ============================================================

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Charge un fichier CSV dans un DataFrame pandas.

    Paramètres
    ----------
    csv_path : str
        Chemin vers le fichier CSV.

    Retour
    ------
    df : pd.DataFrame
        DataFrame contenant les données chargées.
    """
    df = pd.read_csv(csv_path)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage basique des données :
    - Supprime les doublons
    - (Option) supprime des lignes complètement vides, si besoin

    Paramètres
    ----------
    df : pd.DataFrame
        Données brutes.

    Retour
    ------
    df_clean : pd.DataFrame
        Données nettoyées.
    """
    # Supprimer les doublons (mêmes lignes)
    df_clean = df.drop_duplicates()

    # Exemple : supprimer les lignes 100% NaN (optionnel)
    df_clean = df_clean.dropna(how="all")

    return df_clean


# ============================================================
# 2. Fonctions utilitaires de prétraitement
# ============================================================

def get_feature_target(df: pd.DataFrame, target_col: str):
    """
    Sépare les features (X) et la variable cible (y).

    Paramètres
    ----------
    df : pd.DataFrame
        Données nettoyées.
    target_col : str
        Nom de la colonne cible.

    Retour
    ------
    X : pd.DataFrame
        Variables explicatives.
    y : pd.Series
        Variable à prédire.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def detect_column_types(X: pd.DataFrame):
    """
    Détecte automatiquement les colonnes numériques et catégorielles.

    Paramètres
    ----------
    X : pd.DataFrame
        Variables explicatives.

    Retour
    ------
    numeric_cols : list
        Noms des colonnes numériques.
    categorical_cols : list
        Noms des colonnes catégorielles (object / string).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


# ============================================================
# 3. Différentes méthodes de preprocessing
#    (imputation, scaling, encodage)
# ============================================================

def build_numeric_pipeline(strategy_impute: str = "median",
                           scaling: str = "standard") -> Pipeline:
    """
    Construit un pipeline de prétraitement pour les colonnes numériques.

    - Imputation des valeurs manquantes :
        * "mean"   : moyenne
        * "median" : médiane
        * "most_frequent" : valeur la plus fréquente
    - Scaling :
        * "standard" : StandardScaler (moyenne 0, écart-type 1)
        * "minmax"   : MinMaxScaler (entre 0 et 1)
        * None       : pas de scaling

    Paramètres
    ----------
    strategy_impute : str
        Stratégie d'imputation des NaN.
    scaling : str
        Type de normalisation / standardisation.

    Retour
    ------
    num_pipeline : Pipeline
        Pipeline sklearn pour les colonnes numériques.
    """
    steps = []

    # Étape 1 : Imputation des valeurs manquantes
    steps.append(
        ("imputer", SimpleImputer(strategy=strategy_impute))
    )

    # Étape 2 : Scaling (optionnel)
    if scaling == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaling == "minmax":
        steps.append(("scaler", MinMaxScaler()))
    else:
        # Aucun scaling si scaling == None
        pass

    num_pipeline = Pipeline(steps)
    return num_pipeline


def build_categorical_pipeline(handle_unknown: str = "ignore") -> Pipeline:
    """
    Construit un pipeline pour les colonnes catégorielles.

    Étapes :
    - Imputation de la valeur manquante par la catégorie la plus fréquente.
    - Encodage One-Hot (création de colonnes binaires 0/1).

    Paramètres
    ----------
    handle_unknown : str
        Comportement si une catégorie inconnue apparaît en prédiction.
        "ignore" est souvent utilisé pour éviter les erreurs.

    Retour
    ------
    cat_pipeline : Pipeline
        Pipeline sklearn pour les colonnes catégorielles.
    """
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown=handle_unknown))
        ]
    )
    return cat_pipeline


def build_preprocessor(X: pd.DataFrame,
                       strategy_impute_num: str = "median",
                       scaling_num: str = "standard") -> ColumnTransformer:
    """
    Crée un préprocesseur complet (ColumnTransformer) combinant :
    - Pipeline numérique
    - Pipeline catégoriel

    Paramètres
    ----------
    X : pd.DataFrame
        Données d'entrée (features uniquement).
    strategy_impute_num : str
        Stratégie d'imputation pour les colonnes numériques.
    scaling_num : str
        Type de scaling pour les colonnes numériques.

    Retour
    ------
    preprocessor : ColumnTransformer
        Objet sklearn qui applique les bons traitements aux bonnes colonnes.
    numeric_cols : list
    categorical_cols : list
        Listes des noms de colonnes pour information.
    """
    numeric_cols, categorical_cols = detect_column_types(X)

    num_pipeline = build_numeric_pipeline(
        strategy_impute=strategy_impute_num,
        scaling=scaling_num
    )
    cat_pipeline = build_categorical_pipeline()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


# ============================================================
# 4. Split train / validation / test + application du preprocessing
# ============================================================

def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Sépare les données en train / validation / test.

    - On commence par séparer test.
    - Puis on coupe le train restant en train / val.

    Paramètres
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
        Proportion pour le test (ex : 0.2 = 20 %).
    val_size : float
        Proportion pour la validation par rapport au total (ex : 0.1 = 10 %).
    random_state : int
        Graines aléatoire pour reproductibilité.

    Retour
    ------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 1) Split train+val / test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2) Calcul de la part validation sur le train_val
    val_ratio = val_size / (1.0 - test_size)

    # 3) Split train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_dataset(df: pd.DataFrame,
                       target_col: str,
                       strategy_impute_num: str = "median",
                       scaling_num: str = "standard"):
    """
    Fonction "tout-en-un" :

    1. Nettoyage basique
    2. Séparation X / y
    3. Split train / val / test
    4. Construction du préprocesseur
    5. Fit du préprocesseur sur le train puis transformation de tous les splits

    Paramètres
    ----------
    df : pd.DataFrame
        Données brutes.
    target_col : str
        Nom de la colonne cible.
    strategy_impute_num : str
        Stratégie d'imputation numérique.
    scaling_num : str
        Type de scaling numérique.

    Retour
    ------
    X_train_p, X_val_p, X_test_p : np.ndarray
        Features prétraitées (prêtes pour un modèle).
    y_train, y_val, y_test : pd.Series
        Cible pour chaque split.
    preprocessor : ColumnTransformer
        Objet préprocesseur entraîné (pour réutilisation / sauvegarde).
    """
    # Nettoyage
    df_clean = basic_cleaning(df)

    # Séparation features / cible
    X, y = get_feature_target(df_clean, target_col=target_col)

    # Split train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Construction du préprocesseur
    preprocessor, num_cols, cat_cols = build_preprocessor(
        X_train,
        strategy_impute_num=strategy_impute_num,
        scaling_num=scaling_num
    )

    print("Colonnes numériques :", num_cols)
    print("Colonnes catégorielles :", cat_cols)

    # Fit du préprocesseur sur le train, puis transformation
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    return X_train_p, X_val_p, X_test_p, y_train, y_val, y_test, preprocessor


# ============================================================
# 5. Exemple d'utilisation dans un main
# ============================================================

if __name__ == "__main__":
    """
    Exemple concret :

    Supposons un fichier 'data/clients.csv' avec une colonne cible 'churn'
    (0/1 : le client quitte ou non le service).

    Colonnes possibles :
    - age (numérique)
    - revenu (numérique)
    - ville (catégorielle)
    - type_contrat (catégorielle)
    - churn (cible binaire)
    """

    # 1) Chemin vers le CSV (à adapter à votre cas)
    csv_path = "data/clients.csv"

    # 2) Chargement des données
    try:
        df = load_data(csv_path)
    except FileNotFoundError:
        print(f"⚠️ Fichier non trouvé : {csv_path}")
        print("Créez un CSV d'exemple ou changez le chemin.")
        exit(1)

    print("Aperçu des données brutes :")
    print(df.head())

    # 3) Prétraitement complet
    X_train_p, X_val_p, X_test_p, y_train, y_val, y_test, preprocessor = preprocess_dataset(
        df,
        target_col="churn",          # ⚠️ à adapter au nom de votre colonne cible
        strategy_impute_num="median",
        scaling_num="standard"       # ou "minmax" ou None
    )

    # 4) Affichage des shapes finales
    print("\nShapes après preprocessing :")
    print("X_train :", X_train_p.shape)
    print("X_val   :", X_val_p.shape)
    print("X_test  :", X_test_p.shape)
    print("y_train :", y_train.shape)
    print("y_val   :", y_val.shape)
    print("y_test  :", y_test.shape)

    # Ici vous pouvez ensuite entraîner un modèle, par ex :
    # from sklearn.linear_model import LogisticRegression
    # model = LogisticRegression(max_iter=1000)
    # model.fit(X_train_p, y_train)
    # print("Score val :", model.score(X_val_p, y_val))
