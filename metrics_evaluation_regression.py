"""
metrics_regression.py
---------------------
Exemple simple de calcul des métriques de régression :
MAE, MSE, RMSE, R², MAPE.

⚙️ Prérequis :
    pip install numpy scikit-learn
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calcul du MAPE = moyenne des |erreur relative| en pourcentage.
    ⚠️ Ne pas utiliser lorsque certaines valeurs de y_true sont nulles.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # On ajoute une petite valeur pour éviter la division par zéro
    epsilon = 1e-8
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def main():
    # =======================
    # 1. Données d'exemple
    # =======================
    # y_true : vraies valeurs continues
    y_true = np.array([100, 150, 200, 250, 300])

    # y_pred : valeurs prédites par le modèle
    y_pred = np.array([110, 140, 195, 260, 310])

    # =======================
    # 2. Calcul des métriques
    # =======================
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print("==== MÉTRIQUES DE RÉGRESSION ====\n")
    print(f"MAE  (Mean Absolute Error)           : {mae:.3f}")
    print(f"MSE  (Mean Squared Error)           : {mse:.3f}")
    print(f"RMSE (Racine du MSE)                : {rmse:.3f}")
    print(f"R²   (Coefficient de détermination) : {r2:.3f}")
    print(f"MAPE (Erreur absolue en %)          : {mape:.2f} %")


if __name__ == "__main__":
    main()
