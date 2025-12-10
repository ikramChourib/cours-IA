"""
metrics_regression.py
---------------------
Exemple de calcul des métriques de régression :
MAE, MSE, RMSE, R², MAPE.
"""

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    """Calcul du MAPE en % (attention aux valeurs nulles)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # On évite les divisions par zéro en filtrant les 0
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def main():
    # 1. Charger un dataset de régression
    data = load_diabetes()
    X, y = data.data, data.target

    # 2. Train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Modèle de régression
    reg = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)

    # 4. Prédictions
    y_pred = reg.predict(X_test)

    # 5. Métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("=== Métriques de régression ===")
    print(f"MAE  : {mae:.3f}")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print(f"MAPE : {mape:.2f} %")


if __name__ == "__main__":
    main()
