#linear_regression
#Problème : prédire le prix d’une maison
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Création du dataset
# -----------------------------
# Exemple : relation entre surface (m²) et prix d'une maison
X = np.array([[50], [70], [100], [120], [150]])   # surface
y = np.array([100000, 130000, 180000, 210000, 260000])  # prix

# -----------------------------
# 2) Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3) Modèle de régression linéaire
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 4) Prédiction
# -----------------------------
prediction = model.predict([[80]])
print("Prix estimé pour 80 m² :", prediction[0])

# -----------------------------
# 5) Visualisation de la droite de régression
# -----------------------------
plt.scatter(X, y, color='blue', label="Données réelles")  

# Générer la droite de régression
X_line = np.linspace(40, 160, 100).reshape(-1, 1)
y_line = model.predict(X_line)

plt.plot(X_line, y_line, color='red', label="Régression linéaire")
plt.scatter(80, prediction, color='green', label=f"Prédiction (80 m²) : {int(prediction[0])} €")

plt.xlabel("Surface (m²)")
plt.ylabel("Prix (€)")
plt.title("Régression linéaire : Prix en fonction de la surface")
plt.legend()
plt.grid(True)
plt.show()
# -----------------------------
# 6) Le coefficient du modèle
# -----------------------------
print("Pente (coef) :", model.coef_[0])
print("Ordonnée à l'origine (intercept) :", model.intercept_)



