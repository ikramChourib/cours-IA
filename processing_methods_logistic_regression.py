#score d'examen -> admission 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Données simulées : score d'examen -> admission (0/1)
X = np.array([[40], [50], [60], [65], [70], [75], [80], [85], [90]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])

# Split train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions continue (probabilités)
X_range = np.linspace(40, 90, 300).reshape(-1, 1)
prob = model.predict_proba(X_range)[:, 1]

# Graphique
plt.figure(figsize=(7,5))
plt.scatter(X, y, color="blue", label="Données")
plt.plot(X_range, prob, color="red", label="Courbe sigmoïde")
plt.xlabel("Score d'examen")
plt.ylabel("Probabilité d'admission")
plt.title("Régression Logistique")
plt.legend()
plt.grid()
plt.show()

# Exemple de prédiction
p = model.predict_proba([[72]])[0][1]
print("Probabilité d'être admis avec 72 :", p)
