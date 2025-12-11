import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Création d’un dataset simple
# ---------------------------
X = np.array([
    [1, 2], [2, 3], [3, 1],
    [6, 5], [7, 7], [8, 6]
])
y = np.array([0, 0, 0, 1, 1, 1])  # 0 = Classe A, 1 = Classe B

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------
# 2. Modèle KNN
# ---------------------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Prédiction d’un nouveau point
new_point = np.array([[4, 3]])
prediction = model.predict(new_point)
print("Classe prédite :", prediction[0])
#affiche = Classe prédite : 0

# ---------------------------
# 3. Graphe
# ---------------------------
plt.figure(figsize=(6, 6))

# Points d'entraînement
plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='bwr', s=80, label="Train")

# Nouveau point
plt.scatter(new_point[:,0], new_point[:,1], c="green", s=150, marker="X", label="Nouveau point")

plt.title("Classification KNN (k=3)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True)
plt.show()

