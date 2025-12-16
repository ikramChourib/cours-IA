#classification binaire sur un dataset artificiel
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1) Générer un dataset simple
X, y = make_classification(n_samples=500,
                           n_features=2,
                           n_redundant=0,
                           n_clusters_per_class=1,
                           random_state=42)

# 2) Diviser train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) Créer un perceptron (single-layer)
model = Perceptron(max_iter=1000, eta0=0.1)

# 4) Entraînement
model.fit(X_train, y_train)

# 5) Prédiction
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
