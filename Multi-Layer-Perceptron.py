import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) Dataset artificiel
X, y = make_classification(n_samples=1000,
                           n_features=20,
                           n_classes=2,
                           random_state=42)

# Normalisation
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2) Construire un deep neural network
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),   # couche cachée 1
    Dense(32, activation='relu'),                      # couche cachée 2
    Dense(16, activation='relu'),                      # couche cachée 3
    Dense(1, activation='sigmoid')                     # sortie (binaire)
])

# 3) Compiler le modèle
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4) Entraîner le modèle
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 5) Évaluer
loss, acc = model.evaluate(X_test, y_test)

print("Accuracy :", acc)
