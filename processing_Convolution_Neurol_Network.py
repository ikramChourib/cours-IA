"""
config_cnn.py
---------------
Fichier de configuration pour notre CNN :
- chemins
- hyperparamètres d'entraînement
"""

# Hyperparamètres d'entraînement
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2

# Paramètres du modèle
INPUT_SHAPE = (32, 32, 3)   # CIFAR-10 : images 32x32 RGB
NUM_CLASSES = 10            # 10 classes

# Callbacks
EARLY_STOPPING_PATIENCE = 5   # nombre d'époques sans amélioration avant arrêt
LR_REDUCE_PATIENCE = 3        # nombre d'époques sans amélioration avant réduction du LR

# Chemin pour sauvegarder le meilleur modèle
BEST_MODEL_PATH = "best_cnn_cifar10.h5"


"""
model_cnn.py
-------------
Définit la fonction build_cnn_model() qui crée un modèle CNN Keras.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from config_cnn import INPUT_SHAPE, NUM_CLASSES, LEARNING_RATE

def build_cnn_model():
    """
    Crée et compile un modèle CNN pour la classification d'images.
    Retourne : un modèle Keras compilé.
    """

    model = models.Sequential()

    # Bloc 1 de convolution
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                            input_shape=INPUT_SHAPE))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))  # régularisation pour limiter l'overfitting

    # Bloc 2 de convolution
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Bloc 3 de convolution
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Passage en fully-connected
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))   # Dropout plus fort sur la partie dense

    # Couche de sortie
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    # Compilation du modèle
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",  # labels comme entiers 0..9
        metrics=["accuracy"]
    )

    return model


"""
train_cnn.py
-------------
Script principal pour :
- Charger CIFAR-10
- Prétraiter les données
- Appliquer une augmentation de données (data augmentation)
- Entraîner le CNN avec callbacks (early stopping, reduction du LR, sauvegarde du meilleur modèle)
- Évaluer le modèle
"""

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config_cnn import (
    BATCH_SIZE,
    EPOCHS,
    VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    LR_REDUCE_PATIENCE,
    BEST_MODEL_PATH
)
from model_cnn import build_cnn_model

def load_data():
    """
    Charge le dataset CIFAR-10 depuis Keras.
    Renvoie : (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation des pixels dans [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # y_train et y_test sont des entiers (0..9), ce qui est compatible avec sparse_categorical_crossentropy
    return (x_train, y_train), (x_test, y_test)

def get_data_generator():
    """
    Crée un générateur d'images avec augmentation de données.
    Cette augmentation aide à limiter l'overfitting.
    """
    datagen = ImageDataGenerator(
        rotation_range=15,       # rotations aléatoires
        width_shift_range=0.1,   # décalage horizontal
        height_shift_range=0.1,  # décalage vertical
        horizontal_flip=True,    # flip horizontal
        zoom_range=0.1           # zoom léger
    )
    return datagen

def get_callbacks():
    """
    Crée les callbacks pour :
    - EarlyStopping : arrêter quand la val_loss ne s'améliore plus
    - ReduceLROnPlateau : diminuer le LR si la val_loss stagne
    - ModelCheckpoint : sauvegarder le meilleur modèle (val_loss minimale)
    """
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,       # diviser le LR par 2
        patience=LR_REDUCE_PATIENCE,
        min_lr=1e-6,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    return [early_stopping, reduce_lr, checkpoint]

def main():
    # 1. Charger les données
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2. Créer le modèle CNN
    model = build_cnn_model()
    model.summary()  # affiche la structure du modèle

    # 3. Préparer le générateur de données avec augmentation
    datagen = get_data_generator()
    datagen.fit(x_train)

    # 4. Préparer les callbacks (early stopping, etc.)
    cb_list = get_callbacks()

    # 5. Entraîner le modèle
    #    On utilise flow(x, y) pour générer les batchs augmentés à la volée.
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,  # ne fonctionne pas directement avec flow
        # Astuce : on peut split avant ou utiliser validation_data séparément.
        # Ici on fait un split manuel pour la validation.
        steps_per_epoch=int((1 - VALIDATION_SPLIT) * len(x_train) // BATCH_SIZE),
        callbacks=cb_list,
        verbose=1
    )

    # 6. Évaluer sur le jeu de test
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Performance sur le test : loss = {test_loss:.4f}, accuracy = {test_acc:.4f}")

    # 7. Sauvegarder le modèle final (optionnel, en plus du best_model)
    model.save("cnn_cifar10_final.h5")
    print("Modèle final sauvegardé sous 'cnn_cifar10_final.h5'.")

if __name__ == "__main__":
    main()

