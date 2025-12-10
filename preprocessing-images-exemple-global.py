import os
import glob
import random

import cv2
import numpy as np
from sklearn.model_selection import train_test_split  # pour le split du dataset


# -------------------------------------------------------------------
# 1) Fonction d’augmentation d’images
# -------------------------------------------------------------------
def augment_image(img):
    """
    Applique des augmentations de données simples sur une image :
    - flip horizontal aléatoire
    - légère rotation aléatoire
    - ajout d'un léger bruit gaussien

    Paramètres
    ----------
    img : np.ndarray
        Image au format (H, W, C), en RGB, type float32, normalisée entre [0, 1].

    Retour
    ------
    np.ndarray
        Image augmentée, même format que l'entrée.
    """

    # Flip horizontal aléatoire avec probabilité 0.5
    # cv2.flip(img, 1) : 1 = flip horizontal (gauche/droite)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Rotation légère (-15°, +15°) avec probabilité 0.5
    if random.random() < 0.5:
        # Angle tiré au hasard entre -15 et 15 degrés
        angle = random.uniform(-15, 15)

        # Récupération de la hauteur (h) et la largeur (w) de l'image
        h, w = img.shape[:2]

        # Construction de la matrice de rotation 2D autour du centre de l'image
        # (w / 2, h / 2) = centre de l'image
        # angle = angle de rotation
        # 1.0 = facteur d'échelle (on ne zoome pas)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        # Application de la transformation affine (rotation) à l'image
        # borderMode=BORDER_REFLECT_101 : gère les pixels en dehors de l'image
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    # Ajout d'un léger bruit gaussien avec probabilité 0.5
    if random.random() < 0.5:
        # Génère un bruit gaussien de moyenne 0 et d'écart-type 0.02
        # même shape que l'image
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)

        # Ajout du bruit à l'image
        img = img + noise

        # On s'assure que les valeurs restent dans [0, 1]
        img = np.clip(img, 0.0, 1.0)

    return img


# -------------------------------------------------------------------
# 2) Fonction de prétraitement + augmentation (optionnelle)
# -------------------------------------------------------------------
def preprocess_and_augment(path, target_size=(224, 224), train=True):
    """
    Lit une image sur le disque, la convertit au bon format, la normalise,
    puis applique éventuellement une augmentation (pour le train).

    Paramètres
    ----------
    path : str
        Chemin vers l'image sur le disque.
    target_size : tuple
        Taille cible (largeur, hauteur) pour le redimensionnement de l'image.
    train : bool
        Si True, on applique les augmentations de données.

    Retour
    ------
    np.ndarray
        Image prétraitée, de shape (1, H, W, C), prête à être envoyée au modèle.
    """

    # Lecture de l'image avec OpenCV (par défaut en BGR)
    img = cv2.imread(path)
    if img is None:
        # Si l'image ne peut pas être chargée, on lève une erreur explicite
        raise ValueError(f"Impossible de lire l'image : {path}")

    # Conversion BGR -> RGB (plus standard pour la plupart des modèles)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionnement à la taille cible, ex: (224, 224)
    img = cv2.resize(img, target_size)

    # Conversion en float32 puis normalisation dans [0, 1]
    img = img.astype(np.float32) / 255.0

    # Si on est en mode entraînement, on applique l’augmentation
    if train:
        img = augment_image(img)

    # Standardisation optionnelle avec une moyenne et un écart-type
    # typiques de modèles pré-entraînés sur ImageNet (ResNet, etc.)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # (img - mean) / std est appliqué channel-wise (R,G,B)
    img = (img - mean) / std

    # Ajout d'une dimension batch : (H, W, C) -> (1, H, W, C)
    img = np.expand_dims(img, axis=0)

    return img


# -------------------------------------------------------------------
# 3) Fonction utilitaire pour lister les chemins d'images
# -------------------------------------------------------------------
def list_image_paths(data_dir, extensions=("*.jpg", "*.jpeg", "*.png")):
    """
    Récupère tous les chemins d'images dans un dossier donné pour une liste d'extensions.

    Paramètres
    ----------
    data_dir : str
        Dossier racine contenant les images.
    extensions : tuple
        Extensions de fichiers à rechercher.

    Retour
    ------
    list of str
        Liste des chemins d'images trouvés.
    """
    all_paths = []
    for ext in extensions:
        # glob.glob permet de lister les fichiers avec un pattern donné
        # ex: data/images/*.jpg
        all_paths.extend(glob.glob(os.path.join(data_dir, ext)))
    return all_paths


# -------------------------------------------------------------------
# 4) Split du dataset (train / validation / test)
# -------------------------------------------------------------------
def split_dataset(image_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Sépare une liste de chemins d'images en trois sous-ensembles :
    train, validation et test.

    Paramètres
    ----------
    image_paths : list of str
        Liste complète des chemins d'images.
    train_ratio : float
        Proportion d'images pour le train.
    val_ratio : float
        Proportion d'images pour la validation.
    test_ratio : float
        Proportion d'images pour le test.
    random_state : int
        Graine pour la reproductibilité du split.

    Retour
    ------
    (train_paths, val_paths, test_paths) : tuple of lists
        Listes de chemins d'images pour chaque sous-ensemble.
    """

    # Vérification grossière : les ratios doivent faire 1.0
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio doit être égal à 1.0")

    # D'abord on sépare train et (val+test)
    train_paths, temp_paths = train_test_split(
        image_paths,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        shuffle=True,
    )

    # Ensuite on sépare (val+test) en val et test
    # proportion de validation par rapport à temp_paths
    val_relative = val_ratio / (val_ratio + test_ratio)

    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=(1.0 - val_relative),
        random_state=random_state,
        shuffle=True,
    )

    return train_paths, val_paths, test_paths


# -------------------------------------------------------------------
# 5) Exemple d'utilisation dans un "main"
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Dossier contenant les images brutes
    # À adapter selon ton organisation (ex: "data/images")
    data_dir = "data/images"

    # 1) Lister tous les chemins d'images
    image_paths = list_image_paths(data_dir)

    print(f"Nombre total d'images trouvées : {len(image_paths)}")

    if len(image_paths) == 0:
        raise RuntimeError(
            f"Aucune image trouvée dans {data_dir}. "
            f"Vérifie le chemin et les extensions."
        )

    # 2) Split en train / val / test
    train_paths, val_paths, test_paths = split_dataset(image_paths)

    print(f"Train : {len(train_paths)} images")
    print(f"Val   : {len(val_paths)} images")
    print(f"Test  : {len(test_paths)} images")

    # 3) Exemple : prétraiter quelques images du train
    print("\n=== Exemple de prétraitement + augmentation (train) ===")
    for i, img_path in enumerate(train_paths[:3]):  # on prend les 3 premières
        img_tensor = preprocess_and_augment(img_path, target_size=(224, 224), train=True)
        print(f"{i+1}) {img_path} -> tensor shape : {img_tensor.shape}")

    # 4) Exemple : prétraiter une image de validation sans augmentation
    print("\n=== Exemple de prétraitement (validation, sans augmentation) ===")
    if len(val_paths) > 0:
        img_path = val_paths[0]
        img_tensor = preprocess_and_augment(img_path, target_size=(224, 224), train=False)
        print(f"Val image : {img_path} -> tensor shape : {img_tensor.shape}")

    # Ici, dans un vrai projet, tu enverrais `img_tensor` dans ton modèle :
    #   model.predict(img_tensor)
    # ou tu construirais un DataLoader / Generator qui appelle preprocess_and_augment
