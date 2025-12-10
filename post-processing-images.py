"""
post_processing.py

Exemple complet de post-traitement pour des masques de segmentation :
- Binarisation à partir d'une carte de probabilité
- Opérations morphologiques (érosion, dilatation, ouverture, fermeture)
- Seuil de décision
- Fusion d'objets voisins / superposés
- Filtrage par taille minimale

Dépendances :
    pip install opencv-python scikit-image numpy
"""

import os
import glob
import numpy as np
import cv2
from skimage import measure


# ---------------------------------------------------------------------
# 1. BINARISATION / SEUIL DE DÉCISION
# ---------------------------------------------------------------------

def binarize_from_prob_map(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarise une carte de probabilité (valeurs flottantes entre 0 et 1).

    Paramètres
    ----------
    prob_map : np.ndarray
        Carte de probabilité, shape (H, W), valeurs entre [0, 1] ou [0, 255].
    threshold : float
        Seuil de décision. Les pixels > threshold sont considérés comme 1 (objet).

    Retour
    ------
    binary_mask : np.ndarray
        Masque binaire uint8 (0 ou 255).
    """
    # Normaliser si la carte est en 0-255 (uint8)
    if prob_map.dtype != np.float32 and prob_map.dtype != np.float64:
        prob_map_norm = prob_map.astype(np.float32) / 255.0
    else:
        prob_map_norm = prob_map.copy()

    # Binarisation : 1 si proba > threshold, sinon 0
    binary = (prob_map_norm > threshold).astype(np.uint8)

    # Convention masque binaire OpenCV : 0 (fond) ou 255 (objet)
    binary_mask = binary * 255
    return binary_mask


# ---------------------------------------------------------------------
# 2. OPÉRATIONS MORPHOLOGIQUES : ÉROSION, DILATATION, OUVERTURE, FERMETURE
# ---------------------------------------------------------------------

def morphological_ops(
    binary_mask: np.ndarray,
    op_type: str = "open",
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Applique une opération morphologique sur un masque binaire.

    Paramètres
    ----------
    binary_mask : np.ndarray
        Masque binaire (0 / 255).
    op_type : str
        Type d'opération : "erode", "dilate", "open", "close".
    kernel_size : int
        Taille du noyau structurant (carré kernel_size x kernel_size).
    iterations : int
        Nombre de répétitions de l'opération.

    Retour
    ------
    processed : np.ndarray
        Masque après l'opération morphologique.
    """
    # Création du noyau structurant (carré)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if op_type == "erode":
        processed = cv2.erode(binary_mask, kernel, iterations=iterations)
    elif op_type == "dilate":
        processed = cv2.dilate(binary_mask, kernel, iterations=iterations)
    elif op_type == "open":
        # Ouverture = érosion puis dilatation (supprime le bruit)
        processed = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif op_type == "close":
        # Fermeture = dilatation puis érosion (remplit les trous)
        processed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError(f"Type d'opération morphologique inconnu : {op_type}")

    return processed


# ---------------------------------------------------------------------
# 3. FILTRAGE PAR TAILLE MINIMALE
# ---------------------------------------------------------------------

def remove_small_objects(binary_mask: np.ndarray, min_size: int = 100) -> np.ndarray:
    """
    Supprime les objets (composantes connexes) dont l'aire est < min_size.

    Paramètres
    ----------
    binary_mask : np.ndarray
        Masque binaire (0 / 255).
    min_size : int
        Aire minimale (en nombre de pixels) pour garder un objet.

    Retour
    ------
    cleaned_mask : np.ndarray
        Masque après suppression des petits objets.
    """
    # Convertir en booléen pour skimage (True = objet)
    bool_mask = binary_mask > 0

    # Labeling des composantes connexes (4-connexité)
    labeled = measure.label(bool_mask, connectivity=1)

    cleaned = np.zeros_like(bool_mask, dtype=bool)

    # Parcourir chaque label (1..num_labels)
    for region in measure.regionprops(labeled):
        if region.area >= min_size:
            cleaned[labeled == region.label] = True

    cleaned_mask = (cleaned.astype(np.uint8)) * 255
    return cleaned_mask


# ---------------------------------------------------------------------
# 4. FUSION D'OBJETS VOISINS / SUPERPOSÉS
# ---------------------------------------------------------------------

def merge_close_components(binary_mask: np.ndarray, merge_distance: int = 3) -> np.ndarray:
    """
    Fusionne les objets proches (ex : séparés par quelques pixels) en utilisant
    une dilatation suivie d'un re-labeling.

    Idée :
    - On dilate légèrement les objets pour que ceux qui sont très proches
      se "touchent" et deviennent une seule composante.
    - On re-binarise ensuite pour revenir à la taille d'origine approximative.

    Paramètres
    ----------
    binary_mask : np.ndarray
        Masque binaire (0 / 255).
    merge_distance : int
        Rayon (en pixels) pour dilater les objets (distance à laquelle
        on souhaite fusionner deux objets proches).

    Retour
    ------
    merged_mask : np.ndarray
        Masque avec objets fusionnés (0 / 255).
    """
    if merge_distance <= 0:
        # Rien à faire si distance <= 0
        return binary_mask

    # 1) Dilatation pour que les objets voisins se rejoignent
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_distance + 1, 2 * merge_distance + 1))
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)

    # 2) Optionnel : petite érosion pour recentrer un peu les objets fusionnés
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # eroded est déjà binaire (0 / 255), on peut le renvoyer tel quel
    merged_mask = eroded
    return merged_mask


# ---------------------------------------------------------------------
# 5. PIPELINE COMPLET POUR UNE SEULE IMAGE
# ---------------------------------------------------------------------

def post_process_single_mask(
    prob_or_mask: np.ndarray,
    is_prob_map: bool = True,
    threshold: float = 0.5,
    morph_op: str = "open",
    morph_kernel: int = 3,
    morph_iter: int = 1,
    min_size: int = 100,
    merge_distance: int = 3,
) -> np.ndarray:
    """
    Pipeline complet de post-processing pour un seul masque.

    Étapes :
      1. Binarisation (seuil de décision)
      2. Opérations morphologiques (érosion/dilatation/ouverture/fermeture)
      3. Filtrage des petits objets (taille minimale)
      4. Fusion des objets voisins / superposés

    Paramètres
    ----------
    prob_or_mask : np.ndarray
        Soit une carte de probabilité (float32/64), soit un masque déjà binaire.
    is_prob_map : bool
        True si `prob_or_mask` est une carte de probabilité, False si c'est déjà un masque binaire.
    threshold : float
        Seuil de binarisation (utilisé seulement si is_prob_map=True).
    morph_op : str
        Opération morphologique : "erode", "dilate", "open", "close".
    morph_kernel : int
        Taille du noyau pour la morphologie.
    morph_iter : int
        Nombre d'itérations pour la morphologie.
    min_size : int
        Aire minimale des objets à conserver.
    merge_distance : int
        Distance de fusion des objets proches (en pixels).

    Retour
    ------
    final_mask : np.ndarray
        Masque binaire final (0 / 255).
    """
    # 1) Binarisation
    if is_prob_map:
        binary = binarize_from_prob_map(prob_or_mask, threshold=threshold)
    else:
        # On s'assure bien que le masque est binaire 0 / 255
        tmp = (prob_or_mask > 0).astype(np.uint8)
        binary = tmp * 255

    # 2) Morphologie (ex : ouverture pour enlever le bruit)
    binary_morph = morphological_ops(
        binary_mask=binary,
        op_type=morph_op,
        kernel_size=morph_kernel,
        iterations=morph_iter,
    )

    # 3) Filtrage par taille minimale
    binary_clean = remove_small_objects(binary_morph, min_size=min_size)

    # 4) Fusion objets proches
    final_mask = merge_close_components(binary_clean, merge_distance=merge_distance)

    return final_mask


# ---------------------------------------------------------------------
# 6. EXEMPLE D’UTILISATION SUR UN DOSSIER DE MASQUES
# ---------------------------------------------------------------------

def process_folder(
    input_folder: str,
    output_folder: str,
    is_prob_map: bool = True,
    threshold: float = 0.5,
    morph_op: str = "open",
    morph_kernel: int = 3,
    morph_iter: int = 1,
    min_size: int = 100,
    merge_distance: int = 3,
    pattern: str = "*.png",
):
    """
    Applique le post-processing à tous les fichiers d'un dossier.

    Paramètres
    ----------
    input_folder : str
        Dossier contenant les cartes de probabilité ou masques.
    output_folder : str
        Dossier où sauvegarder les masques post-traités.
    is_prob_map : bool
        True si les fichiers sont des cartes de probabilité (0-255), False si masques binaires.
    pattern : str
        Pattern des fichiers (ex: "*.png", "*.jpg").
    """
    os.makedirs(output_folder, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_folder, pattern)))
    print(f"Trouvé {len(files)} fichiers à traiter.")

    for f in files:
        # Lecture de l'image en niveaux de gris
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[AVERTISSEMENT] Impossible de lire {f}, fichier ignoré.")
            continue

        # Post-processing
        processed = post_process_single_mask(
            prob_or_mask=img,
            is_prob_map=is_prob_map,
            threshold=threshold,
            morph_op=morph_op,
            morph_kernel=morph_kernel,
            morph_iter=morph_iter,
            min_size=min_size,
            merge_distance=merge_distance,
        )

        # Sauvegarde
        base_name = os.path.basename(f)
        out_path = os.path.join(output_folder, base_name)
        cv2.imwrite(out_path, processed)
        print(f"Fichier traité et sauvegardé : {out_path}")


# ---------------------------------------------------------------------
# 7. POINT D’ENTRÉE PRINCIPAL (EXEMPLE)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Exemple d'utilisation :
    #
    # Supposons que :
    #   - "probabilities/" contient des cartes de probabilité (grayscale 0-255)
    #   - nous voulons les transformer en masques binaires propres dans "masks_postproc/"
    #
    # Vous pouvez adapter les chemins, seuils et paramètres à vos besoins.

    input_dir = "probabilities"     # dossier d'entrée (à adapter)
    output_dir = "masks_postproc"   # dossier de sortie (sera créé si besoin)

    process_folder(
        input_folder=input_dir,
        output_folder=output_dir,
        is_prob_map=True,       # True si vos images sont des cartes de probabilité
        threshold=0.5,          # seuil de binarisation
        morph_op="open",        # "erode", "dilate", "open", "close"
        morph_kernel=3,         # taille noyau morphologique
        morph_iter=1,           # nombre d'itérations morphologiques
        min_size=200,           # taille minimale des objets (en pixels)
        merge_distance=3,       # distance pour fusionner objets proches
        pattern="*.png",        # extension des fichiers
    )
