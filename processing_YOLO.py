"""
Exemple complet YOLOv8 : Détection + Segmentation
- Entraînement détection
- Entraînement segmentation
- Évaluation
- Prédiction sur nouvelles images
"""

from ultralytics import YOLO
from pathlib import Path


# -------------------------------------------------------------------
# 1. CONFIG GLOBALE
# -------------------------------------------------------------------
# ⚠️ À adapter à ton projet
DETECTION_DATA_YAML = "data/detection.yaml"    # dataset pour la détection
SEGMENTATION_DATA_YAML = "data/segmentation.yaml"  # dataset pour la segmentation

IMGSZ = 640           # taille des images
EPOCHS = 100          # nb max d'époques
BATCH_SIZE = 8        # batch size
PATIENCE = 20         # early stopping (si pas d'amélioration)
PROJECT = "runs_yolo" # dossier où YOLO va sauver les résultats


# -------------------------------------------------------------------
# 2. ENTRAÎNEMENT DÉTECTION (bounding boxes)
# -------------------------------------------------------------------
def train_detection():
    """
    Entraîne un modèle YOLOv8 pour la DÉTECTION (boîtes englobantes).
    On part d'un modèle pré-entraîné (transfer learning).
    """

    # Charge un modèle pré-entraîné léger (yolov8n = nano)
    model = YOLO("yolov8n.pt")  # tu peux mettre 'yolov8s.pt', 'yolov8m.pt', etc.

    # Entraînement
    model.train(
        data=DETECTION_DATA_YAML,  # fichier YAML décrivant le dataset
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        patience=PATIENCE,         # early stopping
        lr0=1e-3,                  # learning rate initial
        weight_decay=5e-4,         # régularisation L2 pour éviter l'overfitting
        optimizer="SGD",           # ou 'AdamW'
        project=PROJECT,
        name="yolo_det",           # sous-dossier runs_yolo/yolo_det
        pretrained=True,           # transfer learning
        augment=True,              # data augmentation par défaut (mosaic, flip, etc.)
        verbose=True
    )

    # Le meilleur modèle est automatiquement sauvegardé dans
    # runs_yolo/yolo_det/weights/best.pt
    return model


# -------------------------------------------------------------------
# 3. ENTRAÎNEMENT SEGMENTATION (masks)
# -------------------------------------------------------------------
def train_segmentation():
    """
    Entraîne un modèle YOLOv8 pour la SEGMENTATION (masks).
    """

    # Modèle de segmentation pré-entraîné
    model = YOLO("yolov8n-seg.pt")  # variante "seg" pour segmentation

    model.train(
        data=SEGMENTATION_DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        lr0=1e-3,
        weight_decay=5e-4,
        optimizer="SGD",
        project=PROJECT,
        name="yolo_seg",
        pretrained=True,
        augment=True,
        verbose=True
    )

    # Meilleur modèle dans runs_yolo/yolo_seg/weights/best.pt
    return model


# -------------------------------------------------------------------
# 4. ÉVALUATION DES MODÈLES (mAP, etc.)
# -------------------------------------------------------------------
def evaluate_model(weights_path, task="detect"):
    """
    Évalue un modèle YOLO (détection ou segmentation) sur le set de validation.

    :param weights_path: chemin vers le .pt entraîné (best.pt)
    :param task: "detect" ou "segment"
    """
    model = YOLO(weights_path)

    # Choix du bon data.yaml pour l'éval
    data_yaml = DETECTION_DATA_YAML if task == "detect" else SEGMENTATION_DATA_YAML

    metrics = model.val(
        data=data_yaml,
        imgsz=IMGSZ,
        project=PROJECT,
        name=f"val_{task}"
    )

    # 'metrics' contient mAP50, mAP50-95, etc.
    print(f"\n=== Résultats {task} ===")
    print(metrics)
    return metrics


# -------------------------------------------------------------------
# 5. PRÉDICTION SUR DE NOUVELLES IMAGES
# -------------------------------------------------------------------
def predict_on_images(weights_path, source="images_test/*.jpg", task="detect"):
    """
    Applique un modèle YOLO entraîné à des nouvelles images.

    :param weights_path: chemin vers le modèle (.pt)
    :param source: chemin vers image / dossier / motif glob
    :param task: "detect" ou "segment" (utilisé seulement pour les noms)
    """
    model = YOLO(weights_path)

    results = model.predict(
        source=source,       # image, dossier, vidéo ou webcam (0)
        imgsz=IMGSZ,
        conf=0.25,           # seuil de confiance
        iou=0.45,            # seuil IoU pour NMS
        project=PROJECT,
        name=f"pred_{task}", # sous-dossier pour les visus
        save=True,           # sauvegarde les images annotées
    )

    print(f"\nPrédictions sauvegardées dans : {Path(PROJECT) / f'pred_{task}'}")
    return results


# -------------------------------------------------------------------
# 6. MAIN : exemple d’utilisation
# -------------------------------------------------------------------
if __name__ == "__main__":

    # ======= 1) Entraînement détection =======
    print(">>> Entraînement YOLOv8 - DÉTECTION")
    det_model = train_detection()

    # Chemin vers le meilleur modèle de détection
    det_best = Path(PROJECT) / "yolo_det" / "weights" / "best.pt"

    # ======= 2) Évaluation détection =======
    print("\n>>> Évaluation modèle de DÉTECTION")
    evaluate_model(str(det_best), task="detect")

    # ======= 3) Prédiction détection =======
    print("\n>>> Prédictions sur nouvelles images (DÉTECTION)")
    predict_on_images(str(det_best), source="images_test/*.jpg", task="detect")

    # ======= 4) Entraînement segmentation =======
    print("\n>>> Entraînement YOLOv8 - SEGMENTATION")
    seg_model = train_segmentation()

    seg_best = Path(PROJECT) / "yolo_seg" / "weights" / "best.pt"

    # ======= 5) Évaluation segmentation =======
    print("\n>>> Évaluation modèle de SEGMENTATION")
    evaluate_model(str(seg_best), task="segment")

    # ======= 6) Prédiction segmentation =======
    print("\n>>> Prédictions sur nouvelles images (SEGMENTATION)")
    predict_on_images(str(seg_best), source="images_test/*.jpg", task="segment")
