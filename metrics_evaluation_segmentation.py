"""
metrics_segmentation.py
-----------------------
Exemple de calcul des métriques de segmentation d'images :
IoU, Dice, Pixel Accuracy, mIoU, precision et recall par classe.

Ici on travaille avec des masques 2D (H x W) d'entiers représentant
les classes : 0 = background, 1, 2, ..., N-1 pour les classes.

⚙️ Prérequis :
    pip install numpy
"""

import numpy as np


def get_confusion_matrix(y_true, y_pred, num_classes):
    """
    Construit une matrice de confusion pour la segmentation.
    y_true, y_pred : tableaux 1D contenant les classes (après flatten).
    """
    mask = (y_true >= 0) & (y_true < num_classes)
    cm = np.bincount(
        num_classes * y_true[mask].astype(int) + y_pred[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return cm


def compute_segmentation_metrics(cm):
    """
    À partir d'une matrice de confusion (num_classes x num_classes),
    calcule :
      - IoU par classe
      - Dice par classe
      - Pixel Accuracy globale
      - mIoU
      - Precision / Recall par classe
    """
    # TP = diag, FP = somme colonne - diag, FN = somme ligne - diag
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)  # pas utilisé ici mais dispo si besoin

    # IoU : TP / (TP + FP + FN)
    iou = tp / (tp + fp + fn + 1e-8)

    # Dice : 2TP / (2TP + FP + FN)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)

    # Pixel accuracy globale
    pixel_acc = tp.sum() / cm.sum()

    # mIoU : moyenne des IoU (on peut exclure le background si souhaité)
    miou = np.mean(iou)

    # Precision : TP / (TP + FP)
    precision = tp / (tp + fp + 1e-8)

    # Recall : TP / (TP + FN)
    recall = tp / (tp + fn + 1e-8)

    metrics = {
        "per_class_iou": iou,
        "per_class_dice": dice,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "pixel_accuracy": pixel_acc,
        "mean_iou": miou,
    }
    return metrics


def main():
    # =======================
    # 1. Masques d'exemple
    # =======================
    # Exemple toy : image 3x3 avec 3 classes (0, 1, 2)
    num_classes = 3

    # Masque "vrai" (ground truth)
    gt_mask = np.array(
        [
            [0, 0, 1],
            [0, 2, 2],
            [1, 1, 2],
        ]
    )

    # Masque prédit par le modèle
    pred_mask = np.array(
        [
            [0, 1, 1],
            [0, 2, 0],
            [1, 2, 2],
        ]
    )

    # =======================
    # 2. Matrice de confusion
    # =======================
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    cm = get_confusion_matrix(gt_flat, pred_flat, num_classes)
    print("Matrice de confusion (Segmentation) :")
    print(cm, "\n")

    # =======================
    # 3. Métriques
    # =======================
    metrics = compute_segmentation_metrics(cm)

    print("==== MÉTRIQUES DE SEGMENTATION ====\n")
    print(f"Pixel Accuracy globale : {metrics['pixel_accuracy']:.3f}")
    print(f"mIoU (mean IoU)        : {metrics['mean_iou']:.3f}\n")

    for cls in range(num_classes):
        print(f"--- Classe {cls} ---")
        print(f"IoU       : {metrics['per_class_iou'][cls]:.3f}")
        print(f"Dice      : {metrics['per_class_dice'][cls]:.3f}")
        print(f"Precision : {metrics['per_class_precision'][cls]:.3f}")
        print(f"Recall    : {metrics['per_class_recall'][cls]:.3f}\n")


if __name__ == "__main__":
    main()
