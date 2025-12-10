"""
metrics_classification.py
-------------------------
Exemple simple de calcul des principales métriques de classification :
Accuracy, Precision, Recall, F1-score, AUC-ROC, matrice de confusion.

⚙️ Prérequis :
    pip install numpy scikit-learn
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt


def main():
    # =======================
    # 1. Données d'exemple
    # =======================
    # y_true : vraies classes (0 = négatif, 1 = positif)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])

    # y_pred : prédictions binaires du modèle
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])

    # y_score : probas ou scores continus (pour AUC-ROC)
    # (souvent la sortie de type "sigmoid" ou "softmax" du modèle)
    y_score = np.array([0.2, 0.9, 0.4, 0.1, 0.8, 0.3, 0.7, 0.45, 0.2, 0.6])

    # =======================
    # 2. Métriques globales
    # =======================
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary")
    recall = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)

    print("==== MÉTRIQUES DE CLASSIFICATION ====\n")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision : {precision:.3f}")
    print(f"Recall (sensibilité) : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"AUC-ROC : {auc:.3f}\n")

    # =======================
    # 3. Matrice de confusion
    # =======================
    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de confusion (lignes = vrai, colonnes = prédit) :")
    print(cm, "\n")

    # Rapport détaillé par classe (precision, recall, f1, support)
    print("Rapport de classification :")
    print(classification_report(y_true, y_pred, digits=3))

    # =======================
    # 4. Courbe ROC (optionnel)
    # =======================
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.title("Courbe ROC")
    plt.show()


if __name__ == "__main__":
    main()
