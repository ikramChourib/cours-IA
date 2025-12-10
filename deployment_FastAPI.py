# ============================================================
# api_nlp.py — API FastAPI pour classifier des images
#
# ▶ Lancer l'API :
#    uvicorn api_nlp:app --host 0.0.0.0 --port 8000
#
# ▶ Tester avec curl :
#    curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
#
# ▶ Tester avec Postman :
#    -> POST
#    -> URL : http://localhost:8000/predict
#    -> Body -> form-data -> key=file (type File)
# ============================================================

# -------- Imports API --------
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# -------- Imports Machine Learning --------
import numpy as np
from io import BytesIO
from PIL import Image
from keras.models import load_model

# -------- Gestion des chemins --------
from pathlib import Path


# ============================================================
# CONFIGURATION GLOBALE
# ============================================================

# Taille des images à utiliser (doit matcher l'entraînement)
IMG_SIZE = 256  

# Noms des classes dans le même ordre que le modèle
CLASS_NAMES = ["benign", "malignant", "normal"]

# Dossiers du projet
THIS_DIR = Path(__file__).resolve().parent      # dossier du fichier actuel
PROJECT_DIR = THIS_DIR.parent                   # dossier parent du projet
MODELS_DIR = PROJECT_DIR / "models"             # dossier contenant le modèle

# Fichier du modèle (h5 ou keras)
MODEL_PATH = MODELS_DIR / "predict_breast_cancer_version.h5"
# Alternative :
# MODEL_PATH = MODELS_DIR / "model.keras"

# Vérification que le modèle existe
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"❌ Modèle introuvable : {MODEL_PATH}\n"
        f"📂 Contenu du dossier {MODELS_DIR} = {list(MODELS_DIR.glob('*'))}"
    )

# Chargement du modèle (compile=False = plus rapide)
MODEL = load_model(MODEL_PATH, compile=False)


# ============================================================
# INITIALISATION DE L'API FASTAPI
# ============================================================

app = FastAPI(
    title="Breast Cancer Image Classification API",
    description="API utilisant un modèle CNN pour prédire la classe d'une image.",
    version="1.0.0",
)

# CORS (autorise les requêtes venant d'autres domaines)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tu peux mettre ton domaine ici
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTE DE TEST
# ============================================================

@app.get("/ping")
async def ping():
    """Route simple pour vérifier que l'API tourne."""
    return {"status": "ok"}


# ============================================================
# FONCTION UTILITAIRE : lecture + prétraitement d'image
# ============================================================

def read_file_as_image(data: bytes) -> np.ndarray:
    """
    Convertit un fichier image brut (bytes) en tableau numpy prêt pour le réseau.
    - conversion en RGB
    - redimensionnement
    - normalisation [0, 1]

    Retour :
        np.ndarray shape (H, W, 3) float32
    """
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))

    # Normalize to [0, 1]
    arr = np.asarray(img, dtype=np.float32) / 255.0  

    return arr


# ============================================================
# ROUTE PRINCIPALE : prédiction
# ============================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Reçoit une image et renvoie :
    - la classe prédite
    - la confiance
    - les probabilités détaillées pour chaque classe
    """
    try:
        # Lecture du fichier envoyé
        raw_bytes = await file.read()

        # Prétraitement image -> tableau numpy
        image = read_file_as_image(raw_bytes)

        # Batch : (1, H, W, 3)
        img_batch = np.expand_dims(image, axis=0)

        # Prédiction du modèle
        preds = MODEL.predict(img_batch)

        # Probabilités sous forme de liste Python
        probs = preds[0].astype(float)

        # Index de la classe prédite
        idx = int(np.argmax(probs))

        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[idx],
            "confidence": float(probs[idx]),
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            },
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur : {str(e)}")


# ============================================================
# LANCEMENT DIRECT
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
