from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
from keras.models import load_model
from pathlib import Path

# -------- Config --------
IMG_SIZE = 256  # adapte à la taille utilisée à l'entraînement
CLASS_NAMES = ['benign', 'malignant', 'normal']
# dossier du fichier: .../end-to-end-potato-disease-classification-using-cnn/api/main.py
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent  # .../end-to-end-potato-disease-classification-using-cnn
MODELS_DIR = PROJECT_DIR / "models"

# --- Choisis le bon fichier modèle :
# Si tu as un .h5:
MODEL_PATH = MODELS_DIR / "predict_breast_cancer_version2.h5"
# Si tu as un .keras à la place :
# MODEL_PATH = MODELS_DIR / "1.keras"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}\n"
                            f"Contenu de {MODELS_DIR} = {list(MODELS_DIR.glob('*'))}")

MODEL = load_model(MODEL_PATH, compile=False)

# -------- App --------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restreins si besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/ping")
async def ping():
    return {"status": "ok"}

def read_file_as_image(data: bytes) -> np.ndarray:
    # Convertit en RGB, redimensionne, normalise
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        image = read_file_as_image(raw)                  # (H, W, 3) float32
        img_batch = np.expand_dims(image, axis=0)        # (1, H, W, 3)

        preds = MODEL.predict(img_batch)                 # (1, num_classes)
        probs = preds[0].astype(float).tolist()
        idx = int(np.argmax(preds[0]))
        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[idx],
            "confidence": float(preds[0][idx]),
            "probabilities": {CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
