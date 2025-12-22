from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
import os

IMG_SIZE = 256

# Dossier du fichier courant
THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
MODELS_DIR = PROJECT_DIR / "/Users/ikram/Desktop/github-project/cours-IA/models"
MODEL_PATH = MODELS_DIR / "predict_breast_cancer_version2.h5"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
CLASS_NAMES = None  # on décidera après avoir vu la shape de sortie


@app.on_event("startup")
def load_tf_model():
    global MODEL, CLASS_NAMES

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable: {MODEL_PATH}\n"
            f"Contenu de {MODELS_DIR} = {list(MODELS_DIR.glob('*'))}"
        )

    # Limite threads (utile sur macOS + TF)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

    from keras.models import load_model

    MODEL = load_model(MODEL_PATH, compile=False)

    # Déduire le nombre de sorties
    out_shape = MODEL.output_shape  # ex: (None, 1) ou (None, 3)
    n_out = int(out_shape[-1])

    if n_out == 1:
        # modèle binaire
        CLASS_NAMES = ["benign", "malignant"]  # adapte si ton mapping est inverse
    else:
        # multi-classes
        CLASS_NAMES = ["benign", "malignant", "normal"]  # adapte à ton entraînement

    print(f"[startup] Model loaded: {MODEL_PATH.name} | output_shape={out_shape} | classes={CLASS_NAMES}")


@app.get("/ping")
async def ping():
    return {"status": "ok"}


def read_file_as_image(data: bytes) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        raw = await file.read()
        image = read_file_as_image(raw)
        img_batch = np.expand_dims(image, axis=0)

        preds = MODEL.predict(img_batch, verbose=0)
        preds = np.asarray(preds)

        # LOG utile
        print(f"[predict] filename={file.filename} batch_shape={img_batch.shape} preds_shape={preds.shape} preds={preds}")

        # Cas binaire: (1,1)
        if preds.shape[-1] == 1:
            score = float(preds[0][0])
            # seuil 0.5 (à adapter)
            idx = 1 if score >= 0.5 else 0
            return {
                "filename": file.filename,
                "predicted_class": CLASS_NAMES[idx],
                "confidence": score if idx == 1 else (1.0 - score),
                "probabilities": {
                    CLASS_NAMES[0]: float(1.0 - score),
                    CLASS_NAMES[1]: score,
                },
                "raw": score,
            }

        # Cas multi-classes: (1, C)
        probs = preds[0].astype(float)
        idx = int(np.argmax(probs))
        return {
            "filename": file.filename,
            "predicted_class": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx),
            "confidence": float(probs[idx]),
            "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(min(len(CLASS_NAMES), len(probs)))},
            "raw": probs.tolist(),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("deployment_FastAPI:app", host="127.0.0.1", port=8000, reload=False)

