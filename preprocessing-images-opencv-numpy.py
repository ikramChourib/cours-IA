import cv2
import numpy as np

def preprocess_image(path, target_size=(224, 224)):
    # 1) Charger l'image (BGR par défaut avec OpenCV)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    
    # 2) Conversion BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3) Redimensionnement
    img = cv2.resize(img, target_size)
    
    # 4) Conversion en float32 + normalisation [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # 5) Optionnel : standardisation (soustraction moyenne, division std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # exemple ImageNet
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # 6) Ajouter la dimension batch : (H, W, C) -> (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    
    return img

# Exemple d'utilisation
preprocessed = preprocess_image("mon_image.jpg", target_size=(224, 224))
print(preprocessed.shape)  # (1, 224, 224, 3)
