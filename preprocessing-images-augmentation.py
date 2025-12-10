import cv2
import numpy as np
import random

def augment_image(img):
    # img supposée au format (H, W, C), RGB, float32 [0,1]
    
    # Flip horizontal aléatoire
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    
    # Rotation légère (-15°, +15°)
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    # Ajout de léger bruit gaussien
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img = img + noise
        img = np.clip(img, 0.0, 1.0)
    
    return img

def preprocess_and_augment(path, target_size=(224, 224), train=True):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0

    if train:
        img = augment_image(img)
    
    # Standardisation optionnelle
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    img = np.expand_dims(img, axis=0)
    return img
