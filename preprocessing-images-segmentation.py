def preprocess_image_and_mask(img_path, mask_path, target_size=(512, 512)):
    # 1) Image couleur
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Impossible de lire l'image : {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0

    # 2) Masque (grayscale, classes codées en entiers)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Impossible de lire le masque : {mask_path}")
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)  # pas de flou !
    
    # 3) Ajouter la dimension canal (H, W) -> (H, W, 1)
    mask = np.expand_dims(mask, axis=-1).astype(np.int32)

    return img, mask

img, mask = preprocess_image_and_mask("img.png", "mask.png", target_size=(512, 512))
print(img.shape, mask.shape)  # (512, 512, 3), (512, 512, 1)
