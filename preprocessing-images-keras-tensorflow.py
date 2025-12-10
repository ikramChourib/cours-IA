import tensorflow as tf

IMG_SIZE = (224, 224)

def load_and_preprocess_tf(path):
    # path : tf.string, chemin vers l'image
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)   # ou decode_png
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    
    # Augmentations simples
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    
    return img

# Exemple : créer un dataset à partir d'une liste de chemins
paths = tf.constant([
    "img1.jpg",
    "img2.jpg",
])
ds = tf.data.Dataset.from_tensor_slices(paths)
ds = ds.map(load_and_preprocess_tf).batch(8)

for batch in ds:
    print(batch.shape)  # (batch_size, 224, 224, 3)
