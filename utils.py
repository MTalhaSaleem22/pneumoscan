# utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.models import load_model
from matplotlib import cm

img_size = 224
class_names = ['lung_cancer', 'normal', 'pneumonia', 'tuberculosis']

# Load the fine-tuned model
def load_trained_model():
    model = load_model("model.keras")
    return model

# Preprocess uploaded image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Predict class
def predict_class(model, img_array):
    prediction = model.predict(img_array)[0]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]
    return class_names[class_index], confidence, prediction

# Generate Grad-CAM heatmap
def generate_gradcam(model, img_array, class_index):
    last_conv_layer_name = 'conv5_block16_concat'  # Adjust based on your model architecture
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    return heatmap_resized  # Return the raw heatmap for later processing


# ---- Grad-CAM Panic-Film (smooth overlay + red circle(s)) ----
# ---- Grad-CAM Panic-Film (improved: percentile threshold + weighted centroid) ----
def generate_gradcam_panic_film(model, img_array, class_index,
                                last_conv_layer_name="conv5_block16_concat",
                                blur_ksize=11, percentile=90, min_area=120):
    """Return (heat, circles) where circles=[(cx, cy, r)] in model input pixels.

    - Uses percentile threshold (default 90th) to adapt per-image.

    - Weighted centroid + equivalent radius for stable circle.

    """
    grad_model = tf.keras.models.Model([model.inputs],
                                        [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    conv_outputs = conv_outputs[0]

    # Raw heat
    heat = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1).numpy()
    heat = np.maximum(heat, 0)
    heat = heat / (heat.max() if heat.max() != 0 else 1.0)
    heat = cv2.resize(heat, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

    # Smooth
    if blur_ksize and blur_ksize >= 3:
        heat = cv2.GaussianBlur(heat, (blur_ksize, blur_ksize), 0)

    # Adaptive threshold (percentile)
    thr = np.percentile(heat, percentile)
    bin_map = (heat >= thr).astype("uint8")

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    bin_map = cv2.morphologyEx(bin_map, cv2.MORPH_CLOSE, kernel, iterations=1)
    bin_map = cv2.morphologyEx(bin_map, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_map, connectivity=8)
    circles = []
    for label in range(1, num_labels):  # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < float(min_area):
            continue
        # Weighted centroid within this component
        mask = (labels == label)
        ys, xs = np.nonzero(mask)
        weights = heat[ys, xs] + 1e-6
        cy = int(np.average(ys, weights=weights))
        cx = int(np.average(xs, weights=weights))
        # Equivalent radius (area -> radius), add small margin
        r = int(np.sqrt(area / np.pi) * 1.10)
        circles.append((cx, cy, r))

    # Keep the most salient region by max integrated heat
    if len(circles) > 1:
        scores = []
        for (cx, cy, r) in circles:
            # score = sum of heat within the circle
            Y, X = np.ogrid[:img_size, :img_size]
            mask = (X - cx)**2 + (Y - cy)**2 <= r**2
            scores.append(heat[mask].sum())
        circles = [circles[int(np.argmax(scores))]]
    elif len(circles) == 0:
        # fallback to hottest spot
        yx = np.unravel_index(np.argmax(heat), heat.shape)
        circles = [(int(yx[1]), int(yx[0]), 16)]
    return heat.astype(np.float32), circles
