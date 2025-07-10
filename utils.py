# utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.models import load_model

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
    last_conv_layer_name = 'conv5_block16_concat'
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
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_HOT)  # Change to HOT for intensity

    return heatmap_colored
