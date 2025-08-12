# app.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.cm as cm
from utils import load_trained_model, preprocess_image, predict_class, generate_gradcam, generate_gradcam_panic_film
from datetime import datetime
import openai

def is_valid_xray(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    mean_intensity = np.mean(gray)
    std_dev = np.std(gray)

    # Keep only minimal intensity filter
    if mean_intensity < 25 or mean_intensity > 240:
        return False

    # Loose edge check — only flag if it's very high
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    if edge_ratio > 0.25:  # reject only extremely detailed images
        return False

    return True

# Face detection-based sanity check
def has_face(image_np):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

# Load API key for Together.ai
openai.api_key = st.secrets["TOGETHER_API_KEY"]
openai.api_base = "https://api.together.xyz/v1"

# Page setup
st.set_page_config(page_title="PneumoScan - Lung Disease Classifier", layout="wide")

# Sidebar
st.sidebar.title("🩺 PneumoScan Controls")
st.sidebar.markdown("Upload a chest X-ray and view the AI-based prediction.")

# Sidebar uploader
uploaded_file = st.sidebar.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
show_probs = st.sidebar.checkbox("Show class probabilities", value=True)
show_gradcam = st.sidebar.checkbox("Show Grad-CAM heatmap", value=True)
chat_enabled = st.sidebar.checkbox("💬 Show PneumoBot", value=True)

# Main Header
title_col, img_col = st.columns([1, 2])
with title_col:
    st.title("PneumoScan")
    st.markdown("""
    PneumoScan is an AI-powered tool for detecting:
    - **Normal**
    - **Pneumonia**
    - **Lung Cancer**
    - **Tuberculosis**
    """)

# Load model
@st.cache_resource
def get_model():
    return load_trained_model()

model = get_model()

# --- MODEL PREDICTION SECTION ---
if uploaded_file is not None:
    image_data = Image.open(uploaded_file)
    image_np = np.array(image_data.resize((224, 224)).convert("RGB"))

    if not is_valid_xray(image_np):
        st.error("🚫 This image doesn't appear to be a valid chest X-ray. Please upload a correct medical image.")
        st.stop()

    if has_face(image_np):
        st.error("🚫 Face detected in image. Please upload a valid chest X-ray.")
        st.stop()

    img_array = preprocess_image(uploaded_file)

    # Predict
    label, confidence, all_probs = predict_class(model, img_array)
    st.session_state["predicted_label"] = label  # Store for chatbot

    with img_col:
        st.image(image_data, caption="📷 Uploaded Chest X-ray", use_container_width=True)

    st.success(f"🎯 **Prediction**: {label.upper()}  |  Confidence: {confidence*100:.2f}%")

    # Disease Info
    st.markdown("### 🩺 About the Disease")
    disease_info = {
        'normal': "**Normal:** No visible signs of lung disease. Consult a doctor if symptoms persist.",
        'pneumonia': "**Pneumonia:** Inflammation and fluid in the lungs. May require antibiotics or hospitalization.",
        'lung_cancer': "**Lung Cancer:** Abnormal cell growth. Early detection is critical. Follow up with imaging or biopsy.",
        'tuberculosis': "**Tuberculosis:** Airborne infection by Mycobacterium tuberculosis. Requires long-term treatment."
    }
    st.info(disease_info.get(label, "No information available."))

    # Probabilities
    if show_probs:
        st.markdown("### 📊 Class Probabilities")
        for idx, class_name in enumerate(['lung_cancer', 'normal', 'pneumonia', 'tuberculosis']):
            prob_value = float(all_probs[idx])
            st.progress(min(1.0, prob_value), text=f"{class_name.capitalize()}: {prob_value*100:.2f}%")

    def make_panic_film(gray_u8, cmap_name='inferno', gamma=1.25, clip_lo=2, clip_hi=98, clahe_clip=1.8, tile=(8,8), sat=0.85):
        """Tone-map to avoid blown highlights and produce a radiology-friendly film look.
        gray_u8: uint8 grayscale at original size
        Returns RGB uint8 film image.
        """
        import numpy as np, cv2, matplotlib.cm as cm
        # Percentile normalization (robust to outliers)
        lo, hi = np.percentile(gray_u8, [clip_lo, clip_hi])
        if hi <= lo: hi = lo + 1
        g = np.clip((gray_u8.astype(np.float32) - lo) / (hi - lo), 0, 1)
        # CLAHE to balance local contrast
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile)
        g8 = (g * 255.0).astype('uint8')
        g8 = clahe.apply(g8)
        # Gamma to pull down highlights slightly
        g = (g8.astype(np.float32) / 255.0) ** gamma
        # Map to calmer colormap (inferno)
        film = (cm.get_cmap(cmap_name)(g)[:, :, :3] * 255).astype('uint8')
        # Mild desaturation for readability
        hsv = cv2.cvtColor(film, cv2.COLOR_RGB2HSV).astype('float32')
        hsv[...,1] = np.clip(hsv[...,1] * sat, 0, 255)
        film = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2RGB)
        return film

    # Grad-CAM
    if show_gradcam:
        st.markdown("### 🔍 Grad-CAM Heatmap")
        # Original grayscale at native resolution (no scaling)
        original_img = Image.open(uploaded_file).convert('L')
        orig_w, orig_h = original_img.size
        xray_gray = np.array(original_img)  # HxW uint8

        # Run Grad-CAM on 224x224 input (already computed img_array/all_probs)
        class_idx = int(np.argmax(all_probs))
        heatmap_224, circles_224 = generate_gradcam_panic_film(
            model, img_array, class_idx,
            last_conv_layer_name="conv5_block16_concat",
            blur_ksize=11, percentile=90, min_area=180
        )

        # Create exposure-controlled 'panic film' from the original-resolution X-ray
        film = make_panic_film(xray_gray, cmap_name='inferno', gamma=1.25, clip_lo=2, clip_hi=98, clahe_clip=1.8, tile=(8,8), sat=0.85)

        # If we have a circle in 224 space, scale it to original size
        cx, cy, r = circles_224[0]
        sx, sy = orig_w / 224.0, orig_h / 224.0
        cx_o, cy_o, r_o = int(cx * sx), int(cy * sy), int(r * (sx + sy) / 2.0)

        # Draw a clean red ring with slight shadow; keep image at original size
        film_draw = film.copy()
        # cv2.circle(film_draw, (cx_o, cy_o), max(r_o, 14), (255, 0, 0), 4, lineType=cv2.LINE_AA)  # red ring
        # cv2.putText(film_draw, "Panic film", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
        # cv2.putText(film_draw, "Panic film", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2, cv2.LINE_AA)        # Right-align without scaling: use spacer columns
        spacer, right = st.columns([3, 2])
        with right:
            st.image(film_draw, caption=None, use_container_width=False)

# --- CHATBOT UI SECTION ---
if chat_enabled:
    if "predicted_label" in st.session_state:
        predicted_label = st.session_state["predicted_label"].upper()

        st.markdown("---")
        st.markdown("### 🤖 PneumoBot")
        st.markdown(
            f"""
            <div style='padding: 1rem; border-radius: 8px;'>
                <b>PneumoBot</b> is your AI assistant inside the PneumoScan app. The model predicts: {predicted_label}. 
                Ask questions about symptoms, treatment, or how the app works.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Init messages
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
            system_msg = {
                "role": "system",
                "content": f"""
                You are PneumoBot, an AI assistant inside the PneumoScan app. 
                PneumoScan is an AI-powered lung disease detection tool developed by M Talha Saleem and Hasnain Ikram using deep learning.
                The model predicts: {predicted_label}.
                Only respond to questions about lung diseases, symptoms, treatment, or the PneumoScan app.
                It is a user-friendly app that allows users to upload their X-Ray scan images and receive a diagnosis within minutes. 
                Keep answers concise, clear, and professional.
                """
            }
            st.session_state.chat_messages.append(system_msg)

        # Display chat
        for msg in st.session_state.chat_messages:
            if msg["role"] != "system":  # Hide system messages from frontend
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])


        # Chat input
        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input("Ask about the disease or app...", key="chat_input", label_visibility="collapsed")
            with col2:
                send = st.button("➤")

            if send and user_input.strip():
                st.chat_message("user").markdown(user_input)
                st.session_state.chat_messages.append({"role": "user", "content": user_input})

                with st.chat_message("assistant"):
                    with st.spinner("PneumoBot is responding..."):
                        try:
                            response = openai.ChatCompletion.create(
                                model="meta-llama/Llama-3-8b-chat-hf",
                                messages=st.session_state.chat_messages,
                                temperature=0.6,
                                max_tokens=500
                            )
                            reply = response.choices[0].message["content"]
                        except Exception as e:
                            reply = f"⚠️ Error: {str(e)}"

                        st.markdown(reply)
                        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
    else:
        st.info("Upload a chest X-ray to activate PneumoBot.")

# --- FOOTER ---
st.markdown("---")
st.markdown("### 👥 Developed By")
team_cols = st.columns(2)
with team_cols[0]:
    st.markdown("""
    **M Talha Saleem**  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/m-talha-saleem/)  
    [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/MTalhaSaleem22)
    """)
with team_cols[1]:
    st.markdown("""
    **Hasnain Ikram**  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/mohammad-hasnian-software/)  
    [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/HasnainMARS)
    """)
