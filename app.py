# app.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils import load_trained_model, preprocess_image, predict_class, generate_gradcam
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

    # Grad-CAM
    if show_gradcam:
        st.markdown("### 🔍 Grad-CAM Heatmap")
        heatmap = generate_gradcam(model, img_array, np.argmax(all_probs))
        
        # Resize and normalize original image
        original = np.array(image_data.resize((224, 224)).convert('RGB'))
        original = (original / 255.0 * 255).astype(np.uint8)  # Ensure original is uint8

        # Create a mask for the infected area based on the heatmap
        # Threshold the heatmap to find areas of interest
        thresholded_heatmap = np.where(heatmap > 0.5, 1, 0).astype(np.uint8)  # Adjust threshold as necessary

        # Find contours of the highlighted areas
        contours, _ = cv2.findContours(thresholded_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a black mask
        mask = np.zeros_like(original)

        # Draw contours on the mask
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small contours
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(mask, center, radius, (255, 0, 0), -1)  # Filled red circle

        # Blend the original image with the mask
        overlay = cv2.addWeighted(original, 1, mask, 0.5, 0)  # Adjust alpha for effect

        # Display the final overlay
        cam_col1, cam_col2 = st.columns([1, 1])
        with cam_col1:
            st.markdown("**Model Focus:** Highlighting infected areas.")
        with cam_col2:
            st.image(overlay, caption="🧭 Infected Area Highlighted", use_container_width=True)

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
