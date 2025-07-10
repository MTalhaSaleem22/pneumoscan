# app.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils import load_trained_model, preprocess_image, predict_class, generate_gradcam
from datetime import datetime
import openai

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
    st.title("🧠 PneumoScan")
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
        original = np.array(image_data.resize((224, 224)).convert('RGB'))
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        gradcam_text = {
            'normal': "Model focused on clear lung fields.",
            'pneumonia': "Highlighted fluid-filled/inflamed regions.",
            'lung_cancer': "Attention on nodules or dense growth.",
            'tuberculosis': "Focus on upper lung areas, patchy shadows."
        }

        cam_col1, cam_col2 = st.columns([1, 1])
        with cam_col1:
            st.markdown(f"**Model Focus:** {gradcam_text.get(label, 'Model highlights uncertain regions.')}")
        with cam_col2:
            st.image(overlay, caption=f"🧭 Grad-CAM Overlay: {label.upper()}", use_container_width=True)

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
