import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from PIL import Image
import pickle
import re
import string
# import streamlit_authenticator as stauth

st.set_page_config(page_title="Multi-Modal Fake News Detection", page_icon="###", layout="wide")

st.title(" Multi-Modal Fake News Detection (Text + Image)")
st.markdown("Analyze both **news text** and **images** to detect fake or AI-generated content.")

@st.cache_resource
def load_text_model():
    try:
        model = load_model("fake_news_model.h5")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f" Failed to load text model or tokenizer: {e}")
        return None, None


@st.cache_resource
def load_image_model():
    try:
        model = tf.keras.models.load_model("real_fake_face_detection_model.h5")
        return model
    except Exception as e:
        st.error(f" Failed to load image model: {e}")
        return None


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict_real_or_fake_image(model, img):
    img = img.resize((96, 96))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)[0][0]
    label = " Real Face" if prediction > 0.5 else " Fake Face"
    confidence = (prediction if prediction > 0.5 else 1 - prediction) * 100
    return label, confidence


text_model, tokenizer = load_text_model()
image_model = load_image_model()
# add side war




tab1, tab2, tab3 = st.tabs([" Text Only", " Image Only", " Text + Image"])

with tab1:
    st.subheader("Fake News Detection (Text)")
    user_input = st.text_area("Enter the News Article:", height=200, placeholder="Paste or type your news text here...")

    if st.button(" Predict Text"):
        if user_input.strip() == "":
            st.warning(" Please enter some text to analyze!")
        elif text_model is not None and tokenizer is not None:
            cleaned = preprocess_text(user_input)
            max_len = 903
            sequence = tokenizer.texts_to_sequences([cleaned])
            padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
            try:
                prediction = text_model.predict(padded_sequence)
                prob = prediction[0][0]
                pred_label = "Fake" if prob > 0.5 else "Real"
                confidence = (prob if prob > 0.5 else 1 - prob) * 100

                st.success(f" The news is **{pred_label}**")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.progress(int(confidence))
            except Exception as e:
                st.error(f"Prediction failed: {e}")


#  TAB 2: IMAGE ONLY
with tab2:
    st.subheader("Fake Face Detection (Image)")
    uploaded_image = st.file_uploader(" Upload an Image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Analyze Image"):
            if image_model is not None:
                with st.spinner("Analyzing image..."):
                    label, confidence = predict_real_or_fake_image(image_model, img)
                st.success(f"**Prediction:** {label}")
                st.metric("Confidence", f"{confidence:.2f}%")
                st.progress(int(confidence))
                if "Fake" in label:
                    st.warning(" This might be an AI-generated face.")
                else:
                    st.success(" This appears to be a real face.")
            else:
                st.error("Image model not loaded properly.")


#  TAB 3: TEXT + IMAGE
with tab3:
    st.subheader("Combined Fake News Detection (Text + Image)")
    combined_text = st.text_area("Enter the News Text:", height=200)
    combined_image = st.file_uploader(" Upload Related Image...", type=["jpg", "jpeg", "png"])

    if st.button("Predict Both"):
        if combined_text.strip() == "" and combined_image is None:
            st.warning(" Please provide either text or image input.")
        else:
            if combined_text.strip() != "" and text_model is not None:
                cleaned = preprocess_text(combined_text)
                max_len = 903
                sequence = tokenizer.texts_to_sequences([cleaned])
                padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
                prediction = text_model.predict(padded_sequence)
                prob = prediction[0][0]
                text_label = "Fake" if prob > 0.5 else "Real"
                text_confidence = (prob if prob > 0.5 else 1 - prob) * 100
                st.write("# Text Analysis")
                st.success(f"Result: **{text_label}**")
                st.metric("Text Confidence", f"{text_confidence:.2f}%")
                st.progress(int(text_confidence))

            if combined_image is not None and image_model is not None:
                img = Image.open(combined_image)
                st.image(img, caption="Uploaded Image", use_container_width=True)
                label, confidence = predict_real_or_fake_image(image_model, img)
                st.write("# Image Analysis")
                st.success(f"Result: **{label}**")
                st.metric("Image Confidence", f"{confidence:.2f}%")
                st.progress(int(confidence))

#  Footer
st.markdown("---")
st.caption("Built with  using Streamlit & TensorFlow | Project by Anoop Ojha")