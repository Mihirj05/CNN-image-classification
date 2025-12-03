
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from data_loader import CIFAR10_LABELS

st.set_page_config(page_title="CNN Image Classifier", layout="centered")

MODEL_DIR = 'outputs/models'

def latest_model_path():
    if not os.path.exists(MODEL_DIR):
        return None
    models = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    if not models:
        return None
    latest = max(models, key=os.path.getctime)
    return latest

@st.cache_resource
def load_model(path):
    if path is None:
        return None
    model = tf.keras.models.load_model(path)
    return model

st.title("CNN Image Classifier — Demo")
st.write("Upload an image (32x32 will be used). Default model: CIFAR-10.")

model_path = latest_model_path()
if model_path:
    st.write("Using model:", model_path)
else:
    st.warning("No saved model found in outputs/models. Train a model first (`run_train.sh`).")
    st.stop()

model = load_model(model_path)

uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])
if uploaded_file and model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded image', use_column_width=True)
    st.write("Classifying...")

   
    img = image.resize((32,32))
    img_arr = np.array(img).astype('float32') / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = preds[0, pred_class]

    st.write(f"Prediction: **{CIFAR10_LABELS[pred_class]}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
