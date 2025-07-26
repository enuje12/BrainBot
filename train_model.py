import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os


#  CONFIGURATION

dataset_path = r"C:\Users\anuja\Downloads\archive\brain_tumor_dataset"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
MODEL_PATH = "brain_tumor_model.h5"


#  STREAMLIT PAGE SETUP

st.set_page_config(page_title="MediBot - Brain Tumor Detection", page_icon="🧠", layout="centered")
st.title("🧠 BrainBot - Brain Tumor Detection")


#  TRAIN MODEL FUNCTION

def train_and_save_model():
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation"
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    with st.spinner("Training model... Please wait (may take a few minutes)"):
        model.fit(train_generator, validation_data=val_generator, epochs=5)
        model.save(MODEL_PATH)
    st.success(" Model trained and saved successfully!")


#  TRAINING BUTTON

if not os.path.exists(MODEL_PATH):
    st.warning("No model found. Click below to train:")
    if st.button("Train Model"):
        train_and_save_model()
else:
    st.success("Model already available. Ready for predictions!")


#  LOAD MODEL

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    model = None


# UPLOAD IMAGE & PREDICT

uploaded_file = st.file_uploader(" Upload an MRI image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing Image..."):
            prediction = model.predict(img_array)[0][0]
            result = "Tumor Detected" if prediction > 0.5 else "No Tumor Detected"
            confidence = round(prediction*100 if prediction > 0.5 else (1-prediction)*100, 2)

        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: **{confidence}%**")
        if result == "Tumor Detected":
            st.error("⚠️ Please consult a neurologist immediately.")
        else:
            st.info("MRI scan seems normal, but regular check-ups are advised.")
