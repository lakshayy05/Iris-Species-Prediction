import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# --- CONFIGURATION ---
st.set_page_config(page_title="Iris Deep Learning Predictor", layout="centered")

# --- LOAD ASSETS ---
@st.cache_resource # Use this for heavy DL models to load them only once
def load_model():
    model = tf.keras.models.load_model('iris_dl_model.h5')
    return model

try:
    model = load_model()
    scaler = joblib.load('iris_scaler.pkl')
    encoder = joblib.load('iris_encoder.pkl')
except OSError:
    st.error("Error: Files not found. Please ensure 'iris_dl_model.h5', 'iris_scaler.pkl', and 'iris_encoder.pkl' are in the same folder.")
    st.stop()

st.title("🌸 Iris Flower Classification (Deep Learning)")
st.write("Enter the flower measurements below to classify the species using a Neural Network.")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width = st.number_input("Sepal Width (cm)", 2.0, 5.0, 3.0)

with col2:
    petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, 4.3)
    petal_width = st.number_input("Petal Width (cm)", 0.1, 3.0, 1.3)

# --- PREDICTION LOGIC ---
if st.button("Classify Flower"):
    try:
        # 1. Prepare Input
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # 2. Scale the Data (Neural Networks are very sensitive to scaling!)
        input_scaled = scaler.transform(input_data)

        # 3. Predict (Returns probabilities, e.g., [0.1, 0.8, 0.1])
        prediction_probs = model.predict(input_scaled)
        
        # 4. Get the class with highest probability (argmax)
        predicted_class_index = np.argmax(prediction_probs)
        
        # 5. Decode label (Convert 0/1/2 back to "Setosa")
        predicted_label = encoder.inverse_transform([predicted_class_index])[0]
        confidence = np.max(prediction_probs) * 100

        # 6. Display Result
        st.success(f"### Species: {predicted_label}")
        st.info(f"Confidence: {confidence:.2f}%")
        
        # Optional: Show the raw probability bars
        st.write("Probability Distribution:")
        probs_df = pd.DataFrame(prediction_probs, columns=encoder.classes_)
        st.bar_chart(probs_df.T)

    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- SIDEBAR ---
st.sidebar.info("Model: TensorFlow/Keras Sequential Neural Network")