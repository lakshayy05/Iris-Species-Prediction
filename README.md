# Iris Species Prediction (Deep Learning) 🌸

A web application that classifies Iris flowers into three species (**Setosa, Versicolor, Virginica**) using a **Deep Learning (Neural Network)** model built with TensorFlow/Keras.

## 🚀 Project Overview

While simple algorithms like KNN work well on Iris, this project demonstrates how to build, train, and deploy a **Multi-Layer Perceptron (MLP)** Neural Network.
* **Backend:** TensorFlow 2.0 / Keras
* **Frontend:** Streamlit

## 📂 Project Structure

```text
Iris-Deep-Learning/
│
├── app.py                       # 🖥️ Frontend: Streamlit Web App
├── Iris_Prediction_using_DL.ipynb # 📓 Backend: Neural Network Training
├── iris_dl_model.h5             # 🧠 Artifact: Saved Keras Model
├── iris_scaler.pkl              # ⚖️ Artifact: StandardScaler
├── iris_encoder.pkl             # 📝 Artifact: LabelEncoder
├── requirements.txt             # ⚙️ Dependencies
└── README.md                    # 📄 Documentation

📊 Model Architecture
I used a Feed-Forward Neural Network (Sequential) with the following structure:
Input Layer: 4 Neurons (Sepal Length, Sepal Width, Petal Length, Petal Width)

Hidden Layers:
Dense Layer (16 neurons, ReLU activation)
Dense Layer (8 neurons, ReLU activation)
Output Layer: 3 Neurons (Softmax activation for multi-class probability)
Optimizer: Adam
Loss Function: Categorical Crossentropy
