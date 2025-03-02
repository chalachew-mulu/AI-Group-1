import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Get the current directory of app.py
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the model
model_path = os.path.join(base_dir, "iris_model.pkl")
label_encoder_path = os.path.join(base_dir,"label_encoder.pkl")
data_path = os.path.join(base_dir,"../data/iris.csv")

print(f"🔍 Checking Model Path: {model_path}")  # Debugging step

# Load the model
model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Define feature names
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# 🌟 Streamlit App UI
st.title("🌸 Iris Flower Classifier")
st.markdown("Predict Iris species based on sepal & petal measurements.")

# 📌 Sidebar for user input
st.sidebar.header("Enter Flower Measurements")
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.sidebar.button("Predict 🌿"):
    # Prepare input as DataFrame
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)
    
    # Model Prediction
    predicted_class = model.predict(input_data)
    predicted_species = label_encoder.inverse_transform(predicted_class)[0]
    
    st.success(f"🌸 Predicted Species: **{predicted_species}**")

# 📊 Data Visualization
st.header("📊 Dataset Visualization")

# Load dataset (ensure correct path)
df = pd.read_csv(data_path)

# Pairplot
st.subheader("Feature Relationships")
sns.pairplot(df, hue="Species", palette="husl")
st.pyplot(plt)

# Class Distribution
st.subheader("Class Distribution")
plt.figure(figsize=(5, 3))
sns.countplot(x="Species", data=df, palette="viridis")
st.pyplot(plt)

# Footer
st.markdown("🚀 Built with **Streamlit** & **Scikit-Learn**")
