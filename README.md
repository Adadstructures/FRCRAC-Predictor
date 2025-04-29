# FRCRAC with FEM Visualisation

This repository hosts a Streamlit-based web application for predicting and visualizing the behavior of fibre-reinforced polymer (FRP)-confined recycled aggregate concrete (FRCRAC) using machine learning and hybrid ML-FEM frameworks.

## 📌 App Features

- Predict ultimate strength, axial strain, and hoop strain of FRP-confined recycled aggregate concrete.
- Choose between two stress-strain prediction models (CATO-MZW and CATO-LSTMO).
- Visualize:
  - Load–Displacement Curves
  - Stress–Strain Curves
  - Stress, Strain, Load, and Displacement Contours in 3D.
- Upload experimental data for comparison.
- Download generated plots as high-quality PNG files.

## 🚀 Access the App

[**Click here to access FRCRAC with FEM Visualisation App**](https://frcrac.streamlit.app/)

## 🛠 Technology Stack

- Python 3.12
- Streamlit
- TensorFlow / Keras
- CatBoost
- Plotly (for 3D visualization)
- SciPy, NumPy, Pandas, Scikit-learn
- Kaleido (for exporting plots)

## 📂 Project Structure

```plaintext
app.py                  # Main Streamlit app
requirements.txt        # Python dependencies
saved_model/            # Pre-trained LSTM models (axial and hoop)
targets_axial.npy       # Target scaling data for axial strain/stress
targets_hoop.npy        # Target scaling data for hoop strain/stress
scaler_axial.pkl        # Input scaler for axial LSTM
scaler_hoop.pkl         # Input scaler for hoop LSTM
CATO_Rupture.pkl        # Trained CatBoost model for rupture strain prediction
CATO_Strain.pkl         # Trained CatBoost model for strain prediction
CATO_Strength.pkl       # Trained CatBoost model for strength prediction
LICENSE                 # Licensing information
README.md               # Project description (this file)
