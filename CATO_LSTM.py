import streamlit as st
import numpy as np
import pandas as pd
import optuna
import catboost
import matplotlib
import tensorflow as tf
import keras
import joblib
import sklearn

# Get the versions of the libraries
versions = {
    "Streamlit": st.__version__,
    "Numpy": np.__version__,
    "Pandas": pd.__version__,
    "Optuna": optuna.__version__,
    "CatBoost": catboost.__version__,
    "Matplotlib": matplotlib.__version__,
    "TensorFlow": tf.__version__,
    "Keras": keras.__version__,
    "Joblib": joblib.__version__,
    "Scikit-learn": sklearn.__version__,
}

# Display the versions in the Streamlit app
st.title("Library Versions on Streamlit Cloud")
st.write("Below are the versions of the installed libraries:")

for library, version in versions.items():
    st.write(f"{library}: {version}")
