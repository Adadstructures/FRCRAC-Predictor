import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model
from io import BytesIO

# Load targets from the new saved .npy files
targets_1_axial = np.load('targets_axial.npy')
targets_1_hoop = np.load('targets_hoop.npy')

# Load scalers and models
scaler_axial = joblib.load('scaler_axial.pkl')
scaler_hoop = joblib.load('scaler_hoop.pkl')
    
# Load your models
cato_rupture = pickle.load(open('CATO_Rupture.pkl', 'rb'))
cato_strain = pickle.load(open('CATO_Strain.pkl', 'rb'))
cato_strength = pickle.load(open('CATO_Strength.pkl', 'rb'))
lstm_model_axial = load_model('axial_lstm_model.h5')
lstm_model_hoop = load_model('hoop_lstm_model.h5')

# Title of the application
st.title("FRCRAC Predictor")

# Aggregate Type Dropdown
aggregate_type = st.selectbox("Aggregate Type", ["RCA", "RCL", "RBA", "NA"], index=0)

# Sections for user inputs
with st.form("input_form"):
    
    # Display RCA properties only if the aggregate type is not 'NA'
    if aggregate_type != 'NA':
        st.subheader("Aggregate Properties")
        percentage_rca = st.number_input("Percentage of RCA replacement by weight", value=50.00, format="%.2f")
        max_rca_size = st.number_input("Maximum diameter of the RCA (mm)", value=31.50, format="%.2f")
    else:
        percentage_rca = 0.0
        max_rca_size = st.number_input("Maximum diameter of the RCA (mm)", value=31.50, format="%.2f")  
        
    st.subheader("Cementitious Property")
    water_cement_ratio = st.number_input("Water-to-cement ratio", value=0.35, format="%.2f")

    st.subheader("Geometry Properties")
    diameter = st.number_input("Diameter of the concrete cylinder (mm)", value=150.00, format="%.2f")
    height = st.number_input("Height of the concrete cylinder (mm)", value=300.00, format="%.2f")

    st.subheader("Concrete Properties")
    unconfined_strength = st.number_input("Unconfined Strength (MPa)", value=50.65, format="%.2f")
    unconfined_strain = st.number_input("Unconfined Strain", value=0.002, format="%.5f")

    st.subheader("FRP Properties")
    fibre_modulus = st.number_input("Fibre Modulus (MPa)", value=272730.0)
    frp_overall_thickness = st.number_input("FRP Overall Thickness (mm)", value=0.167, format="%.3f")
    frp_type = st.selectbox("Fibre Type", ["GFRP", "CFRP"], index=0)
    
    st.subheader("Stress-Strain Model")
    stress_strain_model = st.selectbox("Stress-Strain Model", ["CATO-LSTMO", "CATO-MZW"], index=0)

    # Conditional logic for variables based on aggregate type
    fibre_type = 1 if frp_type == 'GFRP' else 3
    agg_type = 1 if aggregate_type == 'NA' else 2 if aggregate_type == 'RCA' else 3 if aggregate_type == 'RCL' else 4
    
    if aggregate_type != 'NA':
        concrete_modulus = 4120 * (unconfined_strength ** 0.5)
    else:
        concrete_modulus = 4370 * (unconfined_strength ** 0.5)

    rupture_strain = float(cato_rupture.predict([[fibre_type, diameter, height, percentage_rca if aggregate_type != 'NA' else 0, 
                                                  max_rca_size if aggregate_type != 'NA' else 0, water_cement_ratio,
                                                  unconfined_strength, unconfined_strain, fibre_modulus, frp_overall_thickness, 
                                                  agg_type, concrete_modulus]]))

    confinement_stress = 2 * rupture_strain * fibre_modulus * frp_overall_thickness / diameter
    
    
    # Optional tab for uploading experimental data
    st.subheader("Upload Experimental Stress-Strain Data")

    # File uploader for axial and hoop stress-strain CSV files
    uploaded_file_axial = st.file_uploader("Upload axial stress-strain CSV file", type=["csv"])
    uploaded_file_hoop = st.file_uploader("Upload hoop stress-strain CSV file", type=["csv"])

    # Load axial stress-strain data or create an empty DataFrame
    if uploaded_file_axial is not None:
        # Load the uploaded file into a DataFrame
        df_axial = pd.read_csv(uploaded_file_axial)
        st.success("Axial stress-strain data loaded successfully.")
    else:
        # Create an empty DataFrame if no file is uploaded
        df_axial = pd.DataFrame(columns=['Strain', 'Stress'])
        st.warning("No axial data uploaded. Empty DataFrame created.")

    # Load hoop stress-strain data or create an empty DataFrame
    if uploaded_file_hoop is not None:
        # Load the uploaded file into a DataFrame
        df_hoop = pd.read_csv(uploaded_file_hoop)
        st.success("Hoop stress-strain data loaded successfully.")
    else:
        # Create an empty DataFrame if no file is uploaded
        df_hoop = pd.DataFrame(columns=['Strain', 'Stress'])
        st.warning("No hoop data uploaded. Empty DataFrame created.")

    
    # Button to perform calculation
    submit_button = st.form_submit_button("Predict")

if submit_button:
    
    # Input data with scalar values
       
    input_data = [fibre_type, diameter, height, percentage_rca, max_rca_size, water_cement_ratio, unconfined_strength,
                  unconfined_strain, fibre_modulus, frp_overall_thickness, agg_type, concrete_modulus, rupture_strain, confinement_stress]
    
    cato_strength_prediction = float(cato_strength.predict([input_data]))
    cato_strain_prediction = float(cato_strain.predict([input_data]))
    
    input_data_lstm = [fibre_type, diameter, height, percentage_rca, max_rca_size, water_cement_ratio, unconfined_strength,
                  unconfined_strain, fibre_modulus, frp_overall_thickness, agg_type, concrete_modulus, rupture_strain, confinement_stress, 
                  cato_strength_prediction,cato_strain_prediction]
    
    input_data_axial = np.array(input_data_lstm).reshape(1, -1) 
    input_data_hoop = np.array(input_data_lstm).reshape(1, -1) 
    
    new_inputs_normalized_axial = scaler_axial.transform(input_data_axial)
    new_inputs_normalized_hoop = scaler_hoop.transform(input_data_hoop)
    
    
  # Predict on new data
    predicted_targets_axial = lstm_model_axial.predict(new_inputs_normalized_axial)
    predicted_targets_hoop = lstm_model_hoop.predict(new_inputs_normalized_hoop)

    # Denormalize predictions
    predicted_targets_axial_denorm = predicted_targets_axial * targets_1_axial.max(axis=(0, 1))
    predicted_targets_hoop_denorm = predicted_targets_hoop * targets_1_hoop.max(axis=(0, 1))

    strength_prediction = max(predicted_targets_axial_denorm[:, 0])
    strain_prediction = max(predicted_targets_axial_denorm[:, 1])

       
    if stress_strain_model == 'CATO-MZW':
        ultimate_strength = cato_strength_prediction
        ultimate_axial_strain = cato_strain_prediction
        ultimate_hoop_strain = rupture_strain
        strength_enhancement_ratio = cato_strength_prediction / unconfined_strength
        strain_enhancement_ratio = cato_strain_prediction / unconfined_strain
                
        f_o = unconfined_strength + (0.003 * confinement_stress)
        Modulus_1 = concrete_modulus
        Modulus_2 = (cato_strength_prediction - f_o) / cato_strain_prediction

        section_1 = np.arange(0, unconfined_strain + (unconfined_strain / 10), unconfined_strain / 10)
        section_2 = np.arange(unconfined_strain, cato_strain_prediction + (unconfined_strain / 10), unconfined_strain / 10)
        strain_values = np.concatenate((section_1, section_2))

        k_Correction = cato_strain_prediction / (cato_strain_prediction - unconfined_strain)
        stress_values = ((((Modulus_1 * unconfined_strain) - f_o) * np.exp(-strain_values / unconfined_strain)) +
                     f_o + (k_Correction * Modulus_2 * strain_values)) * (1 - np.exp(-strain_values / unconfined_strain))
        
        plt.figure(figsize=(12, 8))
        plt.plot(strain_values * 100, stress_values, label='CATO-MZW',  color='blue')
        
        plt.plot(df_axial['Strain'] * 100, df_axial['Stress'], label='Experiment', color='red')
        plt.xlabel("Strain (%)", fontsize=22)
        plt.ylabel("Stress (MPa)", fontsize=22)
        plt.title("Stress-Strain Curve", fontsize=22)
        plt.legend(fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.grid(True)
        
    else:       
        # Extract and adjust predicted values for axial
        predicted_stress_axial = np.insert(predicted_targets_axial_denorm[0, :, 0], 0, 0)
        predicted_strain_axial = np.insert(predicted_targets_axial_denorm[0, :, 1] * 100, 0, 0)

        # Extract and adjust predicted values for hoop
        predicted_stress_hoop = np.insert(predicted_targets_hoop_denorm[0, :, 0], 0, 0)
        predicted_strain_hoop = np.insert(-predicted_targets_hoop_denorm[0, :, 1] * 100, 0, 0)
        
        last_predicted_point_axial = (predicted_strain_axial[-1], predicted_stress_axial[-1])
        last_predicted_point_hoop = (predicted_strain_hoop[-1], predicted_stress_hoop[-1])

        ultimate_axial_strength = np.max(predicted_stress_axial)
        ultimate_hoop_strength = np.max(predicted_stress_hoop)

        lstm_axial_strain_ultimate = predicted_strain_axial[np.argmax(predicted_stress_axial)]
        lstm_hoop_strain_ultimate = predicted_strain_hoop[np.argmax(predicted_stress_hoop)]

        lstm_strength = max(ultimate_axial_strength, ultimate_hoop_strength)
        
        ultimate_strength = lstm_strength
        ultimate_axial_strain = lstm_axial_strain_ultimate/100
        ultimate_hoop_strain = lstm_hoop_strain_ultimate/100
        strength_enhancement_ratio = lstm_strength / unconfined_strength
        strain_enhancement_ratio = (lstm_axial_strain_ultimate/100) / unconfined_strain

        plt.figure(figsize=(12, 8))
        
        # Plot axial stress-strain
        plt.plot(predicted_strain_axial, predicted_stress_axial,  color='blue', label='CATO-LSTMO')
        plt.plot(predicted_strain_hoop, predicted_stress_hoop, color='blue')
        plt.plot(df_axial['Strain'] * 100, df_axial['Stress'], label='Experiment', color='red')
        plt.plot(df_hoop['Strain'] * 100, df_hoop['Stress'],  color='red')
        plt.xlabel("Strain (%)", fontsize=22)
        plt.ylabel("Stress (MPa)", fontsize=22)
        plt.title("Stress-Strain Curve", fontsize=22)
        plt.legend(fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.grid(True)
        
    
    # Display plot
    st.pyplot(plt)
    
    # Print prediction results
    st.subheader("Prediction Results")
    st.write(f"Ultimate Strength: {ultimate_strength:.3f} MPa")
    st.write(f"Ultimate Axial Strain: {100*ultimate_axial_strain:.3f} %")
    st.write(f"Ultimate Hoop Strain: {100*ultimate_hoop_strain:.3f} %")
    st.write(f"Strength Enhancement: {strength_enhancement_ratio:.3f}")
    st.write(f"Strain Enhancement: {strain_enhancement_ratio:.3f}")
    
        # Add a download button for the plot
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(
        label="Download plot",
        data=buf,
        file_name="stress_strain_curve.png",
        mime="image/png"
    )
    
    # Add footnote
    st.markdown("""
        **Notes**: 
        1. This application predicts the stress-strain behaviours of circular fibre-reinforced polymer confined recycled aggregate concrete (FRCRAC).
        2. Three types of recycled aggregates (RA) were considered, namely recycled concrete aggregate (RCA), recycled concrete lump(RCL) and recycled brick aggregate (RBA).
        3. Two types of fibre-reinforced polymers (FRP) were considered, namely glass (GFRP) and carbon (CFRP).
        4. CATO-MZW: hybridised Categorical Boosting optimised with Optuna with modified Zhou and Wu model, proposed by Dada et al. (2024).
        5. CATO-LSTMO: hybridised Categorical Boosting and Long Short Term Memory optimised with Optuna framework.
        """)
    st.markdown("""
        **References**: 
        1. T.E. Dada, G. Gong, J. Xia, L. Di Sarno, Stress-strain behaviour of axially loaded FRP-confined natural and recycled aggregate concrete using design-oriented and machine learning approaches, Journal of Building Engineering 95 (2024) 110256. https://doi.org/https://doi.org/10.1016/j.jobe.2024.110256.
        2. L. Prokhorenkova, G. Gusev, A. Vorobev, A.V. Dorogush, A. Gulin, CatBoost: unbiased boosting with categorical features, 2018. https://github.com/catboost/catboost.
        3. S. Hochreiter, J. Schmidhuber, Long Short-Term Memory, Neural Comput 9 (1997) 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735.
        4. F.A. Gers, J. Schmidhuber, F. Cummins, Learning to Forget: Continual Prediction with LSTM, Neural Comput 12 (2000) 2451–2471. https://doi.org/10.1162/089976600300015015.
        5. T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A Next-generation Hyperparameter Optimization Framework, in: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Association for Computing Machinery, New York, NY, USA, 2019: pp. 2623–2631. https://doi.org/10.1145/3292500.3330701.
        """)
# Adding a footer with contact information
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    color: #6c757d;
}
</style>
<div class="footer">
    <p>© 2024 My Streamlit App. All rights reserved. |Temitope E. Dada, Guobin Gong, Jun Xia, Luigi Di Sarno | For Queries: <a href="mailto: T.Dada19@student.xjtlu.edu.cn"> T.Dada19@student.xjtlu.edu.cn</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)   