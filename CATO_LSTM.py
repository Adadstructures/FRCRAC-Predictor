# Import required libraries for the FRCRAC Predictor and Visualization app
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model
from scipy import interpolate
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from io import BytesIO
import os

# Define selected frames globally
selected_frames = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]

# Apply custom CSS for responsive design and styling
st.markdown("""
<style>
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    .stSlider > div > div > div {
        width: 100%!important;
    }
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    .plotly-graph-div {
        width: 100%!important;
        height: auto!important;
    }
    .stMarkdown, .stText, .stSubheader {
        font-size: clamp(14px, 3vw, 16px);
    }
    .footer {
        position: relative;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: clamp(10px, 2vw, 12px);
        color: #6c757d;
    }
    @media (max-width: 600px) {
        .stNumberInput > div > div > input {
            font-size: 14px;
        }
        .stSelectbox > div > div > select {
            font-size: 14px;
        }
        .plotly-graph-div {
            height: 50vh !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to create Abaqus-style 3D visualization
def create_abaqus_colorscale(values, num_bands=10):
    abaqus_colors = [
        (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)
    ]
    min_val, max_val = np.min(values), np.max(values)
    if max_val == min_val:
        max_val = min_val + 1e-6
        colorscale = [[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']]
        ticks = [min_val]
        ticktext = [f"{min_val:.2e}"]
    else:
        colorscale = [[i/(len(abaqus_colors)-1), f'rgb{c}'] for i, c in enumerate(abaqus_colors)]
        colorscale[-1][0] = 1.0
        ticks = np.linspace(min_val, max_val, num_bands)
        ticktext = [f"{v:.2e}" for v in ticks]
    return colorscale, ticks, ticktext, min_val, max_val

# Load target data and scalers
try:
    targets_1_axial = np.load('targets_axial.npy')
    targets_1_hoop = np.load('targets_hoop.npy')
    scaler_axial = joblib.load('scaler_axial.pkl')
    scaler_hoop = joblib.load('scaler_hoop.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading data or scaler files: {e}")
    st.stop()

# Load machine learning models
try:
    cato_rupture = pickle.load(open('CATO_Rupture.pkl', 'rb'))
    cato_strain = pickle.load(open('CATO_Strain.pkl', 'rb'))
    cato_strength = pickle.load(open('CATO_Strength.pkl', 'rb'))
    lstm_model_axial = load_model('axial_lstm_model.h5')
    lstm_model_hoop = load_model('hoop_lstm_model.h5')
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'stress_strain_model' not in st.session_state:
    st.session_state.stress_strain_model = None

# Set application title
st.title("FRCRAC Predictor and Visualisation")

# Create input form
with st.form("input_form"):
    aggregate_type = st.selectbox("Aggregate Type", ["RCA", "RCL", "RBA", "NA"], index=0, key="aggregate_type_selectbox")
    
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
    frp_type = st.selectbox("Fibre Type", ["GFRP", "CFRP"], index=0, key="frp_type_selectbox")

    st.subheader("Stress-Strain Model")
    stress_strain_model = st.selectbox("Stress-Strain Model", ["CATO-LSTMO", "CATO-MZW"], index=0, key="stress_strain_model_selectbox")

    st.subheader("Displacement Parameter")
    max_displacement = st.number_input("Maximum Displacement (mm)", min_value=0.1, value=10.0)

    fibre_type = 1 if frp_type == 'GFRP' else 3
    agg_type = 1 if aggregate_type == 'NA' else 2 if aggregate_type == 'RCA' else 3 if aggregate_type == 'RCL' else 4
    concrete_modulus = 4370 * (unconfined_strength ** 0.5) if aggregate_type == 'NA' else 4120 * (unconfined_strength ** 0.5)

    submit_button = st.form_submit_button("Predict")

# Perform predictions
if submit_button:
    rupture_strain = float(cato_rupture.predict([[fibre_type, diameter, height, percentage_rca if aggregate_type != 'NA' else 0,
                                                  max_rca_size if aggregate_type != 'NA' else 0, water_cement_ratio,
                                                  unconfined_strength, unconfined_strain, fibre_modulus,
                                                  frp_overall_thickness, agg_type, concrete_modulus]]))
    
    confinement_stress = 2 * rupture_strain * fibre_mod / diameter

    input_data = [fibre_type, diameter, height, percentage_rca, max_rca_size, water_cement_ratio, unconfined_strength,
                  unconfined_strain, fibre_modulus, frp_overall_thickness, agg_type, concrete_modulus, rupture_strain,
                  confinement_stress]
    cato_strength_prediction = float(cato_strength.predict([input_data]))
    cato_strain_prediction = float(cato_strain.predict([input_data]))
    
    if stress_strain_model == 'CATO-MZW':
        ultimate_strength = cato_strength_prediction
        ultimate_axial_strain = cato_strain_prediction
        ultimate_hoop_strain = rupture_strain
        strength_enhancement_ratio = cato_strength_prediction / unconfined_strength
        strain_enhancement_ratio = cato_strain_prediction / unconfined_strain
        
        f_o = unconfined_strength + (0.003 * confinement_stress)
        Modulus_1 = concrete_modulus
        Modulus_2 = (cato_strength_prediction - f_o) / cato_strain_prediction

        strain_step = unconfined_strain / 20
        section_1 = np.arange(0, unconfined_strain + strain_step, strain_step)
        section_2 = np.arange(unconfined_strain + strain_step, cato_strain_prediction + strain_step, strain_step)
        strain_values = np.concatenate((section_1, section_2))
        
        k_Correction = cato_strain_prediction / (cato_strain_prediction - unconfined_strain)
        stress_values = ((((Modulus_1 * unconfined_strain) - f_o) * np.exp(-strain_values / unconfined_strain)) +
                        f_o + (k_Correction * Modulus_2 * strain_values)) * (1 - np.exp(-strain_values / unconfined_strain))
        
        mask = np.ones(len(stress_values), dtype=bool)
        mask[1:] = np.diff(stress_values) > 0
        axial_stresses = stress_values[mask] * 1e6
        axial_strains = strain_values[mask]
        hoop_stresses = None
        hoop_strains = None

    else:  # CATO-LSTMO
        input_data_lstm = input_data + [cato_strength_prediction, cato_strain_prediction]
        input_data_axial = np.array(input_data_lstm).reshape(1, -1)
        input_data_hoop = np.array(input_data_lstm).reshape(1, -1)
        
        new_inputs_normalized_axial = scaler_axial.transform(input_data_axial)
        new_inputs_normalized_hoop = scaler_hoop.transform(input_data_hoop)
        
        predicted_targets_axial = lstm_model_axial.predict(new_inputs_normalized_axial)
        predicted_targets_hoop = lstm_model_hoop.predict(new_inputs_normalized_hoop)

        predicted_targets_axial_denorm = predicted_targets_axial * targets_1_axial.max(axis=(0, 1))
        predicted_targets_hoop_denorm = predicted_targets_hoop * targets_1_hoop.max(axis=(0, 1))

        predicted_stress_axial = np.insert(predicted_targets_axial_denorm[0, :, 0], 0, 0) * 1e6
        predicted_strain_axial = np.insert(predicted_targets_axial_denorm[0, :, 1], 0, 0)
        predicted_stress_hoop = np.insert(predicted_targets_hoop_denorm[0, :, 0], 0, 0) * 1e6
        predicted_strain_hoop = np.insert(-predicted_targets_hoop_denorm[0, :, 1], 0, 0)
        
        ultimate_axial_strength = np.max(predicted_stress_axial) / 1e6
        ultimate_hoop_strength = np.max(predicted_stress_hoop) / 1e6
        lstm_axial_strain_ultimate = predicted_strain_axial[np.argmax(predicted_stress_axial)]
        lstm_hoop_strain_ultimate = predicted_strain_hoop[np.argmax(predicted_stress_hoop)]

        ultimate_strength = max(ultimate_axial_strength, ultimate_hoop_strength)
        ultimate_axial_strain = lstm_axial_strain_ultimate
        ultimate_hoop_strain = lstm_hoop_strain_ultimate
        strength_enhancement_ratio = ultimate_strength / unconfined_strength
        strain_enhancement_ratio = lstm_axial_strain_ultimate / unconfined_strain

        axial_stresses = predicted_stress_axial
        axial_strains = predicted_strain_axial
        hoop_stresses = predicted_stress_hoop
        hoop_strains = np.abs(predicted_strain_hoop)

    st.session_state.predictions = {
        'axial_stresses': axial_stresses,
        'axial_strains': axial_strains,
        'hoop_stresses': hoop_stresses,
        'hoop_strains': hoop_strains,
        'ultimate_strength': ultimate_strength,
        'ultimate_axial_strain': ultimate_axial_strain,
        'ultimate_hoop_strain': ultimate_hoop_strain,
        'strength_enhancement_ratio': strength_enhancement_ratio,
        'strain_enhancement_ratio': strain_enhancement_ratio,
        'diameter': diameter,
        'height': height,
        'frp_thickness': frp_overall_thickness,
        'max_displacement': max_displacement,
        'rupture_strain': rupture_strain,
        'confinement_stress': confinement_stress,
        'unconfined_strength': unconfined_strength,
        'fibre_modulus': fibre_modulus
    }
    st.session_state.stress_strain_model = stress_strain_model

    st.subheader("Prediction Results")
    st.write(f"Ultimate Strength: {ultimate_strength:.3f} MPa")
    st.write(f"Ultimate Axial Strain: {100 * ultimate_axial_strain:.3f} %")
    st.write(f"Ultimate Hoop Strain: {100 * ultimate_hoop_strain:.3f} %")
    st.write(f"Strength Enhancement: {strength_enhancement_ratio:.3f}")
    st.write(f"Strain Enhancement: {strain_enhancement_ratio:.3f}")

# Visualization section
if st.session_state.predictions:
    preds = st.session_state.predictions
    axial_stresses = preds['axial_stresses']
    axial_strains = preds['axial_strains']
    hoop_stresses = preds['hoop_stresses']
    hoop_strains = preds['hoop_strains']
    rupture_strain = preds['rupture_strain']
    confinement_stress = preds['confinement_stress']
    stress_strain_model = st.session_state.stress_strain_model
    unconfined_strength = preds['unconfined_strength']
    frp_thickness = preds['frp_thickness']
    fibre_modulus = preds['fibre_modulus']
    max_displacement = preds['max_displacement']
    diameter = preds['diameter']
    height = preds['height']

    st.subheader("Post-Processing Visualisation")
    selected_plot = st.selectbox(
        "Select the plot to display",
        ["Load-Displacement Curve", "Stress-Strain Curve", "Stress Contours", "Strain Contours", "Displacement Contours"],
        key="plot_type_selectbox"
    )

    epsilon_frp_ult = rupture_strain if stress_strain_model == 'CATO-MZW' else np.max(hoop_strains) if hoop_strains is not None else 0.0103026467
    stress_threshold = 15e6
    axial_mask = axial_stresses >= stress_threshold
    experimental_axial_stresses_filtered = np.concatenate(([0.], axial_stresses[axial_mask]))
    experimental_axial_strains_filtered = np.concatenate(([0.], axial_strains[axial_mask]))

    if stress_strain_model == 'CATO-LSTMO' and hoop_stresses is not None and len(hoop_stresses[hoop_stresses >= stress_threshold]) > 0:
        hoop_mask = hoop_stresses >= stress_threshold
        experimental_hoop_stresses_filtered = np.concatenate(([0], hoop_stresses[hoop_mask]))
        experimental_hoop_strains_filtered = np.concatenate(([0], hoop_strains[hoop_mask]))
    else:
        experimental_hoop_stresses_filtered = np.array([0., confinement_stress * 1e6])
        experimental_hoop_strains_filtered = np.array([0., epsilon_frp_ult])

    axial_stress_strain_curve = interpolate.CubicSpline(experimental_axial_strains_filtered, experimental_axial_stresses_filtered, extrapolate=True)
    hoop_strain_curve = interpolate.CubicSpline(experimental_hoop_stresses_filtered, experimental_hoop_strains_filtered, extrapolate=True) if stress_strain_model == 'CATO-LSTMO' else None

    indices = np.argsort(experimental_axial_stresses_filtered)
    experimental_axial_stresses_sorted = experimental_axial_stresses_filtered[indices]
    experimental_axial_strains_sorted = experimental_axial_strains_filtered[indices]
    mask = np.ones(len(experimental_axial_stresses_sorted), dtype=bool)
    mask[1:] = np.diff(experimental_axial_stresses_sorted) > 0
    strain_from_stress = interpolate.CubicSpline(experimental_axial_stresses_sorted[mask], experimental_axial_strains_sorted[mask], extrapolate=True)

    diameter_m = diameter / 1000
    height_m = height / 1000
    radius = diameter_m / 2
    area = np.pi * radius**2
    initial_height = height_m
    max_axial_stress = np.max(experimental_axial_stresses_filtered)
    failure_load_value = max_axial_stress * area / 1000

    def load_displacement_curve(max_displacement_m):
        displacement, load, strain, stress = [], [], [], []
        current_displacement, applied_load = 0, 0
        delta_displacement = 0.00001
        max_displacement_m = max_displacement_m / 1000
        while current_displacement < max_displacement_m:
            strain_at_top = current_displacement / initial_height
            stress_at_base = axial_stress_strain_curve(strain_at_top)
            applied_load = stress_at_base * area
            if stress_at_base >= max_axial_stress:
                applied_load = max_axial_stress * area
                displacement.append(current_displacement)
                load.append(applied_load)
                strain.append(strain_at_top)
                stress.append(stress_at_base)
                break
            displacement.append(current_displacement)
            load.append(applied_load)
            strain.append(strain_at_top)
            stress.append(stress_at_base)
            current_displacement += delta_displacement
        return np.array(displacement), np.array(load), np.array(strain), np.array(stress)

    if selected_plot == "Load-Displacement Curve":
        displacement, load, _, _ = load_displacement_curve(max_displacement)
        load_kN = load / 1000
        displacement_mm = displacement * 1000

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(displacement_mm, load_kN, 'b-')
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Load (kN)")
        fig.suptitle("Load-Displacement Curve")
        plt.tight_layout()
        st.pyplot(fig)
        
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download Load-Displacement Curve",
            data=buf,
            file_name="load_displacement_curve.png",
            mime="image/png"
        )
        plt.close(fig)

    elif selected_plot == "Stress-Strain Curve":
        fig, ax = plt.subplots(figsize=(8, 6))
        if stress_strain_model == 'CATO-MZW':
            ax.plot(axial_strains * 100, axial_stresses / 1e6, 'b-', label='CATO-MZW')
        else:
            ax.plot(axial_strains * 100, axial_stresses / 1e6, 'b-', label='CATO-LSTMO')
            ax.plot(-hoop_strains * 100, hoop_stresses / 1e6, 'b-')

        st.subheader("Upload Experimental Data for Comparison")
        uploaded_file_axial = st.file_uploader("Upload axial stress-strain CSV file", type=["csv"], key="axial_exp")
        uploaded_file_hoop = st.file_uploader("Upload hoop stress-strain CSV file", type=["csv"], key="hoop_exp")

        if uploaded_file_axial is not None:
            df_axial = pd.read_csv(uploaded_file_axial)
            ax.plot(df_axial['Strain'] * 100, df_axial['Stress'], 'r-', label='Experiment')
            st.success("Axial experimental data loaded.")
        if uploaded_file_hoop is not None and stress_strain_model == 'CATO-LSTMO':
            df_hoop = pd.read_csv(uploaded_file_hoop)
            ax.plot(df_hoop['Strain'] * 100, df_hoop['Stress'], 'r-')
            st.success("Hoop experimental data loaded.")

        ax.set_xlabel("Strain (%)", fontsize=14)
        ax.set_ylabel("Stress (MPa)", fontsize=14)
        ax.set_title("Stress-Strain Curve", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download Stress-Strain Curve",
            data=buf,
            file_name="stress_strain_curve.png",
            mime="image/png"
        )
        plt.close(fig)

    else:  # 3D Contours
        try:
            stress_model = pickle.load(open('vis_stress_model.pkl', 'rb'))
            strain_model = pickle.load(open('vis_strain_model.pkl', 'rb'))
            disp_model = pickle.load(open('vis_disp_model.pkl', 'rb'))
            stress_scaler = joblib.load('vis_stress_scaler.pkl')
            strain_scaler = joblib.load('vis_strain_scaler.pkl')
            disp_scaler = joblib.load('vis_disp_scaler.pkl')
        except FileNotFoundError as e:
            st.error(f"Error loading visualization model files: {e}")
            st.stop()

        def load_mesh():
            nodes_path = 'frp_mesh_with_manual_caps_nodes.csv'
            faces_path = 'frp_mesh_with_manual_caps_faces.csv'
            try:
                xyz = pd.read_csv(nodes_path)[["X", "Y", "Z"]].values
                faces = pd.read_csv(faces_path)
                i, j, k = faces["i"].values, faces["j"].values, faces["k"].values
                node_count = len(xyz)
                st.write(f"Mesh loaded: {node_count} nodes, {len(i)} faces")
                return xyz, i, j, k, node_count
            except FileNotFoundError as e:
                st.error(f"Error loading mesh files: {e}")
                st.stop()
            except KeyError as e:
                st.error(f"Mesh CSV files missing required columns (X, Y, Z or i, j, k): {e}")
                st.stop()

        xyz, i, j, k, node_count = load_mesh()

        if diameter < 1e-6 or height < 1e-6:
            st.error("Diameter or height too small.")
            st.stop()
        original_diameter = 150.0
        original_height = 300.0
        diameter_scale = diameter / original_diameter
        height_scale = height / original_height

        xyz_scaled = xyz.copy()
        xyz_scaled[:, 0] *= diameter_scale
        xyz_scaled[:, 2] *= diameter_scale
        xyz_scaled[:, 1] *= height_scale
        st.write(f"Input: diameter={diameter:.2f} mm, height={height:.2f} mm")
        st.write(f"Scales: diameter_scale={diameter_scale:.2f}, height_scale={height_scale:.2f}")
        st.write(f"Scaled mesh: X={xyz_scaled[:,0].min():.2f}/{xyz_scaled[:,0].max():.2f}, "
                 f"Y={xyz_scaled[:,1].min():.2f}/{xyz_scaled[:,1].max():.2f}, "
                 f"Z={xyz_scaled[:,2].min():.2f}/{xyz_scaled[:,2].max():.2f}")

        num_stress_points = len(axial_stresses)
        if num_stress_points < 2:
            stress_values = np.array([0, np.max(axial_stresses) / 1e6])
            strain_values = np.array([0, np.max(axial_strains)])
            num_stress_points = 2
        else:
            stress_values = axial_stresses / 1e6
            strain_values = axial_strains
        
        indices = np.argsort(stress_values)
        stress_values = stress_values[indices]
        strain_values = strain_values[indices]
        mask = np.ones(len(stress_values), dtype=bool)
        mask[1:] = np.diff(stress_values) > 0
        stress_values = stress_values[mask]
        strain_values = strain_values[mask]

        stress_interp = interp1d(np.linspace(0, 1, num_stress_points)[mask], stress_values, bounds_error=False, fill_value="extrapolate")
        strain_interp = interp1d(stress_values, strain_values, bounds_error=False, fill_value="extrapolate")
        st.write(f"Stress interpolation range: {stress_interp(0):.2f} to {stress_interp(1):.2f} MPa")
        st.write(f"Strain interpolation max: {strain_interp(stress_interp(1)):.4f}")

        bottom_center_idx = node_count - 2
        top_center_idx = node_count - 1
        y_vals = xyz_scaled[:, 1]
        tol = 1e-2 * height_scale
        bottom_ring = np.where(np.abs(y_vals - y_vals.min()) < tol)[0]
        top_ring = np.where(np.abs(y_vals - y_vals.max()) < tol)[0]

        top_surface_nodes = set()
        for face in zip(i, j, k):
            face_nodes = [face[0], face[1], face[2]]
            if all(np.abs(y_vals[node] - y_vals.max()) < tol for node in face_nodes):
                top_surface_nodes.update(face_nodes)
        top_surface_nodes = np.array(list(top_surface_nodes))

        vis_mode_map = {
            "Stress Contours": "Stress (MPa)",
            "Strain Contours": "Strain (%)",
            "Displacement Contours": "Displacement (mm)"
        }
        vis_mode = vis_mode_map[selected_plot]

        def predict_all_frames(vis_mode, node_count, stress_model, strain_model, disp_model, 
                              stress_scaler, strain_scaler, disp_scaler, unconfined_strength, 
                              frp_thickness, fibre_modulus, bottom_center_idx, top_center_idx, 
                              bottom_ring, top_ring, top_surface_nodes, stress_interp, strain_interp, 
                              max_displacement):
            all_scaled = []
            for idx in selected_frames:
                if idx == 0:
                    scaled = np.zeros(node_count)
                else:
                    df_feat = pd.DataFrame({
                        "Unconfined_Strength": [unconfined_strength] * node_count,
                        "FRP_Thickness": [frp_thickness] * node_count,
                        "Fibre_Modulus": [fibre_modulus] * node_count,
                        "Frame_Index": [idx] * node_count,
                        "Node_Label": list(range(node_count))
                    })
                    st.write(f"df_feat columns: {list(df_feat.columns)}")

                    try:
                        X_stress = stress_scaler.transform(df_feat)
                        X_strain = strain_scaler.transform(df_feat)
                        X_disp = disp_scaler.transform(df_feat)
                    except ValueError:
                        st.warning("Scaler feature name mismatch. Using values directly.")
                        X_stress = stress_scaler.transform(df_feat.values)
                        X_strain = strain_scaler.transform(df_feat.values)
                        X_disp = disp_scaler.transform(df_feat.values)

                    s_vals = stress_model.predict(X_stress)
                    strain_vals = strain_model.predict(X_strain)
                    u_vals = disp_model.predict(X_disp)

                    for arr in [s_vals, strain_vals, u_vals]:
                        if arr[bottom_center_idx] == 0:
                            arr[bottom_center_idx] = np.mean(arr[bottom_ring])
                        if arr[top_center_idx] == 0:
                            arr[top_center_idx] = np.mean(arr[top_ring])

                    norm_factor = np.max(np.abs(s_vals)) or 1.0
                    frac = idx / 21.0
                    if vis_mode == "Stress (MPa)":
                        scaled = s_vals * stress_interp(frac) / norm_factor
                    elif vis_mode == "Strain (%)":
                        stress_scaled = s_vals * stress_interp(frac) / norm_factor
                        scaled = strain_interp(stress_scaled) * 100
                    else:  # Displacement (mm)
                        disp_scale = max_displacement * frac
                        scaled = u_vals * disp_scale / (np.max(np.abs(u_vals)) or 1.0)
                        scaled[top_surface_nodes] = np.mean(scaled[top_surface_nodes])

                all_scaled.append(scaled)
            return np.array(all_scaled)

        try:
            frame_values = predict_all_frames(
                vis_mode=vis_mode,
                node_count=node_count,
                stress_model=stress_model,
                strain_model=strain_model,
                disp_model=disp_model,
                stress_scaler=stress_scaler,
                strain_scaler=strain_scaler,
                disp_scaler=disp_scaler,
                unconfined_strength=unconfined_strength,
                frp_thickness=frp_thickness,
                fibre_modulus=fibre_modulus,
                bottom_center_idx=bottom_center_idx,
                top_center_idx=top_center_idx,
                bottom_ring=bottom_ring,
                top_ring=top_ring,
                top_surface_nodes=top_surface_nodes,
                stress_interp=stress_interp,
                strain_interp=strain_interp,
                max_displacement=max_displacement
            )
            st.write(f"Generated {len(frame_values)} frames, indices: {selected_frames}")
            st.write(f"Frame 11 (index {selected_frames[-1]}): {np.max(np.abs(frame_values[-1])):.2e} {vis_mode}")
        except Exception as e:
            st.error(f"Error generating frames: {e}")
            st.stop()

    def generate_edges():
        edge_set = set()
        for tri in zip(i, j, k):
            for a in, b in [(0, 1), (1, 2), (2, 0)]:
                edge_set.add(tuple(sorted((tri[a], tri[b]))))
        edges = list(edge_set)
        x_lines, y_lines, z_lines = [], [], []
        for e in edges:
            p1, p2 = xyz_scaled[e[0]], xyz_scaled[e[1]]
            x_lines.extend([p1[0], p2[0], None])
            y_lines.extend([p1[1], p2[1], None])
            z_lines.extend([p1[2], p2[2], None])
        return x_lines, y_lines, z_lines

    x_lines, y_lines, z_lines = generate_edges(x_lines)

    frame_idx = st.select_slider("Select Frame", options=selected_frames, value=0, key="frame_idx")
    frame_num = selected_frames.index(frame_idx)

    frames = []
    for idx, val in enumerate(frame_values):
        colorscale, _, _, _, _ = create_abaqus_colorscale(val)
        mesh = go.Mesh3d(
            x=xyz_scaled[:, 0],
            y=xyz_scaled[:, 1],
            z=xyz_scaled[:, 2],
            i=i,
            j=j,
            k=k,
            intensity=val,
            intensitymode="vertex",
            colorscale=colorscale,
            flatshading=False,
            lighting=dict(
                ambient=0.9,
                diffuse=0.1,
                specular=0.0
            ),
            colorbar=dict(
                title=vis_mode,
                len=0.5,
                x=1.02,
                thickness=15,
                tickfont=dict(size=10)
            ),
            showscale=True,
            opacity=1.0
        )
        wire = go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode="lines",
            line=dict(
                color="black",
                width=1
            )
            showlegend=False,
        )
        frames.append(go.Frame(
            data=[mesh, wire],
            name=str(idx),
            layout=dict(
                title_text=f"{vis_mode} – Frame {selected_frames[idx]}"
            )
        ))

    colorscale, _, _, min_v, max_v = create_abaqus_colorscale(frame_values[frame_num])
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=xyz_scaled[:, 0],
                y=xyz_scaled[:, 0],
                z=xyz_scaled[:, 2],
                i=i,
                j=j,
                k=k,
                intensity=frame_values[frame_num],
                intensitymode="vertex",
                colorscale=colorscale,
                cmin=min_v,
                cmax=cmax_v,
                flatshading=False,
                lighting=dict(
                    ambient=0.9,
                    diffuse=0.1,
                    specular=0.0
                ),
                colorbar=dict(
                    title=vis_mode,
                    len=0.5,
                    x=1.02,
                    thickness=15,
                    tickfont=dict(size=10)
                ),
                showscale=True,
                opacity=1.0
            ),
            go.Scatter3d(
                x=x_lines,
                y=y_lines,
                z_lines=z_lines,
                mode="lines",
                line=dict(
                    color="black",
                    width=1
                ),
                showlegend=False
            )
        ],
        frames=frames,
        layout=go.Layout(
            title=f"{vis_mode} – Frame {frame_idx}",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y (Height)",
                zaxis_title="Z",
                aspectmode="data"
            ),
            margin=dict(l=10, r=100, t=100, b=50),
            sliders=[{
                    "steps": [
                        dict(
                            method="animate",
                            "label": str(selected_frames[k]),
                            args=[[str(k)], {
                                "mode": "immediate",
                                "frame": {"duration": 300, "redraw": True},
                                "transition": {"duration": 100}
                            }]
                        )
                    ] for k in range(len(frame_values))
                    ],
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top",
                    "len": 0.9,
                    "font": {"size": 10}
                }],
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 300, "redraw": True},
                            "fromcurrent": True
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "mode": "immediate",
                            "frame": {"duration": 0},
                            "transition": {"duration": 0}
                        }]
                    }
                ],
                "direction": "left",
                "x": 0.1,
                "xanchor": "left",
                "y": -0.1,
                "yanchor": "top",
                "font": {"size": 10}
            }]
        )
    )

    fig.update_layout(
        dragmode="orbit",
        scene_dragmode="orbit",
        hovermode=False
    )

    with st.spinner("Generating 3D visualization..."):
        st.plotly_chart(fig, use_container_width=True)

    buf = BytesIO()
    fig.write_image(buf, format="png", engine="kaleido")
    buf.seek(0)
    st.download_button(
        label=f"Download {vis_mode} Visualization (Frame {frame_idx})",
        data=buf,
        file_name=f"{vis_mode.replace(' ', '_').lower()}_frame_{frame_idx}.png",
        mime="image/png"
    )

st.subheader("Performance Summary")
st.write(f"Maximum Axial Stress: {max_axial_stress / 1e6:.2f} MPa")
st.write(f"Calculated Failure Load: {failure_load_value:.2f} kN")
st.write(f"Maximum Displacement: {max_displacement:.2f} mm")

st.markdown("""
**Notes**: 
1. This app predicts stress-strain behaviors of FRCRAC using ML.
2. Aggregates: RCA, RCL, RBA, NA.
3. FRP: GFRP, CFRP.
4. CATO-MZW: CatBoost + Zhou/Wu (axial).
5. CATO-LSTMO: CatBoost + LSTM (CFEA).
6. Visualizations mimic Abaqus FEA, mapping 12 frames to stress-strain curve.
7. Frame 12 shows maximum load.
8. Contact: T.Dada19@example.com
""")

footer = """
<div class="footer">
    <p>© 2025 | Temitope E. Dada, Guobin Gong, Jun Xia, Luigi Di Sarno | <a href="mailto:T.Dada19@example.com">T.Dada19@example.com</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
