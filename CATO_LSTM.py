import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tensorflow as tf
from keras.models import load_model
from scipy import interpolate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'stress_strain_model' not in st.session_state:
    st.session_state.stress_strain_model = None

# Title
st.title("FRCRAC Predictor and Visualisation")

# Input Form
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

    st.subheader("Finite Element Parameters")
    load_unit = st.selectbox('Select Load Unit', ['kN', 'MPa'], key="load_unit_selectbox")
    if load_unit == 'kN':
        applied_load_input = st.number_input('Applied Load (kN)', min_value=0.0, max_value=5000.0, value=2000.0)
    else:
        applied_load_input = st.number_input('Applied Stress (MPa)', min_value=0.0, max_value=500.0, value=91.39)

    # Conditional logic
    fibre_type = 1 if frp_type == 'GFRP' else 3
    agg_type = 1 if aggregate_type == 'NA' else 2 if aggregate_type == 'RCA' else 3 if aggregate_type == 'RCL' else 4
    concrete_modulus = 4370 * (unconfined_strength ** 0.5) if aggregate_type == 'NA' else 4120 * (unconfined_strength ** 0.5)

    submit_button = st.form_submit_button("Predict")

# Prediction Logic
if submit_button:
    rupture_strain = float(cato_rupture.predict([[fibre_type, diameter, height, percentage_rca if aggregate_type != 'NA' else 0,
                                                  max_rca_size if aggregate_type != 'NA' else 0, water_cement_ratio,
                                                  unconfined_strength, unconfined_strain, fibre_modulus, frp_overall_thickness,
                                                  agg_type, concrete_modulus]]))
    confinement_stress = 2 * rupture_strain * fibre_modulus * frp_overall_thickness / diameter

    input_data = [fibre_type, diameter, height, percentage_rca, max_rca_size, water_cement_ratio, unconfined_strength,
                  unconfined_strain, fibre_modulus, frp_overall_thickness, agg_type, concrete_modulus, rupture_strain, confinement_stress]
    
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

        # Ensure strictly increasing strain values
        strain_step = unconfined_strain / 20
        section_1 = np.arange(0, unconfined_strain + strain_step, strain_step)
        section_2 = np.arange(unconfined_strain + strain_step, cato_strain_prediction + strain_step, strain_step)
        strain_values = np.concatenate((section_1, section_2))
        
        k_Correction = cato_strain_prediction / (cato_strain_prediction - unconfined_strain)
        stress_values = ((((Modulus_1 * unconfined_strain) - f_o) * np.exp(-strain_values / unconfined_strain)) +
                        f_o + (k_Correction * Modulus_2 * strain_values)) * (1 - np.exp(-strain_values / unconfined_strain))
        
        # Ensure strictly increasing stress values
        mask = np.ones(len(stress_values), dtype=bool)
        mask[1:] = np.diff(stress_values) > 0
        axial_strains = strain_values[mask]
        axial_stresses = stress_values[mask] * 1e6  # Pa
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

        predicted_stress_axial = np.insert(predicted_targets_axial_denorm[0, :, 0], 0, 0) * 1e6  # Pa
        predicted_strain_axial = np.insert(predicted_targets_axial_denorm[0, :, 1], 0, 0)
        
        predicted_stress_hoop = np.insert(predicted_targets_hoop_denorm[0, :, 0], 0, 0) * 1e6  # Pa
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

    # Store predictions including rupture_strain
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
        'load_unit': load_unit,
        'applied_load_input': applied_load_input,
        'rupture_strain': rupture_strain,
        'confinement_stress': confinement_stress
    }
    st.session_state.stress_strain_model = stress_strain_model

    st.subheader("Prediction Results")
    st.write(f"Ultimate Strength: {ultimate_strength:.3f} MPa")
    st.write(f"Ultimate Axial Strain: {100 * ultimate_axial_strain:.3f} %")
    st.write(f"Ultimate Hoop Strain: {100 * ultimate_hoop_strain:.3f} %")
    st.write(f"Strength Enhancement: {strength_enhancement_ratio:.3f}")
    st.write(f"Strain Enhancement: {strain_enhancement_ratio:.3f}")

# FE and Visualisation Logic
if st.session_state.predictions:
    preds = st.session_state.predictions
    axial_stresses = preds['axial_stresses']
    axial_strains = preds['axial_strains']
    hoop_stresses = preds['hoop_stresses']
    hoop_strains = preds['hoop_strains']
    rupture_strain = preds['rupture_strain']
    confinement_stress = preds['confinement_stress'] 
    stress_strain_model = st.session_state.stress_strain_model

    # FE Setup
    epsilon_frp_ult = rupture_strain if stress_strain_model == 'CATO-MZW' else np.max(hoop_strains) if hoop_strains is not None else 0.0103026467
    stress_threshold = 15e6
    axial_mask = axial_stresses >= stress_threshold
    experimental_axial_stresses_filtered = np.concatenate(([0.], axial_stresses[axial_mask]))
    experimental_axial_strains_filtered = np.concatenate(([0.], axial_strains[axial_mask]))

    if stress_strain_model == 'CATO-LSTMO' and hoop_stresses is not None and len(hoop_stresses[hoop_stresses >= stress_threshold]) > 0:
        hoop_mask = hoop_stresses >= stress_threshold
        experimental_hoop_stresses_filtered = np.concatenate(([0.], hoop_stresses[hoop_mask]))
        experimental_hoop_strains_filtered = np.concatenate(([0.], hoop_strains[hoop_mask]))
    else:
        experimental_hoop_stresses_filtered = np.array([0., confinement_stress * 1e6])
        experimental_hoop_strains_filtered = np.array([0., epsilon_frp_ult])

    max_axial_stress = np.max(experimental_axial_stresses_filtered)
    axial_stress_strain_curve = interpolate.CubicSpline(experimental_axial_strains_filtered, experimental_axial_stresses_filtered, extrapolate=True)
    hoop_strain_curve = interpolate.CubicSpline(experimental_hoop_stresses_filtered, experimental_hoop_strains_filtered, extrapolate=True) if stress_strain_model == 'CATO-LSTMO' else None

    indices = np.argsort(experimental_axial_stresses_filtered)
    experimental_axial_stresses_sorted = experimental_axial_stresses_filtered[indices]
    experimental_axial_strains_sorted = experimental_axial_strains_filtered[indices]
    mask = np.ones(len(experimental_axial_stresses_sorted), dtype=bool)
    mask[1:] = np.diff(experimental_axial_stresses_sorted) > 0
    strain_from_stress = interpolate.CubicSpline(experimental_axial_stresses_sorted[mask], experimental_axial_strains_sorted[mask], extrapolate=True)

    diameter_m = preds['diameter'] / 1000
    height_m = preds['height'] / 1000
    radius = diameter_m / 2
    frp_radius = radius + (preds['frp_thickness'] / 1000)
    area = np.pi * radius**2
    initial_height = height_m

    applied_load_input_Pa = preds['applied_load_input'] * 1e3 if preds['load_unit'] == 'kN' else preds['applied_load_input'] * 1e6 * area
    failure_load_value = max_axial_stress * area / 1000
    applied_load_input_Pa = min(applied_load_input_Pa, failure_load_value * 1000)

    def load_displacement_curve(current_load_Pa):
        displacement, load, strain, stress = [], [], [], []
        current_displacement, applied_load = 0, 0
        delta_displacement, max_displacement = 0.00001, 0.1
        while current_displacement < max_displacement:
            strain_at_top = current_displacement / initial_height
            stress_at_base = axial_stress_strain_curve(strain_at_top)
            applied_load = stress_at_base * area
            if stress_at_base >= max_axial_stress or applied_load >= current_load_Pa:
                applied_load = min(current_load_Pa, max_axial_stress * area)
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
        load = np.array(load)
        displacement = np.array(displacement)
        strain = np.array(strain)
        stress = np.array(stress)
        indices = np.argsort(load)
        mask = np.ones(len(load[indices]), dtype=bool)
        mask[1:] = np.diff(load[indices]) > 0
        return (displacement[indices][mask], load[indices][mask], strain[indices][mask], stress[indices][mask])

    # Visualisation
    st.subheader("ML-FEM Post-Processing Visualisation")
    selected_plot = st.selectbox(
        "Select the plot to display",
        ["Load-Displacement Curve", "Stress-Strain Curve", "Stress Contours", "Strain Contours",
         "Load Contours", "Displacement Contours"],
        key="plot_type_selectbox"
    )

    current_load_Pa = applied_load_input_Pa
    displacement, load, strain, stress = load_displacement_curve(current_load_Pa)
    load_displacement_curve_interp = interpolate.CubicSpline(load, displacement, extrapolate=True)

    load_kN = load / 1000
    stress_MPa = stress / 1e6
    displacement_mm = displacement * 1000
    strain_percent = strain * 100
    hoop_strain_from_axial = hoop_strain_curve(stress) if stress_strain_model == 'CATO-LSTMO' else None
    hoop_strain_percent = hoop_strain_from_axial * 100 if stress_strain_model == 'CATO-LSTMO' else None

    n_theta, n_z = 50, 50
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(0, height_m, n_z)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    poissons_ratio = 0.2

    applied_stress = applied_load_input_Pa / area
    axial_stress_3d = applied_stress * (1 - z / height_m)
    axial_stress_3d = np.broadcast_to(axial_stress_3d[:, 0][:, np.newaxis], z.shape)
    radial_stress_3d = poissons_ratio * axial_stress_3d * (1 - np.sqrt(x**2 + y**2) / radius)
    total_stress_3d = axial_stress_3d + radial_stress_3d
    strain_3d = strain_from_stress(total_stress_3d)
    strain_3d_percent = strain_3d * 100

    dz = z[1, 0] - z[0, 0]
    displacement_3d = np.cumsum(strain_3d * dz, axis=0)
    max_displacement_from_curve = load_displacement_curve_interp(applied_load_input_Pa)
    displacement_3d = displacement_3d * (max_displacement_from_curve / np.max(displacement_3d))
    displacement_3d_mm = displacement_3d * 1000

    load_3d = total_stress_3d * area
    load_3d_kN = load_3d / 1000

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z_levels = np.linspace(0, height_m, n_z)
    vertices = np.array([[radius * np.cos(t), radius * np.sin(t), z_val] for z_val in z_levels for t in theta])
    vertices = np.vstack([[0, 0, 0], vertices, [0, 0, height_m]])

    full_faces = []
    for i in range(n_z - 1):
        for j in range(n_theta):
            v0 = i * n_theta + j
            v1 = i * n_theta + (j + 1) % n_theta
            v2 = (i + 1) * n_theta + j
            v3 = (i + 1) * n_theta + (j + 1) % n_theta
            full_faces.append([v0 + 1, v1 + 1, v2 + 1])
            full_faces.append([v1 + 1, v3 + 1, v2 + 1])
    full_faces = np.array(full_faces)
    for j in range(n_theta):
        full_faces = np.vstack([full_faces, [0, 1 + j, 1 + (j + 1) % n_theta]])
    top_base_idx = 1 + (n_z - 1) * n_theta
    for j in range(n_theta):
        full_faces = np.vstack([full_faces, [len(vertices) - 1, top_base_idx + (j + 1) % n_theta, top_base_idx + j]])

    total_stress_vertices = np.array([total_stress_3d[0, 0]] + [total_stress_3d[i, j] for i in range(n_z) for j in range(n_theta)] + [total_stress_3d[-1, 0]]) / 1e6
    strain_vertices = np.array([strain_3d_percent[0, 0]] + [strain_3d_percent[i, j] for i in range(n_z) for j in range(n_theta)] + [strain_3d_percent[-1, 0]])
    load_vertices = np.array([load_3d_kN[0, 0]] + [load_3d_kN[i, j] for i in range(n_z) for j in range(n_theta)] + [load_3d_kN[-1, 0]])
    displacement_vertices = np.array([displacement_3d_mm[0, 0]] + [displacement_3d_mm[i, j] for i in range(n_z) for j in range(n_theta)] + [displacement_3d_mm[-1, 0]])

    frp_vertices = np.array([[frp_radius * np.cos(t), frp_radius * np.sin(t), z_val] for z_val in z_levels for t in theta])
    full_frp_faces = []
    for i in range(n_z - 1):
        for j in range(n_theta):
            v0 = i * n_theta + j
            v1 = i * n_theta + (j + 1) % n_theta
            v2 = (i + 1) * n_theta + j
            v3 = (i + 1) * n_theta + (j + 1) % n_theta
            full_frp_faces.append([v0, v1, v2])
            full_frp_faces.append([v1, v3, v2])
    full_frp_faces = np.array(full_frp_faces)

    edges = set(tuple(sorted([face[i], face[(i + 1) % 3]])) for face in full_faces for i in range(3))
    edges = list(edges)
    wireframe_x = [vertices[v0, 0] for v0, v1 in edges] + [vertices[v1, 0] for v0, v1 in edges] + [None] * len(edges)
    wireframe_y = [vertices[v0, 1] for v0, v1 in edges] + [vertices[v1, 1] for v0, v1 in edges] + [None] * len(edges)
    wireframe_z = [vertices[v0, 2] for v0, v1 in edges] + [vertices[v1, 2] for v0, v1 in edges] + [None] * len(edges)

    frp_edges = set(tuple(sorted([face[i], face[(i + 1) % 3]])) for face in full_frp_faces for i in range(3))
    frp_edges = list(frp_edges)
    frp_wireframe_x = [frp_vertices[v0, 0] for v0, v1 in frp_edges] + [frp_vertices[v1, 0] for v0, v1 in frp_edges] + [None] * len(frp_edges)
    frp_wireframe_y = [frp_vertices[v0, 1] for v0, v1 in frp_edges] + [frp_vertices[v1, 1] for v0, v1 in frp_edges] + [None] * len(frp_edges)
    frp_wireframe_z = [frp_vertices[v0, 2] for v0, v1 in frp_edges] + [frp_vertices[v1, 2] for v0, v1 in frp_edges] + [None] * len(frp_edges)

    aspect_ratio = {'x': 1, 'y': 1, 'z': height_m / diameter_m}

    if selected_plot == "Load-Displacement Curve":
        fig, ax = plt.subplots()
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
        fig, ax = plt.subplots(figsize=(12, 8))
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

        ax.set_xlabel("Strain (%)", fontsize=22)
        ax.set_ylabel("Stress (MPa)", fontsize=22)
        ax.set_title("Stress-Strain Curve", fontsize=22)
        ax.legend(fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=22)
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

    elif selected_plot == "Stress Contours":
        fig = go.Figure()
        if len(full_faces) > 0:
            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                                i=full_faces[:, 0], j=full_faces[:, 1], k=full_faces[:, 2], 
                                intensity=total_stress_vertices, colorscale='Viridis', 
                                colorbar=dict(title="Stress (MPa)", tickformat=".2f", len=0.75, x=1.05), 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                name="Concrete Stress"))
        if len(full_frp_faces) > 0:
            fig.add_trace(go.Mesh3d(x=frp_vertices[:, 0], y=frp_vertices[:, 1], z=frp_vertices[:, 2], 
                                i=full_frp_faces[:, 0], j=full_frp_faces[:, 1], k=full_frp_faces[:, 2], 
                                color='gray', opacity=0.3, 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                showlegend=False))
        if len(wireframe_x) > 0:
            fig.add_trace(go.Scatter3d(x=wireframe_x, y=wireframe_y, z=wireframe_z, mode='lines', 
                                    line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        if len(frp_wireframe_x) > 0:
            fig.add_trace(go.Scatter3d(x=frp_wireframe_x, y=frp_wireframe_y, z=frp_wireframe_z, mode='lines', 
                                    line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        fig.update_layout(title="Stress Contours", height=600, 
                        scene=dict(aspectmode="manual", aspectratio=aspect_ratio, 
                                xaxis_title="X", yaxis_title="Y", zaxis_title="Z", 
                                camera=dict(eye=dict(x=1.5, y=1.5, z=2))))
        st.plotly_chart(fig)

        buf = BytesIO()
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        st.download_button(
            label="Download Stress Contours",
            data=buf,
            file_name="stress_contours.png",
            mime="image/png"
        )

    elif selected_plot == "Strain Contours":
        fig = go.Figure()
        if len(full_faces) > 0:
            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                                i=full_faces[:, 0], j=full_faces[:, 1], k=full_faces[:, 2], 
                                intensity=strain_vertices, colorscale='Cividis', 
                                colorbar=dict(title="Strain (%)", tickformat=".2f", len=0.75, x=1.05), 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                name="Concrete Strain"))
        if len(full_frp_faces) > 0:
            fig.add_trace(go.Mesh3d(x=frp_vertices[:, 0], y=frp_vertices[:, 1], z=frp_vertices[:, 2], 
                                i=full_frp_faces[:, 0], j=full_frp_faces[:, 1], k=full_frp_faces[:, 2], 
                                color='gray', opacity=0.3, 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                showlegend=False))
        if len(wireframe_x) > 0:
            fig.add_trace(go.Scatter3d(x=wireframe_x, y=wireframe_y, z=wireframe_z, mode='lines', 
                                    line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        if len(frp_wireframe_x) > 0:
            fig.add_trace(go.Scatter3d(x=frp_wireframe_x, y=frp_wireframe_y, z=frp_wireframe_z, mode='lines', 
                                    line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        fig.update_layout(title="Strain Contours", height=600, 
                        scene=dict(aspectmode="manual", aspectratio=aspect_ratio, 
                                xaxis_title="X", yaxis_title="Y", zaxis_title="Z", 
                                camera=dict(eye=dict(x=1.5, y=1.5, z=2))))
        st.plotly_chart(fig)

        buf = BytesIO()
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        st.download_button(
            label="Download Strain Contours",
            data=buf,
            file_name="strain_contours.png",
            mime="image/png"
        )

    elif selected_plot == "Load Contours":
        fig = go.Figure()
        if len(full_faces) > 0:
            fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
                                i=full_faces[:, 0], j=full_faces[:, 1], k=full_faces[:, 2], 
                                intensity=load_vertices, colorscale='Plasma', 
                                colorbar=dict(title="Internal Load Distribution (kN)", tickformat=".2f", len=0.75, x=1.05), 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                name="Concrete Load"))
        if len(full_frp_faces) > 0:
            fig.add_trace(go.Mesh3d(x=frp_vertices[:, 0], y=frp_vertices[:, 1], z=frp_vertices[:, 2], 
                                i=full_frp_faces[:, 0], j=full_frp_faces[:, 1], k=full_frp_faces[:, 2], 
                                color='gray', opacity=0.3, 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                showlegend=False))
        if len(wireframe_x) > 0:
            fig.add_trace(go.Scatter3d(x=wireframe_x, y=wireframe_y, z=wireframe_z, mode='lines', 
                                    line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        if len(frp_wireframe_x) > 0:
            fig.add_trace(go.Scatter3d(x=frp_wireframe_x, y=frp_wireframe_y, z=frp_wireframe_z, mode='lines', 
                                    line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        fig.update_layout(title="Internal Load Distribution Contours", height=600, 
                        scene=dict(aspectmode="manual", aspectratio=aspect_ratio, 
                                xaxis_title="X", yaxis_title="Y", zaxis_title="Z", 
                                camera=dict(eye=dict(x=1.5, y=1.5, z=2))))
        st.plotly_chart(fig)

        buf = BytesIO()
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        st.download_button(
            label="Download Internal Load Distribution Contours",
            data=buf,
            file_name="load_contours.png",
            mime="image/png"
        )

    elif selected_plot == "Displacement Contours":
        vertices_deformed = vertices.copy()
        vertices_deformed[0, 2] += displacement_vertices[0] / 1000
        for i in range(n_z):
            for j in range(n_theta):
                vertex_idx = 1 + i * n_theta + j
                vertices_deformed[vertex_idx, 2] += displacement_vertices[vertex_idx] / 1000
        vertices_deformed[-1, 2] += displacement_vertices[-1] / 1000

        frp_vertices_deformed = frp_vertices.copy()
        for i in range(n_z):
            for j in range(n_theta):
                vertex_idx = i * n_theta + j
                frp_vertices_deformed[vertex_idx, 2] += displacement_vertices[1 + i * n_theta + j] / 1000

        wireframe_x_deformed = [vertices_deformed[v0, 0] for v0, v1 in edges] + [vertices_deformed[v1, 0] for v0, v1 in edges] + [None] * len(edges)
        wireframe_y_deformed = [vertices_deformed[v0, 1] for v0, v1 in edges] + [vertices_deformed[v1, 1] for v0, v1 in edges] + [None] * len(edges)
        wireframe_z_deformed = [vertices_deformed[v0, 2] for v0, v1 in edges] + [vertices_deformed[v1, 2] for v0, v1 in edges] + [None] * len(edges)

        frp_wireframe_x_deformed = [frp_vertices_deformed[v0, 0] for v0, v1 in frp_edges] + [frp_vertices_deformed[v1, 0] for v0, v1 in frp_edges] + [None] * len(frp_edges)
        frp_wireframe_y_deformed = [frp_vertices_deformed[v0, 1] for v0, v1 in frp_edges] + [frp_vertices_deformed[v1, 1] for v0, v1 in frp_edges] + [None] * len(frp_edges)
        frp_wireframe_z_deformed = [frp_vertices_deformed[v0, 2] for v0, v1 in frp_edges] + [frp_vertices_deformed[v1, 2] for v0, v1 in frp_edges] + [None] * len(frp_edges)

        fig = go.Figure()
        if len(full_faces) > 0:
            fig.add_trace(go.Mesh3d(x=vertices_deformed[:, 0], y=vertices_deformed[:, 1], z=vertices_deformed[:, 2], 
                                i=full_faces[:, 0], j=full_faces[:, 1], k=full_faces[:, 2], 
                                intensity=displacement_vertices, colorscale='Magma', 
                                colorbar=dict(title="Displacement (mm)", tickformat=".2f", len=0.75, x=1.05), 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                name="Concrete Displacement"))
        if len(full_frp_faces) > 0:
            fig.add_trace(go.Mesh3d(x=frp_vertices_deformed[:, 0], y=frp_vertices_deformed[:, 1], z=frp_vertices_deformed[:, 2], 
                                i=full_frp_faces[:, 0], j=full_frp_faces[:, 1], k=full_frp_faces[:, 2], 
                                color='gray', opacity=0.3, 
                                lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0), 
                                showlegend=False))
        if len(wireframe_x_deformed) > 0:
            fig.add_trace(go.Scatter3d(x=wireframe_x_deformed, y=wireframe_y_deformed, z=wireframe_z_deformed, 
                                    mode='lines', line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        if len(frp_wireframe_x_deformed) > 0:
            fig.add_trace(go.Scatter3d(x=frp_wireframe_x_deformed, y=frp_wireframe_y_deformed, z=frp_wireframe_z_deformed, 
                                    mode='lines', line=dict(color='black', width=1), opacity=0.5, showlegend=False))
        fig.update_layout(title="Displacement Contours", height=600, 
                        scene=dict(aspectmode="manual", aspectratio=aspect_ratio, 
                                xaxis_title="X", yaxis_title="Y", zaxis_title="Z", 
                                camera=dict(eye=dict(x=1.5, y=1.5, z=2))))
        st.plotly_chart(fig)

        buf = BytesIO()
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        st.download_button(
            label="Download Displacement Contours",
            data=buf,
            file_name="displacement_contours.png",
            mime="image/png"
        )

    st.write(f"Maximum axial stress (failure criteria): {max_axial_stress / 1e6:.2f} MPa")
    st.write(f"Calculated failure load: {failure_load_value:.2f} kN")
    st.write(f"Current applied load: {current_load_Pa / 1000:.2f} kN")

# Footer
st.markdown("""
    **Notes**: 
    1. This application predicts the stress-strain behaviours of circular fibre-reinforced polymer confined recycled aggregate concrete (FRCRAC) using machine learning framework.
    2. Three types of recycled aggregates (RA) were considered: RCA, RCL, RBA.
    3. Two types of FRP were considered: GFRP and CFRP.
    4. CATO-MZW: Hybridised Categorical Boosting with modified Zhou and Wu model (axial stress-strain only).
    5. CATO-LSTMO: Hybridised Categorical Boosting and Long Short-Term Memory (axial and hoop stress-strain).
    6. Finite Element Method is integrated with the ML framework for visualisation.
""")

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
    <p>© 2024 My Streamlit App. All rights reserved. | Temitope E. Dada, Guobin Gong, Jun Xia, Luigi Di Sarno | For Queries: <a href="mailto: T.Dada19@student.xjtlu.edu.cn"> T.Dada19@student.xjtlu.edu.cn</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)