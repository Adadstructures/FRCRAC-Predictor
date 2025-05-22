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

# Optimized plot_abaqus_style function (unused but kept for reference)
def plot_abaqus_style(vertices, faces, scalar_per_face, title, colorbar_title, min_val=None, max_val=None, aspect_ratio=None, num_bands=10):
    if min_val is None:
        min_val = np.min(scalar_per_face)
    if max_val is None:
        max_val = np.max(scalar_per_face)

    if max_val == min_val:
        max_val = min_val + 1e-10

    normalized_scalars = (scalar_per_face - min_val) / (max_val - min_val)
    normalized_scalars = np.clip(normalized_scalars, 0, 1)

    abaqus_colors = [
        (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)
    ]
    colorscale = []
    for i in range(num_bands):
        frac = i / (num_bands - 1)
        idx = int(frac * (len(abaqus_colors) - 1))
        rgb = abaqus_colors[idx]
        colorscale.append([i / (num_bands - 1), f'rgb{rgb}'])
    colorscale[-1][0] = 1.0

    band_edges = np.linspace(0, 1, num_bands, endpoint=True)
    color_indices = np.digitize(normalized_scalars, band_edges, right=True)
    color_indices = np.clip(color_indices, 0, num_bands - 1)
    face_colors = [colorscale[min(idx, num_bands - 1)][1] for idx in color_indices]

    tick_vals = np.linspace(min_val, max_val, num_bands)

    def format_sci_notation(val):
        if abs(val) < 1e-10:
            return "+0.00e+00"
        coeff, exp = f"{val:.2e}".split('e')
        exp = int(exp)
        sign = '+' if exp >= 0 else '-'
        exp_str = f"{abs(exp):02d}"
        return f"{coeff}e{sign}{exp_str}"

    tick_text = [format_sci_notation(v) for v in tick_vals]

    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        facecolor=face_colors, intensity=scalar_per_face, colorscale=colorscale,
        showscale=True, flatshading=True, opacity=1.0, hoverinfo='skip',
        lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0),
        colorbar=dict(title=colorbar_title, tickvals=tick_vals, ticktext=tick_text, len=0.75, x=1.05)
    ))

    edges = set(tuple(sorted([face[i], face[(i + 1) % 3]])) for face in faces for i in range(3))
    wireframe_x, wireframe_y, wireframe_z = [], [], []
    for v0, v1 in edges:
        wireframe_x.extend([vertices[v0, 0], vertices[v1, 0], None])
        wireframe_y.extend([vertices[v0, 1], vertices[v1, 1], None])
        wireframe_z.extend([vertices[v0, 2], vertices[v1, 2], None])
    
    fig.add_trace(go.Scatter3d(
        x=wireframe_x, y=wireframe_y, z=wireframe_z,
        mode='lines', line=dict(color='black', width=2), showlegend=False, hoverinfo='skip'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="manual", aspectratio=aspect_ratio,
            camera=dict(eye=dict(x=1.5, y=1.5, z=2))
        )
    )

    return fig

# Helper function to compute face scalars
def compute_face_scalars(vertices_scalar, faces):
    face_scalars = np.zeros(len(faces))
    for i, face in enumerate(faces):
        face_scalars[i] = np.mean(vertices_scalar[face])
    return face_scalars

# Load targets and scalers
try:
    targets_1_axial = np.load('targets_axial.npy')
    targets_1_hoop = np.load('targets_hoop.npy')
    scaler_axial = joblib.load('scaler_axial.pkl')
    scaler_hoop = joblib.load('scaler_hoop.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading data or scaler files: {e}")
    st.stop()

# Load models
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

    st.subheader("Displacement Parameter")
    max_displacement = st.number_input("Maximum Displacement (mm)", min_value=0.1, value=10.0)

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

# Visualization Section
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

    # Analytical Setup for Load-Displacement and Stress-Strain Curves
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

    diameter_m = diameter / 1000
    height_m = height / 1000
    radius = diameter_m / 2
    area = np.pi * radius**2
    initial_height = height_m
    failure_load_value = max_axial_stress * area / 1000

    def load_displacement_curve(max_displacement_m):
        displacement, load, strain, stress = [], [], [], []
        current_displacement, applied_load = 0, 0
        delta_displacement = 0.00001
        max_displacement_m = max_displacement_m / 1000  # Convert mm to m
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

    # Visualization Logic
    if selected_plot == "Load-Displacement Curve":
        displacement, load, _, _ = load_displacement_curve(max_displacement)
        load_kN = load / 1000
        displacement_mm = displacement * 1000

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

    else:  # 3D Contours (Stress, Strain, Displacement)
        # Load pretrained models and scalers
        try:
            stress_model = pickle.load(open('vis_stress_model.pkl', 'rb'))
            strain_model = pickle.load(open('vis_strain_model.pkl', 'rb'))
            disp_model = pickle.load(open('vis_disp_model.pkl', 'rb'))
            stress_scaler = joblib.load('vis_stress_scaler.pkl')
            strain_scaler = joblib.load('vis_strain_scaler.pkl')
            disp_scaler = joblib.load('vis_disp_scaler.pkl')
        except FileNotFoundError as e:
            st.error(f"Error loading visualization model or scaler files: {e}")
            st.stop()

        # Load mesh
        try:
            nodes_path = 'frp_mesh_with_manual_caps_nodes.csv'
            faces_path = 'frp_mesh_with_manual_caps_faces.csv'
            xyz = pd.read_csv(nodes_path)[["X", "Y", "Z"]].values
            faces = pd.read_csv(faces_path)
            i, j, k = faces["i"].values, faces["j"].values, faces["k"].values
            node_count = len(xyz)
        except FileNotFoundError as e:
            st.error(f"Error loading mesh files: {e}")
            st.stop()

        # Use predicted stress-strain data
        stress_interp = interp1d(np.linspace(0, 1, len(axial_stresses) if len(axial_stresses) > 1 else 2), 
                                axial_stresses / 1e6 if len(axial_stresses) > 1 else np.array([0, np.max(axial_stresses) / 1e6]))
        strain_interp = interp1d(axial_stresses / 1e6 if len(axial_stresses) > 1 else np.array([0, np.max(axial_stresses) / 1e6]), 
                                axial_strains if len(axial_strains) > 1 else np.array([0, np.max(axial_strains)]), 
                                bounds_error=False, fill_value="extrapolate")

        # Abaqus-like colorscale function
        def create_abaqus_colorscale(values, num_bands=10):
            abaqus_colors = [(0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)]
            min_val, max_val = np.min(values), np.max(values)
            if max_val == min_val:
                max_val = min_val + 1e-6  # Avoid division by zero
                colorscale = [[0, 'rgb(0,0,255)'], [1, 'rgb(0,0,255)']]  # Single color for frame 0
                ticks = [min_val]
                ticktext = [f"{min_val:.2e}"]
            else:
                colorscale = [[i/(len(abaqus_colors)-1), f'rgb{c}'] for i, c in enumerate(abaqus_colors)]
                ticks = np.linspace(min_val, max_val, num_bands)
                ticktext = [f"{v:.2e}" for v in ticks]
            return colorscale, ticks, ticktext, min_val, max_val

        # Map selected plot to visualization mode
        vis_mode_map = {
            "Stress Contours": "Stress (MPa)",
            "Strain Contours": "Strain (%)",
            "Displacement Contours": "Displacement (mm)"
        }
        vis_mode = vis_mode_map[selected_plot]

        # Frame selection
        frame_idx = st.slider("Select Frame", 0, 20, 0, key="frame_idx_slider")

        # Identify bottom and top nodes
        bottom_center_idx = node_count - 2
        top_center_idx = node_count - 1
        y_vals = xyz[:, 1]
        tol = 1e-2
        bottom_ring = np.where(np.abs(y_vals - y_vals.min()) < tol)[0]
        top_ring = np.where(np.abs(y_vals - y_vals.max()) < tol)[0]

        # Identify top surface nodes (top cap)
        top_surface_nodes = set()
        for face in zip(i, j, k):
            face_nodes = [face[0], face[1], face[2]]
            # Check if all nodes in the face are at the maximum y-value (top cap)
            if all(np.abs(y_vals[node] - y_vals.max()) < tol for node in face_nodes):
                top_surface_nodes.update(face_nodes)
        top_surface_nodes = np.array(list(top_surface_nodes))

        # Predict all frame values with explicit zero at frame 0 and top surface matching top ring for displacement
        @st.cache_data
        def predict_all_frames(_unconfined_strength, _frp_thickness, _fibre_modulus, _max_displacement):
            all_scaled = []
            for idx in range(21):
                if idx == 0:
                    # Explicitly set all values to zero for frame 0
                    scaled = np.zeros(node_count)
                else:
                    df_feat = pd.DataFrame({
                        "Unconfined_Strength (MPa)": [unconfined_strength] * node_count,
                        "FRP_Thickness (mm)": [frp_thickness] * node_count,
                        "Fibre_Modulus (MPa)": [fibre_modulus] * node_count,
                        "Frame_Index": [idx] * node_count,
                        "Node_Label": list(range(node_count))
                    })

                    X_stress = stress_scaler.transform(df_feat)
                    X_strain = strain_scaler.transform(df_feat)
                    X_disp = disp_scaler.transform(df_feat)

                    s_vals = stress_model.predict(X_stress)
                    strain_vals = strain_model.predict(X_strain)
                    u_vals = disp_model.predict(X_disp)

                    for arr in [s_vals, strain_vals, u_vals]:
                        if arr[bottom_center_idx] == 0:
                            arr[bottom_center_idx] = np.mean(arr[bottom_ring])
                        if arr[top_center_idx] == 0:
                            arr[top_center_idx] = np.mean(arr[top_ring])

                    if vis_mode == "Stress (MPa)":
                        norm = s_vals / (np.max(np.abs(s_vals)) or 1.0)
                        scaled = norm * stress_interp(idx / 20)
                    elif vis_mode == "Strain (%)":
                        stress_scaled = s_vals / (np.max(np.abs(s_vals)) or 1.0) * stress_interp(idx / 20)
                        scaled = strain_interp(stress_scaled) * 100
                    else:  # Displacement (mm)
                        norm = u_vals / (np.max(np.abs(u_vals)) or 1.0)
                        scaled = norm * max_displacement
                        # Set top surface nodes to mean of top ring nodes
                        top_ring_mean = np.mean(scaled[top_ring])
                        scaled[top_surface_nodes] = top_ring_mean

                all_scaled.append(scaled)
            return np.array(all_scaled)

        frame_values = predict_all_frames(unconfined_strength, frp_thickness, fibre_modulus, max_displacement)

        # Generate wireframe
        def generate_edges(i, j, k):
            edge_set = set()
            for tri in zip(i, j, k):
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    edge_set.add(tuple(sorted((tri[a], tri[b]))))
            return list(edge_set)

        edges = generate_edges(i, j, k)
        x_lines, y_lines, z_lines = [], [], []
        for e in edges:
            p1, p2 = xyz[e[0]], xyz[e[1]]
            x_lines += [p1[0], p2[0], None]
            y_lines += [p1[1], p2[1], None]
            z_lines += [p1[2], p2[2], None]

        # Create Plotly frames
        frames = []
        for idx, val in enumerate(frame_values):
            colorscale, tick_vals, tick_text, min_v, max_v = create_abaqus_colorscale(val)
            mesh = go.Mesh3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                i=i, j=j, k=k,
                intensity=val,
                intensitymode="vertex",
                colorscale=colorscale,
                cmin=min_v,
                cmax=max_v,
                flatshading=True,
                lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
                colorbar=dict(
                    title=vis_mode,
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    len=0.75,
                    x=1.05,
                    thickness=20
                ),
                showscale=True,
                opacity=1.0
            )
            wire = go.Scatter3d(
                x=x_lines, y=y_lines, z=z_lines,
                mode="lines",
                line=dict(color="black", width=1),
                showlegend=False
            )
            frames.append(go.Frame(
                data=[mesh, wire],
                name=str(idx),
                layout=go.Layout(title=f"{vis_mode} – Frame {idx}")
            ))

        # Create Plotly figure with animation
        colorscale, tick_vals, tick_text, min_v, max_v = create_abaqus_colorscale(frame_values[frame_idx])
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                    i=i, j=j, k=k,
                    intensity=frame_values[frame_idx],
                    intensitymode="vertex",
                    colorscale=colorscale,
                    cmin=min_v,
                    cmax=max_v,
                    flatshading=True,
                    lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
                    colorbar=dict(
                        title=vis_mode,
                        tickvals=tick_vals,
                        ticktext=tick_text,
                        len=0.75,
                        x=1.05,
                        thickness=20
                    ),
                    showscale=True,
                    opacity=1.0
                ),
                go.Scatter3d(
                    x=x_lines, y=y_lines, z=z_lines,
                    mode="lines",
                    line=dict(color="black", width=1),
                    showlegend=False
                )
            ],
            frames=frames,
            layout=go.Layout(
                title=f"{vis_mode} – Frame {frame_idx}",
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y (Height)',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                scene_camera=dict(up=dict(x=0, y=1, z=0)),
                width=1000,
                height=800,
                sliders=[{
                    "steps": [
                        dict(
                            method="animate",
                            label=str(k),
                            args=[[str(k)], {"mode": "immediate", "frame": {"duration": 300, "redraw": True}, "transition": {"duration": 100}}]
                        ) for k in range(len(frames))
                    ],
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top"
                }],
                updatemenus=[{
                    "type": "buttons",
                    "buttons": [
                        {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True}]},
                        {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0}, "transition": {"duration": 0}}]}
                    ],
                    "direction": "left",
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 0,
                    "yanchor": "top"
                }]
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download button for current frame
        buf = BytesIO()
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        st.download_button(
            label=f"Download {vis_mode} Visualization (Frame {frame_idx})",
            data=buf,
            file_name=f"{vis_mode.replace(' ', '_').lower()}_frame_{frame_idx}.png",
            mime="image/png"
        )

    # Performance Summary
    st.subheader("Performance Summary")
    st.write(f"Maximum Axial Stress (failure criteria): {max_axial_stress / 1e6:.2f} MPa")
    st.write(f"Calculated Failure Load: {failure_load_value:.2f} kN")
    st.write(f"Maximum Displacement: {max_displacement:.2f} mm")

# Footer
st.markdown("""
    **Notes**: 
    1. This application predicts the stress-strain behaviours of circular fibre-reinforced polymer confined recycled aggregate concrete (FRCRAC) using machine learning framework.
    2. Three types of recycled aggregates (RA) were considered: RCA, RCL, RBA.
    3. Two types of FRP were considered: GFRP and CFRP.
    4. CATO-MZW: Hybridised Categorical Boosting with modified Zhou and Wu model (axial stress-strain only).
    5. CATO-LSTMO: Hybridised Categorical Boosting with Long-Short Term Memory Optimisation (axial and hoop stress-strains).
    6. Visualizations are generated using ML predictions, styled to resemble Abaqus FEA outputs for familiarity.
""")
