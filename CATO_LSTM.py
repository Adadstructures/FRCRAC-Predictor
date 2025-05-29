# Import required libraries for the FRCRAC Predictor and Visualization app
import streamlit as st  # Web app framework
import pickle  # Model loading
import numpy as np  # Numerical computations
import matplotlib.pyplot as plt  # 2D plotting
import pandas as pd  # Data handling
import joblib  # Scaler loading
import tensorflow as tf  # ML model framework
from keras.models import load_model  # LSTM model loading
from scipy import interpolate  # Interpolation for stress-strain curves
from scipy.interpolate import interp1d  # 1D interpolation
import plotly.graph_objects as go  # 3D visualization
from io import BytesIO  # Image downloading
import os  # File operations

# Apply custom CSS for responsive design and styling
st.markdown("""
<style>
    /* Main container padding and width */
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    /* Ensure sliders are full-width */
    .stSlider > div > div > div {
        width: 100% !important;
    }
    /* Style buttons for full-width and spacing */
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    /* Ensure Plotly graphs are responsive */
    .plotly-graph-div {
        width: 100% !important;
        height: auto !important;
    }
    /* Responsive font sizes */
    .stMarkdown, .stText, .stSubheader {
        font-size: clamp(14px, 3vw, 16px);
    }
    /* Footer styling */
    .footer {
        position: relative;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: clamp(10px, 2vw, 12px);
        color: #6c757d;
    }
    /* Mobile-specific adjustments */
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
def create_abaqus_plot(vertices, faces, scalar_per_face, title, colorbar_title, min_val=None, max_val=None, num_bands=10):
    """
    Generate a 3D mesh plot styled like Abaqus FEA output.
    
    Args:
        vertices (np.ndarray): Vertex coordinates (N, 3)
        faces (np.ndarray): Face indices (M, 3)
        scalar_per_face (np.ndarray): Scalar values per face
        title (str): Plot title
        colorbar_title (str): Colorbar label
        min_val (float, optional): Minimum scalar value
        max_val (float, optional): Maximum scalar value
        num_bands (int): Number of color bands
    
    Returns:
        go.Figure: Plotly figure object
    """
    # Set min/max if not provided
    min_val = np.min(scalar_per_face) if min_val is None else min_val
    max_val = np.max(scalar_per_face) if max_val is None else max_val
    if max_val == min_val:
        max_val = min_val + 1e-10  # Avoid division by zero

    # Normalize scalars for coloring
    normalized_scalars = np.clip((scalar_per_face - min_val) / (max_val - min_val), 0, 1)

    # Define Abaqus-like color scale
    abaqus_colors = [
        (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 165, 0), (255, 0, 0)
    ]
    colorscale = [[i / (len(abaqus_colors) - 1), f'rgb{c}'] for i, c in enumerate(abaqus_colors)]
    colorscale[-1][0] = 1.0

    # Assign colors based on bands
    band_edges = np.linspace(0, 1, num_bands, endpoint=True)
    color_indices = np.clip(np.digitize(normalized_scalars, band_edges, right=True), 0, num_bands - 1)
    face_colors = [colorscale[min(idx, num_bands - 1)][1] for idx in color_indices]

    # Set colorbar ticks
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

    # Create 3D mesh
    fig = go.Figure()
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        facecolor=face_colors, intensity=scalar_per_face, colorscale=colorscale,
        showscale=True, flatshading=True, opacity=1.0, hoverinfo='skip',
        lighting=dict(ambient=1.0, diffuse=0.1, specular=0.0, roughness=1.0),
        colorbar=dict(title=colorbar_title, tickvals=tick_vals, ticktext=tick_text, len=0.75, x=1.05)
    ))

    # Add wireframe
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

    # Configure layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

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

# Initialize session state for storing predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'stress_strain_model' not in st.session_state:
    st.session_state.stress_strain_model = None

# Set application title
st.title("FRCRAC Predictor and Visualisation")

# Create input form for user parameters
with st.form("input_form"):
    # Aggregate type selection
    aggregate_type = st.selectbox("Aggregate Type", ["RCA", "RCL", "RBA", "NA"], index=0, key="aggregate_type_selectbox")
    
    # Aggregate properties (conditional on type)
    if aggregate_type != 'NA':
        st.subheader("Aggregate Properties")
        percentage_rca = st.number_input("Percentage of RCA replacement by weight", value=50.00, format="%.2f")
        max_rca_size = st.number_input("Maximum diameter of the RCA (mm)", value=31.50, format="%.2f")
    else:
        percentage_rca = 0.0
        max_rca_size = st.number_input("Maximum diameter of the RCA (mm)", value=31.50, format="%.2f")

    # Cementitious properties
    st.subheader("Cementitious Property")
    water_cement_ratio = st.number_input("Water-to-cement ratio", value=0.35, format="%.2f")

    # Geometry properties
    st.subheader("Geometry Properties")
    diameter = st.number_input("Diameter of the concrete cylinder (mm)", value=150.00, format="%.2f")
    height = st.number_input("Height of the concrete cylinder (mm)", value=300.00, format="%.2f")

    # Concrete properties
    st.subheader("Concrete Properties")
    unconfined_strength = st.number_input("Unconfined Strength (MPa)", value=50.65, format="%.2f")
    unconfined_strain = st.number_input("Unconfined Strain", value=0.002, format="%.5f")

    # FRP properties
    st.subheader("FRP Properties")
    fibre_modulus = st.number_input("Fibre Modulus (MPa)", value=272730.0)
    frp_overall_thickness = st.number_input("FRP Overall Thickness (mm)", value=0.167, format="%.3f")
    frp_type = st.selectbox("Fibre Type", ["GFRP", "CFRP"], index=0, key="frp_type_selectbox")

    # Stress-strain model selection
    st.subheader("Stress-Strain Model")
    stress_strain_model = st.selectbox("Stress-Strain Model", ["CATO-LSTMO", "CATO-MZW"], index=0, key="stress_strain_model_selectbox")

    # Displacement parameter
    st.subheader("Displacement Parameter")
    max_displacement = st.number_input("Maximum Displacement (mm)", min_value=0.1, value=10.0)

    # Process input parameters
    fibre_type = 1 if frp_type == 'GFRP' else 3
    agg_type = 1 if aggregate_type == 'NA' else 2 if aggregate_type == 'RCA' else 3 if aggregate_type == 'RCL' else 4
    concrete_modulus = 4370 * (unconfined_strength ** 0.5) if aggregate_type == 'NA' else 4120 * (unconfined_strength ** 0.5)

    submit_button = st.form_submit_button("Predict")

# Perform predictions when form is submitted
if submit_button:
    # Predict rupture strain
    rupture_strain = float(cato_rupture.predict([[fibre_type, diameter, height, percentage_rca if aggregate_type != 'NA' else 0,
                                                  max_rca_size if aggregate_type != 'NA' else 0, water_cement_ratio,
                                                  unconfined_strength, unconfined_strain, fibre_modulus, frp_overall_thickness,
                                                  agg_type, concrete_modulus]]))
    
    # Calculate confinement stress
    confinement_stress = 2 * rupture_strain * fibre_modulus * frp_overall_thickness / diameter

    # Prepare input data for strength and strain predictions
    input_data = [fibre_type, diameter, height, percentage_rca, max_rca_size, water_cement_ratio, unconfined_strength,
                  unconfined_strain, fibre_modulus, frp_overall_thickness, agg_type, concrete_modulus, rupture_strain, confinement_stress]
    
    # Predict ultimate strength and strain
    cato_strength_prediction = float(cato_strength.predict([input_data]))
    cato_strain_prediction = float(cato_strain.predict([input_data]))
    
    # Handle stress-strain model predictions
    if stress_strain_model == 'CATO-MZW':
        ultimate_strength = cato_strength_prediction
        ultimate_axial_strain = cato_strain_prediction
        ultimate_hoop_strain = rupture_strain
        strength_enhancement_ratio = cato_strength_prediction / unconfined_strength
        strain_enhancement_ratio = cato_strain_prediction / unconfined_strain
        
        # Generate stress-strain curve for CATO-MZW
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
        
        # Filter for positive stress increments
        mask = np.ones(len(stress_values), dtype=bool)
        mask[1:] = np.diff(stress_values) > 0
        axial_stresses = stress_values[mask] * 1e6
        axial_strains = strain_values[mask]
        hoop_stresses = None
        hoop_strains = None

    else:  # CATO-LSTMO
        # Prepare input for LSTM models
        input_data_lstm = input_data + [cato_strength_prediction, cato_strain_prediction]
        input_data_axial = np.array(input_data_lstm).reshape(1, -1)
        input_data_hoop = np.array(input_data_lstm).reshape(1, -1)
        
        # Normalize inputs
        new_inputs_normalized_axial = scaler_axial.transform(input_data_axial)
        new_inputs_normalized_hoop = scaler_hoop.transform(input_data_hoop)
        
        # Predict axial and hoop stress-strains
        predicted_targets_axial = lstm_model_axial.predict(new_inputs_normalized_axial)
        predicted_targets_hoop = lstm_model_hoop.predict(new_inputs_normalized_hoop)

        # Denormalize predictions
        predicted_targets_axial_denorm = predicted_targets_axial * targets_1_axial.max(axis=(0, 1))
        predicted_targets_hoop_denorm = predicted_targets_hoop * targets_1_hoop.max(axis=(0, 1))

        # Extract stress and strain
        predicted_stress_axial = np.insert(predicted_targets_axial_denorm[0, :, 0], 0, 0) * 1e6
        predicted_strain_axial = np.insert(predicted_targets_axial_denorm[0, :, 1], 0, 0)
        predicted_stress_hoop = np.insert(predicted_targets_hoop_denorm[0, :, 0], 0, 0) * 1e6
        predicted_strain_hoop = np.insert(-predicted_targets_hoop_denorm[0, :, 1], 0, 0)
        
        # Compute ultimate values
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

    # Store predictions in session state
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

    # Display prediction results
    st.subheader("Prediction Results")
    st.write(f"Ultimate Strength: {ultimate_strength:.3f} MPa")
    st.write(f"Ultimate Axial Strain: {100 * ultimate_axial_strain:.3f} %")
    st.write(f"Ultimate Hoop Strain: {100 * ultimate_hoop_strain:.3f} %")
    st.write(f"Strength Enhancement: {strength_enhancement_ratio:.3f}")
    st.write(f"Strain Enhancement: {strain_enhancement_ratio:.3f}")

# Visualization section
if st.session_state.predictions:
    # Extract predictions
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

    # Display visualization options
    st.subheader("Post-Processing Visualisation")
    selected_plot = st.selectbox(
        "Select the plot to display",
        ["Load-Displacement Curve", "Stress-Strain Curve", "Stress Contours", "Strain Contours", "Displacement Contours"],
        key="plot_type_selectbox"
    )

    # Set up analytical curves
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

    # Interpolate stress-strain curves
    max_axial_stress = np.max(experimental_axial_stresses_filtered)
    axial_stress_strain_curve = interpolate.CubicSpline(experimental_axial_strains_filtered, experimental_axial_stresses_filtered, extrapolate=True)
    hoop_strain_curve = interpolate.CubicSpline(experimental_hoop_stresses_filtered, experimental_hoop_strains_filtered, extrapolate=True) if stress_strain_model == 'CATO-LSTMO' else None

    # Prepare strain-from-stress interpolation
    indices = np.argsort(experimental_axial_stresses_filtered)
    experimental_axial_stresses_sorted = experimental_axial_stresses_filtered[indices]
    experimental_axial_strains_sorted = experimental_axial_strains_filtered[indices]
    mask = np.ones(len(experimental_axial_stresses_sorted), dtype=bool)
    mask[1:] = np.diff(experimental_axial_stresses_sorted) > 0
    strain_from_stress = interpolate.CubicSpline(experimental_axial_stresses_sorted[mask], experimental_axial_strains_sorted[mask], extrapolate=True)

    # Convert dimensions to meters
    diameter_m = diameter / 1000
    height_m = height / 1000
    radius = diameter_m / 2
    area = np.pi * radius**2
    initial_height = height_m
    failure_load_value = max_axial_stress * area / 1000

    # Function to compute load-displacement curve
    def load_displacement_curve(max_displacement_m):
        """
        Compute load-displacement curve based on stress-strain data.
        
        Args:
            max_displacement_m (float): Maximum displacement in meters
        
        Returns:
            tuple: Displacement, load, strain, and stress arrays
        """
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

    # Handle selected visualization
    if selected_plot == "Load-Displacement Curve":
        # Generate and plot load-displacement curve
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
        
        # Offer download option
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
        # Generate and plot stress-strain curve
        fig, ax = plt.subplots(figsize=(8, 6))
        if stress_strain_model == 'CATO-MZW':
            ax.plot(axial_strains * 100, axial_stresses / 1e6, 'b-', label='CATO-MZW')
        else:
            ax.plot(axial_strains * 100, axial_stresses / 1e6, 'b-', label='CATO-LSTMO')
            ax.plot(-hoop_strains * 100, hoop_stresses / 1e6, 'b-')

        # Allow experimental data upload
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

        # Configure plot
        ax.set_xlabel("Strain (%)", fontsize=14)
        ax.set_ylabel("Stress (MPa)", fontsize=14)
        ax.set_title("Stress-Strain Curve", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        # Offer download option
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
        # Load visualization models and scalers
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

        # Load and validate mesh files
        def load_mesh():
            """
            Load mesh nodes and faces from CSV files.
            
            Returns:
                tuple: Vertices, face indices (i, j, k), and node count
            """
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

        # Scale mesh to user-specified dimensions
        if diameter < 1e-6 or height < 1e-6:
            st.error("Diameter or height too small.")
            st.stop()
        original_diameter = 150.0  # Reference diameter (mm)
        original_height = 300.0    # Reference height (mm)
        diameter_scale = diameter / original_diameter
        height_scale = height / original_height

        xyz_scaled = xyz.copy()
        xyz_scaled[:, 0] *= diameter_scale  # Scale X (radial)
        xyz_scaled[:, 2] *= diameter_scale  # Scale Z (radial)
        xyz_scaled[:, 1] *= height_scale    # Scale Y (height)
        st.write(f"Input: diameter={diameter:.2f} mm, height={height:.2f} mm")
        st.write(f"Scales: diameter_scale={diameter_scale:.2f}, height_scale={height_scale:.2f}")
        st.write(f"Scaled mesh: X={xyz_scaled[:,0].min():.2f}/{xyz_scaled[:,0].max():.2f}, "
                 f"Y={xyz_scaled[:,1].min():.2f}/{xyz_scaled[:,1].max():.2f}, "
                 f"Z={xyz_scaled[:,2].min():.2f}/{xyz_scaled[:,2].max():.2f}")

        # Define interpolation for stress and strain
        num_stress_points = len(axial_stresses) if len(axial_stresses) > 1 else 2
        stress_values = axial_stresses / 1e6 if len(axial_stresses) > 1 else np.array([0, np.max(axial_stresses) / 1e6])
        strain_values = axial_strains if len(axial_strains) > 1 else np.array([0, np.max(axial_strains)])
        stress_interp = interp1d(np.linspace(0, 1, num_stress_points), stress_values, bounds_error=False, fill_value="extrapolate")
        strain_interp = interp1d(stress_values, strain_values, bounds_error=False, fill_value="extrapolate")
        st.write(f"Stress interpolation range: {stress_interp(0):.2f} to {stress_interp(1):.2f} MPa")
        st.write(f"Strain interpolation max: {strain_interp(stress_interp(1)):.4f}")

        # Create Abaqus-like colorscale
        def create_abaqus_colorscale(values, num_bands=10):
            """
            Create an Abaqus-style colorscale for 3D visualization.
            
            Args:
                values (np.ndarray): Scalar values to color
                num_bands (int): Number of color bands
            
            Returns:
                tuple: Colorscale, tick values, tick text, min/max values
            """
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

        # Identify boundary nodes
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

        # Map visualization mode
        vis_mode_map = {
            "Stress Contours": "Stress (MPa)",
            "Strain Contours": "Strain (%)",
            "Displacement Contours": "Displacement (mm)"
        }
        vis_mode = vis_mode_map[selected_plot]

        # Generate frame values for visualization
        def predict_all_frames(vis_mode, node_count, stress_model, strain_model, disp_model, 
                              stress_scaler, strain_scaler, disp_scaler, unconfined_strength, 
                              frp_thickness, fibre_modulus, bottom_center_idx, top_center_idx, 
                              bottom_ring, top_ring, top_surface_nodes, stress_interp, strain_interp, 
                              max_displacement):
            """
            Predict stress, strain, or displacement for selected frames.
            
            Args:
                vis_mode (str): Visualization mode (Stress, Strain, Displacement)
                node_count (int): Number of mesh nodes
                stress_model, strain_model, disp_model: ML models
                stress_scaler, strain_scaler, disp_scaler: Feature scalers
                unconfined_strength, frp_thickness, fibre_modulus (float): Material properties
                bottom_center_idx, top_center_idx (int): Boundary node indices
                bottom_ring, top_ring (np.ndarray): Boundary ring node indices
                top_surface_nodes (np.ndarray): Top surface node indices
                stress_interp, strain_interp: Interpolation functions
                max_displacement (float): Maximum displacement (mm)
            
            Returns:
                tuple: Frame values array, frame indices
            """
            all_scaled = []
            frame_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21]  # Selected model frames
            
            for idx in frame_indices:
                if idx == 0:
                    scaled = np.zeros(node_count)  # Initial frame (zero load)
                else:
                    # Prepare feature DataFrame
                    df_feat = pd.DataFrame({
                        "Unconfined_Strength (MPa)": [unconfined_strength] * node_count,
                        "FRP_Thickness (mm)": [frp_thickness] * node_count,
                        "Fibre_Modulus (MPa)": [fibre_modulus] * node_count,
                        "Frame_Index": [idx] * node_count,
                        "Node_Label": list(range(node_count))
                    })

                    # Scale features
                    X_stress = stress_scaler.transform(df_feat)
                    X_strain = strain_scaler.transform(df_feat)
                    X_disp = disp_scaler.transform(df_feat)

                    # Predict nodal values
                    s_vals = stress_model.predict(X_stress)
                    strain_vals = strain_model.predict(X_strain)
                    u_vals = disp_model.predict(X_disp)

                    # Correct boundary conditions
                    for arr in [s_vals, strain_vals, u_vals]:
                        if arr[bottom_center_idx] == 0:
                            arr[bottom_center_idx] = np.mean(arr[bottom_ring])
                        if arr[top_center_idx] == 0:
                            arr[top_center_idx] = np.mean(arr[top_ring])

                    # Scale predictions based on mode
                    if vis_mode == "Stress (MPa)":
                        scaled = s_vals * stress_interp(idx / 21.0) / (np.max(np.abs(s_vals)) or 1.0)
                    elif vis_mode == "Strain (%)":
                        stress_scaled = s_vals * stress_interp(idx / 21.0) / (np.max(np.abs(s_vals)) or 1.0)
                        scaled = strain_interp(stress_scaled) * 100
                    else:  # Displacement (mm)
                        scaled = u_vals * max_displacement / (np.max(np.abs(u_vals)) or 1.0)
                        top_ring_mean = np.mean(scaled[top_ring])
                        scaled[top_surface_nodes] = top_ring_mean

                all_scaled.append(scaled)
            return np.array(all_scaled), frame_indices

        # Compute frame values
        frame_values, frame_indices = predict_all_frames(
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
        st.write(f"Generated {len(frame_values)} frames, indices: {frame_indices}")
        st.write(f"Frame 11 (index {frame_indices[-1]}) max value: {np.max(np.abs(frame_values[-1])):.2e} {vis_mode.split(' ')[-1]}")

        # Generate wireframe edges
        def generate_edges(i, j, k):
            """
            Generate wireframe edges from triangular faces.
            
            Args:
                i, j, k (np.ndarray): Face vertex indices
            
            Returns:
                list: List of edge tuples
            """
            edge_set = set()
            for tri in zip(i, j, k):
                for a, b in [(0, 1), (1, 2), (2, 0)]:
                    edge_set.add(tuple(sorted((tri[a], tri[b]))))
            return list(edge_set)

        edges = generate_edges(i, j, k)
        x_lines, y_lines, z_lines = [], [], []
        for e in edges:
            p1, p2 = xyz_scaled[e[0]], xyz_scaled[e[1]]
            x_lines.extend([p1[0], p2[0], None])
            y_lines.extend([p1[1], p2[1], None])
            z_lines.extend([p1[2], p2[2], None])

        # Select animation frame
        frame_idx = st.slider("Select Frame", 0, len(frame_values)-1, 0, key="frame_idx_slider")

        # Create animation frames
        frames = []
        for idx, val in enumerate(frame_values):
            colorscale, tick_vals, tick_text, min_v, max_v = create_abaqus_colorscale(val)
            mesh = go.Mesh3d(
                x=xyz_scaled[:, 0], y=xyz_scaled[:, 1], z=xyz_scaled[:, 2],
                i=i, j=j, k=k,
                intensity=val,
                intensitymode="vertex",
                colorscale=colorscale,
                cmin=min_v,
                cmax=max_v,
                flatshading=True,
                lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
                colorbar=dict(
                    title=dict(text=vis_mode, side="right", font=dict(size=12)),
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    len=0.5,
                    x=0.9,
                    thickness=15,
                    tickfont=dict(size=10)
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
                layout=go.Layout(title=dict(text=f"{vis_mode} – Frame {frame_indices[idx]}", font=dict(size=14)))
            ))

        # Create Plotly figure
        colorscale, tick_vals, tick_text, min_v, max_v = create_abaqus_colorscale(frame_values[frame_idx])
        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=xyz_scaled[:, 0], y=xyz_scaled[:, 1], z=xyz_scaled[:, 2],
                    i=i, j=j, k=k,
                    intensity=frame_values[frame_idx],
                    intensitymode="vertex",
                    colorscale=colorscale,
                    cmin=min_v,
                    cmax=max_v,
                    flatshading=True,
                    lighting=dict(ambient=0.9, diffuse=0.1, specular=0.0),
                    colorbar=dict(
                        title=dict(text=vis_mode, side="right", font=dict(size=12)),
                        tickvals=tick_vals,
                        ticktext=tick_text,
                        len=0.5,
                        x=0.9,
                        thickness=15,
                        tickfont=dict(size=10)
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
                title=dict(text=f"{vis_mode} – Frame {frame_indices[frame_idx]}", font=dict(size=14)),
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y (Height)',
                    zaxis_title='Z',
                    aspectmode='data',
                    camera=dict(
                        up=dict(x=0, y=1, z=0),
                        eye=dict(x=1.5, y=1.5, z=2)
                    )
                ),
                margin=dict(l=10, r=10, t=50, b=50),
                sliders=[{
                    "steps": [
                        dict(
                            method="animate",
                            label=str(frame_indices[k]),
                            args=[[str(k)], {"mode": "immediate", "frame": {"duration": 300, "redraw": True}, "transition": {"duration": 100}}]
                        ) for k in range(len(frame_values))
                    ],
                    "x": 0.1,
                    "xanchor": "left",
                    "y": 0.1,
                    "yanchor": "top",
                    "len": 0.9,
                    "font": {"size": 10}
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
                    "yanchor": "top",
                    "font": {"size": 10}
                }]
            )
        )

        # Enable touch interactions
        fig.update_layout(
            dragmode="orbit",
            scene_dragmode="orbit",
            hovermode=False
        )

        # Render visualization
        with st.spinner("Generating 3D visualization..."):
            st.plotly_chart(fig, use_container_width=True)

        # Offer download option
        buf = BytesIO()
        fig.write_image(buf, format="png", engine="kaleido")
        buf.seek(0)
        st.download_button(
            label=f"Download {vis_mode} Visualization (Frame {frame_indices[frame_idx]})",
            data=buf,
            file_name=f"{vis_mode.replace(' ', '_').lower()}_frame_{frame_indices[frame_idx]}.png",
            mime="image/png"
        )

    # Display performance summary
    st.subheader("Performance Summary")
    st.write(f"Maximum Axial Stress (failure criteria): {max_axial_stress / 1e6:.2f} MPa")
    st.write(f"Calculated Failure Load: {failure_load_value:.2f} kN")
    st.write(f"Maximum Displacement: {max_displacement:.2f} mm")

# Add footer with application notes and credits
st.markdown("""
**Notes**: 
1. This application predicts the stress-strain behaviours of circular fibre-reinforced polymer confined recycled aggregate concrete (FRCRAC) using machine learning framework.
2. Three types of recycled aggregates (RA) were considered: RCA, RCL, RBA.
3. Two types of FRP were considered: GFRP and CFRP.
4. CATO-MZW: Hybridised Categorical Boosting with modified Zhou and Wu model (axial stress-strain).
5. CATO-LSTMO: Hybridised Categorical Boosting with Long-Short Term Memory Optimisation (axial and hoop stress-strains).
6. Visualizations are generated using ML predictions, styled to resemble Abaqus FEA outputs for familiarity.
""")
footer = """
<div class="footer">
    <p>© 2025 My Streamlit App. All rights reserved. | Temitope E. Dada, Guobin Gong, Jun Xia, Luigi Di Sarno | For Queries: <a href="mailto: T.Dada19@example.com"> T.Dada19@example.com</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
