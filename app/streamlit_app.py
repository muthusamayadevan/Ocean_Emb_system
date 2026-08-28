"""
streamlit_app.py

Streamlit demonstration dashboard for the OceanEncoderDecoder model.
Allows selecting real profiles from the North Indian Ocean dataset, running model inference,
and visualizing "predicted vs real" vertical ocean temperature profiles.

Running Standalone:
streamlit run app/streamlit_app.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import streamlit as st

# Import the model definition from model.py
from src.model import OceanEncoderDecoder


# Page Configuration
st.set_page_config(
    page_title="OceanEmbed — Temperature Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Apply custom styling to the Streamlit app to create a premium, modern design
st.markdown(
    """
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Card Container */
    .metric-card {
        background-color: #0E1624;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #1E2D4A;
        margin-bottom: 20px;
    }
    
    .metric-title {
        color: #8A99AD;
        font-size: 14px;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #00D2FF;
        font-size: 28px;
        font-weight: 700;
    }
    
    .highlight-text {
        color: #00E5FF;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_and_scalers():
    """
    Loads and caches the trained PyTorch model and scalers so they only load once at startup.
    """
    model_path = "data/processed/model.pth"
    input_scaler_path = "data/processed/input_scaler.pkl"
    output_scaler_path = "data/processed/output_scaler.pkl"
    
    # Check if files exist
    if not (os.path.exists(model_path) and os.path.exists(input_scaler_path) and os.path.exists(output_scaler_path)):
        st.error("Error: Trained model weights or scaler files not found in `data/processed/`.")
        st.info("Please ensure that you have run the training pipeline first: `python -m src.train`.")
        st.stop()

    # Load scalers
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)

    # Reconstruct model and load weights
    model = OceanEncoderDecoder(input_dim=10, embedding_dim=24, output_dim=8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    return model, input_scaler, output_scaler


@st.cache_data
def load_datasets():
    """
    Loads and caches the processed matchup table and original argo profiles
    so the user can choose profiles using real date strings.
    """
    matchup_path = "data/processed/matchup_table.csv"
    argo_path = "data/raw/argo_profiles.csv"

    if not (os.path.exists(matchup_path) and os.path.exists(argo_path)):
        st.error("Error: Required datasets `matchup_table.csv` or `argo_profiles.csv` not found.")
        st.stop()

    df_matchup = pd.read_csv(matchup_path)
    df_argo = pd.read_csv(argo_path)
    
    # Align rows. Since build_matchup.py maps 1-to-1 but drops date strings, 
    # we copy the date column from df_argo by joining on coordinate keys or matching indices.
    # To be extremely safe, we assign the date directly by index matching.
    if len(df_matchup) == len(df_argo):
        df_matchup["date"] = df_argo["date"]
    else:
        # Fallback: Merge on lat/lon coordinates
        df_matchup = df_matchup.merge(df_argo[["lat", "lon", "date"]].drop_duplicates(subset=["lat", "lon"]), on=["lat", "lon"], how="left")
        
    return df_matchup


# --- Main Application UI Flow ---

# Header block
st.title("🌊 OceanEmbed")
st.subheader("Subsurface Ocean Temperature Profile Prediction Prototype")
st.write("Extracts surface satellite observations and maps them to vertical subsurface profiles using an artificial neural network.")

# Sidebar specification info
st.sidebar.image("https://img.icons8.com/color/150/sea-waves.png", width=80)
st.sidebar.markdown("### 📋 Model Specifications")
st.sidebar.markdown(
    """
    - **Architecture**: Neural Network (Encoder-Decoder)
    - **Input Space**: 10 surface variables from CMEMS/ERA5
    - **Output Space**: Temperature at 8 standard depths (0m to 500m)
    - **Spatial Region**: North Indian Ocean (5°N–30°N, 45°E–105°E)
    - **Evaluation Data**: Quality-filtered Argo float profiles
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 12px; color: #8A99AD;'>"
    "This prototype was designed for matching point-based surface data with deep profiles "
    "to reconstruct thermal structures from satellite records."
    "</div>",
    unsafe_allow_html=True,
)

# Step 1: Load resources and data
model, input_scaler, output_scaler = load_model_and_scalers()
df_matchup = load_datasets()

# Columns defined in train.py
input_cols = ["lat", "lon", "day_of_year", "sst", "sss", "ssh", "current_u", "current_v", "wind_u", "wind_v"]
output_depths = [0, 10, 30, 50, 75, 100, 200, 500]
output_cols = [f"temp_{depth}m" for depth in output_depths]

# Step 2: Dropdown selector for profile selection
st.markdown("### 🔍 Select Profile Location and Date")

# Create formatting labels for the dropdown selectbox
options = []
for idx, row in df_matchup.iterrows():
    options.append((idx, f"Profile #{idx+1:03d} | Lat: {row['lat']:.3f}°N | Lon: {row['lon']:.3f}°E | Date: {str(row['date'])[:16]}"))

selected_option = st.selectbox(
    "Choose a real profile from the test/train database to evaluate predicted vs actual values:",
    options=options,
    format_func=lambda x: x[1]
)

selected_idx = selected_option[0]
row_data = df_matchup.iloc[selected_idx]

# Split row into inputs and target actuals
inputs_raw = row_data[input_cols].values.reshape(1, -1)
actual_temps = row_data[output_cols].values.astype(float)

# Step 3: Run Model Inference
# Standardize inputs using the fitted input scaler
inputs_scaled = input_scaler.transform(inputs_raw)
inputs_tensor = torch.FloatTensor(inputs_scaled)

with torch.no_grad():
    pred_scaled = model(inputs_tensor).numpy()

# Inverse transform outputs from standardized values back to physical Celsius degrees
predicted_temps = output_scaler.inverse_transform(pred_scaled).squeeze()

# Calculate profile-specific RMSE
profile_rmse = np.sqrt(np.mean((actual_temps - predicted_temps) ** 2))

# Layout: Split into two columns for input features and prediction results
col_left, col_right = st.columns([2, 3])

with col_left:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.write("#### 📡 Matched Surface Observations (Model Inputs)")
    
    # Render inputs in a neat key-value table
    input_data = {
        "Variable": [
            "Latitude", "Longitude", "Day of Year", 
            "Sea Surface Temperature (SST)", "Sea Surface Salinity (SSS)", 
            "Sea Surface Height (SSH)", "Current Speed U (Eastward)", "Current Speed V (Northward)",
            "Wind Velocity U (Eastward)", "Wind Velocity V (Northward)"
        ],
        "Value": [
            f"{row_data['lat']:.4f} °N", f"{row_data['lon']:.4f} °E", f"{int(row_data['day_of_year'])}",
            f"{row_data['sst']:.3f} °C", f"{row_data['sss']:.3f} PSU",
            f"{row_data['ssh']:.3f} m", f"{row_data['current_u']:.3f} m/s", f"{row_data['current_v']:.3f} m/s",
            f"{row_data['wind_u']:.3f} m/s", f"{row_data['wind_v']:.3f} m/s"
        ]
    }
    st.table(pd.DataFrame(input_data))
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    # Display overall Profile RMSE in a card
    st.markdown(
        f"""
        <div class="metric-card" style="text-align: center;">
            <div class="metric-title">Single Profile Prediction Error</div>
            <div class="metric-value">{profile_rmse:.4f} °C</div>
            <div style="font-size: 13px; color: #8A99AD; margin-top: 5px;">Root Mean Squared Error (RMSE) across all 8 depths</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.write("#### 📈 Temperature Profile (Predicted vs Real)")
    
    # Render Matplotlib Chart
    fig, ax = plt.subplots(figsize=(7, 6))
    # Style customization for dark theme matching
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1624')
    
    # Plot predicted and actual temperature curves
    ax.plot(predicted_temps, output_depths, marker='o', color='#00D2FF', linewidth=2.5, label='Predicted Profile')
    ax.plot(actual_temps, output_depths, marker='s', color='#FF4B4B', linewidth=2, linestyle='--', label='Actual Argo Profile')
    
    # Configure axes, labels, and gridlines
    ax.set_xlabel('Temperature (°C)', color='#FFFFFF', fontsize=12)
    ax.set_ylabel('Depth (meters)', color='#FFFFFF', fontsize=12)
    ax.set_title(f'Vertical Temperature Profile Comparison (Argo Profile #{selected_idx+1})', color='#FFFFFF', fontsize=13, pad=15)
    
    # Invert y-axis so 0m (sea surface) is at the top of the graph, and 500m is at the bottom
    ax.invert_yaxis()
    
    # Axis styling
    ax.spines['bottom'].set_color('#1E2D4A')
    ax.spines['top'].set_color('#1E2D4A')
    ax.spines['left'].set_color('#1E2D4A')
    ax.spines['right'].set_color('#1E2D4A')
    ax.tick_params(colors='#FFFFFF', labelsize=10)
    
    ax.grid(True, which='both', color='#1E2D4A', linestyle=':', linewidth=0.8)
    
    # Legend customization
    legend = ax.legend(facecolor='#0E1624', edgecolor='#1E2D4A')
    for text in legend.get_texts():
        text.set_color('#FFFFFF')
        
    plt.tight_layout()
    st.pyplot(fig)

# Step 4: Comparison Table
st.markdown("### 📊 Depth-by-Depth Prediction Table")

comparison_data = []
for i, depth in enumerate(output_depths):
    pred = predicted_temps[i]
    act = actual_temps[i]
    abs_err = abs(pred - act)
    
    comparison_data.append({
        "Depth (m)": f"{depth}m",
        "Predicted Temperature (°C)": f"{pred:.4f} °C",
        "Actual Argo Temperature (°C)": f"{act:.4f} °C",
        "Absolute Error (°C)": f"{abs_err:.4f} °C"
    })

st.table(pd.DataFrame(comparison_data))
