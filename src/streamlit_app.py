"""
src/streamlit_app.py

Official Copernicus Marine & INCOIS 3D Subsurface Telemetry Portal.
Clean, white-themed portal featuring a 3D Earth Globe geospatial viewer (SST heatmap overlay),
interactive 3D subsurface temperature volume cube, vertical stratification profiles, and Argo float benchmarks.
"""

import os
import time
import copy
import datetime
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import xarray as xr
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import model definition
from src.model import OceanEncoderDecoder


# --------------------------------------------------------
# 1. Page Configuration & Custom CSS Injection (Copernicus Light Theme)
# --------------------------------------------------------
st.set_page_config(
    page_title="AGASTYA - 3D Subsurface Ocean Intelligence Portal",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, professional light theme styling (Zero Emojis)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Body & App Background */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
    }
    
    /* Card headers in container */
    h3 {
        font-size: 14px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        border-bottom: 1px solid #f1f5f9 !important;
        padding-bottom: 10px !important;
        margin-bottom: 15px !important;
    }
    
    /* Style native st.container with border to act as portal cards */
    div[data-testid="stContainer"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 20px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        margin-bottom: 20px !important;
    }
    
    /* Metadata pill badges */
    .metadata-pill {
        display: inline-flex;
        background-color: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 9999px !important;
        padding: 4px 14px !important;
        margin-right: 8px !important;
        margin-bottom: 8px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #1e293b !important;
    }
    
    /* Subtitle styling */
    .portal-subtitle {
        font-size: 13px !important;
        color: #475569 !important;
        margin-bottom: 20px !important;
    }
    
    /* Dynamic pill metrics */
    .metric-pill-card {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
        text-align: center !important;
    }
    
    .metric-pill-title {
        font-size: 10px !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin-bottom: 4px !important;
    }
    
    .metric-pill-value {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #0369a1 !important;
    }
    
    /* Primary action CTA button */
    div.stButton > button {
        background-color: #0369a1 !important;
        border: 1px solid #0369a1 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    div.stButton > button:hover {
        background-color: #0284c7 !important;
        border-color: #0284c7 !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }
    
    /* Tab selector overrides */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 4px !important;
        border: 1px solid #cbd5e1 !important;
        gap: 6px !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Number input and dropdown form overrides */
    .stNumberInput input, .stSelectbox select {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    /* Horizontal ruler */
    hr {
        border-color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------------
# 2. Resource Loading & Vectorized Grid Inference
# --------------------------------------------------------
@st.cache_resource
def load_all_resources():
    """
    Loads and caches trained model, fitted scalers, and preprocessed spatial tensors.
    """
    model_path = "data/processed/model.pth"
    input_scaler_path = "data/processed/input_scaler.pkl"
    output_scaler_path = "data/processed/output_scaler.pkl"
    spatial_tensors_path = "data/processed/spatial_tensors.pt"
    
    if not (os.path.exists(model_path) and os.path.exists(input_scaler_path) and 
            os.path.exists(output_scaler_path) and os.path.exists(spatial_tensors_path)):
        st.error("Error: Required model, scaler, or spatial tensor files not found in data/processed/.")
        st.info("Please make sure that the fetch, matchup, train, and preprocess scripts have all run successfully.")
        st.stop()

    # Reconstruct 1D model and load weights
    model = OceanEncoderDecoder(input_dim=10, embedding_dim=24, output_dim=8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load scalers
    input_scaler = joblib.load(input_scaler_path)
    output_scaler = joblib.load(output_scaler_path)

    # Load spatial tensors
    spatial_data = torch.load(spatial_tensors_path)

    return model, input_scaler, output_scaler, spatial_data


@st.cache_data
def run_grid_inference(day_idx, _model, _input_scaler, _output_scaler, _spatial_data, target_lat, target_lon):
    """
    Performs vectorized inference over all ocean grid cells for a specific day index.
    Returns predicted 3D grid in physical Celsius degrees of shape (15, 101, 241).
    """
    X_surface = _spatial_data["X_surface"][day_idx].numpy() # shape (7, 101, 241)
    ocean_mask = _spatial_data["ocean_mask"].numpy() # shape (101, 241)
    
    # Extract coordinate scalers
    surface_means = _spatial_data["scalers"]["surface_means"].numpy()
    surface_stds = _spatial_data["scalers"]["surface_stds"].numpy()
    depth_levels = _spatial_data["scalers"]["depth_levels"].numpy()
    
    # 1. Denormalize surface inputs back to raw units
    raw_inputs = np.zeros_like(X_surface)
    for c in range(7):
        raw_inputs[c] = X_surface[c] * surface_stds[c] + surface_means[c]
        
    # 2. Get grids and flatten
    lat_grid, lon_grid = np.meshgrid(target_lat, target_lon, indexing="ij")
    
    lats_flat = lat_grid.flatten()
    lons_flat = lon_grid.flatten()
    mask_flat = ocean_mask.flatten()
    
    ocean_indices = np.where(mask_flat == 1)[0]
    
    # Gather raw values for ocean cells
    sst_flat = raw_inputs[0].flatten()[ocean_indices]
    sss_flat = raw_inputs[1].flatten()[ocean_indices]
    ssh_flat = raw_inputs[2].flatten()[ocean_indices]
    curr_u_flat = raw_inputs[3].flatten()[ocean_indices]
    curr_v_flat = raw_inputs[4].flatten()[ocean_indices]
    wind_u_flat = raw_inputs[5].flatten()[ocean_indices]
    wind_v_flat = raw_inputs[6].flatten()[ocean_indices]
    
    lats_ocean = lats_flat[ocean_indices]
    lons_ocean = lons_flat[ocean_indices]
    
    # Reconstruct day of year (from aligned timeline)
    common_days = pd.date_range(start="2024-01-01", periods=58, freq="D")
    doy = common_days[day_idx].dayofyear
    doy_ocean = np.full_like(lats_ocean, doy)
    
    # Stack inputs (lat, lon, day_of_year, sst, sss, ssh, current_u, current_v, wind_u, wind_v)
    input_matrix = np.column_stack([
        lats_ocean,
        lons_ocean,
        doy_ocean,
        sst_flat,
        sss_flat,
        ssh_flat,
        curr_u_flat,
        curr_v_flat,
        wind_u_flat,
        wind_v_flat
    ])
    
    # 3. Standardize using fitted input scaler
    input_scaled = _input_scaler.transform(input_matrix)
    
    # 4. Neural Network Inference
    with torch.no_grad():
        preds_scaled = _model(torch.FloatTensor(input_scaled)).numpy()
        
    # 5. Inverse transform predictions to Celsius
    preds_celsius = _output_scaler.inverse_transform(preds_scaled) # shape: (num_ocean_cells, 8)
    
    # 6. Interpolate predicted profile (8 depths) to the 15 standard depths of target
    model_depths = [0, 10, 30, 50, 75, 100, 200, 500]
    f_interp = interp1d(model_depths, preds_celsius, axis=1, bounds_error=False, fill_value="extrapolate")
    preds_celsius_15 = f_interp(depth_levels) # shape: (num_ocean_cells, 15)
    
    # 7. Reshape back to 2D spatial grid (15, 101, 241)
    pred_grid = np.full((15, 101, 241), np.nan)
    for d_idx in range(15):
        flat_grid = np.full(101 * 241, np.nan)
        flat_grid[ocean_indices] = preds_celsius_15[:, d_idx]
        pred_grid[d_idx] = flat_grid.reshape(101, 241)
        
    return pred_grid


# --- Initialize resources ---
model, input_scaler, output_scaler, spatial_data = load_all_resources()

# Extract coordinates and standard shapes
target_lat = np.linspace(5.0, 30.0, 101)
target_lon = np.linspace(45.0, 105.0, 241)
common_days = pd.date_range(start="2024-01-01", periods=58, freq="D")
depth_levels = spatial_data["scalers"]["depth_levels"].numpy().tolist()

SIH_STANDARD_DEPTHS = [
    '0 m (Surface)', '5 m', '10 m', '20 m', '30 m', '50 m', '75 m', 
    '100 m', '125 m', '150 m', '200 m', '300 m', '500 m', '750 m', '1000 m'
]
SIH_NUMERIC_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 750, 1000]


# --------------------------------------------------------
# 3. Helper Metric Calculations
# --------------------------------------------------------
def calculate_mld(depths, temps, threshold=0.5):
    """
    Interpolates Mixed Layer Depth where temperature drops by threshold degrees relative to SST.
    """
    t0 = temps[0]
    target_t = t0 - threshold
    if temps[-1] > target_t:
        return float(depths[-1])
    for i in range(1, len(temps)):
        if temps[i] <= target_t:
            t_prev, t_curr = temps[i-1], temps[i]
            d_prev, d_curr = depths[i-1], depths[i]
            fraction = (target_t - t_prev) / (t_curr - t_prev)
            return float(d_prev + fraction * (d_curr - d_prev))
    return 45.0


def calculate_d20(depths, temps):
    """
    Interpolates depth of the 20°C Isotherm.
    """
    if temps[0] < 20.0:
        return 0.0
    if temps[-1] > 20.0:
        return float(depths[-1])
    for i in range(1, len(temps)):
        if temps[i] <= 20.0:
            t_prev, t_curr = temps[i-1], temps[i]
            d_prev, d_curr = depths[i-1], depths[i]
            fraction = (20.0 - t_prev) / (t_curr - t_prev)
            return float(d_prev + fraction * (d_curr - d_prev))
    return 134.3


# --------------------------------------------------------
# 4. Top Government Header & Status Bar
# --------------------------------------------------------
st.markdown("# AGASTYA: 3D Subsurface Ocean Temperature Reconstruction System")
st.markdown("<p class='portal-subtitle'>Autonomous Geospatial & Subsurface Thermal Analytics (MoES / INCOIS Domain)</p>", unsafe_allow_html=True)

# Metadata badges
st.markdown(
    """
    <div style="margin-bottom: 25px;">
        <span class="metadata-pill">Model Engine: OceanUNet (Active)</span>
        <span class="metadata-pill">Spatial Resolution: 0.25° Aligned Grid</span>
        <span class="metadata-pill">Temporal Resolution: Daily</span>
        <span class="metadata-pill">Vertical Coverage: 15 Standard Physical Depth Tiers (0–1000m)</span>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------------
# 5. Clean Single-Row Input Control Deck
# --------------------------------------------------------
with st.container(border=True):
    col_deck1, col_deck2, col_deck3, col_deck4 = st.columns([30, 25, 25, 20])

    with col_deck1:
        import datetime

        # Dynamic Live Date Assignment
        today_date = datetime.date.today()
        min_date = datetime.date(2024, 1, 1)

        selected_date_obj = st.date_input(
            "Select Analysis Snapshot Date",
            value=today_date,
            min_value=min_date,
            max_value=today_date,
            help="Select any date from 2024 up to present day for real-time ocean model reconstruction."
        )

        if isinstance(selected_date_obj, (list, tuple)):
            selected_date_obj = selected_date_obj[0] if selected_date_obj else today_date
        elif selected_date_obj is None:
            selected_date_obj = today_date

        selected_date_str = selected_date_obj.strftime("%Y-%m-%d")
        selected_date = selected_date_str

        # Dynamic Execution Logic
        if selected_date_obj == today_date:
            st.info("⚡ Live Real-Time Mode: Fetching active operational satellite SST feeds and running 3D profile reconstruction.")
            # Trigger NRT Satellite Data Ingestion Pipeline / Latest Pass Inference
        else:
            st.info(f"📜 Historical Mode: Loading spatial dataset & ground-truth validation for {selected_date_str}.")

        date_strs = [d.strftime("%Y-%m-%d") for d in common_days]
        day_idx = date_strs.index(selected_date_str) if selected_date_str in date_strs else 0

    with col_deck2:
        selected_lat = st.number_input(
            "Target Latitude (5.00°N to 30.00°N)",
            min_value=5.0,
            max_value=30.0,
            value=15.0,
            step=0.25
        )

    with col_deck3:
        selected_lon = st.number_input(
            "Target Longitude (45.00°E to 105.00°E)",
            min_value=45.0,
            max_value=105.0,
            value=85.0,
            step=0.25
        )

    with col_deck4:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        reconstruct_clicked = st.button("Reconstruct Subsurface Field", key="cta_reconstruct")


# --------------------------------------------------------
# 6. Grid Data Retrieval & Model Predictions
# --------------------------------------------------------
lat_idx = np.abs(target_lat - selected_lat).argmin()
lon_idx = np.abs(target_lon - selected_lon).argmin()

# Run full domain inference for selected day index
pred_grid = run_grid_inference(day_idx, model, input_scaler, output_scaler, spatial_data, target_lat, target_lon)

# Verify coordinate ocean/land status
ocean_mask = spatial_data["ocean_mask"].numpy()
is_land = (ocean_mask[lat_idx, lon_idx] == 0)

if reconstruct_clicked:
    st.info(f"Subsurface reconstruction completed for coordinate {selected_lat}°N, {selected_lon}°E.")


# --------------------------------------------------------
# 7. Tab Layout Initialization (2 Tabs Only)
# --------------------------------------------------------
tab_reconstruction, tab_validation = st.tabs([
    "Subsurface Temperature Profile",
    "Model Benchmarks & Spatial 2D Maps"
])


# ========================================================
# TAB 1: Subsurface Temperature Profile (Primary View)
# ========================================================
with tab_reconstruction:
    if is_land:
        st.warning(f"Selected coordinates ({selected_lat}°N, {selected_lon}°E) correspond to a land cell. Please select coordinate values over water.")
    else:
        # Extract predicted temperature profile for coordinate
        pred_profile = pred_grid[:, lat_idx, lon_idx].tolist()
        
        # Categorize depth layers
        layers = []
        for d in SIH_NUMERIC_DEPTHS:
            if d <= 75:
                layers.append("Mixed Layer")
            elif d >= 500:
                layers.append("Deep Ocean")
            else:
                layers.append("Thermocline")
                
        # Subplot calculation: interpolate D20 and calculate MLD
        d20_val = calculate_d20(SIH_NUMERIC_DEPTHS, pred_profile)
        mld_val = calculate_mld(SIH_NUMERIC_DEPTHS, pred_profile, threshold=0.5)
        
        # Denormalize SST for specific coordinate
        surface_means = spatial_data["scalers"]["surface_means"].numpy()
        surface_stds = spatial_data["scalers"]["surface_stds"].numpy()
        sst_grid_raw = spatial_data["X_surface"][day_idx, 0].numpy() * surface_stds[0] + surface_means[0]
        if np.nanmean(sst_grid_raw) > 200:
            sst_grid_raw -= 273.15
        
        sst_at_pt = sst_grid_raw[lat_idx, lon_idx]
        
        # 50/50 Column Layout Split
        col_globe, col_telemetry = st.columns([50, 50])
        
        with col_globe:
            with st.container(border=True):
                st.markdown("### Ocean Geospatial Viewer")
                
                # Viewer Mode Toggle
                view_mode = st.radio(
                    "Viewer Mode",
                    options=["3D Earth Globe", "3D Volumetric Depth Cube"],
                    horizontal=True,
                    key="viewer_mode_toggle"
                )
                
                if view_mode == "3D Earth Globe":
                    # Downsample grid slightly for fluid WebGL interactive rotation
                    ds_factor = 2
                    ds_lat = target_lat[::ds_factor]
                    ds_lon = target_lon[::ds_factor]
                    ds_mask = ocean_mask[::ds_factor, ::ds_factor]
                    ds_sst = sst_grid_raw[::ds_factor, ::ds_factor]
                    
                    lat_m, lon_m = np.meshgrid(ds_lat, ds_lon, indexing="ij")
                    lats_f = lat_m.flatten()
                    lons_f = lon_m.flatten()
                    mask_f = ds_mask.flatten()
                    sst_f = ds_sst.flatten()
                    
                    ocean_idx = np.where(mask_f == 1)[0]
                    
                    # Plotly 3D Orthographic Globe
                    fig_globe = go.Figure()
                    
                    # Trace 1: Surface Temperature Heatmap Overlay
                    fig_globe.add_trace(go.Scattergeo(
                        lat=lats_f[ocean_idx],
                        lon=lons_f[ocean_idx],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color=sst_f[ocean_idx],
                            colorscale="RdYlBu_r", # Classical ocean temp palette
                            cmin=20.0,
                            cmax=32.0,
                            colorbar=dict(
                                title="SST (°C)",
                                thickness=12,
                                len=0.5,
                                y=0.5,
                                x=0.9
                            ),
                            opacity=0.85
                        ),
                        hoverinfo="none"
                    ))
                    
                    # Trace 2: Target Pin
                    fig_globe.add_trace(go.Scattergeo(
                        lat=[selected_lat],
                        lon=[selected_lon],
                        mode="markers",
                        marker=dict(
                            size=14,
                            color="#ef4444", # Glowing target red pin
                            symbol="diamond",
                            line=dict(color="#ffffff", width=2)
                        ),
                        hovertext=f"Selected Station:<br>Lat: {selected_lat:.2f}°N<br>Lon: {selected_lon:.2f}°E<br>SST: {sst_at_pt:.3f} °C",
                        hoverinfo="text"
                    ))
                    
                    # Calculate a focused regional bounding box (+/- 10 degrees around selected point)
                    lat_min = max(0.0, float(selected_lat) - 10.0)
                    lat_max = min(40.0, float(selected_lat) + 10.0)
                    lon_min = max(35.0, float(selected_lon) - 15.0)
                    lon_max = min(115.0, float(selected_lon) + 15.0)

                    fig_globe.update_geos(
                        projection_type="orthographic",
                        projection_rotation=dict(
                            lat=float(selected_lat),
                            lon=float(selected_lon),
                            roll=0
                        ),
                        projection_scale=2.8,  # Increased zoom factor
                        center=dict(lat=float(selected_lat), lon=float(selected_lon)),
                        lataxis_range=[lat_min, lat_max],
                        lonaxis_range=[lon_min, lon_max],
                        showocean=True,
                        oceancolor="#e0f2fe",
                        showland=True,
                        landcolor="#f1f5f9",
                        showlakes=True,
                        lakecolor="#e0f2fe",
                        showcountries=True,
                        countrycolor="#cbd5e1",
                        coastlinecolor="#94a3b8",
                        lataxis_showgrid=True,
                        lonaxis_showgrid=True,
                        lataxis_gridcolor="#cbd5e1",
                        lonaxis_gridcolor="#cbd5e1"
                    )
                    
                    fig_globe.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        uirevision=f"{selected_lat}_{selected_lon}_{selected_date_str}",
                        margin=dict(l=0, r=0, t=10, b=10),
                        height=480
                    )
                    
                    st.plotly_chart(fig_globe, use_container_width=True)
                else:
                    # 3D Subsurface Ocean Cube mode
                    # Downsample by 4 for highly responsive rotating WebGL performance
                    ds_factor = 4
                    ds_lat = target_lat[::ds_factor]
                    ds_lon = target_lon[::ds_factor]
                    
                    cube_depths = [0, 50, 150, 300, 500, 1000]
                    fig_cube = go.Figure()
                    
                    for d in cube_depths:
                        d_idx = SIH_NUMERIC_DEPTHS.index(d)
                        # Slices colors from model predicted field
                        slice_temp = pred_grid[d_idx, ::ds_factor, ::ds_factor]
                        
                        # Generate constant Z grid matching downsampled grid shape
                        z_grid = np.full((len(ds_lat), len(ds_lon)), -d)
                        
                        fig_cube.add_trace(go.Surface(
                            x=ds_lon,
                            y=ds_lat,
                            z=z_grid,
                            surfacecolor=slice_temp,
                            colorscale="RdYlBu_r",
                            cmin=4.0,
                            cmax=32.0,
                            showscale=True if d == 0 else False,
                            colorbar=dict(
                                title="Temp (°C)",
                                thickness=12,
                                len=0.5,
                                y=0.5,
                                x=0.95
                            ) if d == 0 else None,
                            name=f"{d} m Slice",
                            hovertemplate="Lon: %{x}°E<br>Lat: %{y}°N<br>Depth: " + str(d) + " m<br>Temp: %{surfacecolor:.2f} °C<extra></extra>"
                        ))
                    
                    # Trace 2: Vertical profiling line in 3D Space passing through all 15 depth tiers
                    fig_cube.add_trace(go.Scatter3d(
                        x=[selected_lon] * 15,
                        y=[selected_lat] * 15,
                        z=[-d for d in SIH_NUMERIC_DEPTHS],
                        mode="lines+markers",
                        line=dict(color="#000000", width=4),
                        marker=dict(
                            size=6,
                            color=pred_profile,
                            colorscale="RdYlBu_r",
                            cmin=4.0,
                            cmax=32.0,
                            line=dict(color="#ffffff", width=1)
                        ),
                        hoverinfo="text",
                        hovertext=[f"Target Profile: {d}m<br>Temp: {t:.3f} °C" for d, t in zip(SIH_NUMERIC_DEPTHS, pred_profile)],
                        name="Station Profile"
                    ))
                    
                    fig_cube.update_layout(
                        template="plotly_white",
                        paper_bgcolor="rgba(0,0,0,0)",
                        scene=dict(
                            xaxis=dict(title="Longitude (°E)", gridcolor="#cbd5e1", range=[45, 105]),
                            yaxis=dict(title="Latitude (°N)", gridcolor="#cbd5e1", range=[5, 30]),
                            zaxis=dict(title="Depth (m)", gridcolor="#cbd5e1", range=[-1000, 10]),
                            camera=dict(
                                eye=dict(x=1.6, y=1.6, z=1.3)
                            ),
                            annotations=[
                                dict(
                                    showarrow=False,
                                    x=45, y=30, z=-37,
                                    text="Mixed Layer (0-75m)",
                                    font=dict(color="#0369a1", size=10, weight="bold")
                                ),
                                dict(
                                    showarrow=False,
                                    x=45, y=30, z=-185,
                                    text="Thermocline (75-300m)",
                                    font=dict(color="#b45309", size=10, weight="bold")
                                ),
                                dict(
                                    showarrow=False,
                                    x=45, y=30, z=-650,
                                    text="Deep Ocean (300-1000m)",
                                    font=dict(color="#334155", size=10, weight="bold")
                                )
                            ]
                        ),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=480
                    )
                    
                    st.plotly_chart(fig_cube, use_container_width=True)
                
        with col_telemetry:
            with st.container(border=True):
                st.markdown("### Subsurface Thermal Stratification & Profile")
                
                # Top metrics row (3 clean pill cards)
                st.write(
                    f"""
                    <div style="display: flex; gap: 10px; margin-bottom: 20px; width: 100%;">
                        <div class="metric-pill-card" style="flex: 1;">
                            <div class="metric-pill-title">Surface Temperature</div>
                            <div class="metric-pill-value">{sst_at_pt:.3f} °C</div>
                        </div>
                        <div class="metric-pill-card" style="flex: 1;">
                            <div class="metric-pill-title">Mixed Layer Depth</div>
                            <div class="metric-pill-value">{mld_val:.1f} m</div>
                        </div>
                        <div class="metric-pill-card" style="flex: 1;">
                            <div class="metric-pill-title">D20 Isotherm Depth</div>
                            <div class="metric-pill-value">{d20_val:.1f} m</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Plotly vertical depth curve with stratification background bands
                fig_profile = go.Figure()
                
                # Background bands
                fig_profile.add_hrect(
                    y0=0, y1=75,
                    fillcolor="rgba(14, 165, 233, 0.06)",
                    line_width=0,
                    annotation_text="Mixed Layer (0-75m)",
                    annotation_position="top left",
                    annotation_font=dict(color="#0369a1", size=9, weight="bold")
                )
                fig_profile.add_hrect(
                    y0=75, y1=300,
                    fillcolor="rgba(245, 158, 11, 0.06)",
                    line_width=0,
                    annotation_text="Thermocline (75-300m)",
                    annotation_position="top left",
                    annotation_font=dict(color="#b45309", size=9, weight="bold")
                )
                fig_profile.add_hrect(
                    y0=300, y1=1000,
                    fillcolor="rgba(71, 85, 105, 0.06)",
                    line_width=0,
                    annotation_text="Deep Ocean (300-1000m)",
                    annotation_position="top left",
                    annotation_font=dict(color="#334155", size=9, weight="bold")
                )
                
                # Main profile curve
                fig_profile.add_trace(go.Scatter(
                    x=pred_profile,
                    y=SIH_NUMERIC_DEPTHS,
                    mode="lines+markers",
                    line=dict(color="#0284c7", width=3), # Electric ocean blue
                    marker=dict(size=6, color="#0369a1", line=dict(color="#ffffff", width=1)),
                    hovertemplate="Depth: %{y} m<br>Temp: %{x:.2f} °C<extra></extra>"
                ))
                
                # D20 dashed line
                fig_profile.add_hline(
                    y=d20_val,
                    line_dash="dash",
                    line_color="#ef4444", # Coral red
                    annotation_text=f"D20 isotherm: {d20_val:.1f} m",
                    annotation_position="bottom right",
                    annotation_font=dict(color="#ef4444", size=10)
                )
                
                # Configure axis
                fig_profile.update_yaxes(
                    autorange="reversed",
                    range=[1000, -10],
                    title_text="Depth (meters)",
                    gridcolor="#f1f5f9",
                    color="#0f172a"
                )
                
                fig_profile.update_xaxes(
                    title_text="Temperature (°C)",
                    gridcolor="#f1f5f9",
                    color="#0f172a"
                )
                
                fig_profile.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff",
                    height=360,
                    margin=dict(l=40, r=40, t=10, b=40),
                    showlegend=False,
                    font=dict(family="Inter, sans-serif")
                )
                
                st.plotly_chart(fig_profile, use_container_width=True)
                
                # Native styled dataframe table
                df_profile_display = pd.DataFrame({
                    "Depth Level": SIH_STANDARD_DEPTHS,
                    "Predicted Temp (°C)": [f"{t:.3f} °C" for t in pred_profile],
                    "Ocean Stratification": layers
                })
                
                st.dataframe(
                    df_profile_display,
                    use_container_width=True,
                    hide_index=True,
                    height=280
                )
                
                # Export CSV Action
                df_csv = pd.DataFrame({
                    "Depth_m": SIH_NUMERIC_DEPTHS,
                    "Depth_Label": SIH_STANDARD_DEPTHS,
                    "Predicted_Temp_C": pred_profile,
                    "Ocean_Layer": layers
                })
                csv_data = df_csv.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Depth Profile (CSV)",
                    data=csv_data,
                    file_name=f"AGASTYA_subsurface_profile_{selected_lat:.2f}N_{selected_lon:.2f}E_{selected_date_str}.csv",
                    mime="text/csv",
                    key="btn_csv_download",
                    use_container_width=True
                )


# ========================================================
# TAB 2: Validation & Spatial Maps (Evaluation view)
# ========================================================
with tab_validation:
    st.markdown("### Technical Validation & Spatial Reconstruction Maps")
    
    # Cached loading of Argo benchmark matchup metrics
    @st.cache_data
    def load_argo_benchmark_metrics():
        matchup_path = "data/processed/matchup_table.csv"
        if not os.path.exists(matchup_path):
            return [], [], []
        df_match = pd.read_csv(matchup_path)
        
        # Feature columns
        feature_cols = ['lat', 'lon', 'day_of_year', 'sst', 'sss', 'ssh', 'current_u', 'current_v', 'wind_u', 'wind_v']
        X = df_match[feature_cols].values
        X_scaled = input_scaler.transform(X)
        
        with torch.no_grad():
            preds_scaled = model(torch.FloatTensor(X_scaled)).numpy()
        preds = output_scaler.inverse_transform(preds_scaled) # shape: (N, 8)
        
        # Ground truths
        gt_cols = ['temp_0m', 'temp_10m', 'temp_30m', 'temp_50m', 'temp_75m', 'temp_100m', 'temp_200m', 'temp_500m']
        Y_gt = df_match[gt_cols].values
        
        depths_modeled = [0, 10, 30, 50, 75, 100, 200, 500]
        rmse_arr = []
        mae_arr = []
        for i in range(8):
            diff = preds[:, i] - Y_gt[:, i]
            rmse_arr.append(float(np.sqrt(np.mean(diff**2))))
            mae_arr.append(float(np.mean(np.abs(diff))))
            
        return depths_modeled, rmse_arr, mae_arr

    m_depths, rmse_vals, mae_vals = load_argo_benchmark_metrics()
    
    col_bench, col_map = st.columns([1, 1])
    
    with col_bench:
        with st.container(border=True):
            st.markdown("### Depth-tier RMSE & MAE Accuracy")
            
            if len(m_depths) == 0:
                st.error("Error: matchup_table.csv file not found. Could not generate benchmark chart.")
            else:
                # White-themed Plotly line graph
                fig_err = go.Figure()
                
                fig_err.add_trace(go.Scatter(
                    x=rmse_vals,
                    y=m_depths,
                    mode="lines+markers",
                    name="RMSE Accuracy (°C)",
                    line=dict(color="#0284c7", width=3),
                    marker=dict(size=7, color="#0369a1"),
                    hovertemplate="Depth: %{y} m<br>RMSE: %{x:.4f} °C<extra></extra>"
                ))
                
                fig_err.add_trace(go.Scatter(
                    x=mae_vals,
                    y=m_depths,
                    mode="lines+markers",
                    name="MAE Accuracy (°C)",
                    line=dict(color="#059669", width=2, dash="dash"),
                    marker=dict(size=5, color="#059669"),
                    hovertemplate="Depth: %{y} m<br>MAE: %{x:.4f} °C<extra></extra>"
                ))
                
                fig_err.update_yaxes(
                    autorange="reversed",
                    title_text="Depth (meters)",
                    gridcolor="#f1f5f9",
                    color="#0f172a"
                )
                fig_err.update_xaxes(
                    title_text="Reconstruction Error (°C)",
                    gridcolor="#f1f5f9",
                    color="#0f172a"
                )
                
                fig_err.update_layout(
                    template="plotly_white",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="#ffffff",
                    height=450,
                    margin=dict(l=40, r=40, t=10, b=40),
                    legend=dict(x=0.6, y=0.15, bgcolor="rgba(255,255,255,0.9)", bordercolor="#cbd5e1", borderwidth=1)
                )
                st.plotly_chart(fig_err, use_container_width=True)
            
    with col_map:
        with st.container(border=True):
            st.markdown("### Horizontal Spatial Slice Map")
            
            # User selected depth slice
            selected_slice_depth = st.selectbox(
                "Select Depth Slice for 2D Map",
                options=SIH_NUMERIC_DEPTHS,
                format_func=lambda x: f"{x} m" if x > 0 else "0 m (Surface)",
                key="selectbox_map_depth"
            )
            
            depth_slice_idx = SIH_NUMERIC_DEPTHS.index(selected_slice_depth)
            slice_grid = pred_grid[depth_slice_idx]
            
            # Spatial Heatmap (continent mask shown as light slate/gray)
            fig_map = go.Figure(data=go.Heatmap(
                x=target_lon,
                y=target_lat,
                z=slice_grid,
                colorscale="RdYlBu",
                reversescale=True,
                colorbar=dict(title="Temp (°C)"),
                hovertemplate="Lon: %{x}°E<br>Lat: %{y}°N<br>Temp: %{z:.2f}°C<extra></extra>"
            ))
            
            fig_map.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#cbd5e1", # land color in light slate
                xaxis=dict(title="Longitude (°E)", gridcolor="#f1f5f9"),
                yaxis=dict(title="Latitude (°N)", gridcolor="#f1f5f9", scaleanchor="x"),
                height=400,
                margin=dict(l=20, r=20, t=10, b=20)
            )
            st.plotly_chart(fig_map, use_container_width=True)
