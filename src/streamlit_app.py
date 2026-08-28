"""
app/streamlit_app.py

Enterprise-grade, publication-ready research dashboard for the OceanEmbed framework.
Structured to provide direct depth-by-depth predicted temperature tables and vertical profiles 
as the primary view, with verification, benchmarking, and spatial maps in a secondary tab.
Upgraded to feature a dual Plotly subplot rendering a water column thermal ribbon and interactive curve.

Running Standalone:
streamlit run app/streamlit_app.py
"""

import os
import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import streamlit as st
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Import model definition from model.py
from src.model import OceanEncoderDecoder


# Page Configuration
st.set_page_config(
    page_title="OceanEmbed: Subsurface Temperature Reconstruction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean, Minimalist CSS Styling Injection (No Emojis)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Metric Badges */
    .metric-badge {
        display: inline-block;
        background-color: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 4px;
        padding: 6px 12px;
        margin-right: 10px;
        font-size: 12px;
        font-weight: 500;
        color: #475569;
    }
    
    /* Custom Slate Container Cards */
    .dashboard-card {
        background-color: rgba(148, 163, 184, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 6px;
        padding: 18px;
        margin-bottom: 18px;
    }
    
    .card-title {
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #64748b;
        margin-bottom: 6px;
    }
    
    .card-value {
        font-size: 26px;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Dark Mode Compatibility adjustments for cards */
    @media (prefers-color-scheme: dark) {
        .card-value {
            color: #f8fafc;
        }
        .metric-badge {
            color: #cbd5e1;
            background-color: rgba(203, 213, 225, 0.08);
            border: 1px solid rgba(203, 213, 225, 0.2);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


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


# --- Initialize Resources ---
model, input_scaler, output_scaler, spatial_data = load_all_resources()

# Extract coordinates and lists
target_lat = np.linspace(5.0, 30.0, 101)
target_lon = np.linspace(45.0, 105.0, 241)
common_days = pd.date_range(start="2024-01-01", periods=58, freq="D")
depth_levels = spatial_data["scalers"]["depth_levels"].numpy().tolist()

# --------------------------------------------------------
# 1. Application Header & Top Navigation
# --------------------------------------------------------
st.markdown("## OceanEmbed: Satellite-Driven 3D Subsurface Ocean Temperature Reconstruction")
st.markdown(
    "<p style='color: #64748b; font-size: 14px; margin-top: -10px;'>"
    "North Indian Ocean Domain (5°N–30°N, 45°E–105°E) | 0.25° Spatial Resolution | 15 Standard Depths (0–1000m)"
    "</p>",
    unsafe_allow_html=True
)

# Text Badges for System Status
st.markdown(
    f"""
    <div>
        <span class="metric-badge">Model Status: OceanUNet Architecture Active</span>
        <span class="metric-badge">Temporal Bounds: 2024-01-01 to 2024-02-27</span>
        <span class="metric-badge">Reconstruction Domain: Aligned 0.25 Grid</span>
    </div>
    <br>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# 2. Input Selectors (Top Section)
# --------------------------------------------------------
st.markdown("### Reconstruction Slicer Settings")
col_date, col_lat, col_lon = st.columns(3)

with col_date:
    date_strs = [d.strftime("%Y-%m-%d") for d in common_days]
    selected_date_str = st.selectbox(
        "Analysis Snapshot Date",
        options=date_strs,
        index=0
    )
    day_idx = date_strs.index(selected_date_str)

with col_lat:
    selected_lat = st.number_input(
        "Latitude (5.0°N to 30.0°N)",
        min_value=5.0,
        max_value=30.0,
        value=15.0,
        step=0.25
    )

with col_lon:
    selected_lon = st.number_input(
        "Longitude (45.0°E to 105.0°E)",
        min_value=45.0,
        max_value=105.0,
        value=85.0,
        step=0.25
    )

# Prominent Action Button
reconstruct_triggered = st.button(
    "Reconstruct Subsurface Temperature", 
    type="primary", 
    use_container_width=True
)

# Manage state for reconstruct button
if reconstruct_triggered or 'calculated' not in st.session_state:
    st.session_state.calculated = True

# --------------------------------------------------------
# Grid Data Calculations
# --------------------------------------------------------
X_surface_norm = spatial_data["X_surface"][day_idx].numpy()
surface_means = spatial_data["scalers"]["surface_means"].numpy()
surface_stds = spatial_data["scalers"]["surface_stds"].numpy()
ocean_mask = spatial_data["ocean_mask"].numpy()

# Denormalize variables
sst_grid = X_surface_norm[0] * surface_stds[0] + surface_means[0]
if np.nanmean(sst_grid) > 200:
    sst_grid = sst_grid - 273.15
    
sss_grid = X_surface_norm[1] * surface_stds[1] + surface_means[1]
ssh_grid = X_surface_norm[2] * surface_stds[2] + surface_means[2]
curr_u = X_surface_norm[3] * surface_stds[3] + surface_means[3]
curr_v = X_surface_norm[4] * surface_stds[4] + surface_means[4]
wind_u = X_surface_norm[5] * surface_stds[5] + surface_means[5]
wind_v = X_surface_norm[6] * surface_stds[6] + surface_means[6]

current_speed = np.sqrt(curr_u**2 + curr_v**2)
wind_speed = np.sqrt(wind_u**2 + wind_v**2)

# Mask land cells with NaN for maps plotting
for grid in [sst_grid, sss_grid, ssh_grid, current_speed, wind_speed]:
    grid[ocean_mask == 0] = np.nan

# Perform grid-wise model inference
pred_grid = run_grid_inference(day_idx, model, input_scaler, output_scaler, spatial_data, target_lat, target_lon)

# Get ground truth 3D subsurface temperature
subsurface_means = spatial_data["scalers"]["subsurface_means"].numpy()
subsurface_stds = spatial_data["scalers"]["subsurface_stds"].numpy()
gt_grid = np.zeros_like(pred_grid)
for d in range(15):
    gt_grid[d] = spatial_data["Y_subsurface"][day_idx, d].numpy() * subsurface_stds[d] + subsurface_means[d]
    gt_grid[d][ocean_mask == 0] = np.nan

# Find closest grid indices for selected Lat/Lon
lat_idx = np.abs(target_lat - selected_lat).argmin()
lon_idx = np.abs(target_lon - selected_lon).argmin()

actual_lat_val = target_lat[lat_idx]
actual_lon_val = target_lon[lon_idx]


# --------------------------------------------------------
# 3. Main Dashboard Panels (Tabs)
# --------------------------------------------------------
tab_reconstruction, tab_verification = st.tabs([
    "Subsurface Temperature Profile",
    "Validation and Spatial Maps"
])

# --- Tab 1: Subsurface Temperature Profile (Primary View) ---
with tab_reconstruction:
    if ocean_mask[lat_idx, lon_idx] == 0:
        st.warning(f"Selected coordinates ({selected_lat}°N, {selected_lon}°E) correspond to a land cell. Please select coordinate values over water.")
    else:
        # Extract predictions for the specific point
        raw_pt_inputs = np.array([
            sst_grid[lat_idx, lon_idx],
            sss_grid[lat_idx, lon_idx],
            ssh_grid[lat_idx, lon_idx],
            curr_u[lat_idx, lon_idx],
            curr_v[lat_idx, lon_idx],
            wind_u[lat_idx, lon_idx],
            wind_v[lat_idx, lon_idx]
        ])
        
        doy = common_days[day_idx].dayofyear
        pt_input = np.array([[
            actual_lat_val,
            actual_lon_val,
            doy,
            raw_pt_inputs[0],
            raw_pt_inputs[1],
            raw_pt_inputs[2],
            raw_pt_inputs[3],
            raw_pt_inputs[4],
            raw_pt_inputs[5],
            raw_pt_inputs[6]
        ]])
        
        # Scale and run model inference
        pt_scaled = input_scaler.transform(pt_input)
        with torch.no_grad():
            pt_pred_scaled = model(torch.FloatTensor(pt_scaled)).numpy()
        pred_profile_8 = output_scaler.inverse_transform(pt_pred_scaled).squeeze()
        
        # Interpolate predictions to standard 15 depths
        model_depths = [0, 10, 30, 50, 75, 100, 200, 500]
        pred_profile = np.interp(depth_levels, model_depths, pred_profile_8)
        
        # Calculate Mixed Layer Depth (MLD) for ocean layer classification
        surface_t = pred_profile[0]
        mld_idxs = np.where(pred_profile <= (surface_t - 0.5))[0]
        mld_depth = depth_levels[mld_idxs[0]] if len(mld_idxs) > 0 else 40.0
        
        # Classify Ocean Layers
        layers = []
        for d in depth_levels:
            if d <= mld_depth:
                layers.append("Mixed Layer")
            elif d >= 500.0:
                layers.append("Deep Ocean")
            else:
                layers.append("Thermocline")

        # Two Column Layout (35% Left, 65% Right)
        col_table, col_profile = st.columns([35, 65])
        
        with col_table:
            st.markdown("#### Layer-by-Layer Predicted Temperature")
            
            # Format and display output table
            df_profile_display = pd.DataFrame({
                "Depth Level (m)": [f"{d:.2f} m" for d in depth_levels],
                "Predicted Temp (°C)": [f"{t:.3f} °C" for t in pred_profile],
                "Ocean Layer": layers
            })
            
            st.dataframe(
                df_profile_display,
                use_container_width=True,
                hide_index=True,
                height=560
            )
            
            # Download Depth Profile CSV Button directly under table
            csv_profile_data = df_profile_display.to_csv(index=False)
            st.download_button(
                label="Download Depth Profile (CSV)",
                data=csv_profile_data,
                file_name=f"oceanembed_profile_{selected_lat:.2f}N_{selected_lon:.2f}E_{selected_date_str}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with col_profile:
            st.markdown("#### Dual Subsurface Analysis")
            
            # Calculate Thermocline Depth (D20)
            try:
                sort_idx = np.argsort(pred_profile)
                d20_depth = float(np.interp(20.0, pred_profile[sort_idx], np.array(depth_levels)[sort_idx]))
                if d20_depth < 0 or d20_depth > 1000:
                    d20_depth = 120.0
            except:
                d20_depth = 120.0

            # 1. Setup Combined Plotly Dual Subplot figure
            fig = make_subplots(
                rows=1, cols=2,
                shared_yaxes=True,
                column_widths=[0.25, 0.75],
                horizontal_spacing=0.08
            )
            
            # Left Subplot: Water Column Heatmap Ribbon
            z_heatmap = [[t] for t in pred_profile]
            fig.add_trace(
                go.Heatmap(
                    x=["Water Column"],
                    y=depth_levels,
                    z=z_heatmap,
                    colorscale="RdYlBu",
                    reversescale=True,
                    showscale=False,
                    hovertemplate="Depth: %{y}m<br>Predicted: %{z:.2f}°C<extra></extra>"
                ),
                row=1, col=1
            )
            
            # Right Subplot: Vertical Temperature Profile Curve
            fig.add_trace(
                go.Scatter(
                    x=pred_profile,
                    y=depth_levels,
                    mode="lines+markers",
                    line=dict(color="#0284c7", width=3),
                    marker=dict(size=7, color="#005f73"),
                    hovertemplate="Depth: %{y:.1f} m<br>Temperature: %{x:.2f} °C<extra></extra>"
                ),
                row=1, col=2
            )
            
            # Add Horizontal Zone Callout Lines & Labels on Heatmap Ribbon
            fig.add_hline(y=50, line_dash="dash", line_color="rgba(148, 163, 184, 0.45)", row=1, col=1)
            fig.add_hline(y=200, line_dash="dash", line_color="rgba(148, 163, 184, 0.45)", row=1, col=1)
            
            # Layer annotations for Heatmap (col 1)
            fig.add_annotation(
                x=0, y=25, text="Mixed Layer",
                showarrow=False, font=dict(size=9, color="#0f172a", weight="bold"),
                xref="x1", yref="y1"
            )
            fig.add_annotation(
                x=0, y=125, text="Thermocline",
                showarrow=False, font=dict(size=9, color="#0f172a", weight="bold"),
                xref="x1", yref="y1"
            )
            fig.add_annotation(
                x=0, y=600, text="Deep Ocean",
                showarrow=False, font=dict(size=9, color="#ffffff", weight="bold"),
                xref="x1", yref="y1"
            )

            # Dashed Reference Line for D20 on Scatter Curve (col 2)
            fig.add_hline(y=d20_depth, line_dash="dash", line_color="#db2777", row=1, col=2,
                          annotation_text=f"D20 ({d20_depth:.1f}m)", annotation_position="bottom right")

            # Axis & Layout Formatting
            fig.update_yaxes(
                autorange="reversed",
                range=[1000, -10],
                title_text="Depth (meters)",
                tickfont=dict(color="#475569"),
                showgrid=False,
                row=1, col=1
            )
            fig.update_xaxes(showgrid=False, showticklabels=False, row=1, col=1)
            
            fig.update_yaxes(
                autorange="reversed",
                range=[1000, -10],
                showgrid=True,
                gridcolor="#f1f5f9",
                row=1, col=2
            )
            fig.update_xaxes(
                title_text="Temperature (°C)",
                showgrid=True,
                gridcolor="#f1f5f9",
                tickfont=dict(color="#475569"),
                row=1, col=2
            )
            
            fig.update_layout(
                height=620,
                margin=dict(l=40, r=40, t=30, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig, use_container_width=True)


# --- Tab 2: Validation and Spatial Maps (Secondary View) ---
with tab_verification:
    if ocean_mask[lat_idx, lon_idx] == 1:
        # Reconstruct ground truth profile
        gt_profile = gt_grid[:, lat_idx, lon_idx]
        
        # Mixed Layer Depth (MLD)
        surface_t = pred_profile[0]
        mld_idxs = np.where(pred_profile <= (surface_t - 0.5))[0]
        mld_depth = depth_levels[mld_idxs[0]] if len(mld_idxs) > 0 else 40.0
        
        # Thermocline Depth (D20)
        try:
            sort_idx = np.argsort(pred_profile)
            d20_depth = float(np.interp(20.0, pred_profile[sort_idx], np.array(depth_levels)[sort_idx]))
            if d20_depth < 0 or d20_depth > 1000:
                d20_depth = 120.0
        except:
            d20_depth = 120.0
            
        thermocline_temp = float(np.interp(d20_depth, depth_levels, pred_profile))
        
        # Error metrics
        rmse_pt = np.sqrt(np.mean((gt_profile - pred_profile)**2))
        pearson_r = np.corrcoef(gt_profile, pred_profile)[0, 1]

        st.markdown("### Technical Verification & Skill Metrics")
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="card-title">Profile RMSE (All Depths)</div>
                    <div class="card-value">{rmse_pt:.4f} °C</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="card-title">Pearson Correlation (R)</div>
                    <div class="card-value">{pearson_r:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="card-title">Mixed Layer Depth</div>
                    <div class="card-value">{mld_depth:.1f} m</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="dashboard-card">
                    <div class="card-title">Thermocline D20 Depth</div>
                    <div class="card-value">{d20_depth:.1f} m</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Sync Plot: Model Predictions vs Ground Truth (GLORYS)
        st.markdown("#### Subsurface Temperature Verification Profile")
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        ax.plot(pred_profile, depth_levels, marker='o', color='#0284c7', linewidth=2.5, label='Reconstruction (OceanEmbed)')
        ax.plot(gt_profile, depth_levels, marker='s', color='#e11d48', linewidth=1.8, linestyle='--', label='Ground Truth (GLORYS)')
        ax.fill_betweenx(depth_levels, pred_profile, gt_profile, color='#0284c7', alpha=0.12, label='Prediction Deviation')
        
        ax.axhline(y=mld_depth, color='#d97706', linestyle=':', linewidth=1.5, label=f'Mixed Layer Depth ({mld_depth:.1f}m)')
        ax.axhline(y=d20_depth, color='#db2777', linestyle='-.', linewidth=1.5, label=f'Thermocline D20 ({d20_depth:.1f}m)')
        
        ax.set_xlabel("Temperature (°C)", fontsize=11, fontweight="semibold", color="#0f172a")
        ax.set_ylabel("Depth (meters)", fontsize=11, fontweight="semibold", color="#0f172a")
        ax.invert_yaxis()
        ax.set_ylim(1000, 0)
        ax.grid(True, which="both", color="#e2e8f0", linestyle=":", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
        
        st.pyplot(fig)

    # Spatial 2D Maps Section
    st.markdown("---")
    st.markdown("### Spatial Map Validation")
    
    surface_channels = {
        "Sea Surface Temperature (SST)": (sst_grid, "coolwarm", "°C"),
        "Sea Surface Salinity (SSS)": (sss_grid, "viridis", "PSU"),
        "Sea Surface Height (SSH)": (ssh_grid, "Spectral_r", "meters"),
        "Surface Current Speed": (current_speed, "magma", "m/s"),
        "Surface Wind Speed": (wind_speed, "plasma", "m/s")
    }
    
    selected_channel = st.selectbox(
        "Select Surface Observation Channel (Left Spatial Map)",
        options=list(surface_channels.keys())
    )
    left_grid, cmap_left_name, left_unit = surface_channels[selected_channel]
    
    selected_verif_depth = st.selectbox(
        "Select Depth for Reconstructed Temperature Map (Right Spatial Map)",
        options=depth_levels,
        format_func=lambda x: f"{x:.3f} m",
        key="verif_depth_select"
    )
    verif_depth_idx = depth_levels.index(selected_verif_depth)

    # Plot spatial maps side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#f8fafc')
    ax2.set_facecolor('#f8fafc')
    
    cmap_left = copy.copy(plt.colormaps[cmap_left_name])
    cmap_left.set_bad(color="#e2e8f0")
    mesh1 = ax1.pcolormesh(target_lon, target_lat, left_grid, cmap=cmap_left, shading="auto")
    plt.colorbar(mesh1, ax=ax1, fraction=0.046, pad=0.04).set_label(f"{selected_channel} ({left_unit})", color="#0f172a")
    ax1.set_title(f"Surface Observation: {selected_channel}", fontsize=12, fontweight="bold", color="#0f172a")
    
    pred_layer = pred_grid[verif_depth_idx]
    cmap_right = copy.copy(plt.colormaps["inferno"])
    cmap_right.set_bad(color="#e2e8f0")
    mesh2 = ax2.pcolormesh(target_lon, target_lat, pred_layer, cmap=cmap_right, shading="auto")
    plt.colorbar(mesh2, ax=ax2, fraction=0.046, pad=0.04).set_label("Reconstructed Temperature (°C)", color="#0f172a")
    ax2.set_title(f"Model Reconstruction: Subsurface Temp at {selected_verif_depth:.2f}m", fontsize=12, fontweight="bold", color="#0f172a")
    
    for ax in [ax1, ax2]:
        ax.set_xlabel("Longitude (°E)", fontsize=9, color="#0f172a")
        ax.set_ylabel("Latitude (°N)", fontsize=9, color="#0f172a")
        ax.tick_params(colors="#0f172a", labelsize=8)
        ax.grid(True, which="both", color="#e2e8f0", linestyle=":", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
            
    plt.tight_layout()
    st.pyplot(fig)

    # Domain Stats
    mean_temp = np.nanmean(pred_layer)
    spatial_var = np.nanvar(pred_layer)
    mean_val_grid = np.nanmean(pred_layer)
    max_anomaly = np.nanmax(np.abs(pred_layer - mean_val_grid))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='dashboard-card'><div class='card-title'>Domain Mean Temp</div><div class='card-value'>{mean_temp:.3f} °C</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='dashboard-card'><div class='card-title'>Spatial Variance</div><div class='card-value'>{spatial_var:.4f} °C²</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='dashboard-card'><div class='card-title'>Max Eddy Anomaly</div><div class='card-value'>{max_anomaly:.3f} °C</div></div>", unsafe_allow_html=True)

    # Depth-wise Benchmark Plot & Table
    st.markdown("---")
    st.markdown("### Depth-wise Error Analysis & CSV Export")
    
    rmse_per_depth = []
    mae_per_depth = []
    mean_gt_temp = []
    mean_pred_temp = []
    for d_idx in range(15):
        gt_layer = gt_grid[d_idx]
        pred_layer = pred_grid[d_idx]
        diff = gt_layer[ocean_mask == 1] - pred_layer[ocean_mask == 1]
        rmse_per_depth.append(np.sqrt(np.mean(diff**2)))
        mae_per_depth.append(np.mean(np.abs(diff)))
        mean_gt_temp.append(np.nanmean(gt_layer))
        mean_pred_temp.append(np.nanmean(pred_layer))
        
    col_bench_plot, col_bench_table = st.columns([1, 1])
    
    with col_bench_plot:
        fig, ax = plt.subplots(figsize=(6.5, 7.5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')
        
        ax.plot(rmse_per_depth, depth_levels, marker='o', color='#2563eb', linewidth=2.5, label='RMSE (°C)')
        ax.plot(mae_per_depth, depth_levels, marker='x', color='#059669', linewidth=1.8, linestyle='-.', label='MAE (°C)')
        
        ax.set_xlabel("Reconstruction Accuracy Error (°C)", fontsize=11, fontweight="semibold", color="#0f172a")
        ax.set_ylabel("Depth (meters)", fontsize=11, fontweight="semibold", color="#0f172a")
        ax.set_title("Reconstruction Error vs Depth Tier", fontsize=12, fontweight="bold", color="#0f172a", pad=15)
        ax.invert_yaxis()
        ax.set_ylim(1000, 0)
        
        ax.tick_params(colors="#0f172a", labelsize=9)
        ax.grid(True, which="both", color="#e2e8f0", linestyle=":", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_color("#cbd5e1")
        ax.legend(facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_bench_table:
        st.markdown("#### Accuracy Summary Table")
        
        accuracy_data = []
        for i, depth in enumerate(depth_levels):
            accuracy_data.append({
                "Depth Tier": f"{depth:.3f} m",
                "Mean Ground Truth (°C)": f"{mean_gt_temp[i]:.3f}",
                "Mean Predicted (°C)": f"{mean_pred_temp[i]:.3f}",
                "RMSE (°C)": f"{rmse_per_depth[i]:.4f}",
                "MAE (°C)": f"{mae_per_depth[i]:.4f}"
            })
        df_bench = pd.DataFrame(accuracy_data)
        st.dataframe(df_bench, use_container_width=True, hide_index=True, height=420)
        
        st.markdown("#### Export 3D Spatial Predictions")
        st.write("Export coordinates, predicted temperatures, and ground truth temperatures as a clean flat CSV.")
        
        # Flattened grid export generation
        export_rows = []
        for d_idx, depth_val in enumerate(depth_levels):
            p_layer = pred_grid[d_idx]
            g_layer = gt_grid[d_idx]
            for lat_i in range(101):
                for lon_i in range(241):
                    if ocean_mask[lat_i, lon_i] == 1:
                        export_rows.append({
                            "Latitude": target_lat[lat_i],
                            "Longitude": target_lon[lon_i],
                            "Depth_m": depth_val,
                            "Predicted_Temp_C": p_layer[lat_i, lon_i],
                            "Ground_Truth_Temp_C": g_layer[lat_i, lon_i]
                        })
                        
        df_export = pd.DataFrame(export_rows)
        csv_data = df_export.to_csv(index=False)
        st.download_button(
            label="Export 3D Reconstructed Field (CSV)",
            data=csv_data,
            file_name=f"oceanembed_3d_reconstruction_{selected_date_str}.csv",
            mime="text/csv",
            use_container_width=True
        )
