"""
src/inspect_and_visualize_nc.py

Automated NetCDF inspection, tabular decoding, and geospatial visualization pipeline
for the OceanEmbed framework (SIH 2026).

This script:
1. Inspections dimensions, coordinate ranges, variable metadata, and missing/NaN values.
2. Extracts the surface snapshot for the North Indian Ocean bounds (5°N-30°N, 45°E-105°E).
3. Exports 2D CSV grids and flattened preview tables of the first 1000 ocean points.
4. Generates individual high-resolution (300 DPI) spatial heatmaps with grey land masking.
5. Generates a combined 5-panel parameter overview plot including a custom info card.
6. Programmatically checks for subsurface depth data and generates transect plots if applicable.

Running Standalone:
python -m src.inspect_and_visualize_nc
"""

import os
import glob
import copy
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def get_time_name(ds):
    """Detect time coordinate name."""
    for name in ["time", "valid_time", "t"]:
        if name in ds.coords:
            return name
    return None


def get_lat_lon_names(ds):
    """Detect latitude and longitude coordinate names."""
    lat_name = None
    lon_name = None
    for name in ["latitude", "lat", "LAT"]:
        if name in ds.coords:
            lat_name = name
            break
    for name in ["longitude", "lon", "LON"]:
        if name in ds.coords:
            lon_name = name
            break
    return lat_name, lon_name


def get_depth_name(ds):
    """Detect depth coordinate name."""
    for name in ["depth", "deptho", "level", "z"]:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def crop_to_nio(ds, lat_name, lon_name):
    """Crop dataset to North Indian Ocean bounds: 5°N-30°N, 45°E-105°E."""
    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values

    # Determine sorting and create slices
    lat_descending = lat_vals[0] > lat_vals[-1]
    if lat_descending:
        lat_slice = slice(30.0, 5.0)
    else:
        lat_slice = slice(5.0, 30.0)

    lon_descending = lon_vals[0] > lon_vals[-1]
    if lon_descending:
        lon_slice = slice(105.0, 45.0)
    else:
        lon_slice = slice(45.0, 105.0)

    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})


def main():
    # Define paths
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    visualizations_dir = "reports/visualizations"

    # Ensure directories exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)

    print("================================================================")
    # Get all .nc files
    nc_files = glob.glob(os.path.join(raw_dir, "*.nc"))
    if not nc_files:
        print(f"[ERROR] No NetCDF files found in '{raw_dir}'.")
        return

    print(f"Found {len(nc_files)} NetCDF file(s) for inspection and processing.")
    print("================================================================")

    # Dictionary to store datasets for combined overview plotting
    surface_datasets = {}

    for filepath in nc_files:
        filename = os.path.basename(filepath)
        print(f"\n--- Processing: {filename} ---")
        
        # Load dataset
        ds = xr.open_dataset(filepath)

        # Detect coordinate names
        time_name = get_time_name(ds)
        lat_name, lon_name = get_lat_lon_names(ds)
        depth_name = get_depth_name(ds)

        # --------------------------------------------------------
        # Task 2: File-by-File Inspection & Metadata Logging
        # --------------------------------------------------------
        dims_str = ", ".join([f"{dim}: {size}" for dim, size in ds.dims.items()])
        print(f"Dimensions: {dims_str}")
        
        # Coordinate ranges
        if lat_name:
            lat_min = float(ds[lat_name].min())
            lat_max = float(ds[lat_name].max())
            print(f"Latitude Range: {lat_min:.3f}° to {lat_max:.3f}°")
        else:
            print("Latitude: Not found")

        if lon_name:
            lon_min = float(ds[lon_name].min())
            lon_max = float(ds[lon_name].max())
            print(f"Longitude Range: {lon_min:.3f}° to {lon_max:.3f}°")
        else:
            print("Longitude: Not found")

        if time_name:
            time_vals = ds[time_name].values
            time_span = f"{pd.to_datetime(time_vals[0]).strftime('%Y-%m-%d')} to {pd.to_datetime(time_vals[-1]).strftime('%Y-%m-%d')}"
            print(f"Time Span: {time_span} ({len(time_vals)} steps)")
        else:
            print("Time: Not found")

        if depth_name:
            depth_vals = ds[depth_name].values
            print(f"Depth Levels: {depth_vals.tolist()} (depth coordinate present)")
        else:
            print("Depth: Not present")

        # Variables inspection
        print("Variables:")
        for var_name in ds.data_vars:
            var = ds[var_name]
            long_name = var.attrs.get("long_name", "N/A")
            units = var.attrs.get("units", "N/A")
            fill_val = var.attrs.get("_FillValue", var.attrs.get("missing_value", "N/A"))
            nan_count = np.isnan(var.values).sum()
            total_points = var.values.size
            nan_pct = (nan_count / total_points) * 100
            
            print(f"  * {var_name}: {long_name}")
            print(f"    Units: {units} | Fill Value: {fill_val}")
            print(f"    NaNs / Total Points: {nan_count:,} / {total_points:,} ({nan_pct:.2f}%)")

        # --------------------------------------------------------
        # Task 3: Decode & Export Readable Tabular Samples (NIO Bounds)
        # --------------------------------------------------------
        # Extract first time step
        if time_name:
            ds_subset = ds.isel({time_name: 0})
            step_date = pd.to_datetime(ds[time_name].values[0]).strftime('%Y-%m-%d')
        else:
            ds_subset = ds
            step_date = "unknown"

        # Select surface level if depth exists
        if depth_name:
            ds_subset = ds_subset.isel({depth_name: 0})

        # Crop to North Indian Ocean bounds
        if lat_name and lon_name:
            ds_subset = crop_to_nio(ds_subset, lat_name, lon_name)

        # Store the subset for mapping
        surface_datasets[filename] = {
            "ds": ds_subset,
            "lat_name": lat_name,
            "lon_name": lon_name,
            "date": step_date
        }

        # Export grid & flattened CSVs
        for var_name in ds_subset.data_vars:
            da = ds_subset[var_name]
            
            # Export 2D Grid CSV: Latitudes as rows, Longitudes as columns
            # Ensure latitude is sorted descending (North at the top of CSV)
            df_grid = da.to_pandas()
            if df_grid.index[0] < df_grid.index[-1]:
                df_grid = df_grid.iloc[::-1] # Reverse rows
            grid_path = os.path.join(processed_dir, f"sample_grid_{var_name}.csv")
            df_grid.to_csv(grid_path)
            print(f"    -> Exported 2D grid table to: {grid_path}")

            # Export Flattened Preview Table (lat, lon, value) for first 1,000 non-NaN points
            df_flat = da.to_dataframe(name="value").reset_index()
            df_flat = df_flat.rename(columns={lat_name: "lat", lon_name: "lon"})
            df_flat = df_flat[["lat", "lon", "value"]].dropna(subset=["value"])
            
            df_flat_1000 = df_flat.head(1000)
            flat_path = os.path.join(processed_dir, f"sample_flat_{var_name}.csv")
            df_flat_1000.to_csv(flat_path, index=False)
            print(f"    -> Exported flattened preview table to: {flat_path}")

        # Close resource
        ds.close()

    # --------------------------------------------------------
    # Task 4: Generate Individual Spatial 2D Heatmaps
    # --------------------------------------------------------
    print("\n================================================================")
    print("Generating Individual Spatial Maps")
    print("================================================================")

    # We map filenames to variable mapping details
    # SST
    if "sst.nc" in surface_datasets:
        data_info = surface_datasets["sst.nc"]
        ds_sst = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        date_str = data_info["date"]
        
        # CMEMS thetao variable check (convert Kelvin to Celsius if necessary)
        sst_val = ds_sst["thetao"].values
        if np.nanmean(sst_val) > 200:
            sst_val = sst_val - 273.15
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = copy.copy(plt.colormaps["coolwarm"])
        cmap.set_bad(color="#e0e0e0")
        
        mesh = ax.pcolormesh(ds_sst[lon_n], ds_sst[lat_n], sst_val, cmap=cmap, shading="auto")
        cbar = plt.colorbar(mesh, ax=ax, label="Sea Surface Temperature (°C)")
        ax.set_title(f"SST - North Indian Ocean ({date_str})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        fig_path = os.path.join(visualizations_dir, "sst_spatial_map.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SST Spatial Map to: {fig_path}")

    # SSS
    if "sss.nc" in surface_datasets:
        data_info = surface_datasets["sss.nc"]
        ds_sss = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        date_str = data_info["date"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = copy.copy(plt.colormaps["viridis"])
        cmap.set_bad(color="#e0e0e0")
        
        mesh = ax.pcolormesh(ds_sss[lon_n], ds_sss[lat_n], ds_sss["so"].values, cmap=cmap, shading="auto")
        cbar = plt.colorbar(mesh, ax=ax, label="Sea Surface Salinity (PSU)")
        ax.set_title(f"SSS - North Indian Ocean ({date_str})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        fig_path = os.path.join(visualizations_dir, "sss_spatial_map.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SSS Spatial Map to: {fig_path}")

    # SSH
    if "ssh.nc" in surface_datasets:
        data_info = surface_datasets["ssh.nc"]
        ds_ssh = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        date_str = data_info["date"]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = copy.copy(plt.colormaps["Spectral_r"])
        cmap.set_bad(color="#e0e0e0")
        
        mesh = ax.pcolormesh(ds_ssh[lon_n], ds_ssh[lat_n], ds_ssh["zos"].values, cmap=cmap, shading="auto")
        cbar = plt.colorbar(mesh, ax=ax, label="Sea Surface Height / SLA (m)")
        ax.set_title(f"SSH - North Indian Ocean ({date_str})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        fig_path = os.path.join(visualizations_dir, "ssh_spatial_map.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved SSH Spatial Map to: {fig_path}")

    # Currents (uo, vo)
    if "currents.nc" in surface_datasets:
        data_info = surface_datasets["currents.nc"]
        ds_curr = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        date_str = data_info["date"]
        
        u = ds_curr["uo"].values
        v = ds_curr["vo"].values
        speed = np.sqrt(u**2 + v**2)
        
        # Save speed grids
        speed_da = xr.DataArray(speed, coords=ds_curr["uo"].coords, dims=ds_curr["uo"].dims)
        df_grid = speed_da.to_pandas()
        if df_grid.index[0] < df_grid.index[-1]:
            df_grid = df_grid.iloc[::-1]
        grid_path = os.path.join(processed_dir, "sample_grid_current_speed.csv")
        df_grid.to_csv(grid_path)
        
        # Flattened table
        df_flat = speed_da.to_dataframe(name="value").reset_index()
        df_flat = df_flat.rename(columns={lat_n: "lat", lon_n: "lon"})
        df_flat = df_flat[["lat", "lon", "value"]].dropna(subset=["value"])
        df_flat.head(1000).to_csv(os.path.join(processed_dir, "sample_flat_current_speed.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = copy.copy(plt.colormaps["magma"])
        cmap.set_bad(color="#e0e0e0")
        
        mesh = ax.pcolormesh(ds_curr[lon_n], ds_curr[lat_n], speed, cmap=cmap, shading="auto")
        cbar = plt.colorbar(mesh, ax=ax, label="Current Speed (m/s)")
        
        # Overplot quiver arrows (subsample for readability)
        skip = 5
        lon_q = ds_curr[lon_n].values[::skip]
        lat_q = ds_curr[lat_n].values[::skip]
        lon_grid, lat_grid = np.meshgrid(lon_q, lat_q)
        u_q = u[::skip, ::skip]
        v_q = v[::skip, ::skip]
        
        ax.quiver(lon_grid, lat_grid, u_q, v_q, color="white", scale=12, width=0.002, alpha=0.8)
        
        ax.set_title(f"Surface Currents - North Indian Ocean ({date_str})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        fig_path = os.path.join(visualizations_dir, "currents_spatial_map.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Currents Spatial Map to: {fig_path}")

    # Winds (u10, v10)
    if "winds.nc" in surface_datasets:
        data_info = surface_datasets["winds.nc"]
        ds_winds = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        date_str = data_info["date"]
        
        u10 = ds_winds["u10"].values
        v10 = ds_winds["v10"].values
        wind_speed = np.sqrt(u10**2 + v10**2)
        
        # Save wind speed grids
        speed_da = xr.DataArray(wind_speed, coords=ds_winds["u10"].coords, dims=ds_winds["u10"].dims)
        df_grid = speed_da.to_pandas()
        if df_grid.index[0] < df_grid.index[-1]:
            df_grid = df_grid.iloc[::-1]
        grid_path = os.path.join(processed_dir, "sample_grid_wind_speed.csv")
        df_grid.to_csv(grid_path)
        
        # Flattened table
        df_flat = speed_da.to_dataframe(name="value").reset_index()
        df_flat = df_flat.rename(columns={lat_n: "lat", lon_n: "lon"})
        df_flat = df_flat[["lat", "lon", "value"]].dropna(subset=["value"])
        df_flat.head(1000).to_csv(os.path.join(processed_dir, "sample_flat_wind_speed.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = copy.copy(plt.colormaps["plasma"])
        cmap.set_bad(color="#e0e0e0")
        
        mesh = ax.pcolormesh(ds_winds[lon_n], ds_winds[lat_n], wind_speed, cmap=cmap, shading="auto")
        cbar = plt.colorbar(mesh, ax=ax, label="Wind Speed (m/s)")
        ax.set_title(f"Surface Winds - North Indian Ocean ({date_str})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        fig_path = os.path.join(visualizations_dir, "winds_spatial_map.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Winds Spatial Map to: {fig_path}")

    # --------------------------------------------------------
    # Task 5: Generate Combined 5-Panel Input Overview
    # --------------------------------------------------------
    print("\n================================================================")
    print("Generating 5-Panel Overview Figure")
    print("================================================================")
    
    # We will lay out the 5 channels on a 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    
    # 1. SST
    if "sst.nc" in surface_datasets:
        ax = axes[0, 0]
        data_info = surface_datasets["sst.nc"]
        ds_sst = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        sst_val = ds_sst["thetao"].values
        if np.nanmean(sst_val) > 200:
            sst_val = sst_val - 273.15
        
        cmap = copy.copy(plt.colormaps["coolwarm"])
        cmap.set_bad(color="#e0e0e0")
        mesh = ax.pcolormesh(ds_sst[lon_n], ds_sst[lat_n], sst_val, cmap=cmap, shading="auto")
        plt.colorbar(mesh, ax=ax, label="SST (°C)", pad=0.02)
        ax.set_title("Sea Surface Temperature (SST)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=8)

    # 2. SSS
    if "sss.nc" in surface_datasets:
        ax = axes[0, 1]
        data_info = surface_datasets["sss.nc"]
        ds_sss = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        
        cmap = copy.copy(plt.colormaps["viridis"])
        cmap.set_bad(color="#e0e0e0")
        mesh = ax.pcolormesh(ds_sss[lon_n], ds_sss[lat_n], ds_sss["so"].values, cmap=cmap, shading="auto")
        plt.colorbar(mesh, ax=ax, label="SSS (PSU)", pad=0.02)
        ax.set_title("Sea Surface Salinity (SSS)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=8)

    # 3. SSH
    if "ssh.nc" in surface_datasets:
        ax = axes[0, 2]
        data_info = surface_datasets["ssh.nc"]
        ds_ssh = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        
        cmap = copy.copy(plt.colormaps["Spectral_r"])
        cmap.set_bad(color="#e0e0e0")
        mesh = ax.pcolormesh(ds_ssh[lon_n], ds_ssh[lat_n], ds_ssh["zos"].values, cmap=cmap, shading="auto")
        plt.colorbar(mesh, ax=ax, label="SSH (m)", pad=0.02)
        ax.set_title("Sea Surface Height (SSH)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=8)

    # 4. Currents
    if "currents.nc" in surface_datasets:
        ax = axes[1, 0]
        data_info = surface_datasets["currents.nc"]
        ds_curr = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        u = ds_curr["uo"].values
        v = ds_curr["vo"].values
        speed = np.sqrt(u**2 + v**2)
        
        cmap = copy.copy(plt.colormaps["magma"])
        cmap.set_bad(color="#e0e0e0")
        mesh = ax.pcolormesh(ds_curr[lon_n], ds_curr[lat_n], speed, cmap=cmap, shading="auto")
        plt.colorbar(mesh, ax=ax, label="Current Speed (m/s)", pad=0.02)
        
        # Subsampled vectors for currents
        skip = 5
        lon_q = ds_curr[lon_n].values[::skip]
        lat_q = ds_curr[lat_n].values[::skip]
        lon_grid, lat_grid = np.meshgrid(lon_q, lat_q)
        u_q = u[::skip, ::skip]
        v_q = v[::skip, ::skip]
        ax.quiver(lon_grid, lat_grid, u_q, v_q, color="white", scale=15, width=0.002, alpha=0.7)
        
        ax.set_title("Surface Currents (Speed & Quivers)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=8)

    # 5. Winds
    if "winds.nc" in surface_datasets:
        ax = axes[1, 1]
        data_info = surface_datasets["winds.nc"]
        ds_winds = data_info["ds"]
        lat_n, lon_n = data_info["lat_name"], data_info["lon_name"]
        u10 = ds_winds["u10"].values
        v10 = ds_winds["v10"].values
        wind_speed = np.sqrt(u10**2 + v10**2)
        
        cmap = copy.copy(plt.colormaps["plasma"])
        cmap.set_bad(color="#e0e0e0")
        mesh = ax.pcolormesh(ds_winds[lon_n], ds_winds[lat_n], wind_speed, cmap=cmap, shading="auto")
        plt.colorbar(mesh, ax=ax, label="Wind Speed (m/s)", pad=0.02)
        ax.set_title("Surface Winds (Speed)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(labelsize=8)

    # 6. Information Card (Row 1, Col 2)
    ax_info = axes[1, 2]
    ax_info.axis("off")
    
    # Construct info content
    common_date = "2024-01-01"
    for k, v in surface_datasets.items():
        common_date = v["date"]
        break

    info_text = (
        "OceanEmbed Framework\n"
        "Input Data Overview Card\n"
        "================================\n\n"
        f"Snapshot Date: {common_date}\n"
        "Spatial Extent: North Indian Ocean\n"
        "  - Latitude:  5.0°N to 30.0°N\n"
        "  - Longitude: 45.0°E to 105.0°E\n\n"
        "Input Parameter Statistics (Mean):\n"
    )
    
    # Calculate means dynamically from the cropped datasets
    stats_lines = []
    if "sst.nc" in surface_datasets:
        sst_val = surface_datasets["sst.nc"]["ds"]["thetao"].values
        if np.nanmean(sst_val) > 200:
            sst_val = sst_val - 273.15
        stats_lines.append(f"  * SST:   {np.nanmean(sst_val):.2f} °C")
    if "sss.nc" in surface_datasets:
        sss_val = surface_datasets["sss.nc"]["ds"]["so"].values
        stats_lines.append(f"  * SSS:   {np.nanmean(sss_val):.2f} PSU")
    if "ssh.nc" in surface_datasets:
        ssh_val = surface_datasets["ssh.nc"]["ds"]["zos"].values
        stats_lines.append(f"  * SSH:   {np.nanmean(ssh_val):.3f} m")
    if "currents.nc" in surface_datasets:
        u_val = surface_datasets["currents.nc"]["ds"]["uo"].values
        v_val = surface_datasets["currents.nc"]["ds"]["vo"].values
        speed_val = np.sqrt(u_val**2 + v_val**2)
        stats_lines.append(f"  * Current Speed: {np.nanmean(speed_val):.2f} m/s")
    if "winds.nc" in surface_datasets:
        u10_val = surface_datasets["winds.nc"]["ds"]["u10"].values
        v10_val = surface_datasets["winds.nc"]["ds"]["v10"].values
        wind_speed_val = np.sqrt(u10_val**2 + v10_val**2)
        stats_lines.append(f"  * Wind Speed: {np.nanmean(wind_speed_val):.2f} m/s")

    info_text += "\n".join(stats_lines) + "\n\n"
    info_text += "Note: Grey shaded areas denote land masses\n(reconstructed from NetCDF NaN masks)."

    # Style the text box card
    ax_info.text(
        0.05, 0.95, info_text, 
        transform=ax_info.transAxes, 
        fontsize=10.5, 
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=1.0", facecolor="#f9f9f9", edgecolor="#cccccc", alpha=0.95)
    )
    
    # Save combined layout
    overview_path = os.path.join(visualizations_dir, "surface_parameters_overview.png")
    plt.savefig(overview_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 5-Panel Input Overview to: {overview_path}")

    # --------------------------------------------------------
    # Task 6: Vertical Cross-Section Check (Subsurface)
    # --------------------------------------------------------
    print("\n================================================================")
    print("Vertical Subsurface Transect Check")
    print("================================================================")
    
    subsurface_plotted = False
    for filepath in nc_files:
        ds = xr.open_dataset(filepath)
        depth_name = get_depth_name(ds)
        if depth_name and len(ds[depth_name]) > 1:
            filename = os.path.basename(filepath)
            print(f"Found subsurface levels in file: {filename} ({len(ds[depth_name])} levels)")
            
            # Extract names
            lat_n, lon_n = get_lat_lon_names(ds)
            time_name = get_time_name(ds)
            
            # Slice along 15.0°N (method nearest), for the first time step
            ds_time0 = ds.isel({time_name: 0})
            slice_15n = ds_time0.sel({lat_n: 15.0}, method="nearest")
            
            # Filter depths to 0-1000m
            depths = slice_15n[depth_name].values
            depth_mask = (depths >= 0.0) & (depths <= 1000.0)
            slice_1000m = slice_15n.sel({depth_name: depths[depth_mask]})
            
            # Find the first data variable to plot
            var_to_plot = list(ds.data_vars.keys())[0]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            cmap = copy.copy(plt.colormaps["viridis"])
            cmap.set_bad(color="#e0e0e0")
            
            data_arr = slice_1000m[var_to_plot]
            
            # Make sure we coordinates align: x=longitude, y=depth
            mesh = ax.pcolormesh(
                slice_1000m[lon_n].values, 
                slice_1000m[depth_name].values, 
                data_arr.values, 
                cmap=cmap, 
                shading="auto"
            )
            
            # Invert depth axis so 0m is at the top
            ax.invert_yaxis()
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Depth (m)")
            ax.set_title(f"Subsurface Transect along 15°N - {var_to_plot} ({slice_1000m[time_name].values})", fontsize=11, fontweight="bold")
            plt.colorbar(mesh, ax=ax, label=f"{var_to_plot} ({data_arr.attrs.get('units', 'N/A')})")
            
            transect_path = os.path.join(visualizations_dir, "subsurface_transect_15N.png")
            plt.savefig(transect_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"Successfully saved subsurface transect plot to: {transect_path}")
            subsurface_plotted = True
            ds.close()
            break
        ds.close()

    if not subsurface_plotted:
        print("No NetCDF files contain subsurface depth coordinates with multiple levels.")
        print("Skipping subsurface vertical transect plot (only surface level data present).")

    print("\n================================================================")
    print("Inspection, Decoding, and Visualization pipeline successfully finished!")
    print("================================================================")


if __name__ == "__main__":
    main()
