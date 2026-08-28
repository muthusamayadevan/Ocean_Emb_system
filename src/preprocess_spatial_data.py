"""
src/preprocess_spatial_data.py

Harmonizes and regrids surface satellite observations and subsurface 3D GLORYS
potential temperature target data into standardized PyTorch tensors for spatial modeling.

Spatial Grid: 0.25° x 0.25° grid over 5°N-30°N (101 lats) and 45°E-105°E (241 lons).
Depth Levels: 15 standard levels.
Time Alignment: Chronologically synchronized daily steps.

Running Standalone:
python -m src.preprocess_spatial_data
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import torch


def main():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Define Target Spatial Reference Coordinates
    target_lat = np.linspace(5.0, 30.0, 101)
    target_lon = np.linspace(45.0, 105.0, 241)
    
    # Standard depth levels
    target_depths = [
        0.494, 5.08, 9.57, 21.6, 29.45, 55.76, 77.85, 109.73, 
        130.67, 155.85, 222.48, 318.13, 541.09, 763.33, 902.34
    ]

    print("================================================================")
    print("OceanEmbed Spatial Data Preprocessing Pipeline")
    print("================================================================")

    # Load datasets
    print("Loading datasets...")
    ds_sst = xr.open_dataset(os.path.join(raw_dir, "sst.nc"))
    ds_sss = xr.open_dataset(os.path.join(raw_dir, "sss.nc"))
    ds_ssh = xr.open_dataset(os.path.join(raw_dir, "ssh.nc"))
    ds_curr = xr.open_dataset(os.path.join(raw_dir, "currents.nc"))
    ds_winds = xr.open_dataset(os.path.join(raw_dir, "winds.nc"))
    ds_target = xr.open_dataset(os.path.join(raw_dir, "glorys_subsurface_temp.nc"))

    # 2. Resample Hourly Winds to Daily Means
    print("Computing daily mean for winds...")
    ds_winds_daily = ds_winds.resample(valid_time="1D").mean()

    # 3. Synchronize Time Alignment
    print("Synchronizing temporal alignment...")
    times_sst = pd.to_datetime(ds_sst.time.values)
    times_sss = pd.to_datetime(ds_sss.time.values)
    times_ssh = pd.to_datetime(ds_ssh.time.values)
    times_curr = pd.to_datetime(ds_curr.time.values)
    times_winds = pd.to_datetime(ds_winds_daily.valid_time.values)
    times_target = pd.to_datetime(ds_target.time.values)

    # Compute intersection of dates
    common_days = times_sst.intersection(times_sss)\
                           .intersection(times_ssh)\
                           .intersection(times_curr)\
                           .intersection(times_winds)\
                           .intersection(times_target)
    
    num_days = len(common_days)
    print(f"  * Chronologically aligned date range: {common_days[0].strftime('%Y-%m-%d')} to {common_days[-1].strftime('%Y-%m-%d')}")
    print(f"  * Total synchronized daily timestamps: {num_days}")

    # Slice datasets to synchronized common dates
    ds_sst_sel = ds_sst.sel(time=common_days)
    ds_sss_sel = ds_sss.sel(time=common_days)
    ds_ssh_sel = ds_ssh.sel(time=common_days)
    ds_curr_sel = ds_curr.sel(time=common_days)
    ds_winds_sel = ds_winds_daily.sel(valid_time=common_days)
    ds_target_sel = ds_target.sel(time=common_days)

    # 4. Standard Depth Coordinate Selection for Target
    print("Selecting 15 standard target depth levels...")
    available_depths = ds_target_sel.depth.values
    selected_depths = []
    for d in target_depths:
        idx = np.abs(available_depths - d).argmin()
        selected_depths.append(available_depths[idx])
    
    selected_depths = sorted(list(set(selected_depths)))
    print(f"  * Closest depth levels selected: {selected_depths}")
    ds_target_sel = ds_target_sel.sel(depth=selected_depths)

    # 5. Regrid to Reference Grid using Bilinear Interpolation
    print("Regridding variables to reference grid (101 lats x 241 lons)...")
    
    print("  * Regridding SST...")
    sst_regrid = ds_sst_sel["thetao"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    # Convert Kelvin to Celsius if necessary
    if np.nanmean(sst_regrid.values) > 200:
        sst_regrid = sst_regrid - 273.15
        
    print("  * Regridding SSS...")
    sss_regrid = ds_sss_sel["so"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    
    print("  * Regridding SSH...")
    ssh_regrid = ds_ssh_sel["zos"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    
    print("  * Regridding Currents (U and V)...")
    curr_u_regrid = ds_curr_sel["uo"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    curr_v_regrid = ds_curr_sel["vo"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    
    print("  * Regridding Winds (U and V)...")
    wind_u_regrid = ds_winds_sel["u10"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    wind_v_regrid = ds_winds_sel["v10"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    
    print("  * Regridding GLORYS Subsurface Temperature...")
    target_regrid = ds_target_sel["thetao"].interp(latitude=target_lat, longitude=target_lon, method="linear")
    if np.nanmean(target_regrid.values) > 200:
        target_regrid = target_regrid - 273.15

    # Remove depth dim of size 1 if present in inputs
    if "depth" in sst_regrid.coords:
        sst_regrid = sst_regrid.squeeze("depth", drop=True)
    if "depth" in sss_regrid.coords:
        sss_regrid = sss_regrid.squeeze("depth", drop=True)
    if "depth" in curr_u_regrid.coords:
        curr_u_regrid = curr_u_regrid.squeeze("depth", drop=True)
    if "depth" in curr_v_regrid.coords:
        curr_v_regrid = curr_v_regrid.squeeze("depth", drop=True)

    # 6. Masking & Ocean Mask Construction
    print("Constructing binary ocean/land mask...")
    # Mask is 1 where data is valid for all ocean inputs, and 0 for land/empty grid points
    valid_sst = ~np.isnan(sst_regrid.isel(time=0).values)
    valid_sss = ~np.isnan(sss_regrid.isel(time=0).values)
    valid_ssh = ~np.isnan(ssh_regrid.isel(time=0).values)
    valid_curr = ~np.isnan(curr_u_regrid.isel(time=0).values)
    valid_target = ~np.isnan(target_regrid.isel(time=0, depth=0).values)
    
    ocean_mask = valid_sst & valid_sss & valid_ssh & valid_curr & valid_target
    ocean_mask = ocean_mask.astype(np.float32)
    ocean_cells = int(np.sum(ocean_mask))
    total_cells = ocean_mask.size
    print(f"  * Ocean Mask coverage: {ocean_cells:,} / {total_cells:,} grid cells ({ocean_cells/total_cells*100:.2f}%)")

    # 7. Stack Variables
    print("Stacking spatial channels...")
    # Shape of surface stack: (time, channel, lat, lon)
    X_surface = np.stack([
        sst_regrid.values,
        sss_regrid.values,
        ssh_regrid.values,
        curr_u_regrid.values,
        curr_v_regrid.values,
        wind_u_regrid.values,
        wind_v_regrid.values
    ], axis=1)
    
    # Shape of subsurface target: (time, depth, lat, lon)
    Y_subsurface = target_regrid.values

    # 8. Standardize values (mean/std per channel over ocean cells)
    print("Normalizing tensors over ocean grid cells...")
    
    # Surface variables normalization
    X_normalized = np.zeros_like(X_surface)
    surface_means = []
    surface_stds = []
    for c in range(7):
        ch_data = X_surface[:, c, :, :]
        ocean_vals = ch_data[:, ocean_mask == 1]
        mean_val = float(np.nanmean(ocean_vals))
        std_val = float(np.nanstd(ocean_vals))
        if std_val == 0:
            std_val = 1.0
            
        normalized = (ch_data - mean_val) / std_val
        normalized[:, ocean_mask == 0] = 0.0 # Zero out land cells
        normalized = np.nan_to_num(normalized, nan=0.0) # Zero out any lingering NaNs
        
        X_normalized[:, c, :, :] = normalized
        surface_means.append(mean_val)
        surface_stds.append(std_val)
        
    # Subsurface target variables normalization
    Y_normalized = np.zeros_like(Y_subsurface)
    subsurface_means = []
    subsurface_stds = []
    for d in range(15):
        dp_data = Y_subsurface[:, d, :, :]
        ocean_vals = dp_data[:, ocean_mask == 1]
        mean_val = float(np.nanmean(ocean_vals))
        std_val = float(np.nanstd(ocean_vals))
        if std_val == 0:
            std_val = 1.0
            
        normalized = (dp_data - mean_val) / std_val
        normalized[:, ocean_mask == 0] = 0.0
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        Y_normalized[:, d, :, :] = normalized
        subsurface_means.append(mean_val)
        subsurface_stds.append(std_val)

    # 9. Save as PyTorch Tensors
    tensor_path = os.path.join(processed_dir, "spatial_tensors.pt")
    print(f"Saving final tensors to: {tensor_path}")
    
    X_tensor = torch.from_numpy(X_normalized.astype(np.float32))
    Y_tensor = torch.from_numpy(Y_normalized.astype(np.float32))
    mask_tensor = torch.from_numpy(ocean_mask)

    torch.save({
        "X_surface": X_tensor,
        "Y_subsurface": Y_tensor,
        "ocean_mask": mask_tensor,
        "scalers": {
            "surface_means": torch.tensor(surface_means, dtype=torch.float32),
            "surface_stds": torch.tensor(surface_stds, dtype=torch.float32),
            "subsurface_means": torch.tensor(subsurface_means, dtype=torch.float32),
            "subsurface_stds": torch.tensor(subsurface_stds, dtype=torch.float32),
            "depth_levels": torch.tensor(selected_depths, dtype=torch.float32)
        }
    }, tensor_path)

    # Close file handles
    ds_sst.close()
    ds_sss.close()
    ds_ssh.close()
    ds_curr.close()
    ds_winds.close()
    ds_target.close()

    print("\n================================================================")
    print("Verification: Preprocessing Completeness")
    print("================================================================")
    print(f"  * X_surface Shape:        {X_tensor.shape}")
    print(f"  * Y_subsurface Shape:     {Y_tensor.shape}")
    print(f"  * Ocean Mask Shape:       {mask_tensor.shape}")
    print(f"  * Non-empty ocean cells:  {ocean_cells}")
    print(f"  * X_surface range:        Min={X_tensor.min():.4f}, Max={X_tensor.max():.4f}")
    print(f"  * Y_subsurface range:     Min={Y_tensor.min():.4f}, Max={Y_tensor.max():.4f}")
    print("================================================================")
    print("Preprocessing completed successfully!")
    print("================================================================")


if __name__ == "__main__":
    main()
