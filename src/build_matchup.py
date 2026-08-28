"""
build_matchup.py

This script merges the processed Argo profiles dataset with matching surface satellite observations
(SST, SSS, SSH, ocean currents) and ERA5 winds at the nearest geographic coordinate and time.

Output:
- A combined training dataset saved at `data/processed/matchup_table.csv`

Running Standalone:
python -m src.build_matchup
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr


def main():
    # Define paths
    argo_path = "data/raw/argo_profiles.csv"
    sst_path = "data/raw/sst.nc"
    sss_path = "data/raw/sss.nc"
    ssh_path = "data/raw/ssh.nc"
    currents_path = "data/raw/currents.nc"
    winds_path = "data/raw/winds.nc"
    output_dir = "data/processed"
    output_path = os.path.join(output_dir, "matchup_table.csv")

    # Step 1: Check existence of raw input datasets
    for path in [argo_path, sst_path, sss_path, ssh_path, currents_path, winds_path]:
        if not os.path.exists(path):
            print(f"[ERROR] Required input file not found: {path}")
            print("Please ensure that both 'src/fetch_argo.py' and 'src/fetch_satellite.py' have been run successfully.")
            sys.exit(1)

    print("----------------------------------------------------------------")
    print("Starting Dataset Matchup and Feature Generation")
    print("----------------------------------------------------------------")

    # Step 2: Load Argo profiles
    print(f"Loading Argo profiles from {argo_path}...")
    df = pd.read_csv(argo_path)
    initial_rows = len(df)
    print(f"Loaded {initial_rows} Argo profiles.")

    # Step 3: Load NetCDF datasets using xarray
    print("Opening and loading NetCDF datasets into memory (this makes matchup searches virtually instantaneous)...")
    print("  Loading SST...")
    ds_sst = xr.open_dataset(sst_path).load()
    print("  Loading SSS...")
    ds_sss = xr.open_dataset(sss_path).load()
    print("  Loading SSH...")
    ds_ssh = xr.open_dataset(ssh_path).load()
    print("  Loading Currents...")
    ds_currents = xr.open_dataset(currents_path).load()
    print("  Loading Winds...")
    ds_winds = xr.open_dataset(winds_path).load()

    # Step 4: Prepare matchup coordinates
    # To execute a fast, vectorized nearest-neighbor coordinate lookup in xarray,
    # we convert the profile's latitude, longitude, and date series into xarray DataArrays.
    # We specify a shared dimension name ('matchup') to trigger vectorized indexing.
    lats_da = xr.DataArray(df["lat"], dims="matchup")
    lons_da = xr.DataArray(df["lon"], dims="matchup")

    # xarray reads NetCDF time coordinates as timezone-naive datetime64[ns] (interpreted as UTC).
    # We strip any timezone information from the Argo date strings to prevent timezone mismatch errors.
    times_da = xr.DataArray(pd.to_datetime(df["date"]).dt.tz_localize(None), dims="matchup")

    print("\nPerforming nearest-neighbor spatial and temporal joins...")

    # Join SST (thetao)
    print("  Matching Sea Surface Temperature (SST)...")
    sst_matched = ds_sst.sel(latitude=lats_da, longitude=lons_da, time=times_da, method="nearest")
    # Squeeze out the depth coordinate of size 1 and load values into the dataframe
    df["sst"] = sst_matched["thetao"].values.squeeze()

    # Apply a safety check to convert Kelvin to Celsius if required
    # (CMEMS thetao is typically in Celsius, but we check to be robust)
    df["sst"] = np.where(df["sst"] > 200, df["sst"] - 273.15, df["sst"])

    # Join SSS (so)
    print("  Matching Sea Surface Salinity (SSS)...")
    sss_matched = ds_sss.sel(latitude=lats_da, longitude=lons_da, time=times_da, method="nearest")
    df["sss"] = sss_matched["so"].values.squeeze()

    # Join SSH (zos)
    print("  Matching Sea Surface Height (SSH)...")
    ssh_matched = ds_ssh.sel(latitude=lats_da, longitude=lons_da, time=times_da, method="nearest")
    df["ssh"] = ssh_matched["zos"].values.squeeze()

    # Join Currents (uo, vo)
    print("  Matching Ocean Currents (U/V)...")
    currents_matched = ds_currents.sel(latitude=lats_da, longitude=lons_da, time=times_da, method="nearest")
    df["current_u"] = currents_matched["uo"].values.squeeze()
    df["current_v"] = currents_matched["vo"].values.squeeze()

    # Join Winds (u10, v10)
    # Note: winds.nc uses 'valid_time' instead of 'time' as its time dimension coordinate name
    print("  Matching Winds (U/V)...")
    winds_matched = ds_winds.sel(latitude=lats_da, longitude=lons_da, valid_time=times_da, method="nearest")
    df["wind_u"] = winds_matched["u10"].values.squeeze()
    df["wind_v"] = winds_matched["v10"].values.squeeze()

    # Close NetCDF files to free system resources
    ds_sst.close()
    ds_sss.close()
    ds_ssh.close()
    ds_currents.close()
    ds_winds.close()

    # Step 5: Feature Engineering - Day of Year
    # Calculate the Day of Year (1-366) to capture seasonal cycle patterns
    print("Computing day of year feature...")
    df["day_of_year"] = pd.to_datetime(df["date"]).dt.dayofyear

    # Step 6: Define final columns list and order
    # First 10 columns are inputs: lat, lon, day_of_year, sst, sss, ssh, current_u, current_v, wind_u, wind_v
    # Last 8 columns are outputs: temp_0m, temp_10m, temp_30m, temp_50m, temp_75m, temp_100m, temp_200m, temp_500m
    input_cols = ["lat", "lon", "day_of_year", "sst", "sss", "ssh", "current_u", "current_v", "wind_u", "wind_v"]
    output_cols = [f"temp_{depth}m" for depth in [0, 10, 30, 50, 75, 100, 200, 500]]
    all_cols = input_cols + output_cols

    # Reorder and filter columns
    df_matchup = df[all_cols].copy()

    # Step 7: Quality Filtering
    print("\nApplying quality control and physical sanity checks...")
    
    # 1. Drop any row containing NaNs in either input features or output depths
    nan_mask = df_matchup.isna().any(axis=1)
    df_clean = df_matchup[~nan_mask].copy()
    nan_dropped = len(df_matchup) - len(df_clean)
    print(f"  Dropped {nan_dropped} rows containing NaN values (due to cloud gaps or missing satellite swaths).")

    # 2. Check temperature limits (physically implausible values: outside -2 to 40 degrees C)
    temp_cols = ["sst"] + output_cols
    temp_mask = ((df_clean[temp_cols] < -2.0) | (df_clean[temp_cols] > 40.0)).any(axis=1)
    df_clean = df_clean[~temp_mask].copy()
    temp_dropped = np.sum(temp_mask)
    print(f"  Dropped {temp_dropped} rows with physically implausible temperature values (outside -2 to 40°C).")

    # 3. Check salinity limits (physically implausible SSS values: outside 0 to 45 PSU)
    sal_mask = (df_clean["sss"] < 0.0) | (df_clean["sss"] > 45.0)
    df_clean = df_clean[~sal_mask].copy()
    sal_dropped = np.sum(sal_mask)
    print(f"  Dropped {sal_dropped} rows with physically implausible sea surface salinity values (outside 0 to 45 PSU).")

    final_count = len(df_clean)
    total_dropped = initial_rows - final_count

    print(f"\nMatchup process complete:")
    print(f"  Initial profiles:  {initial_rows}")
    print(f"  Dropped rows:      {total_dropped}")
    print(f"  Final matched rows: {final_count} ({(final_count / initial_rows * 100):.1f}% retained)")

    # Step 8: Save matchup table
    os.makedirs(output_dir, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved combined matchup table to: {output_path}")

    # Step 9: Print summary statistics for sanity check
    print("\nMatchup Table Summary Statistics (df.describe()):")
    print("==========================================================================================")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(df_clean.describe())
    print("==========================================================================================")


if __name__ == "__main__":
    main()
