"""
fetch_argo.py

This script downloads Argo float profiles (temperature at depth, latitude,
longitude, and date) for the North Indian Ocean region (5°N-30°N, 45°E-105°E).
It uses the `argopy` library to fetch the profiles, performs quality filtering,
interpolates temperature values along the vertical depth axis at standard depths,
and saves the resulting cleaned data in the `data/raw/` directory.

Target Standard Depths:
[0, 10, 30, 50, 75, 100, 200, 500] meters

Output Columns:
lat, lon, date, temp_0m, temp_10m, temp_30m, temp_50m, temp_75m, temp_100m, temp_200m, temp_500m

Running Standalone:
python -m src.fetch_argo
"""

import os
import ssl
import sys
import traceback
import numpy as np
import pandas as pd

# We use the 'truststore' library to inject native OS certificates into Python's SSL context.
# This prevents verification failures (e.g., CERTIFICATE_VERIFY_FAILED) that can happen
# on Windows/macOS environments when standard Python certificates are outdated.
try:
    import truststore
    truststore.inject_into_ssl()
    print("Successfully injected truststore into SSL context.")
except ImportError:
    print("Warning: truststore library not found. Proceeding with standard SSL context.")

# We import argopy and its DataFetcher. argopy is a library developed to make it
# easy to fetch and manipulate Argo float data from global repositories.
try:
    import argopy
    from argopy import DataFetcher as ArgoDataFetcher
except ImportError as e:
    print(f"Error: argopy is not installed. Please install requirements.txt first. Details: {e}")
    sys.exit(1)

# Configurable constants at the top of the file to define bounds and outputs
# Bounding box for North Indian Ocean
LAT_MIN = 5.0
LAT_MAX = 30.0
LON_MIN = 45.0
LON_MAX = 105.0

# Vertical pressure range in decibars (approximately equivalent to meters in depth).
# Standard Argo profiles go down to 2000m.
PRES_MIN = 0
PRES_MAX = 2000

# Date range for fetching profiles
# Configured to cover roughly the last 3-5 years (from 2023-01-01 to 2026-08-27)
START_DATE = "2023-01-01"
END_DATE = "2026-08-27"

# Target standard depths for interpolation
TARGET_DEPTHS = [0, 10, 30, 50, 75, 100, 200, 500]

# Output path for the generated CSV
OUTPUT_PATH = "data/raw/argo_profiles.csv"


def fetch_and_interpolate_argo():
    """
    Fetches raw Argo profiles, filters them by depth coverage, interpolates 
    temperature to standard depths, and saves the cleaned dataset.
    """
    print("----------------------------------------------------------------")
    print(f"Starting Argo Float Profile Fetching & Processing")
    print(f"Spatial bounds: Lon [{LON_MIN}, {LON_MAX}], Lat [{LAT_MIN}, {LAT_MAX}]")
    print(f"Temporal bounds: {START_DATE} to {END_DATE}")
    print(f"Pressure bounds: {PRES_MIN} to {PRES_MAX} dbar")
    print(f"Target depths: {TARGET_DEPTHS} m")
    print("----------------------------------------------------------------")

    # Step 1: Define bounding box query parameters
    # The list format expected by argopy DataFetcher is:
    # [lon_min, lon_max, lat_min, lat_max, pres_min, pres_max, date_min, date_max]
    box = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, PRES_MIN, PRES_MAX, START_DATE, END_DATE]

    print("Initializing DataFetcher and fetching profiles from ERDDAP...")
    
    try:
        # We initialize the default DataFetcher (which defaults to using the ERDDAP server)
        # and query the defined region.
        argo_loader = ArgoDataFetcher()
        query = argo_loader.region(box)
        
        # Load the data and convert it into an xarray Dataset.
        # to_xarray() downloads the data and returns a point-wise 1D representation.
        print("Downloading data (this might take a few moments)...")
        ds_raw = query.to_xarray()
        
        # Check if any data was returned
        if ds_raw is None or len(ds_raw.dims) == 0:
            print("No profiles found matching the query criteria.")
            return
            
    except Exception as e:
        print("\n[ERROR] Failed to fetch data from argopy ERDDAP server.")
        print("Argopy data fetchers can be slow or flaky due to network timeouts or server load.")
        print(f"Details of exception:\n{traceback.format_exc()}")
        return

    # Step 2: Restructure the data from a 1D point collection to 2D vertical profiles
    # point2profile() organizes the dataset dimensions by (N_PROF, N_LEVELS)
    # where N_PROF is the number of unique profiles and N_LEVELS is the number of measurement depths.
    print("Restructuring 1D point measurements into 2D vertical profiles...")
    try:
        ds_profiles = ds_raw.argo.point2profile()
    except Exception as e:
        print(f"[ERROR] Failed to restructure points to profiles: {e}")
        return

    num_raw_profiles = int(ds_profiles.sizes.get("N_PROF", 0))
    print(f"Fetched {num_raw_profiles} unique profiles.")

    if num_raw_profiles == 0:
        print("No profiles available for processing.")
        return

    # Extract coordinates and measurements
    # We load them to numpy arrays for fast processing
    lats = ds_profiles.LATITUDE.values
    lons = ds_profiles.LONGITUDE.values
    dates = ds_profiles.TIME.values
    pressures = ds_profiles.PRES.values  # Shape: (N_PROF, N_LEVELS)
    temperatures = ds_profiles.TEMP.values  # Shape: (N_PROF, N_LEVELS)

    processed_rows = []
    skipped_count = 0

    print("Iterating over profiles, applying quality filters, and interpolating temperature...")

    # Step 3: Iterate through each profile to check depth coverage and interpolate
    for i in range(num_raw_profiles):
        prof_pres = pressures[i, :]
        prof_temp = temperatures[i, :]

        # Find levels that have valid (non-NaN) data for both pressure and temperature
        valid_mask = (~np.isnan(prof_pres)) & (~np.isnan(prof_temp))
        valid_pres = prof_pres[valid_mask]
        valid_temp = prof_temp[valid_mask]

        if len(valid_pres) == 0:
            skipped_count += 1
            continue

        # Sort the valid measurements by pressure to ensure monotonic order for interpolation
        sort_indices = np.argsort(valid_pres)
        valid_pres = valid_pres[sort_indices]
        valid_temp = valid_temp[sort_indices]

        # Filter: Skip/drop profiles that do not have data covering at least down to 500m
        # (This ensures we do not perform wild extrapolation for the deep 500m target depth).
        max_pressure = np.max(valid_pres)
        if max_pressure < 500.0:
            skipped_count += 1
            continue

        # Step 4: Perform linear interpolation along the depth (pressure) axis
        # np.interp(x, xp, fp) performs 1D linear interpolation:
        # x: target standard depths
        # xp: observed pressures (must be increasing)
        # fp: observed temperatures
        # Note: If target depth 0m is shallower than the first measured depth,
        # np.interp repeats the shallowest value to the surface, which is standard.
        try:
            interpolated_temps = np.interp(TARGET_DEPTHS, valid_pres, valid_temp)
        except Exception as e:
            print(f"Warning: Failed to interpolate profile {i}: {e}. Skipping.")
            skipped_count += 1
            continue

        # Keep track of date format. Convert numpy datetime64 to standard string.
        # This makes it readable and clean when saved to CSV.
        date_str = pd.to_datetime(dates[i]).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Compile the profile data row
        row = {
            "lat": lats[i],
            "lon": lons[i],
            "date": date_str
        }
        # Add the interpolated temperature values with clear column names
        for depth, temp in zip(TARGET_DEPTHS, interpolated_temps):
            row[f"temp_{depth}m"] = temp

        processed_rows.append(row)

    # Step 5: Output progress and save the DataFrame to data/raw/argo_profiles.csv
    num_kept = len(processed_rows)
    print(f"Processing complete.")
    print(f"Total profiles processed: {num_raw_profiles}")
    print(f"Profiles kept: {num_kept}")
    print(f"Profiles skipped (did not reach 500m or no valid data): {skipped_count}")

    if num_kept == 0:
        print("No profiles survived the filtering. CSV will not be saved.")
        return

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(processed_rows)

    # Reorder columns to match the exact requirement:
    # lat, lon, date, temp_0m, temp_10m, temp_30m, temp_50m, temp_75m, temp_100m, temp_200m, temp_500m
    expected_columns = ["lat", "lon", "date"] + [f"temp_{d}m" for d in TARGET_DEPTHS]
    df = df[expected_columns]

    # Create target directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save DataFrame
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully saved clean dataset of {num_kept} profiles to: {OUTPUT_PATH}")


def main():
    fetch_and_interpolate_argo()


if __name__ == "__main__":
    main()
