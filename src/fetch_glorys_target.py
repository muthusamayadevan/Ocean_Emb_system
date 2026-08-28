"""
src/fetch_glorys_target.py

Downloads the 3D Subsurface Ocean Temperature Ground Truth (GLORYS12V1)
via Copernicus Marine (CMEMS) for the North Indian Ocean.

Target Bounding Box: 5°N–30°N, 45°E–105°E
Time Range: 2024-01-01 to 2024-02-28
Depth Range: 0 to 1000 meters

Running Standalone:
python -m src.fetch_glorys_target
"""

import os
import sys
import traceback
import xarray as xr
import pandas as pd
from dotenv import load_dotenv
import copernicusmarine

# Load environment variables
load_dotenv()


def main():
    # Credentials setup
    username = os.getenv("CMEMS_USERNAME")
    password = os.getenv("CMEMS_PASSWORD")

    if not username or not password:
        print("[ERROR] CMEMS_USERNAME or CMEMS_PASSWORD not set in .env file.")
        sys.exit(1)

    print("================================================================")
    print("CMEMS 3D Subsurface Temperature Downloader (GLORYS Target)")
    print("================================================================")

    # Dataset configurations
    dataset_id = "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m"
    variables = ["thetao"]
    output_dir = "data/raw"
    output_filename = "glorys_subsurface_temp.nc"
    output_path = os.path.join(output_dir, output_filename)

    # Subsetting parameters
    lon_min, lon_max = 45.0, 105.0
    lat_min, lat_max = 5.0, 30.0
    start_dt = "2024-01-01T00:00:00"
    end_dt = "2024-02-28T23:59:59"
    min_depth = 0.0
    max_depth = 1000.0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Skip download if file already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Target file '{output_filename}' already exists at '{output_path}'. Skipping download.")
    else:
        print("Logging in to Copernicus Marine Service...")
        try:
            is_valid = copernicusmarine.login(
                username=username,
                password=password,
                check_credentials_valid=True
            )
            if not is_valid:
                print("[WARNING] CMEMS login check returned invalid credentials. Subset call might fail.")
            else:
                print("CMEMS Login successful and verified.")
        except Exception as e:
            print(f"[WARNING] Login execution raised an exception: {e}. Proceeding with subset...")

        print(f"Initiating CMEMS subset download for: {dataset_id}")
        print(f"  Variables:        {variables}")
        print(f"  Time Window:      {start_dt} to {end_dt}")
        print(f"  Bounding Box:     Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")
        print(f"  Depth Range:      {min_depth}m to {max_depth}m")
        print(f"  Output Path:      {output_path}")
        print("  Downloading... (This might take a few minutes as it pulls 3D data volumes)")

        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=lon_min,
                maximum_longitude=lon_max,
                minimum_latitude=lat_min,
                maximum_latitude=lat_max,
                start_datetime=start_dt,
                end_datetime=end_dt,
                minimum_depth=min_depth,
                maximum_depth=max_depth,
                username=username,
                password=password,
                output_filename=output_filename,
                output_directory=output_dir,
                force_download=True
            )
            print(f"Successfully downloaded GLORYS 3D subset to: {output_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download GLORYS target dataset: {e}")
            print(traceback.format_exc())
            sys.exit(1)

    # Verification phase
    print("\n================================================================")
    print("Verification: Inspecting Downloaded NetCDF File")
    print("================================================================")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print(f"[ERROR] Downloaded file does not exist or is empty at: {output_path}")
        sys.exit(1)

    try:
        ds = xr.open_dataset(output_path)
        
        # Dimensions & Coordinates Check
        print("File coordinates and dimensions loaded successfully:")
        for dim, size in ds.sizes.items():
            print(f"  * Dimension: {dim} = {size}")
            
        print("\nVariable properties:")
        for var_name in ds.data_vars:
            var = ds[var_name]
            print(f"  * {var_name}: {var.attrs.get('long_name', 'N/A')}")
            print(f"    Shape: {var.shape}")
            print(f"    Units: {var.attrs.get('units', 'N/A')}")

        # Exact depth coordinates downloaded
        if "depth" in ds.coords:
            depths = ds.coords["depth"].values
            print(f"\nExact downloaded vertical depth levels ({len(depths)} levels):")
            print(depths.tolist())
        else:
            print("\n[WARNING] 'depth' coordinate not found in coordinates.")

        ds.close()
        print("\nVerification complete! Target dataset is ready for 2D-to-3D training.")
        print("================================================================")
    except Exception as e:
        print(f"[ERROR] Failed to read or verify downloaded NetCDF dataset: {e}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
