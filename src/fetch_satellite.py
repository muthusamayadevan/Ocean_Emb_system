"""
fetch_satellite.py

This script downloads matching surface satellite observations (SST, SSS, SSH, currents)
and ERA5 wind components (U/V) for the region and time period derived from the
Argo profiles stored in `data/raw/argo_profiles.csv`.

Datasets:
- SST (Sea Surface Temperature): CMEMS Daily Physical Analysis (thetao)
- SSS (Sea Surface Salinity): CMEMS Daily Physical Analysis (so)
- SSH (Sea Surface Height): CMEMS Daily Physical Analysis (zos)
- Currents (Ocean currents U/V): CMEMS Daily Physical Analysis (uo, vo)
- Winds (Wind U/V): CDS ERA5 Single Levels Reanalysis (u10, v10)

Running Standalone:
python -m src.fetch_satellite
"""

import os
import ssl
import sys
import traceback
import pandas as pd
from datetime import datetime

# Inject truststore to use local OS certificates, preventing SSL errors on Windows/macOS.
try:
    import truststore
    truststore.inject_into_ssl()
    print("Successfully injected truststore into SSL context.")
except ImportError:
    print("Warning: truststore library not found. Proceeding with standard SSL context.")

# Load environment variables from .env using python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv is not installed. Will read environment variables directly.")

# Ensure Copernicus Marine and CDS API client libraries are available
try:
    import copernicusmarine
except ImportError:
    print("Error: copernicusmarine is not installed. Please run: pip install copernicusmarine")
    sys.exit(1)

try:
    import cdsapi
except ImportError:
    print("Error: cdsapi is not installed. Please run: pip install cdsapi")
    sys.exit(1)


def configure_cds_api():
    """
    Reads the CDS_API_KEY from the environment and writes the .cdsapirc configuration
    file to the user's home directory so cdsapi client can authenticate.
    """
    cds_key = os.getenv("CDS_API_KEY")
    if not cds_key:
        print("[WARNING] CDS_API_KEY is not defined in the environment or .env file.")
        print("CDS API calls to fetch winds will fail unless a valid ~/.cdsapirc file already exists.")
        return

    home_dir = os.path.expanduser("~")
    rc_path = os.path.join(home_dir, ".cdsapirc")
    
    print(f"Writing CDS API credentials to: {rc_path}")
    try:
        with open(rc_path, "w") as f:
            f.write("url: https://cds.climate.copernicus.eu/api\n")
            f.write(f"key: {cds_key}\n")
        print("Successfully configured .cdsapirc")
    except Exception as e:
        print(f"[ERROR] Failed to write .cdsapirc file to home directory: {e}")


def derive_bounds_from_argo():
    """
    Reads the downloaded Argo profiles CSV to extract spatial and temporal limits.
    Returns:
        tuple: (lon_min, lon_max, lat_min, lat_max, start_date_str, end_date_str)
    """
    argo_path = "data/raw/argo_profiles.csv"
    if not os.path.exists(argo_path):
        print(f"[ERROR] Argo profiles file not found at: {argo_path}")
        print("Please run 'python -m src.fetch_argo' first to download the Argo data.")
        sys.exit(1)

    print(f"Reading spatial and temporal limits from: {argo_path}")
    df = pd.read_csv(argo_path)
    
    if len(df) == 0:
        print("[ERROR] Argo profiles CSV is empty. Cannot derive bounds.")
        sys.exit(1)

    # Spatial bounding box
    raw_lon_min = df["lon"].min()
    raw_lon_max = df["lon"].max()
    raw_lat_min = df["lat"].min()
    raw_lat_max = df["lat"].max()

    # Apply a 0.5-degree spatial buffer to ensure spatial interpolation matches correctly
    lon_min = max(45.0, raw_lon_min - 0.5)
    lon_max = min(105.0, raw_lon_max + 0.5)
    lat_min = max(5.0, raw_lat_min - 0.5)
    lat_max = min(30.0, raw_lat_max + 0.5)

    # Temporal range
    dates = pd.to_datetime(df["date"])
    min_date = dates.min()
    max_date = dates.max()

    # Format dates as YYYY-MM-DD for the download queries
    start_date_str = min_date.strftime("%Y-%m-%d")
    end_date_str = max_date.strftime("%Y-%m-%d")

    print(f"Derived bounds from Argo data:")
    print(f"  Longitude: [{raw_lon_min:.3f}, {raw_lon_max:.3f}] -> Buffered: [{lon_min:.3f}, {lon_max:.3f}]")
    print(f"  Latitude:  [{raw_lat_min:.3f}, {raw_lat_max:.3f}] -> Buffered: [{lat_min:.3f}, {lat_max:.3f}]")
    print(f"  Dates:     [{min_date.strftime('%Y-%m-%d %H:%M')}, {max_date.strftime('%Y-%m-%d %H:%M')}] -> Range: {start_date_str} to {end_date_str}")
    
    return lon_min, lon_max, lat_min, lat_max, start_date_str, end_date_str


def download_cmems_dataset(dataset_id, variables, lon_min, lon_max, lat_min, lat_max, 
                           start_date, end_date, output_filename, username=None, password=None, min_depth=None, max_depth=None):
    """
    Helper function to download a specific dataset from Copernicus Marine (CMEMS)
    using the subset() API.
    """
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"\nTarget file {output_filename} already exists, skipping.")
        return

    # Format start and end date/time strings for the CMEMS query
    start_dt = f"{start_date}T00:00:00"
    end_dt = f"{end_date}T23:59:59"

    print(f"\nSubsetting dataset: {dataset_id}")
    print(f"  Variables: {variables}")
    print(f"  File name: {output_filename}")
    
    try:
        # subset() downloads the subset data as a NetCDF directly
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
        print(f"Successfully downloaded {output_filename}")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset {dataset_id}:")
        print(traceback.format_exc())


def download_era5_winds(lon_min, lon_max, lat_min, lat_max, start_date, end_date):
    """
    Downloads ERA5 wind components U and V from the Copernicus CDS API.
    """
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "winds.nc")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print("\nTarget file winds.nc already exists, skipping.")
        return

    print("\nRequesting ERA5 Wind components from Copernicus CDS...")
    
    # Parse years and months covered by the date range
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Get lists of unique years and months in the range
    years = sorted(list(set([str(y) for y in range(dt_start.year, dt_end.year + 1)])))
    
    # Determine all unique month strings in range
    months = []
    curr = dt_start
    while curr <= dt_end:
        months.append(f"{curr.month:02d}")
        # Move to next month
        if curr.month == 12:
            curr = datetime(curr.year + 1, 1, 1)
        else:
            curr = datetime(curr.year, curr.month + 1, 1)
    months = sorted(list(set(months)))

    # Fetch all days
    days = [f"{d:02d}" for d in range(1, 32)]

    # Bounding box format for CDS API: [North, West, South, East]
    area = [lat_max, lon_min, lat_min, lon_max]

    print(f"  Years: {years}, Months: {months}")
    print(f"  Bounding Area (N/W/S/E): {area}")

    try:
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind"
                ],
                "year": years,
                "month": months,
                "day": days,
                "time": [
                    "00:00", "01:00", "02:00",
                    "03:00", "04:00", "05:00",
                    "06:00", "07:00", "08:00",
                    "09:00", "10:00", "11:00",
                    "12:00", "13:00", "14:00",
                    "15:00", "16:00", "17:00",
                    "18:00", "19:00", "20:00",
                    "21:00", "22:00", "23:00"
                ],
                "area": area,
            },
            output_path
        )
        print(f"Successfully downloaded ERA5 wind data to: {output_path}")
    except Exception as e:
        print("[ERROR] Failed to download ERA5 winds from CDS:")
        print(traceback.format_exc())


def main():
    # Step 1: Configure CDS API credentials
    configure_cds_api()

    # Step 2: Derive spatial and temporal bounds from Argo profiles
    lon_min, lon_max, lat_min, lat_max, start_date, end_date = derive_bounds_from_argo()

    # Step 3: Login to Copernicus Marine
    cmems_username = os.getenv("CMEMS_USERNAME")
    cmems_password = os.getenv("CMEMS_PASSWORD")
    if not cmems_username or not cmems_password:
        print("[WARNING] CMEMS_USERNAME or CMEMS_PASSWORD not defined in .env.")
        print("Copernicus Marine downloads will fail unless you are already logged in via CLI.")
    else:
        print("Logging in to Copernicus Marine Service...")
        try:
            is_valid = copernicusmarine.login(
                username=cmems_username, 
                password=cmems_password,
                check_credentials_valid=True
            )
            if not is_valid:
                print("[WARNING] Invalid CMEMS credentials. Downloads may fail.")
            else:
                print("CMEMS Login successful and verified.")
        except Exception as e:
            print(f"[WARNING] CMEMS login failed: {e}. Proceeding hoping credentials are cached.")

    # Step 4: Download CMEMS datasets
    
    # 1. SST (Sea Surface Temperature)
    # cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m contains daily 3D potential temperature (thetao).
    # Depth level ~0.49m represents Sea Surface Temperature.
    download_cmems_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        start_date=start_date,
        end_date=end_date,
        output_filename="sst.nc",
        username=cmems_username,
        password=cmems_password,
        min_depth=0.49,
        max_depth=0.5
    )

    # 2. SSS (Sea Surface Salinity)
    # cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m contains daily 3D salinity (so).
    # Depth level ~0.49m represents Sea Surface Salinity.
    download_cmems_dataset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        variables=["so"],
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        start_date=start_date,
        end_date=end_date,
        output_filename="sss.nc",
        username=cmems_username,
        password=cmems_password,
        min_depth=0.49,
        max_depth=0.5
    )

    # 3. SSH (Sea Surface Height)
    # cmems_mod_glo_phy_anfc_0.083deg_P1D-m contains daily 2D Sea Surface Height (zos).
    download_cmems_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["zos"],
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        start_date=start_date,
        end_date=end_date,
        output_filename="ssh.nc",
        username=cmems_username,
        password=cmems_password
    )

    # 4. Currents (Sea Water Velocity U/V)
    # cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m contains daily 3D currents (uo, vo).
    # Depth level ~0.49m represents surface currents.
    download_cmems_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        start_date=start_date,
        end_date=end_date,
        output_filename="currents.nc",
        username=cmems_username,
        password=cmems_password,
        min_depth=0.49,
        max_depth=0.5
    )

    # Step 5: Download Winds from CDS API
    download_era5_winds(
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        start_date=start_date,
        end_date=end_date
    )


if __name__ == "__main__":
    main()
