# OceanEmbed

A subsurface ocean temperature prediction prototype (SIH 2026 Hackathon, 4-day timeline).

## Project Goal
The goal of this project is to predict ocean temperature at 8 depth levels from surface satellite data (SST, SSH, SSS, ocean currents) and ERA5 winds using an embedding-based neural network (encoder-decoder).

## Region of Interest
- **North Indian Ocean**
- **Latitude**: 5°N to 30°N
- **Longitude**: 45°E to 105°E

## Folder Structure
```
oceanembed/
├── data/
│   ├── raw/          # Downloaded satellite + argo files
│   └── processed/    # Cleaned matchup spreadsheet
├── src/
│   ├── __init__.py
│   ├── fetch_argo.py       # Download Argo float profiles (temp at depth, lat, lon, date)
│   ├── fetch_satellite.py  # Download CMEMS SST/SSH/SSS/currents + ERA5 winds
│   ├── build_matchup.py    # Match each Argo reading to satellite data at same place/date
│   ├── model.py            # Encoder-decoder neural network (PyTorch)
│   ├── train.py            # Training script
│   └── evaluate.py         # RMSE, correlation, bias metrics + plots
├── app/
│   └── streamlit_app.py    # Demo web app
├── notebooks/
│   └── exploration.ipynb   # For quick data checks
├── requirements.txt
├── README.md
└── .gitignore
```

## Quickstart Guide for Team Collaboration

To clone the repository and set up the development environment, follow these steps:

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd oceanembed
```

### 2. Environment Setup
Create a Python virtual environment and install the required dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your CMEMS and Copernicus/CDS credentials:
```bash
# On macOS/Linux:
cp .env.example .env

# On Windows PowerShell:
Copy-Item .env.example .env
```

### 4. Fetch the Data
Run the scripts to fetch the required satellite and Argo float profiles datasets:
```bash
# Fetch Satellite (SST/SSS/SSH/Currents/Winds) data
python -m src.fetch_satellite

# Fetch in-situ Argo float profiles data
python -m src.fetch_argo
```

### 5. Run the Matchup and Modeling Pipeline
Match observations, train the model, and evaluate performance:
```bash
# Build matchup table
python -m src.build_matchup

# Train the model
python -m src.train

# Evaluate model performance
python -m src.evaluate
```

### 6. Run the NetCDF Inspection and Visualization Script
Inspect NetCDF metadata, crop to the North Indian Ocean region, export CSV samples, and generate individual/combined geospatial plots:
```bash
python -m src.inspect_and_visualize_nc
```
Generated reports are stored at `reports/visualizations/` and processed tables at `data/processed/`.

### 7. Launch the Streamlit Demo Dashboard
Launch the interactive dashboard to visualize profiles and predictions:
```bash
streamlit run app/streamlit_app.py
```

