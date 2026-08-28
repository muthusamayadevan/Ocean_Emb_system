"""
train.py

Loads the matchups dataset, standardizes features, splits the data, trains
the OceanEncoderDecoder model, and evaluates metrics per depth level.

Running Standalone:
python -m src.train
"""

import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Import the model architecture defined in model.py
from src.model import OceanEncoderDecoder


def main():
    # File paths
    matchup_path = "data/processed/matchup_table.csv"
    output_dir = "data/processed"
    input_scaler_path = os.path.join(output_dir, "input_scaler.pkl")
    output_scaler_path = os.path.join(output_dir, "output_scaler.pkl")
    model_weights_path = os.path.join(output_dir, "model.pth")

    # Step 1: Check if matchup data exists
    if not os.path.exists(matchup_path):
        print(f"[ERROR] Matchup table not found at: {matchup_path}")
        print("Please run 'python -m src.build_matchup' first to generate the matchups.")
        return

    # Step 2: Load the matchups table
    print("----------------------------------------------------------------")
    print("Starting PyTorch Model Training Pipeline")
    print("----------------------------------------------------------------")
    print(f"Loading matchup table from: {matchup_path}")
    df = pd.read_csv(matchup_path)
    print(f"Dataset shape: {df.shape}")

    # Step 3: Segment into inputs (first 10 columns) and targets (last 8 columns)
    # Inputs: lat, lon, day_of_year, sst, sss, ssh, current_u, current_v, wind_u, wind_v
    # Targets: temp_0m, temp_10m, temp_30m, temp_50m, temp_75m, temp_100m, temp_200m, temp_500m
    input_cols = ["lat", "lon", "day_of_year", "sst", "sss", "ssh", "current_u", "current_v", "wind_u", "wind_v"]
    output_cols = [f"temp_{depth}m" for depth in [0, 10, 30, 50, 75, 100, 200, 500]]

    X = df[input_cols].values
    y = df[output_cols].values

    # Step 4: Split data into training (80%) and testing (20%) sets
    # We use a fixed random state to ensure identical split splits across different runs (reproducibility).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train split size: {X_train.shape[0]} rows")
    print(f"Test split size:  {X_test.shape[0]} rows")

    # Step 5: Standardization (Scaling)
    #
    # WHAT IS STANDARDIZATION?
    # It scales values so they have a mean of 0 and a standard deviation of 1: z = (x - mean) / std.
    # Neural networks perform best when all input features have similar ranges.
    # For example, day_of_year ranges from 1 to 365, while ssh ranges from -1 to +1.
    # Without scaling, features with larger magnitudes would dominate the gradient updates.
    #
    # WHY SAVE THE SCALERS?
    # We must apply the exact same transformation (using the mean and std of the train split) to
    # future inputs (e.g. in the Streamlit UI) before running model inference.
    # Similarly, we need to save the output scaler to inverse-transform the model's scaled
    # outputs back into physical degrees Celsius (°C) for display.
    #
    # PREVENTING INFORMATION LEAKAGE:
    # We fit the scalers ONLY on the train split (X_train, y_train). Fitting on the test split
    # or the entire dataset would leak summary statistics (mean, std) of the test set into training.
    print("\nFitting scalers and standardizing train/test sets...")
    
    input_scaler = StandardScaler()
    output_scaler = StandardScaler()

    # Fit on training data
    input_scaler.fit(X_train)
    output_scaler.fit(y_train)

    # Transform both training and testing datasets
    X_train_scaled = input_scaler.transform(X_train)
    X_test_scaled = input_scaler.transform(X_test)
    y_train_scaled = output_scaler.transform(y_train)
    y_test_scaled = output_scaler.transform(y_test)

    # Save scalers for later deployment use
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(input_scaler, input_scaler_path)
    joblib.dump(output_scaler, output_scaler_path)
    print(f"Saved input scaler to: {input_scaler_path}")
    print(f"Saved output scaler to: {output_scaler_path}")

    # Step 6: Convert to PyTorch Tensors and build DataLoaders
    # We use FloatTensors because models expect float32 values by default.
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    # A small batch size (16) is selected because our dataset size is small (~411 train rows)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Step 7: Instantiate model, loss function, and optimizer
    model = OceanEncoderDecoder(input_dim=10, embedding_dim=24, output_dim=8)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Step 8: Training Loop
    epochs = 200
    print(f"\nTraining model for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)

        # Evaluate on the test set every 20 epochs
        if epoch == 1 or epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_predictions = model(X_test_tensor)
                test_loss = criterion(test_predictions, y_test_tensor).item()
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss (MSE): {train_loss:.4f} | Test Loss (MSE): {test_loss:.4f}")

    # Step 9: Save the trained model weights
    torch.save(model.state_dict(), model_weights_path)
    print(f"\nSuccessfully saved trained model weights to: {model_weights_path}")

    # Step 10: Final Evaluation and Metric Computation
    print("\nEvaluating model performance on the test split...")
    model.eval()
    with torch.no_grad():
        test_preds_scaled = model(X_test_tensor).numpy()

    # Inverse-transform predictions and targets back to physical units (degrees Celsius)
    test_preds = output_scaler.inverse_transform(test_preds_scaled)
    test_targets = y_test  # y_test is already unscaled original values

    # Compute RMSE and R2 score per depth level
    metrics = []
    for i, col in enumerate(output_cols):
        y_true = test_targets[:, i]
        y_pred = test_preds[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        metrics.append({
            "Depth": col.replace("temp_", ""),
            "RMSE (°C)": rmse,
            "R² Score": r2
        })

    # Print out results in a clean table format
    df_metrics = pd.DataFrame(metrics)
    print("\nFinal Test Metrics per Depth Level:")
    print("=========================================")
    print(df_metrics.to_string(index=False))
    print("=========================================")


if __name__ == "__main__":
    main()
