import numpy as np
import pandas as pd
from pathlib import Path

# load CSV (adjust path if needed)
csv_path = Path("car_fuel_efficiency.csv")
if not csv_path.exists():
    raise FileNotFoundError(f"CSV file not found at {csv_path.resolve()}")

df = pd.read_csv(csv_path)

# 1) Select all cars from Asia
asia = df[df["origin"] == "Asia"]

# 2) Select only vehicle_weight and model_year columns
cols = ["vehicle_weight", "model_year"]
if not all(c in asia.columns for c in cols):
    raise KeyError(f"Required columns not present: {cols}")

# 3) Select first 7 values
X = asia[cols].iloc[:7].to_numpy(dtype=float)

if X.shape[0] < 7:
    raise ValueError(f"Need 7 rows but found {X.shape[0]} Asian rows.")

# 4) Compute XTX and invert
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

# 5) Create y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200], dtype=float)

# 6) Compute w = (XTX_inv) @ X.T @ y
w = XTX_inv @ X.T @ y

# 7) Sum the elements of w
sum_w = w.sum()

# Print results and match to the choices
choices = {
    "0.051": 0.051,
    "0.51" : 0.51,
    "5.1"  : 5.1,
    "51"   : 51.0
}

print("w vector:", w)
print("Sum of w elements:", sum_w)
print("Rounded (3 decimals):", round(sum_w, 3))

# Find which choice (if any) matches within a small tolerance
matched = None
tol = 1e-3
for label, val in choices.items():
    if abs(sum_w - val) <= tol:
        matched = label
        break

if matched:
    print(f"\nAnswer matches choice: {matched}")
else:
    # If exact match not found, show closest
    closest = min(choices.items(), key=lambda kv: abs(sum_w - kv[1]))
    print(f"\nNo exact match within tol={tol}. Closest choice: {closest[0]} (value {closest[1]})")

