#GETTING 'f' AND 'B'

import numpy as np
import pandas as pd

filepath="/content/Dataset_Arbitrary_2_float.csv"
df = pd.read_csv(filepath)
# Constants
g = 9.8  # m/s^2
rho_s = 2650  # kg/m^3
rho_water = 1000  # kg/m^3
tau_c = 0.047  # critical shear stress
df["tau_c"]=0.047
# Convert units
df["D50_M"] = df["D50"] / 1000.0  # Convert mm to meters

# Area of pipe rectagular cs:
df["Area_m2"] = df["depth"]*df["width"]

# Flow velocity: u = Q / A
df["u"] = df["Dischage"] / df["Area_m2"]

# Calculate tau: τ = ρ_water * g * R * slope
df["tau"] = rho_water * g * df["depth"] * df["slope"]

# Calculate tau_1: τ₁ = τ / ((ρ_s - ρ_water) * g * D)
df["tau_1"] = df["tau"] / ((rho_s - rho_water) * g * df["D50_M"])

# Calculate B: B = 8 * (τ₁ - τ_c)^1.5
df["B"] = 8 * np.power(abs((df["tau_1"] - tau_c)), 1.5)

# Calculate f: f = 0.75 * ((g * D_50) / u^2)
df["f"] = 0.75 * ((g * df["D50_M"]) / np.power(df["u"], 2))

# Show the new columns
df[["f", "B"]].head()
# Save the updated DataFrame with 'f' and 'B' columns to a new CSV
output_path = "dataset_with_f_B.csv"
df.to_csv(output_path, index=False)
