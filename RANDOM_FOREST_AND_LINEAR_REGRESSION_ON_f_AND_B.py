#RANDOM FOREST AND LINEAR REGRESSION ON f AND B

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("/content/preprocessed_dataset.csv")  # Update path if needed

# Feature/target split
X = df.drop(columns=["f", "B"])
y_f = df["f"]  # Log-transform due to extreme small values
y_B = df["B"]

# Train/test split
X_train, X_test, y_f_train, y_f_test, y_B_train, y_B_test = train_test_split(
    X, y_f, y_B, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

# Evaluate models
for name, model in models.items():
    print(f"\n{name}")

    # Model for f
    model.fit(X_train_scaled, y_f_train)
    y_f_pred_log = model.predict(X_test_scaled)
    y_f_pred = y_f_pred_log  # Inverse log1p
    y_f_true = y_f_test

    print("  [f] Mean Squared Error:", mean_squared_error(y_f_true, y_f_pred))
    print("  [f] R2 Score:", r2_score(y_f_true, y_f_pred))

    # Model for B
    model.fit(X_train_scaled, y_B_train)
    y_B_pred = model.predict(X_test_scaled)

    print("  [B] Mean Squared Error:", mean_squared_error(y_B_test, y_B_pred))
    print("  [B] R2 Score:", r2_score(y_B_test, y_B_pred))
