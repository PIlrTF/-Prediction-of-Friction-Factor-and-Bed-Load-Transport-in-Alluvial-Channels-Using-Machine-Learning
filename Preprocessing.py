#PREPROCESING


# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv('/content/dataset_with_f_B.csv')

# 2. Quick inspection
print(df.head())
print(df.info())
print(df.describe())

# 3. Handle missing values
# Convert numeric-looking columns to actual numbers
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with mean (can adjust strategy if needed)
df = df.fillna(df.mean())

# 4. Select only relevant numeric columns
numeric_cols=[]
for col in df.columns:
    numeric_cols.append(col)
df = df[numeric_cols]

# 5. Outlier detection and handling
# Visualize outliers
for col in numeric_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Cap outliers at 1st and 99th percentiles
for col in numeric_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# 6. Feature scaling (optional but recommended for ML)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 7. Final checks
print(df_scaled.head())
print(df_scaled.describe())

# Save preprocessed dataset if needed
df_scaled.to_csv('preprocessed_dataset.csv', index=False)
