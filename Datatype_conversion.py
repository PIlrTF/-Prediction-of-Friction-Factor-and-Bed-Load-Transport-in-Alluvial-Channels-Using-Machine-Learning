#CONVERTING ALL COLUMNS TO FLOAT
import pandas as pd

# Read the uploaded file
file_path = "/content/dataset_with_f_B(7kdataset).csv"
df = pd.read_csv(file_path, encoding='latin-1')  # Specify the encoding

# Convert all columns to float, forcing conversion
# Non-convertible values will become NaN
df_float = df.apply(pd.to_numeric, errors='coerce')

# Save the converted DataFrame
output_file = "Dataset_Arbitrary_2_float.csv"
df_float.to_csv(output_file, index=False)

print(f"File converted successfully! Saved as: {output_file}")
