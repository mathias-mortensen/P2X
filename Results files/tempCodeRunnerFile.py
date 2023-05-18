import pandas as pd

# Load first Excel file
df1 = pd.read_excel("V1_year_2020-01-01_2020-12-31_k.xlsx")

# Load second Excel file
df2 = pd.read_excel("Version_1_2020-01-01_2020-12-31.xlsx")

# Get list of column names from both dataframes
columns = set(df1.columns.tolist() + df2.columns.tolist())

# Iterate over column names
for col in columns:
    # Check if the column exists in both dataframes
    if col in df1.columns and col in df2.columns:
        # Get the unique values for each dataframe
        unique_values_1 = df1[col].unique()
        unique_values_2 = df2[col].unique()

        # Check if the unique values for each dataframe are similar
        if len(set(unique_values_1).intersection(unique_values_2)) > 0:
            print(f"The values in column {col} are similar.")
        else:
            print(f"The values in column {col} are different.")




