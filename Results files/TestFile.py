import pandas as pd

# Load first Excel file
df1 = pd.read_excel("V1_2020-01-01_2020-12-31_k.xlsx")

# Load second Excel file
df2 = pd.read_excel("Version_1_2020-01-01_2020-12-31.xlsx")

# Get list of column names from both dataframes
columns = set(df1.columns.tolist() + df2.columns.tolist())

df1['P_PEM'].iloc[12] - df2['P_PEM'].iloc[12]

df_error = pd.DataFrame(columns=columns,index=range(len(df1))) 


col = 'P_PEM'


df2['beta_aFRR_up'].iloc[2]


for col in columns:
    if col in df1.columns and col in df2.columns:

        for i in range(len(df1)):
            df_error[col].iloc[i] = df1[col].iloc[i] - df2[col].iloc[i] 





# Iterate over column names
for col in columns:
    # Check if the column exists in both dataframes
    if col in df1.columns and col in df2.columns:
        # Compare the values in the column for each row
        for i in range(len(df1)):

            if round(df1[col].iloc[i],1) != round(df2[col].iloc[i],1):
                print(f"The values in column {col} are different in row {i+1}.")
                break
        else:
            print(f"The values in column {col} are identical for each row.")




#save to Excel 
df_error.to_excel("df_error.xlsx")
