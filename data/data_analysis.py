import pandas as pd

df = pd.read_csv("NPs_BHVO_Oct23_full.csv", index_col=False)

print("Sample: 5\n", df.head())

print(f"\nTotal number of rows: {len(df)}")
print(f"Total number of columns: {df.shape[1]}")

# min and max time stamp
if "Time (ms)" in df.columns:
    print(f"Minimum timestamp (ms): {df['Time (ms)'].min()}")
    print(f"Maximum timestamp (ms): {df['Time (ms)'].max()}")

# Highest val
print("Highest recorded peaks for all ions")
for col in df.columns:
    max_val = df[col].max()
    print(f"{col}: {max_val}")