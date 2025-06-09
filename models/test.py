import pandas as pd

df_d = pd.read_csv("detected_bursts.csv")
print(df_d.head(10))

df = pd.read_csv("burst_clusters.csv")
print(df.head(10))