import pandas as pd

df = pd.read_csv('wandb_export.csv')
df_stored = df.iloc[19::20]
best = df_stored['SugaiNetArchitecture_ema999 - simple'].idxmin()
print(best+1)

