import pandas as pd
from tqdm import tqdm

df = pd.read_csv("combined_csv.csv")
idx = 1

for i, row in tqdm(df.iterrows()):
    row = list(row)
    if i == 0:
        row[0] = 0
        df.loc[i] = row
        prev_row = row
        continue
    if row[1] == 0 and prev_row[1] != 0:
        idx += 1
    row[0] = idx
    df.loc[i] = row
    prev_row = row
df = df.rename(columns={'Unnamed: 0': 'Conv_id'})
df.to_csv("combined_mod.csv")