import pandas as pd
import os

folder_p = r'C:\Users\Jeff\Documents\Trading\Futures\Strategy Info\data'
all_files = os.listdir(folder_p)
txt_files = [file for file in all_files if file.endswith('.txt')]

for f in txt_files:
    df = pd.read_csv(f'{folder_p}\\{f}')

    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df_subset = df[df['Date'] <= pd.to_datetime('04/03/2024', format='%m/%d/%Y')]

    df_subset.to_csv(f'{folder_p}\\{f}', index=False)

