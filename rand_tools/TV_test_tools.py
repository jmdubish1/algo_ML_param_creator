import pandas as pd
import glob

main_folder = \
    r'C:\Users\Jeff\Documents\Trading\Futures\Strategy Info\Double Candles\NQ\15min\15min_test_8years\k_means_model_69_steps_OHLB_tanh_4-6LB\Results'
side = "Bull"

file_paths = glob.glob(f'{main_folder}\\*{side}.csv')

dfs = [pd.read_csv(file) for file in file_paths]

concatenated_df = pd.concat(dfs, ignore_index=True)
concatenated_df.dropna(subset=['Trade #'], inplace=True)

filtered_df = concatenated_df[~concatenated_df['Type'].str.contains('Entry')]
final_df = filtered_df.sort_values(by='Date/Time')
final_df = final_df.drop_duplicates(subset=['Date/Time'])
final_df.to_csv(f'{main_folder}\\{side}_algo_trades.csv', index=False)

