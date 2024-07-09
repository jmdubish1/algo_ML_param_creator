import os
import pandas as pd
from algo_tools import algo_data_tools_v2 as at


def find_feather_files(folder_path):
    """
    Find .feather files within a folder.

    Parameters:
        folder_path (str): Path to the folder.

    Returns:
        List of file paths.
    """
    feather_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".feather"):
            feather_files.append(os.path.join(folder_path, file))

    return feather_files


def concatenate_feather_files(feather_files):
    """
    Concatenate feather files into a single DataFrame.

    Parameters:
        feather_files (list): List of file paths.

    Returns:
        DataFrame containing concatenated data.
    """
    dfs = []
    for file in feather_files:
        print(file)
        df = pd.read_feather(file)
        dfs.append(df)
    concatenated_df = pd.concat(dfs)
    return concatenated_df


def remove_duplicates(df):
    """
    Remove duplicates from DataFrame.

    Parameters:
        df (DataFrame): Input DataFrame.

    Returns:
        DataFrame with duplicates removed.
    """
    df_no_duplicates = df.drop_duplicates()
    return df_no_duplicates


def save_dataframe_chunks_as_feather(df, folder_path, chunk_s, security, timeframe):
    """
    Save DataFrame chunks as .feather files.

    Parameters:
        df (DataFrame): Input DataFrame.
        folder_path (str): Path to the folder where .feather files will be saved.
        chunk_s (int): Size of each chunk.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    num_chunks = len(df) // chunk_s + 1

    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_s: (i + 1) * chunk_s]
        if not chunk.empty:
            file_path = os.path.join(folder_path, f"{security}_{timeframe}_Double_Candles_{i}.feather")
            chunk.reset_index(drop=True).to_feather(file_path)
            print(f"Chunk {i} saved as {file_path}")


if __name__ == "__main__":
    feather_main = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles'
    sec_name = 'CL'
    time_frame = '10min'
    time_len = '10years'
    dbl_candle_cols = ['finalcandleratio', 'fastemalen', 'mincandlepercent', 'finalcandlepercent',
                       'stoplosspercent', 'takeprofitpercent', 'dayslow']
    dbl_ascending = [False, True, False, False, True, False, True]

    ema_cross_cols = ['MinDiffDist', 'MinTickAway', 'StopLossPercent']
    ema_ascending = [False, False, True]
    feather_output = f'{feather_main}\\{sec_name}\\{time_frame}\\{time_frame}_test_{time_len}'
    feather_repo = f'{feather_output}'

    feather_files = find_feather_files(feather_repo)

    concatenated_df = concatenate_feather_files(feather_files).reset_index(drop=True)
    print(concatenated_df.columns)
    print(len(concatenated_df))
    print(concatenated_df['side'].unique())

    concatenated_df.loc[:, 'finalcandleratio'] = concatenated_df.loc[:, 'finalcandleratio'] * 100
    concatenated_df.loc[:, 'mincandlepercent'] = concatenated_df.loc[:, 'mincandlepercent'] * 10000
    concatenated_df.loc[:, 'finalcandlepercent'] = concatenated_df.loc[:, 'finalcandlepercent'] * 10000
    concatenated_df.loc[:, 'stoplosspercent'] = concatenated_df.loc[:, 'stoplosspercent'] * 10000
    concatenated_df.loc[:, 'takeprofitpercent'] = concatenated_df.loc[:, 'takeprofitpercent'] * 10000

    # check_df = concatenated_df.loc[(concatenated_df['lookback'] == 3) &
    #                                (concatenated_df['year'] == 2024)]
    # check_df.to_csv(f'{feather_output}\\check.csv')
    # breakpoint()

    concatenated_df = at.improve_trade_df(concatenated_df, dbl_candle_cols, dbl_ascending, 'lookback')
    # concatenated_df.to_csv(f'{feather_output}\\check.csv')

    """Ensure all months are represented"""
    for side in concatenated_df['side'].unique():
        tempdf = concatenated_df.loc[concatenated_df['side'] == side]
        print(len(tempdf))
        for year in range(2011, 2025):
            if year != 2024:
                for month in range(1, 13):
                    print(f'{year} : {month}')
                    g_count = tempdf.loc[(tempdf['year'] == year) &
                                         (tempdf['month'] == month)]
                    g_len = len(g_count)
                    if g_len == 0:
                        print(f'Zero Count: {side} : {year} : {month} : trades: {g_len}')
            else:
                for month in range(1, 5):
                    print(f'{year} : {month}')
                    g_count = tempdf.loc[(tempdf['year'] == year) &
                                         (tempdf['month'] == month)]
                    g_len = len(g_count)
                    if g_len == 0:
                        print(f'Zero Count: {side} : {year} : {month} : trades: {g_len}')

    print(len(concatenated_df))

    chunk_size = int(len(concatenated_df)/3)

    save_dataframe_chunks_as_feather(concatenated_df, feather_output, chunk_size, sec_name, time_frame)
    concatenated_df.to_csv(f'{feather_output}\\concatenated_df.csv')