import condense_feather as cf
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from algo_tools import algo_data_tools as at
import matplotlib.pyplot as plt


def main():
    feather_main = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles'
    sec_name = 'CL'
    time_frame = '10min'
    time_len = '10years'
    feather_folder = f'{feather_main}\\{sec_name}\\{time_frame}\\{time_frame}_test_{time_len}'
    # breakdown_stat = 'MinDiffDist'
    breakdown_stat = 'lookback'
    dbl_cndl_params = ['lookback', 'finalcandleratio', 'fastemalen', 'mincandlepercent', 'finalcandlepercent',
                       'stoplosspercent', 'takeprofitpercent', 'dayslow']

    max_sl = False
    n_keep = 50
    # n_clusters_range = range(4, 16, 1)
    n_clusters = 16

    feather_files = cf.find_feather_files(feather_folder)
    print(feather_files)
    print(sec_name)

    concatenated_df = cf.concatenate_feather_files(feather_files)

    no_dup_df = concatenated_df.copy()
    print(no_dup_df)
    print(no_dup_df.columns)
    # print(max(no_dup_df['lookback']))

    if max_sl:
        no_dup_df = no_dup_df[no_dup_df['StopLossPercent'] <= 20]

    pnl_stats = no_dup_df.groupby([breakdown_stat, 'side'])['cumPnl'].agg(['mean', 'median', 'std', 'min', 'max'])
    pnl_stats.rename(columns={'mean': 'pnl_mean', 'median': 'pnl_median',
                              'std': 'pnl_std', 'min': 'pnl_min',
                              'max': 'pnl_max'}, inplace=True)

    draw_stats = no_dup_df.groupby([breakdown_stat, 'side'])['maxDraw'].agg(['mean', 'median', 'std', 'min', 'max'])
    draw_stats.rename(columns={'mean': 'draw_mean', 'median': 'draw_median',
                               'std': 'draw_std', 'min': 'draw_min',
                               'max': 'draw_max'}, inplace=True)

    trades_stats = no_dup_df.groupby([breakdown_stat, 'side'])['trades'].agg(['mean', 'median', 'std', 'min', 'max'])
    trades_stats.rename(columns={'mean': 'trade_mean', 'median': 'trade_median',
                                 'std': 'trade_std', 'min': 'trade_min',
                                 'max': 'trade_max'}, inplace=True)

    trades_pnl = no_dup_df.groupby(['trades', 'side'])['cumPnl'].agg(['mean', 'median', 'std', 'min', 'max'])
    trades_pnl.rename(columns={'mean': 'mean_pnl', 'median': 'median_pnl',
                               'std': 'std_pnl', 'min': 'min_pnl',
                               'max': 'max_pnl'}, inplace=True)

    min_candle = no_dup_df.groupby(['mincandlepercent', 'side'])['cumPnl'].agg(['mean', 'median', 'std', 'min', 'max'])
    min_candle.rename(columns={'mean': 'mean_pnl', 'median': 'median_pnl',
                               'std': 'std_pnl', 'min': 'min_pnl',
                               'max': 'max_pnl'}, inplace=True)

    # Very important to use the percentile for the average best cumPnl even if it's like 98%
    max_pnl = no_dup_df.sort_values(by=['year', 'month', 'cumPnl'], ascending=[True, True, False])
    max_pnl = max_pnl.groupby(['side', 'year', 'month']).head(n_keep)

    # Histogram to see where the number of trades falls
    hist_group = no_dup_df.groupby('side')
    hist_group_list = []
    for side, group in hist_group:
        trade_counts = group['trades'].values
        hist, bins = np.histogram(trade_counts, bins=range(0, 35))
        hist_data = {'Bin': bins[:-1], 'Frequency': hist}
        hist_df = pd.DataFrame(hist_data)
        hist_df['Percentile'] = hist_df['Frequency']/np.sum(hist_df['Frequency'])
        hist_df['CDF'] = hist_df['Percentile'].cumsum()
        hist_df['side'] = side
        hist_df = hist_df[['side', 'Bin', 'Frequency', 'Percentile', 'CDF']]
        hist_group_list.append(hist_df)
    hist_df = pd.concat(hist_group_list)

    # Histogram for stop loss
    sl_pnlstats = no_dup_df.groupby('stoplosspercent')['cumPnl'].agg(['mean', 'median', 'std', 'min', 'max'])
    sl_pnlstats.rename(columns={'mean': 'cumPnl_mean', 'median': 'cumPnl_median',
                                'std': 'cumPnl_std', 'min': 'cumPnl_min',
                                'max': 'cumPnl_max'}, inplace=True)

    sl_tradestats = no_dup_df.groupby('stoplosspercent')['trades'].agg(['mean', 'median', 'std', 'min', 'max'])
    sl_tradestats.rename(columns={'mean': 'trades_mean', 'median': 'trades_median',
                                  'std': 'trades_std', 'min': 'trades_min',
                                  'max': 'trades_max'}, inplace=True)

    no_dup_df.fillna(0, inplace=True)
    # at.k_means_param_opt(no_dup_df, n_clusters_range, dbl_cndl_params, "Bear")

    k_means_df = at.create_kmeans_df(no_dup_df, n_clusters, dbl_cndl_params)

    dfs = {'Pnl_stats': pnl_stats, 'Drawdown_stats': draw_stats, 'Trade_stats': trades_stats, 'Trade_hist': hist_df,
           'Trade_pnl': trades_pnl, 'maxPnl_month': max_pnl, 'SL_PnLStats': sl_pnlstats, 'SL_tradeStats': sl_tradestats,
           'Min_Candle_pnl': min_candle, 'K-Means_Result': k_means_df}

    with pd.ExcelWriter(f'{feather_folder}\\param_stats_{sec_name}_kmeans_{n_clusters}_1.xlsx') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name, index=True)

if __name__ == "__main__":
    main()

