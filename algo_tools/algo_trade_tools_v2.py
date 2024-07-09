import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import deque


def get_num_iterations(values):
    num_iters = 1
    for lst in values:
        num_iters *= len(lst)
    return num_iters


def get_attr(df):
    attrs = df.__dict__.copy()
    exclude = ["names", "row.names", "class"]

    # Remove unwanted attributes
    attrs = {k: v for k, v in attrs.items() if k not in exclude}

    attr_list = []
    for k, v in attrs.items():
        attr_list.append(f"{k}_{v}")

    return attr_list


def transfer_attr(df_output, df_donor):
    for attr_name, attr_value in df_donor.attrs.items():
        df_output.attrs[attr_name] = attr_value


def get_name(attr_list):
    return "_".join(attr_list)


def get_pnl(df):
    df.loc[:, 'PnL'] = df.apply(lambda row: row['EntryPrice'] - row['ExitPrice'] if row['bearTrade']
                                else row['ExitPrice'] - row['EntryPrice'] if row['bullTrade'] else None, axis=1)

    return df


def filter_trades(df):
    df = df[(df['bearTrade']) | (df['bullTrade'])]
    df.loc[:, 'side'] = df.apply(lambda row: 'Bear' if row['bearTrade']
                                 else 'Bull' if row['bullTrade'] else None, axis=1)

    df.loc[:, 'side'] = df.apply(lambda row: 'Bear' if row['bearTrade']
                                 else ('Bull' if row['bullTrade'] else None), axis=1)

    return df


def bear_find_take_profit(wrkng_df, bear_tp_percent):
    for i in wrkng_df.index[wrkng_df['bearTrade']]:
        if not pd.isnull(wrkng_df.at[i, 'ExitInd']):
            tp_price = bear_tp_percent * wrkng_df.at[i, 'EntryPrice']
            if min(wrkng_df.loc[i:i+wrkng_df.at[i, 'ExitInd'], 'Low']) <= tp_price:
                wrkng_df.at[i, 'ExitPrice'] = tp_price
    return wrkng_df


def bull_find_take_profit(wrkng_df, bull_tp_percent):
    for i in wrkng_df.index[wrkng_df['bullTrade']]:
        if not pd.isnull(wrkng_df.at[i, 'ExitInd']):
            tp_price = bull_tp_percent * wrkng_df.at[i, 'EntryPrice']
            if max(wrkng_df.loc[i:i+wrkng_df.at[i, 'ExitInd'], 'High']) >= tp_price:
                wrkng_df.at[i, 'ExitPrice'] = tp_price
    return wrkng_df


def update_daybeartrade(row):
    if row['EntryPrice'] > row['dayEma']:
        return False
    else:
        return row['bearTrade']


def update_daybulltrade(row):
    if row['EntryPrice'] < row['dayEma']:
        return False
    else:
        return row['bullTrade']


def daily_ema_trades(wrking_df, daily_ema_df, ema_len):
    wrking_df2 = pd.merge(wrking_df, daily_ema_df[['Date', f'EMA_{ema_len}']], on='Date')
    transfer_attr(wrking_df2, wrking_df)
    wrking_df2.rename(columns={wrking_df2.columns[-1]: 'dayEma'}, inplace=True)
    wrking_df2['bearTrade'] = wrking_df2.apply(lambda row: update_daybeartrade(row), axis=1)
    wrking_df2['bullTrade'] = wrking_df2.apply(lambda row: update_daybulltrade(row), axis=1)

    return wrking_df2


def save_feather(file_loc, df, side):
    file_name = get_name(get_attr(df))
    file_name = f"{side}_{file_name}"
    df.to_feather(f"{file_loc}{file_name}.feather")


def max_drawdown(pnl):
    cumulative_max = np.maximum.accumulate(pnl)
    drawdown = cumulative_max - pnl
    max_draw = np.max(drawdown)

    return -max_draw


def analyze_params(pnl_df):
    month_df = pnl_df.dropna(subset=['PnL']).groupby(['side', 'year', 'month']).agg(cumPnl=('PnL', 'sum'),
                                                                                    maxDraw=('PnL', max_drawdown),
                                                                                    trades=(
                                                                                    'PnL', 'count')).reset_index()

    for k, v in pnl_df.attrs.items():
        month_df[k] = v

    return month_df


def find_stops_bear(working_df, bear_trade_idx, bear_stop_loss_percent):
    for bidx in bear_trade_idx:
        next_exit = working_df.loc[bidx:, 'bearExit'].idxmax()
        bear_window = working_df.loc[bidx:next_exit]
        sl_exit = \
            bear_window.iloc[0, bear_window.columns.get_loc('Close')] * bear_stop_loss_percent
        bear_sl_hit = bear_window['High'] >= sl_exit
        bear_window.loc[(~bear_window['bearExit']) & bear_sl_hit, 'bearExit'] = True
        next_exit = working_df.loc[bidx:, 'bearExit'].idxmin()
        working_df.loc[bidx, 'ExitPrice'] = working_df.loc[next_exit, 'Close']
        working_df.loc[bidx, 'EntryPrice'] = working_df.iloc[bidx, 'Close']
        working_df.loc[bidx, 'ExitInd'] = next_exit

    return working_df


def find_stops_bull(working_df, bull_trade_idx, bull_stop_loss_percent):
    for bidx in bull_trade_idx:
        next_exit = working_df.loc[bidx:, 'bullExit'].idxmax()
        bull_window = working_df.loc[bidx:next_exit]
        sl_exit = \
            bull_window.iloc[0, bull_window.columns.get_loc('Close')] * bull_stop_loss_percent
        bull_sl_hit = bull_window['Low'] <= sl_exit
        bull_window.loc[(~bull_window['bullExit']) & bull_sl_hit, 'bullExit'] = True
        next_exit = working_df.loc[bidx:, 'bullExit'].idxmin()
        working_df.loc[bidx, 'ExitPrice'] = working_df.loc[next_exit, 'Close']
        working_df.loc[bidx, 'EntryPrice'] = working_df.loc[bidx, 'Close']
        working_df.loc[bidx, 'ExitInd'] = next_exit

    return working_df


def ema_slide_engine(t, lookback, df_working, bull_emas_trend_list, bear_emas_trend_list):
    bull_box_list = []
    bear_box_list = []

    if not np.any(bull_box_list[(t - lookback):t]) and not np.any(bear_box_list[(t - lookback):t]):
        box_window = df_working.iloc[(t - lookback):t + 1]
        last_ind = box_window.index[-1]

        top_of_box = box_window.loc[last_ind, 'topOfBull']
        bot_of_box = box_window.loc[last_ind, 'fastEma']

        bull_box_tight = (top_of_box >= box_window.loc[:, 'Open']) & \
                         (bot_of_box <= box_window.loc[:, 'Open']) & \
                         (top_of_box >= box_window.loc[:, 'Close']) & \
                         (bot_of_box <= box_window.loc[:, 'Close'])
        bull_emas_trend = all(bull_emas_trend_list[(t - lookback):t])
        bull_box_list.append(all(bull_box_tight) and bull_emas_trend)

        bot_of_box = df_working.loc[last_ind, 'botOfBear']
        top_of_box = df_working.loc[last_ind, 'fastEma']

        bear_box_tight = (top_of_box >= box_window.loc[:, 'Open']) & \
                         (bot_of_box <= box_window.loc[:, 'Open']) & \
                         (top_of_box >= box_window.loc[:, 'Close']) & \
                         (bot_of_box <= box_window.loc[:, 'Close'])
        bear_emas_trend = all(bear_emas_trend_list[(t - lookback):t])
        bear_box_list.append(all(bear_box_tight) and bear_emas_trend)
    else:
        bull_box_list.append(False)
        bear_box_list.append(False)

    return bull_box_list, bear_box_list


def period_save(monthly_list, file_output, security, timeframe, algo_name, combo):
    print("Saving Model")
    monthly_df = pd.concat(monthly_list, ignore_index=True)
    save_aggregated_data(monthly_df, file_output, security, timeframe, algo_name, combo)
    monthly_list = []

    return monthly_list


def save_aggregated_data(df, save_loc, security, timeframe, algo_name, combo):
    print(f"{security}_{timeframe}_{algo_name}_{combo}")
    save_file = f"{save_loc}\\{security}_{timeframe}_{algo_name}_{combo}.feather"
    df.to_feather(save_file)


def ema_cross_engine(_min_diff, _df_working, _uptrend_dist_list, _dntrend_dist_list):
    '''adjust logic later for entry price slightly beyond the slowEma by switching to High'''

    bull_cross_list = list((_df_working['Open'] < _df_working['fastEma']) & \
                           (_df_working['Close'] > _df_working['slowEma']) & _dntrend_dist_list)
    # bull_cross_list = list(bull_cross_list)
    # bull_cross_list.insert(0, False)
    # bull_cross_list.pop()

    bear_cross_list = list((_df_working['Open'] > _df_working['fastEma']) & \
                           (_df_working['Close'] < _df_working['slowEma']) & _dntrend_dist_list)
    # bear_cross_list.insert(0, False)
    # bear_cross_list.pop()

    return bull_cross_list, bear_cross_list


def last_n_true(series, n):
    if len(series) < n:
        return False
    return all(series[-n:])


def adj_series_forward(series, n_times):
    lst = series.tolist()
    lst = lst[n_times:]
    lst.extend([0] * n_times)
    lst = np.array(lst).astype(bool)

    return lst


def adj_dblc_series(series):
    lst = series.tolist()
    lst.pop()
    lst.insert(0, np.nan)
    lst = [0.0 if np.isnan(x) else x for x in lst]
    lst = np.array(lst).astype(bool)

    return lst


def shift_append_arr(arr):
    arr = list(arr)
    arr.pop(0)
    arr.append(0)

    return np.array(arr)












