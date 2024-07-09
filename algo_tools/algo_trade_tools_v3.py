import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import deque
import time
import timeit
import os
from numba import jit


def get_attr(df):
    attrs = df.__dict__.copy()
    exclude = ["names", "row.names", "class"]

    # Remove unwanted attributes
    attrs = {k: v for k, v in attrs.items() if k not in exclude}

    attr_list = []
    for k, v in attrs.items():
        attr_list.append(f"{k}_{v}")

    return attr_list


def make_exit_conds(df_working, eod_exit):
    df_working['bullExit'] = np.array(df_working['Close'] <= df_working['fastEma']) & \
                             np.array(df_working['Close'].shift(1) > df_working['fastEma'].shift(1))
    df_working.loc[(~df_working['bullExit']) & eod_exit, 'bullExit'] = True

    df_working['bearExit'] = np.array(df_working['Close'] >= df_working['fastEma']) & \
                             np.array(df_working['Close'].shift(1) < df_working['fastEma'].shift(1))
    df_working.loc[(~df_working['bearExit']) & eod_exit, 'bearExit'] = True

    return df_working


def transfer_attr(df_output, df_donor):
    for attr_name, attr_value in df_donor.attrs.items():
        df_output.attrs[attr_name] = attr_value


def add_attrs(df_output, attr_dict):
    for attr_name, attr_value in attr_dict.items():
        df_output.attrs[attr_name] = attr_value


def get_name(attr_list):
    return "_".join(attr_list)


def get_pnl(df):
    df['PnL'] = np.where(df['bearTrade'], df['EntryPrice'] - df['ExitPrice'],
                         np.where(df['bullTrade'], df['ExitPrice'] - df['EntryPrice'], np.nan))

    return df


def filter_trades(df):
    df = df[(df['bearTrade']) | (df['bullTrade'])]
    df['side'] = np.where(df['bearTrade'], 'Bear', np.where(df['bullTrade'], 'Bull', np.nan))

    return df


def bear_find_take_profit(wrkng_df, bear_tp_percent):
    for i in wrkng_df.index[wrkng_df['bearTrade']]:
        if not pd.isnull(wrkng_df.at[i, 'ExitInd']):
            tp_price = bear_tp_percent * wrkng_df.at[i, 'EntryPrice']
            exit_ind = wrkng_df.at[i, 'ExitInd']
            tp_window = wrkng_df.loc[i:exit_ind]
            tp_hit = list(tp_window.index[tp_window['Low'] <= tp_price])
            if len(tp_hit) > 0:
                if min(tp_hit) <= exit_ind:
                    exit_ind = min([wrkng_df.at[i, 'ExitInd']] + tp_hit)
                    wrkng_df.at[i, 'ExitInd'] = exit_ind
                    wrkng_df.at[i, 'ExitPrice'] = tp_price

    return wrkng_df


def bull_find_take_profit(wrkng_df, bull_tp_percent):
    for i in wrkng_df.index[wrkng_df['bullTrade']]:
        if not pd.isnull(wrkng_df.at[i, 'ExitInd']):
            tp_price = bull_tp_percent * wrkng_df.at[i, 'EntryPrice']
            exit_ind = wrkng_df.at[i, 'ExitInd']
            tp_window = wrkng_df.loc[i:exit_ind]
            tp_hit = list(tp_window.index[tp_window['High'] >= tp_price])
            if len(tp_hit) > 0:
                if min(tp_hit) <= exit_ind:
                    exit_ind = min([wrkng_df.at[i, 'ExitInd']] + tp_hit)
                    wrkng_df.at[i, 'ExitInd'] = exit_ind
                    wrkng_df.at[i, 'ExitPrice'] = tp_price

    return wrkng_df


def daily_ema_trades(wrking_df):
    wrking_df.loc[(wrking_df['bearTrade']) & (wrking_df['dayEma'] < wrking_df['EntryPrice']), 'bearTrade'] = False
    wrking_df.loc[(wrking_df['bullTrade']) & (wrking_df['dayEma'] > wrking_df['EntryPrice']), 'bullTrade'] = False

    return wrking_df


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

    win_df = pnl_df[pnl_df['PnL'] > 0].groupby(['side', 'year', 'month']).size().reset_index(name='win_count')
    loss_df = pnl_df[pnl_df['PnL'] <= 0].groupby(['side', 'year', 'month']).size().reset_index(name='loss_count')
    combined_counts = pd.merge(win_df, loss_df, on=['side', 'year', 'month'], how='outer')
    combined_counts = combined_counts.fillna(0)
    month_df = pd.merge(month_df, combined_counts, on=['side', 'year', 'month'], how='left')
    month_df['win_percent'] = month_df['win_count'] / month_df['trades']

    transfer_attr(month_df, pnl_df)
    for k, v in pnl_df.attrs.items():
        month_df[k] = v

    return month_df


def find_stops_bear(working_df, bear_stop_loss_percent):
    max_ind = working_df.index[-1]
    for bidx in working_df.index[working_df['bearTrade']]:
        sl_exit = \
            working_df.at[bidx, 'EntryPrice'] * bear_stop_loss_percent
        next_exit = working_df.loc[bidx+1:bidx+min(bidx+31, max_ind), 'bearExit'].idxmax()
        sl_window = working_df.loc[bidx+1:next_exit]
        sl_hit_list = list(sl_window.index[sl_window['High'] >= sl_exit])
        if len(sl_hit_list) > 0:
            if min(sl_hit_list) <= next_exit:
                working_df.at[bidx, 'ExitInd'] = min(sl_hit_list)
                working_df.at[bidx, 'ExitPrice'] = sl_exit
            else:
                working_df.at[bidx, 'ExitInd'] = next_exit
                working_df.at[bidx, 'ExitPrice'] = working_df.at[next_exit, 'Close']
        else:
            working_df.at[bidx, 'ExitInd'] = next_exit
            working_df.at[bidx, 'ExitPrice'] = working_df.at[next_exit, 'Close']

    return working_df


def find_stops_bull(working_df, bull_stop_loss_percent):
    max_ind = working_df.index[-1]
    for bidx in working_df.index[working_df['bullTrade']]:
        sl_exit = \
            working_df.at[bidx, 'EntryPrice'] * bull_stop_loss_percent
        next_exit = working_df.loc[bidx+1:min(bidx+31, max_ind), 'bullExit'].idxmax()
        sl_window = working_df.loc[bidx+1:next_exit]
        sl_hit_list = list(sl_window.index[sl_window['Low'] <= sl_exit])
        if len(sl_hit_list) > 0:
            if min(sl_hit_list) <= next_exit:
                working_df.at[bidx, 'ExitInd'] = min(sl_hit_list)
                working_df.at[bidx, 'ExitPrice'] = sl_exit
            else:
                working_df.at[bidx, 'ExitInd'] = next_exit
                working_df.at[bidx, 'ExitPrice'] = working_df.at[next_exit, 'Close']
        else:
            working_df.at[bidx, 'ExitInd'] = next_exit
            working_df.at[bidx, 'ExitPrice'] = working_df.at[next_exit, 'Close']

    return working_df


def ema_slide_engine(t, lookback, df_working, bull_emas_trend_list, bear_emas_trend_list):
    bull_box_list = []
    bear_box_list = []

    if not np.any(bull_box_list[(t - lookback):t]) and not np.any(bear_box_list[(t - lookback):t]):
        box_window = df_working.iloc[(t - lookback):t + 1]
        last_ind = box_window.index[-1]

        top_of_box = box_window.at[last_ind, 'topOfBull']
        bot_of_box = box_window.at[last_ind, 'fastEma']

        bull_box_tight = (top_of_box >= box_window.loc[:, 'Open']) & \
                         (bot_of_box <= box_window.loc[:, 'Open']) & \
                         (top_of_box >= box_window.loc[:, 'Close']) & \
                         (bot_of_box <= box_window.loc[:, 'Close'])
        bull_emas_trend = all(bull_emas_trend_list[(t - lookback):t])
        bull_box_list.append(all(bull_box_tight) and bull_emas_trend)

        bot_of_box = df_working.at[last_ind, 'botOfBear']
        top_of_box = df_working.at[last_ind, 'fastEma']

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


def ema_cross_engine(min_diff, df_working, uptrend_dist_list, dntrend_dist_list):
    '''adjust logic later for entry price slightly beyond the slowEma by switching to High'''

    bull_cross_list = list((df_working['Open'] < df_working['fastEma']) & \
                           (df_working['Close'] > df_working['slowEma']) & dntrend_dist_list)

    bear_cross_list = list((df_working['Open'] > df_working['fastEma']) & \
                           (df_working['Close'] < df_working['slowEma']) & dntrend_dist_list)

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


def adj_dblc_series(series, push_up=False):
    lst = series.tolist()
    if push_up:
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


def create_ema_combos(fast_range, slow_range):
    ema_combinations = [(fast, slow) for fast in fast_range for slow in slow_range]
    ema_combinations = [combo for combo in ema_combinations if combo[0] < combo[1]]

    return ema_combinations


def apply_dayslow_ema(ds_working, attr_dict, ema_len):
    ds_working = ds_working.copy()
    attr_dict["DaySlowEmaLen"] = ema_len
    add_attrs(ds_working, attr_dict)

    day_slow_ema_working = daily_ema_trades(ds_working)

    day_slow_ema_analyzed_df = get_pnl(day_slow_ema_working)
    day_slow_ema_analyzed_df = filter_trades(day_slow_ema_analyzed_df)
    day_slow_ema_analyzed_df = analyze_params(day_slow_ema_analyzed_df)

    return day_slow_ema_analyzed_df


"""--------------------------------------------------EMA Tools-------------------------------------------------------"""


def calculate_ema_numba(df, price_colname, window_size, smoothing_factor=2):
    result = calculate_ema_inner(
        price_array=df[price_colname].to_numpy(),
        window_size=window_size,
        smoothing_factor=smoothing_factor
    )

    return pd.Series(result, index=df.index, name="result", dtype=float)


@jit(nopython=True)
def calculate_ema_inner(price_array, window_size, smoothing_factor):
    result = np.empty(len(price_array), dtype="float64")
    sma_list = list()
    for i in range(len(result)):

        if i < window_size - 1:
            # assign NaN to row, append price to simple moving average list
            result[i] = np.NaN
            sma_list.append(price_array[i])
        elif i == window_size - 1:
            # calculate simple moving average
            sma_list.append(price_array[i])
            result[i] = sum(sma_list) / len(sma_list)
        else:
            # compute exponential moving averages according to formula
            result[i] = ((price_array[i] * (smoothing_factor / (window_size + 1))) +
                         (result[i - 1] * (1 - (smoothing_factor / (window_size + 1)))))

    return result


def create_ema_df(df, ema_len_list, daily=False):
    for ema_len in ema_len_list:
        df.loc[:, f'EMA_{ema_len}'] = calculate_ema_numba(
            df=df,
            price_colname='Open',
            window_size=ema_len
        )

    if daily:
        df = df[['Date', 'Open'] + [f'EMA_{ema_len}' for ema_len in ema_len_list]]
    else:
        df = df[['DateTime', 'Open'] + [f'EMA_{ema_len}' for ema_len in ema_len_list]]

    return df


def peek_csv_columns(file_path):
    df = pd.read_csv(file_path, nrows=0)
    return df.columns.tolist()


def check_create_emas(df, ema_range, strat_loc, security, timeframe, daily=False):
    if daily:
        file_output = f'{strat_loc}\\{security}\\{timeframe}\\{security}_daily_EMAs.csv'
    else:
        file_output = f'{strat_loc}\\{security}\\{timeframe}\\{security}_{timeframe}_EMAs.csv'

    file_exists = os.path.exists(file_output)
    if file_exists:
        col_list = peek_csv_columns(file_output)
        ema_list = list(ema_range)
        missing_columns = [col for col in ema_list if f'EMA_{col}' not in col_list]

        if len(missing_columns) > 0:
            ema_df = pd.read_csv(file_output)
            for missing_ema in missing_columns:
                ema_df.loc[:, f'EMA_{missing_ema}'] = calculate_ema_numba(
                    df=ema_df,
                    price_colname='Open',
                    window_size=missing_ema
                )
            ema_df.to_csv(file_output, index=False)
            del ema_df

    else:
        ema_df = create_ema_df(df, ema_range, daily)
        ema_df.to_csv(file_output, index=False)
        del ema_df


def get_ema_dat(ema_len, strat_loc, security, timeframe):
    file_path = f'{strat_loc}\\{security}\\{timeframe}\\{security}_{timeframe}_EMAs.csv'
    col_to_load = [f'EMA_{ema_len}', 'DateTime']
    df = pd.read_csv(file_path, usecols=col_to_load)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')

    return df


def adjust_datetime(df):
    try:
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    except ValueError:
        df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')







