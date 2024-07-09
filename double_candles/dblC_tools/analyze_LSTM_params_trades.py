import os
import pandas as pd
from algo_tools import algo_trade_tools_v3 as att
from double_candles.dblC_tools import dblC_quick_logic as dql

pd.options.mode.chained_assignment = None

timeframe = "15min"
security = "NQ"
timelength = "8years"

file_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data'
strat_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles'
file_output = f'{strat_loc}\\{security}\\{timeframe}\\{timeframe}_test_{timelength}\\k_means_model_92_steps_alleq_LB37'
os.makedirs(file_output, exist_ok=True)


intra_day_data_file = f'{file_loc}\\{security}_{timeframe}_20240505_20040401.txt'

daily_data_file = f'{file_loc}\\{security}_daily_20240505_20040401.txt'

algo_name = "Double_Candle"
begin_date = pd.to_datetime("2016-04-01", format='%Y-%m-%d')
end_date = begin_date + pd.DateOffset(months=1) - pd.Timedelta(days=1)
start_time = 7
eod_time = "15:00"
# Ensure this is adjusted for every security. This is $1 or 4 ticks for NQ
tick_size = 0.1

param_df = []
for side in ['Bull', 'Bear']:
    param_temp = pd.read_csv(f'{file_output}\\full_param_list_{side}.csv')
    # param_temp.drop(['Unnamed: 0'], inplace=True)
    param_temp['side'] = side
    param_df.append(param_temp)

param_df = pd.concat(param_df)

# Loading and cleaning the intraday data
df_working_clean = pd.read_csv(intra_day_data_file)
cols_to_remove = ["AvgExp12", "AvgExp24", "Bearish_Double_Candle", "Bullish_Double_Candle",
                  "Up", "Down", "Vol", "VolAvg", "OpenInt"]
df_working_clean = df_working_clean.drop(columns=[col for col in df_working_clean.columns if col in cols_to_remove])
# Combine Date and Time columns into a single DateTime column
df_working_clean['DateTime'] = pd.to_datetime(df_working_clean['Date'] + ' ' + df_working_clean['Time'],
                                              format='%m/%d/%Y %H:%M')
df_working_clean['Date'] = pd.to_datetime(df_working_clean['Date'], format='%m/%d/%Y')

print('Creating EMA Dfs')
df_daily_ema = pd.read_csv(f'{strat_loc}\\{security}\\{timeframe}\\{security}_daily_EMAs.csv')

try:
    df_daily_ema['Date'] = pd.to_datetime(df_daily_ema['Date'], format='%Y-%m-%d')
except ValueError:
    df_daily_ema['Date'] = pd.to_datetime(df_daily_ema['Date'], format='%m/%d/%Y')

# Creating EMAs for intraday
fastEmaLens = param_df['FastEmaLen'].unique()
att.check_create_emas(df_working_clean, fastEmaLens, strat_loc, security, timeframe)

df_daily_ema.rename(columns={'EMA_8': 'dayEma'}, inplace=True)
df_working_clean = df_working_clean.merge(df_daily_ema[['Date', 'dayEma']], on=['Date'])

time_mask = (df_working_clean['DateTime'].dt.hour >= start_time) & \
            (df_working_clean['DateTime'].dt.time <= pd.to_datetime(eod_time, format='%H:%M').time())

df_working_clean = df_working_clean.loc[time_mask]

trade_dfs = []
month_stats = []
for month in param_df['month_to_test'].unique():
    print(month)
    month_df = param_df.loc[param_df['month_to_test'] == month]
    bull_params = month_df.loc[month_df['side'] == 'Bull'].reset_index(drop=True)
    bear_params = month_df.loc[month_df['side'] == 'Bear'].reset_index(drop=True)

    try:
        end_date = pd.to_datetime(month, format='%Y-%m-%d') + pd.DateOffset(months=1)
    except ValueError:
        end_date = pd.to_datetime(month, format='%m/%d/%Y') + pd.DateOffset(months=1)

    start_date = end_date - pd.DateOffset(months=2)
    test_df_clean = df_working_clean.loc[(df_working_clean['Date'] >= start_date) &
                                         (df_working_clean['Date'] <= end_date)]
    if not ((len(bull_params) > 0) and (len(bear_params) > 0)):
        break
    else:

        bull_paramset = dql.DblCandleParamset(bull_params)
        bull_paramset.ticksize = tick_size

        bear_paramset = dql.DblCandleParamset(bear_params)
        bear_paramset.ticksize = tick_size

    test_db = dql.DblCandleTrades(bull_paramset, bear_paramset)
    test_db.adj_oh_parameters()
    test_db.add_working_df(test_df_clean)
    test_db.double_candle_work(eod_time)
    test_db.find_stops()
    test_db.apply_takeprofit_dayslow()
    test_db.get_pnl(month, end_date)
    print(test_db.working_df['PnL'].sum())
    trade_dfs.append(test_db.working_df.loc[test_db.working_df['bearTrade'] | test_db.working_df['bullTrade']])
    month_stats.append([month, test_db.working_df['PnL'].sum()])

trade_dfs = pd.concat(trade_dfs)
month_stats = pd.DataFrame(month_stats)
dfs = {'Trades': trade_dfs, 'Month_stats': month_stats}

with pd.ExcelWriter(f'{file_output}\\all_trades.xlsx') as writer:
    for sheet_name, df in dfs.items():
        df.to_excel(writer, sheet_name, index=True)

