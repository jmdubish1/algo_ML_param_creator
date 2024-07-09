from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from algo_tools import algo_trade_tools_v3 as att

pd.options.mode.chained_assignment = None


def check_final_candle(x, ratio):
    return ratio < x < (1/ratio)


@dataclass
class DblCandleParamset:
    def __init__(self, paramset):
        self.side = paramset['side']
        self.lookback = int(paramset.loc[0, 'LookBack'])
        self.fastEmaLen = int(paramset.loc[0, 'FastEmaLen'])
        self.minCndlSize = int(paramset.loc[0, 'MinCndlSize'])
        self.finalCndlSize = int(paramset.loc[0, 'FinalCndlSize'])
        self.finalCndlRatio = float(paramset.loc[0, 'FinalCandlRatio'] / 100)
        self.takeProfitOH = str(paramset.loc[0, 'TakeProfit_oh'])
        self.takeProfitPercent = float(paramset.loc[0, 'takeProfit'] / 10000)
        self.stopLossPercent = float(paramset.loc[0, 'StopLossPercent'] / 10000)
        self.daySlowOH = str(paramset.loc[0, 'DaySlow_oh'])
        self.ticksize = 0.0
        self.daySlowEmaLen = 8.0


@dataclass
class DblCandleTrades:
    bull_params: DblCandleParamset
    bear_params: DblCandleParamset

    def __post_init__(self):
        working_df: pd.DataFrame()
        dayslow_df: pd.DataFrame()
        trade_df: pd.DataFrame()

    def adj_oh_parameters(self):
        # self.bull_params = bull_paramset
        # self.bear_params = bear_paramset

        if (self.bull_params.takeProfitOH != 'On') and (self.bull_params.takeProfitOH != 'Off'):
            self.bull_params.takeProfitOH = float(self.bull_params.takeProfitOH)

        if (self.bull_params.daySlowOH != 'On') and (self.bull_params.daySlowOH != 'Off'):
            self.bull_params.daySlowOH = float(self.bull_params.daySlowOH)

        if (self.bear_params.takeProfitOH != 'On') and (self.bear_params.takeProfitOH != 'Off'):
            self.bear_params.takeProfitOH = float(self.bear_params.takeProfitOH)

        if (self.bear_params.daySlowOH != 'On') and (self.bear_params.daySlowOH != 'Off'):
            self.bear_params.daySlowOH = float(self.bear_params.daySlowOH)

    def add_working_df(self, working_df):
        self.working_df = working_df

    def double_candle_work(self, eod_time):
        self.get_row_diff()
        self.set_green_candle()
        self.find_reversals()
        self.decide_good_lookback()
        self.get_candle_ratios()
        self.decide_good_candle_ratio()
        self.get_candle_sizes()
        self.decide_good_candle_sizes()
        self.decide_double_candles()
        self.decide_basic_exits(eod_time)

    def get_row_diff(self):
        self.working_df['row_diff'] = pd.Series(self.working_df['Close'] - self.working_df['Open'])

    def set_green_candle(self):
        self.working_df['green_candle'] = self.working_df['Close'] > self.working_df['Open']

    def find_reversals(self):
        self.working_df['bull_reversal'] = np.array((self.working_df['row_diff'] > 0) &
                                                    (self.working_df['row_diff'].shift(1) < 0))
        self.working_df['bear_reversal'] = np.array((self.working_df['row_diff'] < 0) &
                                                    (self.working_df['row_diff'].shift(1) > 0))

    def decide_good_lookback(self):
        bull_red_good_list = ((~self.working_df['green_candle']).
                              rolling(window=self.bull_params.lookback).apply(lambda x: x.all(), raw=True))
        self.working_df['bull_red_good'] = bull_red_good_list.shift(1).fillna(False)

        bear_green_good_list = ((self.working_df['green_candle']).
                                rolling(window=self.bear_params.lookback).apply(lambda x: x.all(), raw=True))
        self.working_df['bear_green_good'] = bear_green_good_list.shift(1).fillna(False)

    def get_candle_ratios(self):
        self.working_df['candle_ratio'] = np.abs(self.working_df['row_diff'] / self.working_df['row_diff'].shift(1))
        self.working_df['candle_ratio'].fillna(0)
        self.working_df['candle_ratio'] = self.working_df['candle_ratio'].replace([np.inf, -np.inf], 0)

    def decide_good_candle_ratio(self):
        self.working_df['bull_finCndl_rat_g'] = (
            self.working_df['candle_ratio'].apply(lambda x: check_final_candle(x, self.bull_params.finalCndlRatio)))

        self.working_df['bear_finCndl_rat_g'] = (
            self.working_df['candle_ratio'].apply(lambda x: check_final_candle(x, self.bear_params.finalCndlRatio)))

    def get_candle_sizes(self):
        candle_size = self.bull_params.minCndlSize * self.bull_params.ticksize
        self.working_df['bull_cndl_size'] = np.abs(self.working_df['row_diff']) > candle_size

        final_candle_size = self.bull_params.finalCndlSize * self.bull_params.ticksize
        self.working_df['bull_final_cndl_size'] = np.abs(self.working_df['row_diff']) > final_candle_size

        candle_size = self.bear_params.minCndlSize * self.bear_params.ticksize
        self.working_df['bear_cndl_size'] = np.abs(self.working_df['row_diff']) > candle_size

        final_candle_size = self.bear_params.finalCndlSize * self.bear_params.ticksize
        self.working_df['bear_final_cndl_size'] = np.abs(self.working_df['row_diff']) > final_candle_size

    def decide_good_candle_sizes(self):
        self.working_df['bull_cndl_size_g'] = (self.working_df['bull_cndl_size'].
                                               rolling(window=self.bull_params.lookback).apply(lambda x: x.all(),
                                                                                               raw=True))

        self.working_df['bull_final_cndl_size_g'] = (self.working_df['bull_final_cndl_size'].
                                                     rolling(window=self.bull_params.lookback).apply(lambda x: x.all(),
                                                                                                     raw=True))

        self.working_df['bear_cndl_size_g'] = (self.working_df['bear_cndl_size'].
                                               rolling(window=self.bear_params.lookback).apply(lambda x: x.all(),
                                                                                               raw=True))

        self.working_df['bear_final_cndl_size_g'] = (self.working_df['bear_final_cndl_size'].
                                                     rolling(window=self.bear_params.lookback).apply(lambda x: x.all(),
                                                                                                     raw=True))

    def decide_double_candles(self):
        bull_cndl_made = (self.working_df['bull_reversal'] & self.working_df['bull_finCndl_rat_g'] &
                          self.working_df['bull_red_good'] & self.working_df['bull_cndl_size_g'] &
                          self.working_df['bull_final_cndl_size_g'])
        self.working_df['bullTrade'] = [False if pd.isna(x) else x for x in bull_cndl_made]

        bear_cndl_made = (self.working_df['bear_reversal'] & self.working_df['bear_finCndl_rat_g'] &
                          self.working_df['bear_green_good'] & self.working_df['bear_cndl_size_g'] &
                          self.working_df['bear_final_cndl_size_g'])
        self.working_df['bearTrade'] = [False if pd.isna(x) else x for x in bear_cndl_made]
        self.working_df.loc[self.working_df['bullTrade'] | self.working_df['bearTrade'], 'EntryPrice'] \
            = self.working_df['Close']

    def decide_basic_exits(self, eod_time):
        self.working_df['bullExit'] = False
        self.working_df.loc[(self.working_df['bullTrade']) &
                            (self.working_df['bullExit']), 'bullExit'] = False
        self.working_df.loc[self.working_df['bullTrade'], 'bearExit'] = True
        self.working_df.loc[self.working_df['Time'] == eod_time, 'bullTrade'] = False
        self.working_df.loc[self.working_df['Time'] == eod_time, 'bullExit'] = True

        self.working_df['bearExit'] = False
        self.working_df.loc[(self.working_df['bearTrade']) &
                            (self.working_df['bearExit']), 'bearExit'] = False
        self.working_df.loc[self.working_df['bearTrade'], 'bullExit'] = True
        self.working_df.loc[self.working_df['Time'] == eod_time, 'bearTrade'] = False
        self.working_df.loc[self.working_df['Time'] == eod_time, 'bearExit'] = True

    def find_stops(self):
        bull_sl_percent = 1 - self.bull_params.stopLossPercent
        bear_sl_percent = 1 + self.bear_params.stopLossPercent

        self.working_df = att.find_stops_bull(self.working_df, bull_sl_percent)
        self.working_df = att.find_stops_bear(self.working_df, bear_sl_percent)

    def apply_takeprofit_dayslow(self):
        if (self.bull_params.takeProfitOH == 'On') or (self.bull_params.takeProfitOH == 1):
            bull_tp_percent = 1 + self.bull_params.takeProfitPercent
            self.working_df = att.bull_find_take_profit(self.working_df, bull_tp_percent)

        if (self.bull_params.daySlowOH == 'On') or (self.bull_params.daySlowOH == 1):
            self.working_df.loc[(self.working_df['bullTrade']) &
                                (self.working_df['dayEma'] < self.working_df['EntryPrice']), 'bullTrade'] = False

        if (self.bear_params.takeProfitOH == 'On') or (self.bear_params.takeProfitOH == 1):
            bear_tp_percent = 1 - self.bear_params.takeProfitPercent
            self.working_df = att.bear_find_take_profit(self.working_df, bear_tp_percent)

        if (self.bull_params.daySlowOH == 'On') or (self.bull_params.daySlowOH == 1):
            self.working_df.loc[(self.working_df['bearTrade']) &
                                (self.working_df['dayEma'] < self.working_df['EntryPrice']), 'bearTrade'] = False

    def get_pnl(self, start_date, end_date):
        self.working_df['PnL'] = np.where(self.working_df['bearTrade'],
                                          self.working_df['EntryPrice'] - self.working_df['ExitPrice'],
                                          np.where(self.working_df['bullTrade'],
                                                   self.working_df['ExitPrice'] - self.working_df['EntryPrice'],
                                                   np.nan))
        self.working_df = self.working_df.loc[(self.working_df['Date'] >= start_date) &
                                              (self.working_df['Date'] < end_date)]












