import dataclasses
import random

import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import time
import datetime as dt
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.cluster import KMeans
from dateutil.relativedelta import relativedelta
import itertools
import os
from scipy.stats import percentileofscore
import pykalman as pykal
pd.options.mode.chained_assignment = None  # default='warn'
from dataclasses import dataclass, field



def create_atr(data, n=12):
    high_low = data['High'] - data['Low']
    high_prev_close = np.abs(data['High'] - data['Close'].shift(1))
    low_prev_close = np.abs(data['Low'] - data['Close'].shift(1))
    true_range = np.maximum(high_low, high_prev_close)
    true_range = np.maximum(true_range, low_prev_close)

    # Calculate Average True Range (ATR)
    atr = np.zeros_like(data['Close'])
    atr[n - 1] = np.mean(true_range[:n])  # Initial ATR calculation

    for i in range(n, len(data['Close'])):
        atr[i] = ((atr[i - 1] * (n - 1)) + true_range[i]) / n

    return atr


def convert_date_time(data):
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data['Date'] = pd.to_datetime(data['Date']).dt.date
    data['Month'] = data['Datetime'].dt.month
    data['Year'] = data['Datetime'].dt.year

    return data


@dataclass
class KMeansParams:
    side: str
    min_trades: int
    max_trades: int
    n_clusters: int
    kmeans_cols: list
    min_win_loss_rat: float
    min_kept: float
    include_bad_cluster: bool
    name: str

    def __post_init__(self):
        self.orig_n_clusters = self.n_clusters
        self.orig_min_win_loss_rat = self.min_win_loss_rat
        self.orig_len = float
        self.good_clusters = []
        self.good_cluster_len = float
        self.working_df = pd.DataFrame()
        self.combined_df = pd.DataFrame()
        self.complete = False

    def add_dataframe(self, df):
        self.combined_df = df

    def copy_with_changes(self, **changes):
        return dataclasses.replace(self, **changes)

    def re_run(self):
        if self.n_clusters == 8:
            self.n_clusters = self.orig_n_clusters
            self.min_win_loss_rat = max(self.min_win_loss_rat - .1, 1.2)

            self.print_rerun()
            self.eval_win_loss()

        elif (self.n_clusters == 8) and (self.min_win_loss_rat == 1.2):
            self.n_clusters = self.orig_n_clusters
            self.min_win_loss_rat = self.orig_min_win_loss_rat

            self.print_rerun()
            self.eval_win_loss()

        else:
            self.n_clusters = max(self.n_clusters - 2, 8)
            self.print_rerun()
            self.run_clusters()

    def decide_rerun(self):
        if (float(self.good_cluster_len) < (float(self.orig_len) * self.min_kept)) and not self.complete:
            self.re_run()
        else:
            self.complete = True

    def run_clusters(self):
        if not self.complete:
            self.build_clusters()
            self.get_good_clusters()
            self.decide_rerun()

    def eval_win_loss(self):
        if not self.complete:
            self.get_good_clusters()
            self.decide_rerun()

    def finish_rankings(self):
        self.print_good_cluster_params()
        self.analyze_clusters()

    def create_clusers(self):
        self.run_clusters()

        if self.complete:
            self.finish_rankings()
            print(f'Completed Cluster Analysis: {self.name}')

            return self.working_df

    def subset_side_trades(self):
        self.combined_df = self.combined_df.loc[self.combined_df['side'] == self.side]
        self.orig_len = float(len(self.combined_df.index))

        temp_dfs = []
        group_df = self.combined_df.groupby([self.combined_df['Year'], self.combined_df['Month']])
        for name, group in group_df:
            temp_df = find_min_max_trade_dfs(group, self.min_trades, self.max_trades)
            temp_dfs.append(temp_df)

        self.combined_df = pd.concat(temp_dfs)

    def build_clusters(self):
        """K-means clustering to identify best param sets for ranking"""
        print(f'Running clusters: {self.n_clusters}')
        self.working_df = self.combined_df.copy()
        train_df = self.working_df[self.kmeans_cols]
        scaler = RobustScaler(quantile_range=(1, 99))
        normalized_data = scaler.fit_transform(train_df)
        model = KMeans(n_clusters=self.n_clusters)
        model.fit(normalized_data)
        yhat = model.predict(normalized_data)
        self.working_df['cluster'] = yhat

    def get_good_clusters(self):
        """Returns TF for a re-run"""

        self.good_clusters = []
        for i in self.working_df['cluster'].unique():
            subset_df = self.working_df[self.working_df['cluster'] == i]
            win_tot = sum([i if i > 0 else 0 for i in subset_df['CumSum']])
            loss_tot = sum([i if i < 0 else 0 for i in subset_df['CumSum']])
            win_loss_rat = -win_tot/loss_tot

            if win_loss_rat >= self.min_win_loss_rat:
                self.good_clusters.append(i)

        if len(self.good_clusters) > 0:
            self.agg_good_clusters()

    def agg_good_clusters(self):
        if len(self.good_clusters) > 0:
            print(self.good_clusters)
            self.good_cluster_len = self.working_df['cluster'].isin(self.good_clusters).value_counts().loc[True]
            if not self.good_cluster_len > 0:
                self.good_cluster_len = 0

    def analyze_clusters(self,):
        print('Making Rankings')
        sorted_groups = []

        tempdf_clusterdf = self.working_df[self.working_df['cluster'].isin(self.good_clusters)]
        tempdf_clusterdf = tempdf_clusterdf.groupby([tempdf_clusterdf['Year'], tempdf_clusterdf['Month']])
        for name, group in tempdf_clusterdf:
            temp_percentile = (
                group['avg_trade'].apply(lambda x: percentileofscore(group['avg_trade'], x)))
            group.loc[:, 'Rank'] = lin_rescale(temp_percentile)
            sorted_groups.append(group)

        tempdf_clusterdf = self.working_df[~self.working_df['cluster'].isin(self.good_clusters)]
        tempdf_clusterdf = tempdf_clusterdf.groupby([tempdf_clusterdf['Year'], tempdf_clusterdf['Month']])
        for name, group in tempdf_clusterdf:
            if (group['avg_trade'] < 0).all():
                group = group.sort_values(by='avg_trade', ascending=True).reset_index(drop=True)
                temp_percentile = (group['avg_trade'].apply(lambda x: percentileofscore(group['avg_trade'], x)))
                group.loc[:, 'Rank'] = lin_rescale(temp_percentile, min_new=-100, max_new=0)
            else:
                temp_percentile = (
                    group['avg_trade'].apply(lambda x: percentileofscore(group['avg_trade'], x)))
                group.loc[:, 'Rank'] = lin_rescale(temp_percentile, min_new=-100, max_new=95)

            sorted_groups.append(group)

        self.working_df = pd.concat([group for group in sorted_groups])

    def print_good_cluster_params(self):
        print(f'Found Param Sets:\n'
              f'Kept {self.good_cluster_len} out of {self.orig_len}\n'
              f'Targeting more than: {round((float(self.orig_len) * self.min_kept), 0)}\n'
              f'{round(float(self.good_cluster_len) / float(self.orig_len), 2)} Kept\n'
              f'Running LSTM with {self.n_clusters} clusters\n'
              f'WL Ratio: {self.min_win_loss_rat: .2f}\n')

    def print_rerun(self):
        if not isinstance(self.good_cluster_len, type):
            print(f'Too many cut: Kept {self.good_cluster_len} out of {self.orig_len}\n'
                  f'Targeting more than: {round((float(self.orig_len) * self.min_kept), 0)}\n'
                  f'{round(float(self.good_cluster_len) / float(self.orig_len), 2)} Kept\n')
        else:
            print('Empty Cluster')


def lin_rescale(values, min_orig=0, max_orig=100, min_new=-100, max_new=100):
    return [((x - min_orig) / (max_orig - min_orig)) * (max_new - min_new) + min_new for x in values]


def pad_months(data, n_days):
    full_df = []
    for year in data['Year'].unique():
        for month in data['Month'].unique():
            month_df = data.loc[(data['Year'] == year) &
                                (data['Month'] == month)]

            rows_2_add = n_days - len(month_df) + 1
            add_row_beginning = []
            add_row_end = []
            if not month_df.empty:
                if len(month_df) < 23:
                    for i in range(1, rows_2_add):
                        if i % 2 == 1:
                            add_row_beginning.append(list(month_df.iloc[0]))
                        else:
                            add_row_end.append(list(month_df.iloc[-1]))

                    add_row_beginning = pd.DataFrame(add_row_beginning)
                    add_row_beginning.columns = month_df.columns
                    month_df = pd.concat([add_row_beginning, month_df])

                    if len(add_row_end) > 0:
                        add_row_end = pd.DataFrame(add_row_end)
                        add_row_end.columns = month_df.columns
                        month_df = pd.concat([month_df, add_row_end])
                # print(f'{year} : {month} : {len(month_df)}')

                full_df.append(month_df)

    full_df = pd.concat(full_df).reset_index(drop=True)

    return full_df


def create_random_df_discrete_rank(data, rank=0):
    random_df_list = []
    uniq_months = data.drop_duplicates(subset=['Year', 'Month']).reset_index(drop=True)
    uniq_month_list = []
    for row in range(0, len(uniq_months)):
        uniq_month_list.append([uniq_months.at[row, 'Year'], uniq_months.at[row, 'Month']])
    if rank == 0:
        for month_year in uniq_month_list:
            rand_row = data.loc[(data['Year'] == month_year[0]) &
                                (data['Month'] == month_year[1])]
            if len(rand_row.index) > 0:
                random_df_list.append(rand_row.sample(n=1))

    else:
        for month_year in uniq_month_list:
            rand_row = data.loc[(data['Year'] == month_year[0]) &
                                (data['Month'] == month_year[1])]
            rand_row_rank = rand_row.loc[rand_row['Rank'] == rank]
            if len(rand_row_rank.index) > 0:
                random_df_list.append(rand_row_rank.sample(n=1))
            else:
                random_df_list.append(rand_row.sample(n=1))

    out_df = pd.concat(random_df_list, ignore_index=True)

    return out_df


def create_random_df_percent_rank_dblc(data, lower_lookback_df, rank=0.0):
    """Has a provision that fixes the data for when there is an empty set when attempting to target higher lookbacks"""
    random_df_list = []
    uniq_months = lower_lookback_df.drop_duplicates(subset=['Year', 'Month']).reset_index(drop=True)
    uniq_month_list = []
    for row in range(0, len(uniq_months)):
        uniq_month_list.append([uniq_months.at[row, 'Year'], uniq_months.at[row, 'Month']])

    for month_year in uniq_month_list:
        rand_row = data.loc[(data['Year'] == month_year[0]) &
                            (data['Month'] == month_year[1])]

        lower_lookback_temp_full = lower_lookback_df.loc[(lower_lookback_df['Year'] == month_year[0]) &
                                                         (lower_lookback_df['Month'] == month_year[1])]

        if rank == 0:
            if len(rand_row.index) > 0:
                random_df_list.append(rand_row.sample(n=23, replace=True))
            else:
                random_df_list.append(lower_lookback_temp_full.sample(n=23, replace=True))

        elif rank > 0:
            rand_row_rank = rand_row.loc[rand_row['Rank'] >= rank]
            lower_lookback_temp = lower_lookback_temp_full.loc[lower_lookback_temp_full['Rank'] >= rank]

            if len(rand_row_rank.index) > 0:
                random_df_list.append(rand_row_rank.sample(n=23, replace=True))
            elif len(lower_lookback_temp) > 0:
                random_df_list.append(lower_lookback_temp.sample(n=23, replace=True))
            elif len(rand_row) > 0:
                random_df_list.append(rand_row.sample(n=23, replace=True))
            else:
                random_df_list.append(lower_lookback_temp_full.sample(n=23, replace=True))

        elif rank < 0:
            rand_row_rank = rand_row.loc[rand_row['Rank'] <= rank]
            lower_lookback_temp = lower_lookback_temp_full.loc[lower_lookback_temp_full['Rank'] <= rank]

            if len(rand_row_rank.index) > 0:
                random_df_list.append(rand_row_rank.sample(n=23, replace=True))
            elif len(lower_lookback_temp) > 0:
                random_df_list.append(lower_lookback_temp.sample(n=23, replace=True))
            elif len(rand_row) > 0:
                random_df_list.append(rand_row.sample(n=23, replace=True))
            else:
                random_df_list.append(lower_lookback_temp_full.sample(n=23, replace=True))

    out_df = pd.concat(random_df_list, ignore_index=True)

    return out_df


def create_random_df_percent_rank_dblc2(data, lower_lookback_df, rankhigh, ranklow):
    """Has a provision that fixes the data for when there is an empty set when attempting to target higher lookbacks"""

    random_df_list = []
    uniq_months = lower_lookback_df.drop_duplicates(subset=['Year', 'Month']).reset_index(drop=True)
    uniq_month_list = []
    for row in range(0, len(uniq_months)):
        uniq_month_list.append([uniq_months.at[row, 'Year'], uniq_months.at[row, 'Month']])

    for month_year in uniq_month_list:
        rand_row = data.loc[(data['Year'] == month_year[0]) &
                            (data['Month'] == month_year[1])]

        lower_lookback_temp_full = lower_lookback_df.loc[(lower_lookback_df['Year'] == month_year[0]) &
                                                         (lower_lookback_df['Month'] == month_year[1])]

        rand_row_rank = rand_row.loc[rand_row['Rank'] >= rankhigh]
        lower_lookback_temp = lower_lookback_temp_full.loc[lower_lookback_temp_full['Rank'] >= rankhigh]

        if len(rand_row_rank.index) > 0:
            random_df_list.append(rand_row_rank.sample(n=12, replace=True))
        elif len(lower_lookback_temp) > 0:
            random_df_list.append(lower_lookback_temp.sample(n=12, replace=True))
        elif len(rand_row) > 0:
            random_df_list.append(rand_row.sample(n=12, replace=True))
        else:
            random_df_list.append(lower_lookback_temp_full.sample(n=12, replace=True))

        rand_row_rank = rand_row.loc[rand_row['Rank'] <= ranklow]
        lower_lookback_temp = lower_lookback_temp_full.loc[lower_lookback_temp_full['Rank'] <= ranklow]

        if len(rand_row_rank.index) > 0:
            random_df_list.append(rand_row_rank.sample(n=11, replace=True))
        elif len(lower_lookback_temp) > 0:
            random_df_list.append(lower_lookback_temp.sample(n=11, replace=True))
        elif len(rand_row) > 0:
            random_df_list.append(rand_row.sample(n=11, replace=True))
        else:
            random_df_list.append(lower_lookback_temp_full.sample(n=11, replace=True))

    out_df = pd.concat(random_df_list, ignore_index=True)

    return out_df


def create_random_df_percent_rank_emac(data, backup_df, rank=0.0):
    """Has a provision that fixes the data for when there is an empty set when attempting to target removed params"""
    random_df_list = []
    uniq_months = backup_df.drop_duplicates(subset=['Year', 'Month']).reset_index(drop=True)
    uniq_month_list = []
    for row in range(0, len(uniq_months)):
        uniq_month_list.append([uniq_months.at[row, 'Year'], uniq_months.at[row, 'Month']])

    for month_year in uniq_month_list:
        rand_row = data.loc[(data['Year'] == month_year[0]) &
                            (data['Month'] == month_year[1])]

        backup_df_full = backup_df.loc[(backup_df['Year'] == month_year[0]) &
                                                 (backup_df['Month'] == month_year[1])]

        if rank == 0:
            if len(rand_row.index) > 0:
                random_df_list.append(rand_row.sample(n=1))
            else:
                random_df_list.append(backup_df_full.sample(n=1))

        elif rank > 0:
            rand_row_rank = rand_row.loc[rand_row['Rank'] >= rank]
            backup_df_full_temp = backup_df_full.loc[backup_df_full['Rank'] >= rank]

            if len(rand_row_rank.index) > 0:
                random_df_list.append(rand_row_rank.sample(n=1))
            elif len(backup_df_full_temp) > 0:
                random_df_list.append(backup_df_full_temp.sample(n=1))
            else:
                random_df_list.append(rand_row.sample(n=1))

        elif rank < 0:
            rand_row_rank = rand_row.loc[rand_row['Rank'] <= rank]
            backup_df_full_temp = backup_df_full.loc[backup_df_full['Rank'] <= rank]

            if len(rand_row_rank.index) > 0:
                random_df_list.append(rand_row_rank.sample(n=1))
            elif len(backup_df_full_temp) > 0:
                random_df_list.append(backup_df_full_temp.sample(n=1))
            else:
                random_df_list.append(backup_df_full.sample(n=1))

    out_df = pd.concat(random_df_list, ignore_index=True)

    return out_df


def make_month_list(start_date, end_date):
    first_of_every_month = []

    # Generate first of every month
    current_date = start_date
    while current_date < end_date:
        add_date = current_date.date()
        first_of_every_month.append(add_date)
        # Move to the first of the next month
        current_date = current_date.replace(day=1) + dt.timedelta(days=32)
        current_date = current_date.replace(day=1)

    return first_of_every_month


def get_number_months(end_date, start_date):
    time_diff = relativedelta(end_date, start_date)
    len_months = time_diff.years * 12 + time_diff.months

    return len_months


def improve_trade_df(df, sort_cols, asc_list, main_sort_col):
    df['avg_trade'] = df.loc[:, 'cumPnl']/df.loc[:, 'trades']
    df_working = df.copy()
    df_working.loc[:, 'cumPnl'] = np.round(df_working.loc[:, 'cumPnl'], 6)
    df_working.fillna(0, inplace=True)
    df_working.reset_index(inplace=True)
    group_cols = ['side', 'year', 'month', main_sort_col, 'cumPnl', 'trades']

    df_working['unique_id'] = df_working.groupby(group_cols).ngroup()
    df_working.sort_values(by=sort_cols, ascending=asc_list, inplace=True)
    df_working.reset_index(inplace=True, drop=True)
    df_keep = df_working.drop_duplicates(subset=['unique_id'], keep='first')

    return df_keep


def scheduler(epoch, lr):
    return np.max([lr**1.0005, .00001])


def convert_to_datetime(date_str):
    return dt.datetime.strptime(date_str, '%m/%d/%Y')


def create_kmeans_df(data, n_clusters, params):
    final_dfs = []
    for side in data['side'].unique():
        tempdf = data[(data['side'] == side)]

        """K-means clustering to identify best param sets for ranking"""
        train_df = tempdf[params]

        scaler = RobustScaler(quantile_range=(1, 99))
        normalized_data = scaler.fit_transform(train_df)

        model = KMeans(n_clusters=n_clusters)

        model.fit(normalized_data)

        yhat = model.predict(normalized_data)
        clusters = np.unique(yhat)
        tempdf['cluster'] = yhat

        for i in clusters:
            subset_df = tempdf[tempdf['cluster'] == i]
            subset_pnl = subset_df['cumPnl'].sum()
            print(f'Side: {side}\n'
                  f'Cluster: {i}\n'
                  f'PnL: {subset_pnl}')
            mean_ = np.mean(subset_df['cumPnl'])
            median_ = np.median(subset_df['cumPnl'])
            std_dev = np.std(subset_df['cumPnl'])
            win_ = sum([1 if i > 0 else 0 for i in subset_df['cumPnl']])
            loss_ = sum([1 if i < 0 else 0 for i in subset_df['cumPnl']])
            win_per = win_/(win_+loss_)
            win_tot = sum([i if i > 0 else 0 for i in subset_df['cumPnl']])
            loss_tot = sum([i if i < 0 else 0 for i in subset_df['cumPnl']])
            win_loss_rat = -win_tot/loss_tot

            kmeans_dict = {f'Side': side,
                           f'Cluster': i,
                           f'PnL': subset_pnl,
                           f'Mean': mean_,
                           f'Median': median_,
                           f'Std_Dev': std_dev,
                           f'Win_Count': win_,
                           f'Loss_Count': loss_,
                           f'Win_Percent': win_per,
                           f'Win_Total': win_tot,
                           f'Loss_Total': loss_tot,
                           f'Win_Loss_Ratio': win_loss_rat}

            for k, v in kmeans_dict.items():
                if type(v) != str:
                    kmeans_dict[k] = round(v, 3)

            kmeans_df = pd.DataFrame.from_dict(kmeans_dict, orient='index').T

            final_dfs.append(kmeans_df)

    final_df = pd.concat(final_dfs)
    final_df = final_df.transpose()

    return final_df


def one_hot_encode_dblc(df, col_name):
    encoder = OneHotEncoder(sparse=False)
    lookback_encoded = encoder.fit_transform(df[[col_name]])
    lookback_encoded_df = pd.DataFrame(lookback_encoded,
                                       columns=encoder.get_feature_names_out([col_name]),
                                       index=df.index)
    df = pd.concat([df, lookback_encoded_df], axis=1)
    df.drop(columns=[col_name], inplace=True)

    return df


class CustomPrintCallback(Callback):
    def __init__(self, output_name):
        super(CustomPrintCallback, self).__init__()
        self.output_name = output_name

    def on_epoch_end(self, epoch, logs=None):
        output_value = round(logs.get(self.output_name), 4)
        print(f"{epoch} : {self.output_name}: {output_value}")


def result_dict_handler_dblc(predictions, y_scaler, target_float_vars, lookback_list, loss, lb_acc, end_date_test):
    y_predictions = y_scaler.inverse_transform(predictions[0][-1].reshape(1, -1))
    result_dict = dict(zip(target_float_vars, y_predictions[0]))
    result_dict['finalcandleratio'] = round(result_dict['finalcandleratio'], 0)
    for k, v in result_dict.items():
        result_dict[k] = round(v, 4)
        print(f'{k} : {result_dict[k]}')

    result_dict['lookback'] = dict(zip(lookback_list, predictions[1][-1]))
    print(result_dict['lookback'])

    result_dict['takeProfit_oh'] = dict(zip(['Off', 'On'], predictions[2][-1]))
    print(f'Take Profit: {result_dict["takeProfit_oh"]}')

    result_dict['daySlowEma_oh'] = dict(zip(['Off', 'On'], predictions[3][-1]))
    print(f'DaySlow: {result_dict["daySlowEma_oh"]}')

    result_dict['month_to_test'] = str(end_date_test)
    print(f"{result_dict['month_to_test']} \n\n")

    result_dict['model_loss'] = round(loss, 4)
    result_dict['lookback_accuracy'] = lb_acc

    result_dict['daySlowEma_oh'] = max(result_dict['daySlowEma_oh'], key=result_dict['daySlowEma_oh'].get)
    result_dict['takeProfit_oh'] = max(result_dict['takeProfit_oh'], key=result_dict['takeProfit_oh'].get)
    result_dict['lookback'] = max(result_dict['lookback'], key=result_dict['lookback'].get)
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')

    return result_df


def result_dict_handler_dblc2(predictions, y_scaler, target_float_vars, loss, end_date_test):
    y_predictions = y_scaler.inverse_transform(predictions)
    result_dict = dict(zip(target_float_vars, y_predictions[0]))
    result_dict['finalcandleratio'] = round(result_dict['finalcandleratio'], 0)
    for k, v in result_dict.items():
        result_dict[k] = round(v, 4)
        print(f'{k} : {result_dict[k]}')

    result_dict['month_to_test'] = str(end_date_test)
    print(f"{result_dict['month_to_test']}\n")

    result_dict['model_loss'] = round(loss, 4)

    result_df = pd.DataFrame.from_dict(result_dict, orient='index')

    return result_df


def result_dict_handler_emac(predictions, target_float_vars, loss, end_date_test):
    result_dict = dict(zip(target_float_vars, predictions[0][-1]))
    for k, v in result_dict.items():
        result_dict[k] = round(v, 4)
        print(f'{k} : {result_dict[k]}')

    result_dict['takeProfit_oh'] = dict(zip(['Off', 'On'], predictions[1][-1]))
    print(f'Take Profit: {result_dict["takeProfit_oh"]}')
    result_dict['daySlowEma_oh'] = dict(zip(['Off', 'On'], predictions[2][-1]))
    print(f'Dayslow: {result_dict["daySlowEma_oh"]}')
    result_dict['month_to_test'] = str(end_date_test)
    print(f"{result_dict['month_to_test']} \n")
    result_dict['model_loss'] = round(loss, 4)
    result_df = pd.DataFrame.from_dict(result_dict, orient='index')

    return result_df


class CustomCallback_dblc(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        loss = logs.get('loss', None)
        val_loss = logs.get('val_loss', None)
        lb_out_accuracy = logs.get('LBack_out_accuracy')
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()

        print(f"Epoch {epoch + 1} - "
              f"Time: {duration:.2f}s - "
              f"Loss: {loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"LBack_out_accuracy: {lb_out_accuracy:.4f} - "
              f"LR: {lr:.6f}")


class CustomCallback_emac(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start_time
        loss = logs.get('loss', None)
        val_loss = logs.get('val_loss', None)
        lb_out_accuracy = logs.get('LBack_out_accuracy')
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()

        print(f"Epoch {epoch + 1} - "
              f"Time: {duration:.2f}s - "
              f"Loss: {loss:.4f} - "
              f"Val Loss: {val_loss:.4f} - "
              f"LR: {lr:.6f}")


def reshape_Y_n_steps(dat_arr, n_lookback, n_steps):
    dat_arr = np.array([dat_arr[i] for i in range(n_lookback-1, len(dat_arr)-1, n_steps)])
    return dat_arr


def reshape_X_n_steps(dat_arr, n_lookback, n_steps):
    dat_arr = np.array([dat_arr[i - n_lookback:i, :] for i in range(n_lookback, len(dat_arr), n_steps)])
    return dat_arr


def complete_intraday_df_work(intraday_df, start_hour):
    intraday_df.rename(columns={'Vol': 'Vol_int'}, inplace=True)
    intraday_df['ATR_int'] = create_atr(intraday_df)
    intraday_df = convert_date_time(intraday_df)
    intraday_df = intraday_df.sort_values(by='Datetime')
    intraday_df = intraday_df[(intraday_df['Datetime'].dt.hour >= start_hour) & (intraday_df['Datetime'].dt.hour <= 15)]
    intraday_df['Month'] = intraday_df['Month'].astype(int)
    intraday_df['Year'] = intraday_df['Year'].astype(int)
    intraAtr = round(intraday_df.groupby(intraday_df['Date'])['ATR_int'].mean(), 5)
    intraVol = round(intraday_df.groupby(intraday_df['Date'])['Vol_int'].mean(), 5)

    return intraday_df, intraAtr, intraVol


def fill_all_zeros(data_df):
    ignore_cols = ['Date', 'Month', 'Year']
    d_cols = [d for d in data_df.columns if d not in ignore_cols]
    for d in d_cols:
        data_df.loc[:, d] = fill_zeros(data_df[d])

    return data_df


def fill_zeros(series):
    mask = (series == 0) | (series < 0)
    while mask.any():
        series[mask] = series.shift()[mask]
        mask = (series == 0) | (series < 0)
    return series


def standardize_data(daily_df):
    # Standardize by day over day % change
    for c in ['ATR_day']:
        daily_df[c] = daily_df[c]/daily_df['Close']

    # Standardize by difference from last 20-day average
    for c in ['Vol', 'OpenInt', 'VolAvg', 'OI']:
        daily_df[c] = daily_df[c].rolling(window=23, min_periods=1).mean()
        daily_df[c] = daily_df[c]/daily_df[c].shift(23)

    return daily_df


def add_high_low_diff(daily_df, daily_set_cols, other_sec, sec_name):
    securities = other_sec + [sec_name]
    for sec in securities:
        daily_df[f'{sec}_HL_diff'] = (daily_df[f'{sec}_High'] - daily_df[f'{sec}_Low'])/daily_df[f'{sec}_Close']
        daily_set_cols = daily_set_cols + [f'{sec}_HL_diff']

    daily_set_cols2 = [col for col in daily_set_cols if not any(sub in col for sub in ['High', 'Low'])]
    daily_set_cols2 = daily_set_cols2 + [f'{sec_name}_High', f'{sec_name}_Low']
    drop_cols = [col for col in daily_set_cols if col not in daily_set_cols2]

    daily_df.drop(columns=drop_cols, inplace=True)
    return daily_df, daily_set_cols2


def standardize_intradata(daily_df, sec_name):
    # Standardize by day over day % change
    for c in ['ATR_int']:
        daily_df[c] = daily_df[c]/daily_df[f'{sec_name}_Close']

    # Standardize by difference from last 20-day average
    for c in ['Vol_int']:
        daily_df[c] = daily_df[c].rolling(window=23, min_periods=1).mean()
        daily_df[c] = daily_df[c].pct_change()

    return daily_df


def get_price_cols(data_df, input_features):
    data_df.reset_index(inplace=True, drop=True)
    price_substrings = ['Open', 'High', 'Low', 'Close']
    price_cols = [string for string in input_features if any(sub in string for sub in price_substrings)]
    price_cols = [string for string in price_cols if not any(sub in string for sub in ['OpenInt'])]

    return price_cols


def standardize_data_kalman(data_df, daily_set_cols):
    data_df.reset_index(inplace=True, drop=True)
    price_substrings = ['Open', 'High', 'Low', 'Close']
    price_cols = [string for string in daily_set_cols if any(sub in string for sub in price_substrings)]
    price_cols = [string for string in price_cols if not any(sub in string for sub in ['OpenInt', 'HL_diff'])]

    for c in price_cols:
        kf = pykal.KalmanFilter(transition_matrices=[1],
                                observation_matrices=[1],
                                initial_state_mean=0,
                                initial_state_covariance=1,
                                observation_covariance=1,
                                transition_covariance=.0001)
        temp_arr = np.array(data_df.loc[:, c])
        temp_mean_arr, _ = kf.filter(temp_arr)
        temp_mean_arr = temp_mean_arr.flatten()
        data_df.loc[:, c] = temp_mean_arr.astype(np.float32)

    data_df.fillna(0, inplace=True)
    data_df.replace([np.inf, -np.inf], 0, inplace=True)

    return data_df


def set_better_data_range(data_df):
    ignore_cols = ['Date', 'Month', 'Year']
    d_cols = [d for d in data_df.columns if d not in ignore_cols]
    ohlc_cols = [string for string in d_cols if any(sub in string for sub in ['Open', 'High', 'Low', 'Close'])]
    other_cols = [string for string in d_cols if not any(sub in string for sub in ohlc_cols)]
    for d in ohlc_cols:
        data_df.loc[:, d] = data_df.loc[:, d]*100
    for d in other_cols:
        data_df.loc[:, d] = (data_df.loc[:, d] - 1) * 100

    return data_df


def complete_daily_df_work(other_sec, sec_name, data_loc):
    daily_sets = []
    equity_set_cols = []
    print('Working on Daily Data')
    for osec in other_sec + [sec_name]:
        daily_file = f'{osec}_daily_20240505_20040401.txt'
        temp_daily = pd.read_csv(f'{data_loc}\\{daily_file}', sep=",")
        temp_daily = convert_date_time(temp_daily)
        temp_daily = temp_daily.sort_values(by='Date')
        """Need to cut data to the full testing range"""

        temp_daily['ATR_day'] = create_atr(temp_daily)
        temp_daily.drop(columns=['Vol.1', 'Time'], inplace=True)
        print(f'{osec} : Padding Months')

        # Standardize ATR, Vol, OI
        temp_daily = standardize_data(temp_daily)
        for c in ['Open', 'High', 'Low', 'Close', 'Vol', 'OI', 'VolAvg', 'OpenInt', 'ATR_day']:
            temp_daily[c] = temp_daily[c].astype(np.float32)

        temp_daily.rename(columns={
            'Open': f'{osec}_Open',
            'High': f'{osec}_High',
            'Low': f'{osec}_Low',
            'Close': f'{osec}_Close',
            'Vol': f'{osec}_Vol',
            'OI': f'{osec}_OI',
            'VolAvg': f'{osec}_VolAvg',
            'OpenInt': f'{osec}_OpenInt',
            'ATR_day': f'{osec}_ATR_day'
        }, inplace=True)
        daily_sets.append(temp_daily)
        equity_set_cols.append([f'{osec}_Open', f'{osec}_High', f'{osec}_Low', f'{osec}_Close', f'{osec}_Vol',
                               f'{osec}_OI', f'{osec}_VolAvg', f'{osec}_OpenInt', f'{osec}_ATR_day'])

    equity_set_cols = list(itertools.chain(*equity_set_cols))
    if len(daily_sets) > 1:
        dailydf = pd.merge(daily_sets[0], daily_sets[1], on=['Date', 'Datetime', 'Month', 'Year'])
        for df in daily_sets[2:]:
            dailydf = pd.merge(dailydf, df, on=['Date', 'Datetime', 'Month', 'Year'])
    else:
        dailydf = daily_sets[0]

    return dailydf, equity_set_cols


def merge_daily_intraday(dailydf, intraAtr, intraVol, equity_set_cols):
    dailydf = pd.merge(dailydf, intraAtr, on=['Date'], how='left')
    dailydf = pd.merge(dailydf, intraVol, on=['Date'], how='left')
    dailydf = dailydf[~((dailydf['Year'] == 2004) & dailydf['Month'].isin([3, 4, 5, 6, 7, 8]))]
    print(dailydf.columns)
    dailydf = dailydf.sort_values(by='Date')
    equity_set_cols = ['ATR_int', 'Vol_int'] + equity_set_cols
    dailydf = dailydf.loc[:, equity_set_cols + ['Date', 'Month', 'Year']]
    print('Market data merged')

    return dailydf, equity_set_cols


def load_feathers_merge(feather_files, strat_dat_loc):
    dfs = []
    for file in feather_files:
        file_path = str(os.path.join(strat_dat_loc, file))
        df = pd.read_feather(file_path)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.fillna(0, inplace=True)


def create_test_df_dblc(combined_df, dailydf, lower_lookback_df, i,
                        equity_set_cols, start_date, end_date, months_to_train, month_size):
    if i <= 5:
        train_df = create_random_df_percent_rank_dblc(combined_df, lower_lookback_df)
    elif i % 3:
        train_df = create_random_df_percent_rank_dblc(combined_df, lower_lookback_df, rank=.85)
    elif i % 4 == 0:
        train_df = create_random_df_percent_rank_dblc(combined_df, lower_lookback_df, rank=-.85)
    else:
        train_df = create_random_df_percent_rank_dblc(combined_df, lower_lookback_df)

    train_df = pd.concat([train_df, dailydf], axis=1)
    train_df = train_df[(train_df['Date'] >= start_date) & (train_df['Date'] < end_date)]
    train_df = train_df.sort_values(by='Date')
    train_df = train_df.fillna(0)
    train_df = train_df.loc[:, equity_set_cols + ['Rank', 'lookback', 'finalcandleratio', 'fastemalen', 'mincandlepercent',
                                                  'finalcandlepercent', 'stoplosspercent', 'takeprofitpercent',
                                                  'takeProfit_oh', 'daySlowEma_oh']]

    df_length = len(train_df)
    print(f'Test len: {df_length}')
    if df_length < months_to_train * month_size:
        print("Retrying for full df")
        create_test_df_dblc(combined_df, dailydf, lower_lookback_df, i,
                            equity_set_cols, start_date, end_date, months_to_train, month_size)

    return train_df


def create_train_df_dblc2(combined_df, dailydf, lower_lookback_df, i, equity_set_cols, start_date, end_date,
                          months_to_train, month_size, rank_high, rank_low):

    if i % 2 == 0:
        train_df = create_random_df_percent_rank_dblc2(combined_df, lower_lookback_df, rank_high, rank_low)
    else:
        train_df = create_random_df_percent_rank_dblc(combined_df, lower_lookback_df)

    daily_temp = dailydf[(dailydf['Date'] >= start_date) & (dailydf['Date'] < end_date)]
    daily_temp['Year_Month'] = daily_temp['Year'].astype(str) + "_" + daily_temp['Month'].astype(str)
    train_df['Year_Month'] = train_df['Year'].astype(str) + "_" + train_df['Month'].astype(str)

    uniq_months = daily_temp.drop_duplicates(subset=['Year_Month']).reset_index(drop=True)
    uniq_month_list = uniq_months['Year_Month']

    train_df = train_df[train_df['Year_Month'].isin(uniq_month_list)]
    daily_temp = daily_temp[daily_temp['Year_Month'].isin(uniq_month_list)]
    train_df.reset_index(inplace=True, drop=True)
    daily_temp.reset_index(inplace=True, drop=True)

    train_df = pd.concat([train_df, daily_temp], axis=1)
    train_df = train_df.sort_values(by='Date')
    train_df = train_df.fillna(0)
    train_df = train_df.loc[:, equity_set_cols + ['Rank', 'lookback', 'finalcandleratio', 'fastemalen', 'mincandlepercent',
                                                  'finalcandlepercent', 'stoplosspercent', 'takeprofitpercent',
                                                  'takeProfit_oh', 'daySlowEma_oh']]

    df_length = len(train_df)
    print(f'Test len: {df_length}')
    if df_length < months_to_train * month_size:
        print("Retrying for full df")
        create_train_df_dblc2(combined_df, dailydf, lower_lookback_df, i,
                              equity_set_cols, start_date, end_date, months_to_train, month_size, rank_high, rank_low)

    return train_df


def create_test_df_emac(combined_df, dailydf, backup_df, i,
                        daily_cols, start_date, end_date, months_to_train, month_size):
    if i % 3 == 0:
        test_df = create_random_df_percent_rank_emac(combined_df, backup_df, rank=.7)
    elif i % 4 == 0:
        test_df = create_random_df_percent_rank_emac(combined_df, backup_df, rank=-.8)
    else:
        test_df = create_random_df_percent_rank_emac(combined_df, backup_df)

    test_df = pd.merge(dailydf, test_df, on=['Year', 'Month'])

    test_df = test_df.loc[:, daily_cols + ['Rank', 'MinDiffDist', 'FastEmaLen', 'SlowEmaLen', 'MinTicksAway',
                                           'StopLossPercent', 'MinTickAway', 'TakeProfitPercent', 'takeProfit_oh',
                                           'daySlowEma_oh']]

    first_df = test_df[(test_df['Date'] >= start_date) & (test_df['Date'] < end_date)]
    first_df = first_df.sort_values(by='Date')
    first_df = first_df.fillna(0)

    df_length = len(first_df)
    print(f'Test len: {df_length}')

    if df_length < months_to_train * month_size:
        print("Retrying for full df")
        create_test_df_dblc(combined_df, dailydf, backup_df, i,
                            daily_cols, start_date, end_date, months_to_train, month_size)

    return test_df, first_df


def find_min_max_trade_dfs(group_df, min_trades, max_trades):
    temp_df = group_df.loc[(group_df['trades'] >= min_trades) & (group_df['trades'] <= max_trades)]
    if len(group_df) > 10:
        return temp_df
    else:
        return group_df
        # max_trades += 1
        # find_min_max_trade_dfs(group_df, min_trades, max_trades)
        # print(temp_df.loc[0, 'month'])
        # print(max_trades)


def bootstrap_lookbacks(data_df, lookback_range):
    data_df['year_month'] = data_df['Year'].astype(str) + '-' + data_df['Month'].astype(str)
    num_3_lookbacks = data_df.loc[data_df['lookback'] == lookback_range[0]]
    num_high_lookbacks = data_df.loc[data_df['lookback'] >= lookback_range[1]]

    bootstrapped = []
    for yearmonth in data_df['year_month'].unique():
        n_3 = len(num_3_lookbacks.loc[num_3_lookbacks['year_month'] == yearmonth])
        n_4_df = num_high_lookbacks.loc[num_high_lookbacks['year_month'] == yearmonth]
        n_4 = len(n_4_df)
        if n_4 < n_3:
            n_diff = int((n_3 - n_4)/3)
            sample_df = num_high_lookbacks.sample(n=n_diff, replace=True).reset_index(drop=True)
            bootstrapped.append(sample_df)

    bootstrapped = pd.concat(bootstrapped)
    data_df = pd.concat([data_df, bootstrapped], ignore_index=True)
    data_df = data_df.sort_values(by=['Year', 'Month'])
    data_df = data_df.drop(columns=['year_month'])
    data_df.reset_index(drop=True)

    return data_df


def vectorize_data(data_df, tgt_vars, scaler, rank_scaler):
    dat_vals = data_df[tgt_vars].values
    data_scaled = scaler.transform(dat_vals)
    ranks = data_df[['Rank', 'Rank2']].values
    rank_scaled = rank_scaler.transform(ranks)
    data_scaled = np.concatenate((data_scaled, rank_scaled), axis=1)

    return data_scaled


def standardize_data_v4(df, price_cols):
    df.reset_index(inplace=True, drop=True)
    for c in price_cols:
        first_price = df.loc[min(df.index), c]
        temp_arr = df.loc[:, c] / first_price
        df.loc[:, c] = temp_arr.astype(np.float32)

        df.fillna(0, inplace=True)
        df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def separate_train_test(combined_working, daily_working, input_features, test_size):
    combined_working = combined_working.loc[(combined_working['Year'] == combined_working['Year'].iloc[-1]) &
                                            (combined_working['Month'] == combined_working['Month'].iloc[-1])]
    combined_working.reset_index(drop=True, inplace=True)

    daily_working['Rank'] = 0
    daily_working['Rank2'] = 0
    X_train = daily_working[input_features + ['Rank', 'Rank2']]
    X_test = daily_working[input_features + ['Rank', 'Rank2']]

    test_int = int(len(combined_working) * test_size)
    test_idx = np.random.choice(range(0, len(combined_working)), size=test_int, replace=False)
    train_idx = np.array([int(i) for i in range(0, len(combined_working)) if i not in test_idx])

    y_train = combined_working.iloc[train_idx]
    y_test = combined_working.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def subset_by_rank(combineddf, rankhigh, ranklow):
    combineddf = combineddf.loc[(combineddf['Rank'] >= rankhigh) | (combineddf['Rank'] <= ranklow)]

    return combineddf


def prep_ranks(data_df):
    data_df['Rank2'] = 1 / data_df['Rank']
    data_df['Rank2'].replace([np.inf, -np.inf], 100, inplace=True)
    data_df.loc[data_df['Rank2'] > 100, 'Rank2'] = 100

    return data_df


def make_input_dataset(X, y_float, full_size, batch_s):
    y_arr = []
    x_arr = []
    randints = random.sample(range(len(y_float)), int(full_size))
    for i in randints:
        X[:, -2:] = y_float[i, -2:]
        x_arr.append(X)
        y_arr.append(y_float[i, :-2])
    y_arr = np.stack(y_arr, axis=0)
    x_arr = np.stack(x_arr, axis=0)

    y_arr = y_arr.reshape((y_arr.shape[0], 1, y_arr.shape[1]))

    dataset = tf.data.Dataset.from_tensor_slices((x_arr, y_arr))
    dataset = dataset.batch(batch_s)

    return dataset


def make_year_month_combos(start_date, end_date):
    year_month_combinations = []

    current_date = start_date
    while current_date < end_date:
        year_month_combinations.append([current_date.year, current_date.month])
        next_month = current_date.month + 1
        if next_month > 12:
            next_month = 1
            current_date = current_date.replace(year=current_date.year + 1, month=next_month)
        else:
            current_date = current_date.replace(month=next_month)

    return year_month_combinations


def subset_combined_daily_dfs(combineddf, dailydf, year_months):
    """Has a provision that fixes the data for when there is an empty set when attempting to target higher lookbacks"""

    combineddf.reset_index(drop=True, inplace=True)
    dailydf.reset_index(drop=True, inplace=True)

    combined_subset = []
    daily_subset = []
    for ym in year_months:
        year = ym[0]
        month = ym[1]

        temp_cmb = combineddf.loc[(combineddf['Year'] == year) & (combineddf['Month'] == month)]
        combined_subset.append(temp_cmb)

        temp_day = dailydf.loc[(dailydf['Year'] == year) & (dailydf['Month'] == month)]
        daily_subset.append(temp_day)

    combined_subset = pd.concat(combined_subset)
    daily_subset = pd.concat(daily_subset)

    return combined_subset, daily_subset












