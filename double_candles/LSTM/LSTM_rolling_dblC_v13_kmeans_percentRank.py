import math
import random

import numpy as np
import pandas as pd
from algo_tools import algo_data_tools_v3 as at
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input, TimeDistributed, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.regularizers import l2
import tensorflow as tf
import datetime as dt
import itertools
from rand_tools import rand_tools as rt
pd.options.mode.chained_assignment = None  # default='warn'

"""v3:  -Ensures each month is 23 days long
        -Ranks Params as -1 for bad, 0 for mids, 1 for good.
   v4:  -Months are fed as one complete window. Batch size is 4.
   v5:  -Uses K-means clustering to rank
   v5:  -One hot encoding lookback
   v8:  -New ranking system that sorts by PnL per trade
   v9:  -Fixed scaling on X and Y, targeting the right Y values. And did a lot of work to standardize data consistently
   v12: -Attempts to change the ranking and output for each batch
   v13: -Stacking DFs doesn't work that well, attempting to provide a larger dataset to the random batches"""

sec_name = 'CL'

other_sec = ['YM', 'ES']
time_frame = '10min'
time_len = '10years'
sides = ['Bull', 'Bear']
lookback_ranges = [[3, 7]]
min_kepts = [.4, .4]
start_hour = 6

for lookback_range in lookback_ranges:
    for side in sides:
        # Model Params
        n_lookback = 60 * 23
        n_steps = 23
        min_trades = 2
        max_trades = 15
        # K-means Cluster
        min_win_loss_rat = 1.7
        n_clusters = 16
        if side == 'Bull':
            min_kept = min_kepts[0]
        else:
            min_kept = min_kepts[1]

        rank_above = .85
        rank_below = -.5

        kmeans_cols = ['lookback', 'fastemalen', 'stoplosspercent', 'takeprofitpercent', 'dayslow', 'mincandlepercent',
                       'finalcandlepercent', 'finalcandleratio']

        target_float_vars = ['fastemalen', 'stoplosspercent', 'takeprofitpercent', 'mincandlepercent',
                             'finalcandlepercent', 'finalcandleratio', 'lookback', 'takeProfit_oh', 'daySlowEma_oh']

        stop_loss_max = 100
        stop_loss_min = 0
        min_candle_size = 0

        test_size = .2
        month_size = 23
        epochs = 100
        batch_s = 12

        min_dfs = 1
        max_dfs = 10
        max_acceptable_loss = .4
        min_lookback_accuracy = .925

        kmeans_params = at.KMeansParams(
            side=side,
            min_trades=min_trades,
            max_trades=max_trades,
            n_clusters=n_clusters,
            kmeans_cols=kmeans_cols,
            min_win_loss_rat=min_win_loss_rat,
            min_kept=min_kept,
            include_bad_cluster=True,
            name='MainDF'
        )

        lower_lb_params = at.KMeansParams(
            side=side,
            min_trades=0,
            max_trades=100,
            n_clusters=n_clusters,
            kmeans_cols=kmeans_cols,
            min_win_loss_rat=0,
            min_kept=1,
            include_bad_cluster=True,
            name='BackupDF'
        )

        data_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\data'
        strat_dat_loc = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double Candles'
        strat_dat_loc = f'{strat_dat_loc}\\{sec_name}\\{time_frame}\\{time_frame}_test_{time_len}'
        print(strat_dat_loc)
        model_loc = f'{strat_dat_loc}\\k_means_model_{n_lookback}_steps_alleq_LB{lookback_range[0]}{lookback_range[1]}'
        os.makedirs(model_loc, exist_ok=True)
        intra_file = f'{sec_name}_{time_frame}_20240505_20040401.txt'
        model_name = f'{time_frame}_SSR_CuDNN_SL{stop_loss_max}'

        # make lists for rolling through dates
        start_dates = at.make_month_list(dt.datetime(2014, 4, 1), dt.datetime(2019, 5, 1))
        end_dates = at.make_month_list(dt.datetime(2021, 4, 1), dt.datetime(2024, 5, 1))
        end_dates_test = at.make_month_list(dt.datetime(2021, 5, 1), dt.datetime(2024, 6, 1))

        # Set up intraday data to fit with daily data
        intradf = pd.read_csv(f'{data_loc}\\{intra_file}', sep=",")
        print(intradf.columns)
        intradf, intraAtr, intraVol = at.complete_intraday_df_work(intradf, start_hour)

        # Set up daily data
        dailydf, input_features = at.complete_daily_df_work(other_sec, sec_name, data_loc)
        dailydf = dailydf.loc[(dailydf['Date'] >= dt.datetime(2011, 4, 1).date()) &
                              (dailydf['Date'] < end_dates_test[len(end_dates_test) - 1])].reset_index(drop=True)

        # merge daily and intraday
        dailydf, daily_cols = at.merge_daily_intraday(dailydf, intraAtr, intraVol, input_features)
        dailydf = at.pad_months(dailydf, 23)
        dailydf = at.standardize_intradata(dailydf, sec_name)

        print('Loading .feather')
        # EMA Slide data wrangling.
        feather_files = [file for file in os.listdir(strat_dat_loc) if file.endswith('.feather')]

        dfs = []
        for file in feather_files:
            file_path = os.path.join(strat_dat_loc, file)
            df = pd.read_feather(file_path)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.fillna(0, inplace=True)

        for c in ['lookback', 'finalcandleratio', 'fastemalen', 'mincandlepercent', 'finalcandlepercent',
                  'stoplosspercent', 'takeprofitpercent', 'dayslow', 'month', 'trades']:
            combined_df[c] = combined_df[c].astype(int)
        combined_df.rename(columns={'year': 'Year',
                                    'month': 'Month',
                                    'cumPnl': 'CumSum',
                                    'maxDraw': 'MaxDraw'}, inplace=True)

        combined_df['CumSum'] = combined_df['CumSum'].astype(float)
        combined_df['MaxDraw'] = combined_df['MaxDraw'].astype(float)
        combined_df = combined_df[~((combined_df['Year'] == 2004) & combined_df['Month'].isin([3, 4, 5, 6, 7, 8]))]
        lookback_list = sorted(combined_df['lookback'].unique())

        # Display the combined DataFrame
        print("Data results aggregated")
        combined_df['Rank'] = 0.0
        combined_df = combined_df[(combined_df['stoplosspercent'] >= stop_loss_min) &
                                  (combined_df['stoplosspercent'] <= stop_loss_max)]

        combined_df = combined_df[(combined_df['mincandlepercent'] >= min_candle_size)]

        lower_lookback_df = combined_df.copy()

        combined_df = combined_df[(combined_df['lookback'] >= lookback_range[0]) &
                                  (combined_df['lookback'] <= lookback_range[1])]

        """This is important to put in because y_reshape cannot look past the end of its data. Normally, you wouldn't 
        need to prime this, but you want to look 23 days forward at each step, meaning that you would need to handle
        this by putting a larger dataset into y then trimming later"""
        combined_df['Month'] -= 1
        month_zero_mask = combined_df['Month'] == 0
        combined_df.loc[month_zero_mask, 'Month'] = 12
        combined_df.loc[month_zero_mask, 'Year'] -= 1

        kmeans_params.add_dataframe(combined_df)
        kmeans_params.subset_side_trades()
        combined_df = kmeans_params.filter_results()
        combined_df = at.bootstrap_lookbacks(combined_df, lookback_range)

        lower_lb_params.add_dataframe(lower_lookback_df)
        lower_lb_params.subset_side_trades()
        lower_lookback_df = lower_lb_params.filter_results()
        lower_lookback_df = at.bootstrap_lookbacks(lower_lookback_df, [3, 7])

        print(combined_df.shape)
        print(combined_df.columns)
        combined_df = combined_df.drop(columns=['CumSum', 'MaxDraw', 'trades'])

        # One-Hotting DaySlowEma and TakeProfit
        combined_df['takeProfit_oh'] = combined_df['takeprofitpercent'].apply(lambda x: 1 if x > 0 else 0)
        combined_df['daySlowEma_oh'] = combined_df['dayslow'].apply(lambda x: 1 if x > 0 else 0)

        lower_lookback_df['takeProfit_oh'] = lower_lookback_df['takeprofitpercent'].apply(lambda x: 1 if x > 0 else 0)
        lower_lookback_df['daySlowEma_oh'] = lower_lookback_df['dayslow'].apply(lambda x: 1 if x > 0 else 0)

        # start looping through dates
        loss = max_acceptable_loss + 1
        lb_acc = min_lookback_accuracy - 1
        param_list = []

        dailydf = at.standardize_data_kalman(dailydf, input_features)

        y_float_scaler = RobustScaler()
        y_float_scaler.fit(lower_lookback_df[target_float_vars].values)

        for d in range(0, len(end_dates)):
            start_date = start_dates[d]
            end_date = end_dates[d]
            end_date_test = end_dates_test[d]

            months_to_train = at.get_number_months(end_date, start_date)

            i = 1
            tot_df = min_dfs
            df_length = 0
            val_loss_avg = 0
            model_file = f'{model_loc}\\model_{side}_{str(start_date)}_{str(end_date)}_{model_name}.tf'
            while i < max_dfs:
                print(f'\n'
                      f'Side: {side}\n'
                      f'Start: {start_date}, End: {end_date}\n'
                      f'DF number: {i}/{tot_df}\n'
                      f'Batch_s: {batch_s}')

                """Ensures that at least one iteration has all top params, some months with small sets will not have bottom 
                params, so they are untestable for complete bottom sets without some fancier code"""
                # Create Training DF
                train_df = at.create_train_df_dblc2(combined_df, dailydf, lower_lookback_df, i,
                                                    input_features, start_date, end_date, months_to_train,
                                                    month_size, rank_above, rank_below)
                train_df.to_csv(f'{strat_dat_loc}\\train_df.csv')

                df_length = len(train_df)

                # Create Test DF
                y_float = at.vectorize_y2(train_df, target_float_vars, y_float_scaler)

                # y_lookback = OneHotEncoder(sparse_output=False, categories=[lookback_list]).fit_transform(y_lookback)
                # y_takeprofit = OneHotEncoder(sparse_output=False, categories=[[0, 1]]).fit_transform(y_takeprofit)
                # y_dayslow = OneHotEncoder(sparse_output=False, categories=[[0, 1]]).fit_transform(y_dayslow)

                X_train, X_test, y_train, y_test = at.separate_train_test(train_df, y_float, n_steps, n_lookback,
                                                                          test_size=.2,
                                                                          input_features=input_features)
                # X_train_csv = pd.DataFrame(X_train[-1])
                # print(X_train_csv)
                # X_train_csv.to_csv(f'{strat_dat_loc}\\X_train_csv.csv')
                # breakpoint()

                # y_lookback_train, y_lookback_test = (
                #     at.separate_train_test_x(y_lookback, n_steps, n_lookback, test_idx))
                #
                # y_takeprofit_train, y_takeprofit_test = (
                #     at.separate_train_test_x(y_takeprofit, n_steps, n_lookback, test_idx))
                #
                # y_dayslow_train, y_dayslow_test = (
                #     at.separate_train_test_x(y_dayslow, n_steps, n_lookback, test_idx))

                # Build the LSTM model
                if os.path.exists(model_file):
                    # Load the existing model
                    model = load_model(model_file)
                    print("Existing model loaded.")
                else:
                    # Create a new model if the model file doesn't exist
                    initializer1 = tf.keras.initializers.glorot_uniform()
                    initializer2 = tf.keras.initializers.glorot_uniform()
                    initializer3 = tf.keras.initializers.glorot_uniform()
                    initializer4 = tf.keras.initializers.glorot_uniform()

                    print(f'X_shape : {X_train[0].shape}')
                    input_layer = Input(shape=(X_train[0].shape[0], X_train[0].shape[1]))
                    conv_1 = Conv1D(filters=128, kernel_size=3, activation='tanh', padding='same',
                                    name='conv1')(input_layer)
                    lstm_1 = LSTM(units=64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True,
                                  unroll=False, recurrent_dropout=0.0,
                                  kernel_initializer=initializer1,
                                  kernel_regularizer=l2(0.05),
                                  name='lstm1')(conv_1)
                    drop_out1 = Dropout(0.1, name='drop1')(lstm_1)
                    lstm_2 = LSTM(units=128, activation='tanh', recurrent_activation='sigmoid', return_sequences=False,
                                  unroll=False, recurrent_dropout=0.0,
                                  kernel_initializer=initializer2,
                                  kernel_regularizer=l2(0.05),
                                  name='lstm2')(drop_out1)
                    drop_out2 = Dropout(0.1, name='drop2')(lstm_2)
                    dense_1 = Dense(units=64, activation='softplus',
                                    kernel_initializer=initializer3,
                                    name='dense1')(drop_out2)
                    float_output = Dense(units=9, activation='tanh', name='float_output')(dense_1)

                    model = Model(inputs=input_layer,
                                  outputs=[float_output])

                    # Compile the model
                    optimizer = Adam(learning_rate=.005)

                    model.compile(optimizer=optimizer,
                                  loss={'float_output': 'mse'},
                                  metrics={'float_output': 'mse'})
                    print("New model created.")

                loss_avg = []
                val_loss_avg = []
                for epoch in range(epochs):
                    train_dataset = at.make_input_dataset(X_train, y_train, batch_s)
                    for step, (batch_inputs, batch_outputs) in enumerate(train_dataset):
                        # Train on the modified batch
                        with tf.GradientTape() as tape:
                            predictions = model(batch_inputs, training=True)
                            loss = tf.keras.losses.mean_squared_error(batch_outputs, predictions)
                            loss = tf.reduce_mean(tf.concat(loss, axis=0))
                            loss_avg.append(loss)

                        gradients = tape.gradient(loss, model.trainable_variables)
                        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    if epoch % 5 == 0:
                        test_dataset = at.make_input_dataset(X_test, y_test, batch_s)
                        total_loss = 0
                        num_batches = 0
                        for step, (batch_inputs, batch_outputs) in enumerate(test_dataset):
                            predictions = model(batch_inputs, training=False)
                            loss = tf.keras.losses.mean_squared_error(batch_outputs, predictions)
                            total_loss += tf.reduce_mean(loss).numpy()
                            num_batches += 1
                            loss = total_loss / num_batches
                            loss = tf.reduce_mean(tf.concat(loss, axis=0))
                            val_loss_avg.append(loss)

                        loss_avg_out = np.mean([tensor.numpy() for tensor in loss_avg])
                        print(f'Epoch {epoch}/{epochs} : AvgLoss: {loss_avg_out:.5f} '
                              f'val_loss_avg: {val_loss_avg[0]:.5f}\n')

                loss_avg = list(itertools.chain(*loss_avg))
                if np.isnan(loss_avg).any():
                    train_df.to_csv(f'{strat_dat_loc}\\train_df_nan.csv')
                    i -= 1
                    print(f'Model created nan, re-trying: {end_date}')

                else:
                    model.save(model_file)

                if (np.mean(val_loss_avg) < max_acceptable_loss) and (i >= min_dfs):
                    i = max_dfs

                i += 1

            # Test the model
            val_loss_avg = np.mean(val_loss_avg)
            dailydf.sort_values(by='Date', inplace=True)
            fin_testDf = dailydf.loc[:, input_features + ['Date', 'Month', 'Year']]
            fin_testDf = fin_testDf[fin_testDf['Date'] < end_date_test]
            fin_testDf = fin_testDf.tail(n_lookback)
            fin_testDf['Rank'] = 1

            model = load_model(model_file)
            print("Existing model loaded.")

            price_cols = at.get_price_cols(fin_testDf, input_features)
            X_next_month_scaled = at.vectorize_x2(fin_testDf, input_features)
            X_reshaped = tf.expand_dims(X_next_month_scaled, axis=0)

            test_inputs = tf.convert_to_tensor(X_reshaped)
            predictions = model(test_inputs, training=False)
            predictions = predictions[:, -1]

            result_df = at.result_dict_handler_dblc2(predictions, y_float_scaler, target_float_vars, val_loss_avg,
                                                     end_date_test)
            # print(result_df)
            param_list.append(result_df)
            param_df = pd.concat(param_list, axis=1).T
            param_df = param_df.reindex(columns=['lookback', 'finalcandleratio', 'fastemalen', 'mincandlepercent',
                                                 'finalcandlepercent', 'stoplosspercent', 'takeProfit_oh',
                                                 'takeprofitpercent', 'daySlowEma_oh', 'month_to_test', 'model_loss'])
            param_df.to_csv(f'{model_loc}\\param_list_{side}.csv')
