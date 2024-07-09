import time

import numpy as np
import pandas as pd
from algo_tools import algo_data_tools_v4 as at
import os
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (Conv2D, LSTM, Dense, Dropout, Input,
                                     MaxPooling2D, TimeDistributed, Conv1D, MaxPooling1D, Lambda)
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import tensorflow as tf
import datetime as dt
import itertools

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

        subset_rank = True
        rank_above = .65
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
        epochs = 75
        batch_s = 32
        data_per_epoch = 1 #percentage

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

        print(combined_df.shape)
        print(combined_df.columns)

        lower_lb_params.add_dataframe(lower_lookback_df)
        lower_lb_params.subset_side_trades()
        lower_lookback_df = lower_lb_params.create_clusers()
        lower_lookback_df = at.bootstrap_lookbacks(lower_lookback_df, [3, 7])
        # One-Hotting DaySlowEma and TakeProfit
        lower_lookback_df['takeProfit_oh'] = lower_lookback_df['takeprofitpercent'].apply(lambda x: 1 if x > 0 else 0)
        lower_lookback_df['daySlowEma_oh'] = lower_lookback_df['dayslow'].apply(lambda x: 1 if x > 0 else 0)

        # start looping through dates
        param_list = []

        dailydf = at.standardize_data_kalman(dailydf, input_features)

        y_float_scaler = RobustScaler()
        y_float_scaler.fit(lower_lookback_df[target_float_vars].values)

        for d in range(0, len(end_dates)):
            start_date = start_dates[0]
            end_date = end_dates[d]
            end_date_test = end_dates_test[d]

            months_to_train = at.get_number_months(end_date, start_date)
            n_lookback = months_to_train * 23

            model_loc = \
                f'{strat_dat_loc}\\k_means_model_{n_lookback}_steps_alleq_LB{lookback_range[0]}{lookback_range[1]}'
            model_file = f'{model_loc}\\model_{side}_{str(start_date)}_{str(end_date)}_{model_name}.tf'
            os.makedirs(model_loc, exist_ok=True)

            year_months = at.make_year_month_combos(start_date, end_date)
            combined_working, daily_working = at.subset_combined_daily_dfs(combined_df, dailydf, year_months)

            kmeans_params.add_dataframe(combined_working)
            kmeans_params.subset_side_trades()
            combined_working = kmeans_params.create_clusers()
            combined_working = at.bootstrap_lookbacks(combined_working, lookback_range)

            combined_working = combined_working.drop(columns=['CumSum', 'MaxDraw', 'trades'])

            # One-Hotting DaySlowEma and TakeProfit
            combined_working['takeProfit_oh'] = combined_working['takeprofitpercent'].apply(lambda x: 1 if x > 0 else 0)
            combined_working['daySlowEma_oh'] = combined_working['dayslow'].apply(lambda x: 1 if x > 0 else 0)

            if subset_rank:
                combined_working = at.subset_by_rank(combined_working, rank_above, rank_below)

            X_train, X_test, y_train, y_test = at.separate_train_test(combined_working, daily_working, input_features,
                                                                      target_float_vars, test_size=.1)

            x_float_scaler = RobustScaler()
            x_float_scaler.fit(X_train[input_features].values)

            X_train = at.vectorize_data(X_train, input_features, x_float_scaler)
            X_test = at.vectorize_data(X_test, input_features, x_float_scaler)
            y_train = at.vectorize_data(y_train, target_float_vars, y_float_scaler)
            y_test = at.vectorize_data(y_test, target_float_vars, y_float_scaler)

            full_size = int(y_train.shape[0]*data_per_epoch)

            print(f'\n'
                  f'Side: {side}\n'
                  f'Start: {start_date}, End: {end_date}\n'
                  f'Batch_s: {batch_s}')

            """The plan is to add all available data to the model, then split along Ys and Rankings"""
            # Create Training DFs

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

                print(f'X_shape : {X_train.shape}')
                print(f'Y_shape : {y_train.shape}')
                input_layer = Input(shape=(X_train.shape[0], X_train.shape[1]))
                conv_1 = TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='tanh', padding='same',
                                                name='conv1'))(input_layer)
                reshape_layer = Lambda(lambda x: K.reshape(x, (-1, K.shape(x)[2], 128)))(conv_1)
                lstm_1 = LSTM(units=256, activation='tanh', recurrent_activation='sigmoid', return_sequences=True,
                              unroll=False, recurrent_dropout=0.0,
                              kernel_initializer=initializer1,
                              kernel_regularizer=l2(0.05),
                              name='lstm1')(reshape_layer)
                drop_out1 = Dropout(0.15, name='drop1')(lstm_1)
                lstm_2 = LSTM(units=128, activation='tanh', recurrent_activation='sigmoid', return_sequences=False,
                              unroll=False, recurrent_dropout=0.0,
                              kernel_initializer=initializer2,
                              kernel_regularizer=l2(0.05),
                              name='lstm2')(drop_out1)
                drop_out2 = Dropout(0.15, name='drop2')(lstm_2)
                dense_1 = Dense(units=64, activation='softplus',
                                kernel_initializer=initializer3,
                                name='dense1')(drop_out2)
                float_output = Dense(units=9, activation='tanh', name='float_output')(dense_1)

                model = Model(inputs=input_layer,
                              outputs=[float_output])

                # Compile the model
                optimizer = Adam(learning_rate=.01)

                model.compile(optimizer=optimizer,
                              loss={'float_output': 'mse'},
                              metrics={'float_output': 'mse'})
                print("New model created.")

            loss_avg = []
            val_loss_avg = []
            for epoch in range(epochs):
                loss_avg = []
                start_time = time.time()
                train_dataset = at.make_input_dataset(X_train, y_train, full_size, batch_s)
                for step, (batch_inputs, batch_outputs) in enumerate(train_dataset):
                    # Train on the modified batch
                    with tf.GradientTape() as tape:
                        predictions = model(batch_inputs, training=True)
                        loss = tf.keras.losses.mean_squared_error(batch_outputs, predictions)
                        loss = tf.reduce_mean(tf.concat(loss, axis=0))
                        loss_avg.append(loss)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                if epoch % 2 == 0:
                    val_loss_avg = []
                    test_dataset = at.make_input_dataset(X_test, y_test, full_size=10, batch_s=batch_s)
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
                    val_loss_avg_out = np.mean([tensor.numpy() for tensor in val_loss_avg])
                    lapsed_time = time.time() - start_time
                    print(f'Epoch {epoch+2}/{epochs} : AvgLoss: {loss_avg_out:.5f} '
                          f'val_loss_avg: {val_loss_avg_out:.5f} Seconds: {lapsed_time:.2f}\n')

            loss_avg_out = np.mean([tensor.numpy() for tensor in loss_avg])
            val_loss_avg_out = np.mean([tensor.numpy() for tensor in val_loss_avg])

            model.save(model_file)

            # Test the model
            dailydf.sort_values(by='Date', inplace=True)
            fin_testDf = dailydf.loc[:, input_features + ['Date', 'Month', 'Year']]
            fin_testDf = fin_testDf[fin_testDf['Date'] < end_date_test]
            fin_testDf = fin_testDf.tail(n_lookback)
            fin_testDf['Rank'] = 1

            model = load_model(model_file)
            print("Existing model loaded.")

            x_test_scaled = RobustScaler()
            X_next_month_scaled = at.vectorize_data(fin_testDf, input_features, x_test_scaled)

            test_inputs = tf.convert_to_tensor(X_next_month_scaled)
            X_next_month_scaled = (
                tf.reshape(X_next_month_scaled, (-1, X_next_month_scaled.shape[0], X_next_month_scaled.shape[1])))
            predictions = model(X_next_month_scaled, training=False)
            # print(predictions)
            # predictions = predictions[:, -1]

            result_df = at.result_dict_handler_dblc2(predictions, y_float_scaler, target_float_vars, val_loss_avg_out,
                                                     end_date_test)
            # print(result_df)
            param_list.append(result_df)
            param_df = pd.concat(param_list, axis=1).T
            param_df = param_df.reindex(columns=['lookback', 'finalcandleratio', 'fastemalen', 'mincandlepercent',
                                                 'finalcandlepercent', 'stoplosspercent', 'takeProfit_oh',
                                                 'takeprofitpercent', 'daySlowEma_oh', 'month_to_test', 'model_loss'])
            param_df.to_csv(f'{model_loc}\\param_list_{side}.csv')
