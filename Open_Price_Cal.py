from datetime import date, timedelta
import datetime
import copy
import h5py
import joblib
import math
import numpy as np
import os
import pandas as pd
import keras as k
import csv
import time
import datetime as dt
import urllib.request, json
from pandas_datareader import data
import re
import pytz
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
import requests
np.random.seed(seed_value)
import tensorflow as tf
tf.set_random_seed(seed_value)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#keras used for LSTM
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import LeakyReLU
from keras.layers import Activation, Dense
import warnings
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
def time_cal(now):
    import datetime
    print("utc time-----",now)
    hour = now.strftime("%H")
    hour = int(hour)
    minute = now.strftime("%M")
    minute = int(minute)
    year = now.strftime("%Y")
    year = int(year)
    mon = now.strftime("%m")
    mon = int(mon)
    day = now.strftime("%d")
    day = int(day)
    est = pytz.timezone('US/Eastern')
    utc = pytz.utc
    fmt = '%Y-%m-%d %H:%M %Z%z'
    winter = datetime.datetime(year, mon, day, hour, minute, tzinfo=utc)
    txt = winter.astimezone(est).strftime(fmt)
    print(txt)
    x = txt.split()
    Time = x[1]
    print("est time----",Time)
    Date = x[0]
    print("current_date",Date)
    return Time,Date

def date_cal(symbol):
    try:
        file2 = "stock_market_data-%s.csv"%symbol
        dataset = pd.read_csv('/opt/sma-files/sma-stock-csv-files/'+file2)
        dataset.drop("Unnamed: 0", axis = 1,inplace=True)
        B = dataset.iloc[:,0].values
        A = B[-1]
        import datetime
        cr_date = datetime.datetime.strptime(A, '%Y-%m-%d')
        start_date = cr_date + timedelta(1)
        last_date = start_date + timedelta(12)
        def daterange(date1, date2):
            for n in range(int ((date2 - date1).days)+1):
                yield date1 + timedelta(n)

        start_dt = start_date
        end_dt = last_date
        weekdays = [6,7]
        new = []
        for dt in daterange(start_dt, end_dt):
            if dt.isoweekday() not in weekdays:
                New = dt.strftime("%Y-%m-%d")
                Holiday = ['2019-11-28','2019-12-25','2020-01-01','2020-01-20','2020-02-17','2020-04-20','2020-04-10','2020-05-25','2020-07-03','2020-09-07','2020-11-26',
                           '2020-12-25','2021-01-01','2021-01-18','2021-02-15','2021-04-02','2021-05-31','2021-07-05','2021-09-06','2021-11-25','2021-12-24']
                if New in Holiday:
                    pass
                else:
                    new.append(New)
        final_date = new[0:5]
    except:
        pass
    return final_date

def token_generation():
    url = 'https://apis.tradetipsapp.com/api/auth/appSignIn?userName=govinda&password=govinda'
    headers = {"Content-Type": "json"}
    response = requests.post(url, headers = headers)
    data = response.json()
    AAA = data['accessToken']
    return AAA
Holiday = ['2019-11-28','2019-12-25','2020-01-01','2020-01-20','2020-02-17','2020-04-20','2020-04-10','2020-05-25','2020-07-03','2020-09-07','2020-11-26',
           '2020-12-25','2021-01-01','2021-01-18','2021-02-15','2021-04-02','2021-05-31','2021-07-05','2021-09-06','2021-11-25','2021-12-24']
now = datetime.now()
time,Date = time_cal(now)
print(Date)
holiday1 =[]
for j in Holiday:
    if j == Date:
        holiday1.append(j)
print(holiday1)
list_symbol = ["MSFT","AAPL","JPM","GOOG","SPY","IWM","QQQ","DIA"]
#list_symbol = ["QQQ"]
list_main = list_symbol[0:8]
for symbol in list_main:
    try:
        if len(holiday1)!=0:
            url = 'https://apistest.tradetipsapp.com/api/GraphParameter/getNormalPriceByStockName'
            AAA = token_generation()
            token = AAA
            PARAMS = {"stockName" : symbol}
            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
            response = requests.post(url,params = PARAMS , headers=headers)
            data = response.json()
            print(data)
            list1 = []
            for i in data:
                A = i['time']
                print(A)
                if A == '15:50':
                    list1.append(i)
            print(list1)
            if len(list1)!=0:
                last_entry = list1[-1]
                plot_normalized = last_entry['normalizedPrice']
                print(plot_normalized)
                date = date_cal(symbol)
                print(date)
                file2 = "stock_market_data-%s.csv"%symbol
                final_date = pd.read_csv('/opt/sma-files/sma-stock-csv-files/'+file2)
                final_date = final_date.tail(1)
                A = final_date["Date"].values
                last_date = A[0]
                date.insert(0, last_date)
                print("holiday date------------")
                print(date)
                plot_vxx = last_entry['vxxaffected']
                print(plot_vxx)
                plot_news1 = last_entry['newsAffected']
                print(plot_news1)
                AAA = token_generation()
                token = AAA
##                sector_url = 'https://apis.tradetipsapp.com/api/GraphParameter/addGraphParameter'
##                PARAMS = {"stockName": symbol,
##                          "normalizedPrice":plot_normalized,
##                          "dates":date,
##                          "VXXAffected": plot_vxx,
##                          "newsAffected": plot_news1,
##                          "time": '09:30'
##                            }
##                headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
##                response = requests.post(sector_url, params = PARAMS, headers=headers)
##                data = response.json()
##                print(data)
                sector_url = 'https://apistest.tradetipsapp.com/api/GraphParameter/addGraphParameter'
                PARAMS = {"stockName": symbol,
                          "normalizedPrice":plot_normalized,
                          "dates":date,
                          "VXXAffected": plot_vxx,
                          "newsAffected": plot_news1,
                          "time": '09:30'
                            }
                headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
                response = requests.post(sector_url, params = PARAMS, headers=headers)
                data = response.json()
                print(data)
            else:
                pass
        else:
            print("Symbol--------------------")
            print(symbol)
            look_back    = 200
            forward_days = 5
            num_periods  = 205
            train_fraction = float(2.0 / 3.0)
            #print("train_fraction-------",train_fraction)
            if train_fraction > 1. or train_fraction < 0.:
                raise ValueError("ERROR:: train_fraction must have a value between 0 and 1")
            test_fraction = (1 - train_fraction)
            #print("test fraction--------",test_fraction)
            assert(test_fraction <= 1.0 and test_fraction > 0.)
            use_stateful_model = True
            file = "stock_market_data-%s.csv"%symbol
            dataset = pd.read_csv('/opt/sma-files/sma-stock-csv-files/'+file)
            #Del unnamed column from dataset variable Unnamed: 0
            dataset.drop("Unnamed: 0", axis = 1,inplace=True)
            print("dataset-----------------")
            print(dataset)
            #Assert that open, high, low, close are numeric types
            assert (dataset["Open"].dtype  == np.dtype('float64') or dataset["Open"].dtype  == np.dtype('float32'))
            assert (dataset["High"].dtype  == np.dtype('float64') or dataset["High"].dtype  == np.dtype('float32'))
            assert (dataset["Low"].dtype   == np.dtype('float64') or dataset["Low"].dtype   == np.dtype('float32'))
            assert (dataset["Close"].dtype == np.dtype('float64') or dataset["Close"].dtype == np.dtype('float32'))

            dataset["Date"] = pd.to_datetime(dataset["Date"].values, format="%Y-%m-%d")

            assert(pd.api.types.is_datetime64_any_dtype(dataset["Date"]))
            dataset_close = dataset[["Date", "Close"]]
            def difference(dataset, interval=1):
                diff = list()
                for i in range(interval, len(dataset)):
                    value = dataset[i] - dataset[i - interval]    #  for difference in close data
                    diff.append(value)
                return pd.Series(diff)
            dataset_close_diffed = difference(dataset_close["Close"], 1)
            def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
                n_vars = 1 if (type(data) is list or type(data) is np.ndarray) else data.shape[1]
                df = pd.DataFrame(data)
                cols, names = list([]), list([])
                # input sequence (t-n, ... t-1)
                for i in range(n_in, 0, -1):
                    cols.append(df.shift(i))
                    names += [('var%d_t_m_%d' % (j + 1, i)) for j in range(n_vars)]
                for i in range(0, n_out):
                    cols.append(df.shift(-i))
                    if i == 0:
                        names += [('var%d_t' % (j + 1)) for j in range(n_vars)]
                    else:
                        names += [('var%d_t_pl%d' % (j + 1, i)) for j in range(n_vars)]
                agg = pd.concat(cols, axis=1)
                agg.columns = names
                if dropnan:
                    agg.dropna(inplace=True)
                return agg
            def prepare_data(series, n_test, n_lag, n_seq):
                # extract raw values
                raw_values = series.values
                # transform data to be stationary
                diff_series = difference(raw_values, 1)
                diff_values = diff_series.values
                diff_values = diff_values.reshape(len(diff_values), 1)
                # rescale values to -1, 1
                scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
                scaled_values = scaler.fit_transform(diff_values)
                scaled_values = scaled_values.reshape(len(scaled_values), 1)
                # transform into supervised learning problem X, y
                supervised = series_to_supervised(scaled_values, n_in=n_lag, n_out=n_seq, dropnan=True)
                supervised_values = supervised.values
                # split into train and test sets
                train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
                return scaler, train, test

            num_test_start_points  = int(np.floor(dataset_close["Close"].shape[0] * test_fraction))
            num_train_start_points = dataset_close.shape[0] - num_test_start_points
            data_scaler, dataset_train_close, dataset_test_close =            prepare_data(dataset_close["Close"],
                             n_test=int(np.floor(dataset_close["Close"].shape[0] * test_fraction)),
                             n_lag=int(look_back),
                             n_seq=int(forward_days))
            X_train_close = [] #np.empty(shape=(0, look_back, 1))
            y_train_close = [] #np.empty(shape=(0, forward_days, 1))
            X_test_close = [] #np.empty(shape=(0, look_back, 1))
            y_test_close = [] #np.empty(shape=(0, forward_days, 1))
            #subscript t in this case is to denote training dataset
            X_t, y_t = dataset_train_close[:, 0:look_back], dataset_train_close[:, look_back:]
            X_train_close = X_t.reshape(X_t.shape[0], 1, X_t.shape[1])
            y_train_close = y_t
            #v is for validation, just another suffix for "test" dataset
            X_v, y_v = dataset_test_close[:, 0:look_back], dataset_test_close[:, look_back:]
            X_test_close = X_v.reshape(X_v.shape[0], 1, X_v.shape[1])
            y_test_close = y_v
            #fit an LSTM network to training data
            def build_lstm_network(X_seq_length,
                                   X_feature_length,
                                   y_seq_length,
                                   num_sequences_per_batch,
                                   num_cell_units,
                                   keep_state=False,
                                   return_seqs=True):
                # design network
                model = Sequential()
                model.add(LSTM(units=num_cell_units, batch_input_shape=(num_sequences_per_batch, X_seq_length, X_feature_length), stateful=keep_state))
                model.add(Dense(y_seq_length))
                #model.add(LeakyReLU(alpha=0.01))
                model.compile(loss='mean_squared_error', optimizer='sgd')
                return model

            multistep_model_close = build_lstm_network(X_seq_length=X_train_close.shape[1],
                                                      X_feature_length=X_train_close.shape[2],
                                                      y_seq_length=y_train_close.shape[1],
                                                      num_sequences_per_batch=1,
                                                      num_cell_units=12,
                                                      keep_state=use_stateful_model)

            #Set the number of epochs for training
            NUM_EPOCHS_TRAINING = 128

            if use_stateful_model:
                training_history = []
                for i in range(NUM_EPOCHS_TRAINING):
                    print("i = {}".format(i))
                    training_history.append(multistep_model_close.fit(X_train_close, y_train_close, epochs=1, batch_size=1, verbose=1, shuffle=False))
                    multistep_model_close.reset_states()
            else:
                training_history = multistep_model_close.fit(X_train_close, y_train_close, epochs=NUM_EPOCHS_TRAINING, batch_size=0, validation_split=0.25, verbose=1, shuffle=False)

            training_losses = [x.history["loss"][0] for x in training_history]
            num_test_examples = X_test_close.shape[0]
            def forecast_lstm(model, X, n_batch):
                X_feature_length = X.shape[1]
                if X_feature_length is None:
                    raise ValueError("ERROR:: vector length for input is None")
                elif X_feature_length != look_back:
                    raise ValueError("ERROR:: vector length for input is not equal to look back length of {}".format(look_back))
                X_f = X.reshape(1, 1, X_feature_length)
                # make forecast
                forecast = model.predict(X_f, batch_size=n_batch)
                print(forecast)
                # convert to array
                return [x for x in forecast[0, :]]

            #evaluate the persistence model
            def make_forecasts(model, n_batch, train, test, n_test_instances, n_lag, n_seq):
                forecasts = []
                for i in range(n_test_instances):
                    forecast = forecast_lstm(model, test[i], n_batch=int(1))
                    forecasts.append(forecast)
                return forecasts

            #invert differenced forecast
            def inverse_difference(last_ob, forecast):
                # invert first forecast
                inverted = list()
                inverted.append(forecast[0] + last_ob)
                for i in range(1, len(forecast)):
                    inverted.append(forecast[i] + inverted[i - 1])
                return inverted

            #inverse data transform on forecasts
            def inverse_transform(series, forecasts, scaler, n_test):
                inverted = list()

                for i in range(len(forecasts)):
                    forecast = np.array(forecasts[i])
                    forecast = forecast.reshape(1, len(forecast))
                    inv_scale = scaler.inverse_transform(forecast)
                    inv_scale = inv_scale[0, :]
                    index = len(series) - n_test + i - 1
                    last_ob = series.values[index]
                    inv_diff = inverse_difference(last_ob, inv_scale)
                    # store
                    inverted.append(inv_diff)
                return inverted
            #Run the training set through the predictions
            forecasts = make_forecasts(multistep_model_close,
                                       n_batch=1,
                                       train=X_train_close,
                                       test=X_test_close,
                                       n_test_instances=X_test_close.shape[0],
                                       n_lag=look_back,
                                       n_seq=forward_days)

            inverted_forecasts = inverse_transform(dataset_close["Close"],
                                                   forecasts,
                                                   scaler=data_scaler,
                                                   n_test=X_test_close.shape[0] + forward_days - 1)

            actual = [row[look_back:] for row in dataset_test_close]
            inverted_actual = inverse_transform(dataset_close["Close"],
                                                actual,
                                                scaler=data_scaler,
                                                n_test=X_test_close.shape[0] + forward_days - 1)

            def mean_absolute_percentage_error(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)

                return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

            #evaluate the RMSE for each forecast time step
            def evaluate_forecasts(test, forecasts, n_lag, n_seq, metric="rmse"):
                if metric.lower() != "rmse" and metric.lower() != "mape":
                    raise ValueError("ERROR:: model evaluation metric not")

                prediction_errs = []
                for i in range(n_seq):
                    print("i = {}".format(i))
                    actual = [row[i] for row in test]
                    predicted = [forecast[i] for forecast in forecasts]
                    if metric.lower() == "rmse":
                        err = np.sqrt(sklearn.metrics.mean_squared_error(actual, predicted))
                    elif metric.lower() == "mape":
                        err = mean_absolute_percentage_error(actual, predicted)
                    else:
                        raise ValueError("ERROR:: model evaluation metric must be set to either \"rmse\" or \"mape\"")
                    print('t+%d RMSE: %f' % ((i+1), err))
                    prediction_errs.append(err)

                return prediction_errs
            multistep_model_rmse_metrics = evaluate_forecasts(inverted_actual,
                                                              inverted_forecasts,
                                                              n_lag=look_back,
                                                              n_seq=forward_days,
                                                              metric="rmse")

            multistep_model_mape_metrics = evaluate_forecasts(inverted_actual,
                                                              inverted_forecasts,
                                                              n_lag=look_back,
                                                              n_seq=forward_days,
                                                              metric="mape")

            prediction_start_dates = dataset_close[num_train_start_points:]["Date"].values
            prediction_start_dates = np.reshape(prediction_start_dates, (prediction_start_dates.shape[0], 1)).astype(np.float64)
            price_on_start_dates = dataset_close[num_train_start_points:]["Close"].values
            price_on_start_dates = np.reshape(price_on_start_dates, (price_on_start_dates.shape[0], 1)).astype(np.float64)
            actual_price_start_date_plus_five = dataset_close["Close"].values[num_train_start_points+forward_days:]
            forecast_shift = len(inverted_forecasts) - actual_price_start_date_plus_five.shape[0]
            for j in range(forecast_shift):
                actual_price_start_date_plus_five = np.append(actual_price_start_date_plus_five, np.nan)
            actual_price_start_date_plus_five = np.reshape(actual_price_start_date_plus_five, (actual_price_start_date_plus_five.shape[0], 1))
            assert(actual_price_start_date_plus_five.shape[0] == len(inverted_forecasts))
            inverted_forecasts_as_np = np.array(inverted_forecasts)
            inverted_forecasts_with_dates = np.concatenate((prediction_start_dates,
                                                            price_on_start_dates,
                                                            actual_price_start_date_plus_five,
                                                            inverted_forecasts_as_np),
                                                            axis=1)
            df_forecast_five_day = pd.DataFrame(data=inverted_forecasts_with_dates,
                                                columns=["Date",
                                                         "Actual_close",
                                                         "Actual_close_plus_5",
                                                         "Pred_close_plus_1",
                                                         "Pred_close_plus_2",
                                                         "Pred_close_plus_3",
                                                         "Pred_close_plus_4",
                                                         "Pred_close_plus_5"])

            df_forecast_five_day["Date"] = pd.to_datetime(df_forecast_five_day["Date"], unit="ns")
            numeric_columns = [q for q in df_forecast_five_day.columns.tolist() if q != "Date"]
            df_forecast_five_day = df_forecast_five_day.round(decimals=2)
            ud_array = []
            up_array = []
            dn_array = []
            for index, row in df_forecast_five_day.iterrows():
                 up   = ((row["Actual_close_plus_5"] - row["Actual_close"] >= 0) and ((row["Pred_close_plus_5"] - row["Actual_close"] >= 0)))
            down = ((row["Actual_close_plus_5"] - row["Actual_close"] <= 0) and ((row["Pred_close_plus_5"] - row["Actual_close"] <= 0)))
            ud_array = []
            up_array = []
            dn_array = []
            for index, row in df_forecast_five_day.iterrows():
                up   = ((row["Actual_close_plus_5"] - row["Actual_close"] >= 0) and ((row["Pred_close_plus_5"] - row["Actual_close"] >= 0)))
                down = ((row["Actual_close_plus_5"] - row["Actual_close"] <= 0) and ((row["Pred_close_plus_5"] - row["Actual_close"] <= 0)))
                ud = int(up or down)
                ud_array.append(ud)
                if up:
                    up_array.append(up)
                elif down:
                    dn_array.append(down)
            ud_array_as_np = np.array(ud_array)
            ud_array_as_np = np.reshape(ud_array_as_np, (ud_array_as_np.shape[0], 1)).astype(np.int32)
            df_forecast_five_day["ud"] = ud_array_as_np
            batting_average = float(df_forecast_five_day[df_forecast_five_day["ud"] == 1].shape[0] / df_forecast_five_day.shape[0])
            print("last 6 rows------------------------")
            print(df_forecast_five_day.tail(6))
            final_date = date_cal(symbol)
            print("final_date---------------------------")
            print(final_date)
            aa = pd.DataFrame(data = final_date)
            QQ = df_forecast_five_day.tail(1)
            predicted_stock_price_c=[]
            for i in QQ:
                if i ==("Date") or i==("Actual_close") or i==("Actual_close_plus_5") or i==("ud"):
                    pass
                else:
                    A = QQ[i].values
                    final_close = A[0]
                    print(final_close)
                    predicted_stock_price_c.append(final_close)
            print(predicted_stock_price_c)
            aa['Predicted_Stock_Price_Close'] = predicted_stock_price_c
            aa.columns = ["Dates","Predicted_Stock_Price_Close"]
            print("final table--------------------")
            print(aa)
            #url_string =  ("https://cloud.iexapis.com/stable/stock/market/batch?types=quote&symbols="+symbol+"&token=pk_dd324da3fb5f4428a47b05ab12f23ce2".format(symbol))
            url_string =  ("https://sandbox.iexapis.com/stable/stock/market/batch?types=quote&symbols="+symbol+"&token=Tpk_b5eb4116bd1542f99177767878ad0b55".format(symbol))
            file_to_save = 'stock_market_data-%s.csv'%symbol
            url= urllib.request.urlopen(url_string)
            data = json.loads(url.read().decode())
            print("data--------------")
            print(data)
            final_open = (data[symbol]['quote']['latestPrice'])
            print("open(latest price)----------------------")
            print(final_open)
            aa['Actual_Open'] = ''
            aa['Actual_Open'][0] = final_open
            aa['minus'] = ''
            aa['minus'][0] =  aa['Actual_Open'][0] - aa['Predicted_Stock_Price_Close'][0]
            import random
            DD = random.uniform(0.85,0.96)      #taking range between 85% to 95% means 0.85 to 0.95
            aa['diffrence'] = ''
            aa['diffrence'][0] = (aa['minus'][0]*DD)
            aa['Normalized_Price'] = round((aa["Predicted_Stock_Price_Close"] + aa['diffrence'][0]),2)
            print(aa)
            vxx = pd.read_csv("/opt/sma-files/sma-stock-csv-files/stock_market_data-VXX.csv")
            vxx.drop("Unnamed: 0",axis = 1,inplace = True)
            print("Vxx------------------")
            print(vxx)
            vxx_next_day = vxx.tail(5)
            vxx = vxx.drop(vxx.tail(5).index)
            vxx_next_day_open = vxx_next_day.iloc[0:1,1:2].values
            vxx_prev_close = vxx.iloc[-1:,4:5].values
            vxx_percent_change = (vxx_next_day_open - vxx_prev_close)/vxx_prev_close * 100
            vxx_percent_change =  round(float(vxx_percent_change),2)
            print("vxx_percent_change-------------")
            print(vxx_percent_change)
            A1 = abs(vxx_percent_change)
            B1 = A1+3.00
            C1 = (round(B1,0))
            D1 = C1 + 0.001
            change = np.arange(0,D1,.01)
            change_factor = np.linspace(0,3.20,len(change))        #now it's 93% 3/3.2 = 0.93 ~ 93% #71% means 3/4.1 = 0.71 ~ 71%
            list_change = list(zip(change,change_factor))
            def re_round(li, _prec=3):
                try:
                    return round(li, _prec)
                except TypeError:
                    return type(li)(re_round(x, _prec) for x in li)

            list_change = re_round(list_change)
            l = pd.DataFrame(list_change, columns=["Change","Change_Factor"])
            abs(vxx_percent_change)
            xql = l[l["Change"].values == abs(vxx_percent_change)]
            change_multiply_value = xql["Change_Factor"].values
            change_multiply_value = float(change_multiply_value)
            change_multiply_value = round(change_multiply_value,2)/100
            print("change_multiply_value------------------")
            print(change_multiply_value)
            if vxx_percent_change <0 and vxx_percent_change >= -8:
                aa['Predicted_VXX'] = aa["Normalized_Price"] + round(aa["Normalized_Price"]*change_multiply_value,2)
            elif vxx_percent_change >=0 and vxx_percent_change <=8:
                aa['Predicted_VXX'] = aa["Normalized_Price"] - round(aa["Normalized_Price"]*change_multiply_value,2)
            elif vxx_percent_change > 8 or vxx_percent_change < -8:
                aa['Predicted_VXX'] = aa["Normalized_Price"] + round(aa["Normalized_Price"]*.015,2)
            print(aa)
            def words_remove(new):
                try:
                    while (new.count("null")):
                        new.remove("null")
                    while (new.count("")):
                        new.remove("")
                    while (new.count("None")):
                        new.remove("None")
                    while (new.count(None)):
                        new.remove(None)
                except:
                    pass
                return new

            AAA = token_generation()
            url = 'https://apis.tradetipsapp.com/api/sectorNewsSentiment/getSentimentandSMAByStockSymbolResultSet'
            token = AAA
            PARAMS = {"stockSymbol" : symbol}
            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
            response = requests.post(url,params = PARAMS , headers=headers)
            data = response.json()
            print(data)
            new = []
            for x in data:
                new.append(x['stock_sentiment'])
            remove = words_remove(new)
            print(new)
            print(remove)
            if len(new)==0:
                new = []
                for x in data:
                    new.append(x['news_sector_sentiment'])
                print("+++++++++++++++++")
                print(new)
                remove = words_remove(new)
                print(remove)
                print(len(remove))
                news = remove[0]
                print("----------------")
                print(news)
            else:
                news = remove[0]
                print("++++++++++++++")
                print(news)
            print(news)
            if news == "Very Negative":
                aa['News_Affected_Close'] = aa["Normalized_Price"] - round(aa["Normalized_Price"]*.0150,2)
            elif news == "Negative":
                aa['News_Affected_Close'] = aa["Normalized_Price"] - round(aa["Normalized_Price"]*.0100,2)

            elif news == "Neutral":
                aa['News_Affected_Close'] = aa["Normalized_Price"] - round(aa["Normalized_Price"]*.0006,2)
            elif news == "Positive":
                aa['News_Affected_Close'] = aa["Normalized_Price"] + round(aa["Normalized_Price"]*.0100,2)
            elif news == "Very Positive":
                aa['News_Affected_Close'] = aa["Normalized_Price"] + round(aa["Normalized_Price"]*.0150,2)
            print(aa)
            aa.drop(["Actual_Open", "minus","diffrence"],axis = 1,inplace=True)
            print(aa)
            last_value = dataset.tail(1)["Close"].values
            print("last_value--------------------------")
            print(last_value)
            from functools import reduce
            plot_normalized = aa.iloc[:,2:3].values
            print("plot_normalized---------------------")
            print(plot_normalized)
            plot_normalized = np.insert(plot_normalized,0,last_value,axis=0)
            plot_normalized = [i[0] for i in plot_normalized]
            def re_round(li, _prec=3):
                try:
                    return round(li, _prec)
                except TypeError:
                    return type(li)(re_round(x, _prec) for x in li)
            plot_normalized = re_round(plot_normalized)
            plot_normalized = ",".join(str(x) for x in plot_normalized)
            plot_vxx = aa.iloc[:,3:4].values
            plot_vxx = np.insert(plot_vxx,0,last_value,axis=0)
            plot_vxx = [i[0] for i in plot_vxx]
            plot_vxx = re_round(plot_vxx)
            plot_vxx = ",".join(str(x) for x in plot_vxx)
            print("plot_vxx---------------------")
            print(plot_vxx)
            #print("values for News Affected graph")
            plot_news1 = aa.iloc[:,4:5].values
            plot_news1 = np.insert(plot_news1,0,last_value,axis=0)
            plot_news1 = [i[0] for i in plot_news1]
            plot_news1 = re_round(plot_news1)
            plot_news1 = ",".join(str(x) for x in plot_news1)
            print("plot_news1------------------")
            print(plot_news1)
            file2 = "stock_market_data-%s.csv"%symbol
            final_date = pd.read_csv('/opt/sma-files/sma-stock-csv-files/'+file2)
            final_date = final_date.tail(1)
            A = final_date["Date"].values
            last_date = A[0]
            date = aa.iloc[:,0:1].values
            date = np.insert(date,0,last_date,axis=0)
            date = [i[0] for i in date]
            date = ",".join(str(x) for x in date)
            print(date)
            url = 'https://apistest.tradetipsapp.com/api/auth/appSignIn?userName=govinda&password=govinda'
            headers = {"Content-Type": "json"}
            response = requests.post(url, headers = headers)
            data = response.json()
            AAA = data['accessToken']
            token = AAA
##            sector_url = 'https://apis.tradetipsapp.com/api/GraphParameter/addGraphParameter'
##            PARAMS = {"stockName": symbol,
##                      "normalizedPrice":plot_normalized,
##                      "dates":date,
##                      "VXXAffected": plot_vxx,
##                      "newsAffected": plot_news1,
##                      "time": '09:30'
##                      }
##            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
##            response = requests.post(sector_url, params = PARAMS, headers=headers)
##            data = response.json()
##            print(data)
            sector_url = 'https://apistest.tradetipsapp.com/api/GraphParameter/addGraphParameter'
            PARAMS = {"stockName": symbol,
                      "normalizedPrice":plot_normalized,
                      "dates":date,
                      "VXXAffected": plot_vxx,
                      "newsAffected": plot_news1,
                      "time": '09:30'
                      }
            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
            response = requests.post(sector_url, params = PARAMS, headers=headers)
            data = response.json()
            print(data)
    except:
        pass