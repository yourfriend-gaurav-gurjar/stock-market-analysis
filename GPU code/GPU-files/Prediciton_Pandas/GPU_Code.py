#!/usr/bin/env python
# coding: utf-8
# %%

# Just disables the warning, doesn't enable AVX/FMA
#import os
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#CUDA_VISIBLE_DEVICES=0
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#keras used for LSTM
#import tensorflow.compat.v0 as tf
import tensorflow as tf
#tf.disable_v2_behavior()
#tf.debugging.set_log_device_placement(True)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import TimeDistributed
from keras.layers import LeakyReLU
from keras.layers import Activation, Dense
#import matplotlib
#import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import time
import numpy as np
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
#import cudf
#from pandas_datareader import data
import re
import pytz
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
import requests
np.random.seed(seed_value)
#tf.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

#tf.random.set_seed(seed_value)
from keras import backend as K
#import tensorflow as tf

#tf.config.experimental_run_functions_eagerly(True)
#tf.disable_eager_execution()
#tf.compat.v1.disable_eager_execution()
#, config=tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(sess)
'''config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # don't pre-allocate memory; allocate as-needed
config.gpu_options.per_process_gpu_memory_fraction = 0.95  # limit memory to be allocated
K.tensorflow_backend.set_session(tf.Session(config=config))'''



def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]    #  for difference in close data
        diff.append(value)
    return pd.Series(diff)


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


def forecast_lstm(model, X, n_batch):
    X_feature_length = X.shape[1]
    if X_feature_length is None:
        raise ValueError("ERROR:: vector length for input is None")
    elif X_feature_length != look_back:
        raise ValueError("ERROR:: vector length for input is not equal to look back length of {}".format(look_back))
    X_f = X.reshape(1, 1, X_feature_length)
    # make forecast
    forecast = model.predict(X_f, batch_size=n_batch)
    #print(forecast)
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


def Actual(look_back,dataset_test_close):
    A = [row[look_back:] for row in dataset_test_close]
    return A


def sse_numba(symbol,look_back):
    t2 = time.time()
    print("start")
    print("Symbol--------------------")
    print(symbol)
    print(look_back)
    
    
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
    #print(use_stateful_model)
    file = "stock_market_data-%s.csv"%symbol
    dataset = pd.read_csv(file)
    #print(type(dataset))
    #Del unnamed column from dataset variable Unnamed: 0
    dataset.drop("Unnamed: 0", axis = 1,inplace=True)
    print("dataset-----------------")
    print(dataset)
    print(type(dataset))
    
    
    #Assert that open, high, low, close are numeric types

    assert (dataset["Open"].dtype  == np.dtype('float64') or dataset["Open"].dtype  == np.dtype('float32'))
    assert (dataset["High"].dtype  == np.dtype('float64') or dataset["High"].dtype  == np.dtype('float32'))
    assert (dataset["Low"].dtype   == np.dtype('float64') or dataset["Low"].dtype   == np.dtype('float32'))
    assert (dataset["Close"].dtype == np.dtype('float64') or dataset["Close"].dtype == np.dtype('float32'))
##    print(dataset["Date"])
##    print(dataset["Date"].values)
##    print(type(dataset["Date"].values))
    
    dataset["Date"] = pd.to_datetime(dataset["Date"].values, format="%Y-%m-%d")
    print(dataset["Date"])
    
    assert(pd.api.types.is_datetime64_any_dtype(dataset["Date"]))
    dataset_close = dataset[["Date", "Close"]]
    print(dataset_close)
    
    dataset_close_diffed = difference(dataset_close["Close"], 1)
    print(dataset_close_diffed)
    
    num_test_start_points  = int(np.floor(dataset_close["Close"].shape[0] * test_fraction))

    num_train_start_points = dataset_close.shape[0] - num_test_start_points
    
    data_scaler, dataset_train_close, dataset_test_close =            prepare_data(dataset_close["Close"],    
                     n_test=int(np.floor(dataset_close["Close"].shape[0] * test_fraction)),   
                     n_lag=int(look_back),  
                     n_seq=int(forward_days))   
    #print(data_scaler)
    #print(dataset_train_close)
    #print(dataset_test_close)
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



    multistep_model_close = build_lstm_network(X_seq_length=X_train_close.shape[1],
                                              X_feature_length=X_train_close.shape[2],
                                              y_seq_length=y_train_close.shape[1],
                                              num_sequences_per_batch=1,
                                              num_cell_units=12,
                                              keep_state=use_stateful_model)
    print(multistep_model_close)
    
    #Set the number of epochs for training
    NUM_EPOCHS_TRAINING = 128
    #print(NUM_EPOCHS_TRAINING)
    if use_stateful_model:
        training_history = []
        for i in range(NUM_EPOCHS_TRAINING):
            #print("i = {}".format(i))
            training_history.append(multistep_model_close.fit(X_train_close, y_train_close, epochs=1, batch_size=1, verbose=1, shuffle=False, use_multiprocessing=True, workers=8))
            multistep_model_close.reset_states()
    else:
        training_history = multistep_model_close.fit(X_train_close, y_train_close, epochs=NUM_EPOCHS_TRAINING, batch_size=0, validation_split=0.25, verbose=1, shuffle=False, use_multiprocessing=True, workers=8)
    #print("it's over")
    #t4 = time.time()
    #print("time with numba :", t4-t2)
    forecasts = make_forecasts(multistep_model_close, 
                               n_batch=1,
                               train=X_train_close,
                               test=X_test_close,
                               n_test_instances=X_test_close.shape[0],
                               n_lag= 200,
                               n_seq=forward_days)
    #print("forecasts----------------------")
    #print(forecasts)
    inverted_forecasts = inverse_transform(dataset_close["Close"],
                                           forecasts,
                                           scaler=data_scaler,
                                           n_test=X_test_close.shape[0] + forward_days - 1)
    
    #print("inverted_forecasts------------------------------")
    #print(inverted_forecasts)
    #print(dataset_test_close)
    #actual = [row[200:] for row in dataset_test_close]
    actual = Actual(look_back,dataset_test_close)
    #print("actual")
    #print(actual)
    inverted_actual = inverse_transform(dataset_close["Close"],
                                        actual,
                                        scaler=data_scaler,
                                        n_test=X_test_close.shape[0] + forward_days - 1)
  
    #print("inverted_actual-------------------------------------")
    #print(inverted_actual)

    
    
    
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
    #print("last 6 rows------------------------")
    #print(df_forecast_five_day.tail(6))
    result = df_forecast_five_day.tail(6)
    t4 = time.time()
    print("time with numba :", t4-t2)
    print(symbol)
    print("epochs = 128")
    
    
    return result


symbol = 'MSFT'
#t2 = time.time()
look_back = 200
result = sse_numba(symbol,look_back)



