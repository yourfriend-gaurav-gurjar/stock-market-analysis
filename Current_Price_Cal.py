# from datetime import timedelta
import tensorflow as tf
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
import requests
from pytz import timezone
import pytz
from datetime import datetime


def time_cal(now):
    import datetime
    print("utc time-----", now)
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
    print("est time----", Time)
    Date = x[0]
    print("current_date", Date)
    return Time, Date


def date_cal(symbol):
    try:
        file2 = "stock_market_data-%s.csv" % symbol
        dataset = pd.read_csv('/opt/sma-files/sma-stock-csv-files/' + file2)
        dataset.drop("Unnamed: 0", axis=1, inplace=True)
        B = dataset.iloc[:, 0].values
        A = B[-1]
        import datetime
        cr_date = datetime.datetime.strptime(A, '%Y-%m-%d')
        start_date = cr_date + timedelta(1)
        last_date = start_date + timedelta(12)

        def daterange(date1, date2):
            for n in range(int((date2 - date1).days) + 1):
                yield date1 + timedelta(n)

        start_dt = start_date
        end_dt = last_date
        weekdays = [6, 7]
        new = []
        for dt in daterange(start_dt, end_dt):
            if dt.isoweekday() not in weekdays:
                New = dt.strftime("%Y-%m-%d")
                Holiday = ['2019-11-28', '2019-12-25', '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-20',
                           '2020-04-10', '2020-05-25', '2020-07-03', '2020-09-07', '2020-11-26',
                           '2020-12-25', '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31',
                           '2021-07-05', '2021-09-06', '2021-11-25', '2021-12-24']
                if New in Holiday:
                    pass
                else:
                    new.append(New)
        final_date = new[0:5]
    except:
        pass
    return final_date


def token_generation():
    url = 'https://apistest.tradetipsapp.com/api/auth/appSignIn?userName=govinda&password=govinda'
    headers = {"Content-Type": "json"}
    response = requests.post(url, headers=headers)
    data = response.json()
    AAA = data['accessToken']
    return AAA


def current_cal(aa, zz, symbol, time, number):
    try:
        aa['diffrence1'] = ''
        aa.loc[[0], 'diffrence1'] = (aa['minus1'][0] * zz) / 100
        aa['final_Normalized_price'] = ''
        aa.loc[[0], 'final_Normalized_price'] = round((aa['Normalized_Price'][0] + aa['diffrence1'][0]), 2)
        print(aa)
        print("time--------", time)

        def is_between(time, time_range):
            if time_range[1] < time_range[0]:
                return time >= time_range[0] or time <= time_range[1]
            return time_range[0] <= time <= time_range[1]

        Thirteenth = (is_between(time, ("15:50", "09:29")))
        if Thirteenth == True:
            AAA = token_generation()
            url = "https://apistest.tradetipsapp.com/api/GraphParameter/getNormalPriceByStockName"
            token = AAA
            PARAMS = {"stockName": symbol}
            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
            response = requests.post(url, params=PARAMS, headers=headers)
            data = response.json()
            print(data)
            print("--------------------------------------")
            list2 = []
            for i in data:
                A = i['time']
                print(A)
                if A == '15:30':
                    list2.append(i)
            print(list2)
            if len(list2) != 0:
                last_entry = list2[-1]
                print(last_entry)
                Vxx_value = last_entry['vxxaffected']
                News_value = last_entry['newsAffected']
                Vxx_value = Vxx_value.split(",")
                Vxx_value.remove(Vxx_value[0])
                Vxx_value = Vxx_value[0]
                print("vxxxxxxxxxxxxxxxxxx")
                print(Vxx_value)
                News_value = News_value.split(",")
                News_value.remove(News_value[0])
                News_value = News_value[0]
                print("newssssssssssssssss")
                print(News_value)
                aa['Predicted_VXX'] = ''
                aa.loc[[0], 'Predicted_VXX'] = Vxx_value
                aa['News_Affected_Close'] = ''
                aa.loc[[0], 'News_Affected_Close'] = News_value
                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
                print(aa)
                print("final table aa ----------------")
                aa.loc[[1, 2, 3, 4], 'final_Normalized_price'] = aa.loc[[1, 2, 3, 4], 'Normalized_Price']
                aa.loc[[1, 2, 3, 4], 'Predicted_VXX'] = aa.loc[[1, 2, 3, 4], 'Vxx_price']
                aa.loc[[1, 2, 3, 4], 'News_Affected_Close'] = aa.loc[[1, 2, 3, 4], 'News_price']
                aa.drop(['current_price', 'minus1', 'diffrence1', 'Normalized_Price', 'Vxx_price', 'News_price'],
                        axis=1, inplace=True)
                print(aa)
                file = "stock_market_data-%s.csv" % symbol
                dataset = pd.read_csv('/opt/sma-files/sma-stock-csv-files/' + file)
                last_value = dataset.tail(1)["Close"].values
                print("last_value--------------------------")
                print(last_value)
                from functools import reduce
                plot_normalized = aa.iloc[:, 1:2].values
                plot_normalized = np.insert(plot_normalized, 0, last_value, axis=0)
                plot_normalized = [i[0] for i in plot_normalized]
                plot_normalized = ",".join(str(x) for x in plot_normalized)
                print("plot_normalized---------------------")
                print(plot_normalized)
                plot_vxx = aa.iloc[:, 2:3].values
                plot_vxx = np.insert(plot_vxx, 0, last_value, axis=0)
                plot_vxx = [i[0] for i in plot_vxx]
                plot_vxx = ",".join(str(x) for x in plot_vxx)
                print("plot_vxx---------------------")
                print(plot_vxx)
                plot_news1 = aa.iloc[:, 3:4].values
                plot_news1 = np.insert(plot_news1, 0, last_value, axis=0)
                plot_news1 = [i[0] for i in plot_news1]
                plot_news1 = ",".join(str(x) for x in plot_news1)
                print("plot_news1------------------")
                print(plot_news1)
            else:
                pass

        else:
            print("time ok")
            vxx = pd.read_csv("/opt/sma-files/sma-stock-csv-files/stock_market_data-VXX.csv")
            vxx.drop("Unnamed: 0", axis=1, inplace=True)
            print(vxx)
            vxx_next_day = vxx.tail(5)
            vxx = vxx.drop(vxx.tail(5).index)
            vxx_next_day_open = vxx_next_day.iloc[0:1, 1:2].values
            vxx_prev_close = vxx.iloc[-1:, 4:5].values
            vxx_percent_change = (vxx_next_day_open - vxx_prev_close) / vxx_prev_close * 100
            vxx_percent_change = round(float(vxx_percent_change), 2)
            A1 = abs(vxx_percent_change)
            B1 = A1 + 3.00
            C1 = (round(B1, 0))
            D1 = C1 + 0.001
            change = np.arange(0, D1, .01)
            change_factor = np.linspace(0, 3.20, len(change))
            list_change = list(zip(change, change_factor))

            def re_round(li, _prec=3):
                try:
                    return round(li, _prec)
                except TypeError:
                    return type(li)(re_round(x, _prec) for x in li)

            list_change = re_round(list_change)
            l = pd.DataFrame(list_change, columns=["Change", "Change_Factor"])
            xql = l[l["Change"].values == abs(vxx_percent_change)]
            change_multiply_value = xql["Change_Factor"].values
            change_multiply_value = float(change_multiply_value)
            change_multiply_value = round(change_multiply_value, 2) / 100
            print(change_multiply_value)
            aa['Predicted_VXX'] = ''
            if vxx_percent_change < 0 and vxx_percent_change >= -8:
                aa.loc[[0], 'Predicted_VXX'] = aa["final_Normalized_price"][0] + round(
                    aa["final_Normalized_price"][0] * change_multiply_value, 2)
            elif vxx_percent_change >= 0 and vxx_percent_change <= 8:
                aa.loc[[0], 'Predicted_VXX'] = aa["final_Normalized_price"][0] - round(
                    aa["final_Normalized_price"][0] * change_multiply_value, 2)
            elif vxx_percent_change > 8 or vxx_percent_change < -8:
                aa.loc[[0], 'Predicted_VXX'] = aa["final_Normalized_price"][0] + round(
                    aa["final_Normalized_price"][0] * .015, 2)
            print(aa)
            print(number)
            aa['Predicted_VXX_1'] = ''
            aa.loc[[0], ['Predicted_VXX_1']] = aa['Predicted_VXX'][0] + round(
                ((aa["final_Normalized_price"][0] - aa['Predicted_VXX'][0]) * number), 2)
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
            PARAMS = {"stockSymbol": symbol}
            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
            response = requests.post(url, params=PARAMS, headers=headers)
            data = response.json()
            print(data)
            new = []
            for x in data:
                new.append(x['stock_sentiment'])
            remove = words_remove(new)
            print(new)
            print(remove)
            if len(new) == 0:
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
            aa['News_Affected_Close'] = ''
            if news == "Very Negative":
                aa.loc[[0], 'News_Affected_Close'] = aa["final_Normalized_price"][0] - round(
                    aa["final_Normalized_price"][0] * .0150, 2)
            elif news == "Negative":
                aa.loc[[0], 'News_Affected_Close'] = aa["final_Normalized_price"][0] - round(
                    aa["final_Normalized_price"][0] * .0100, 2)

            elif news == "Neutral":
                aa.loc[[0], 'News_Affected_Close'] = aa["final_Normalized_price"][0] - round(
                    aa["final_Normalized_price"][0] * .0006, 2)
            elif news == "Positive":
                aa.loc[[0], 'News_Affected_Close'] = aa["final_Normalized_price"][0] + round(
                    aa["final_Normalized_price"][0] * .0100, 2)
            elif news == "Very Positive":
                aa.loc[[0], 'News_Affected_Close'] = aa["final_Normalized_price"][0] + round(
                    aa["final_Normalized_price"][0] * .0150, 2)
            print(aa)
            aa['News_Affected_Close_1'] = ''
            aa.loc[[0], 'News_Affected_Close_1'] = aa['News_Affected_Close'][0] + round(
                ((aa["final_Normalized_price"][0] - aa['News_Affected_Close'][0]) * number), 2)
            print(aa)
            aa.drop(['current_price', 'minus1', 'diffrence1', 'Predicted_VXX', 'News_Affected_Close'], axis=1,
                    inplace=True)
            # aa.drop(['current_price','minus1','diffrence1','Predicted_VXX'],axis = 1,inplace = True)
            # print(aa)
            print("final table aa ----------------")
            aa.loc[[1, 2, 3, 4], 'final_Normalized_price'] = aa.loc[[1, 2, 3, 4], 'Normalized_Price']
            aa.loc[[1, 2, 3, 4], 'Predicted_VXX_1'] = aa.loc[[1, 2, 3, 4], 'Vxx_price']
            aa.loc[[1, 2, 3, 4], 'News_Affected_Close_1'] = aa.loc[[1, 2, 3, 4], 'News_price']
            # aa.loc[[1,2,3,4],'News_Affected_Close'] = aa.loc[[1,2,3,4],'News_price']
            print(aa)
            aa.drop(['Normalized_Price', 'Vxx_price', 'News_price'], axis=1, inplace=True)
            print(aa)
            file = "stock_market_data-%s.csv" % symbol
            dataset = pd.read_csv('/opt/sma-files/sma-stock-csv-files/' + file)
            last_value = dataset.tail(1)["Close"].values
            print("last_value--------------------------")
            print(last_value)
            from functools import reduce
            plot_normalized = aa.iloc[:, 1:2].values
            plot_normalized = np.insert(plot_normalized, 0, last_value, axis=0)
            plot_normalized = [i[0] for i in plot_normalized]

            def re_round(li, _prec=3):
                try:
                    return round(li, _prec)
                except TypeError:
                    return type(li)(re_round(x, _prec) for x in li)

            plot_normalized = re_round(plot_normalized)
            plot_normalized = ",".join(str(x) for x in plot_normalized)
            print("plot_normalized--------------------")
            print(plot_normalized)
            plot_vxx = aa.iloc[:, 2:3].values
            plot_vxx = np.insert(plot_vxx, 0, last_value, axis=0)
            plot_vxx = [i[0] for i in plot_vxx]
            plot_vxx = re_round(plot_vxx)
            plot_vxx = ",".join(str(x) for x in plot_vxx)
            print("plot_vxx---------------------")
            print(plot_vxx)
            plot_news1 = aa.iloc[:, 3:4].values
            plot_news1 = np.insert(plot_news1, 0, last_value, axis=0)
            plot_news1 = [i[0] for i in plot_news1]
            plot_news1 = re_round(plot_news1)
            plot_news1 = ",".join(str(x) for x in plot_news1)
            print("plot_news1------------------")
            print(plot_news1)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(plot_normalized)
        print(plot_vxx)
        print(plot_news1)
        file1 = "stock_market_data-%s.csv" % symbol
        final_date = pd.read_csv('/opt/sma-files/sma-stock-csv-files/' + file1)
        final_date = final_date.tail(1)
        A = final_date["Date"].values
        last_date = A[0]
        date = aa.iloc[:, 0:1].values
        date = np.insert(date, 0, last_date, axis=0)
        date = [i[0] for i in date]
        date = ",".join(str(x) for x in date)
        print(date)
        AAA = token_generation()
        token = AAA
        ##        sector_url = 'https://apis.tradetipsapp.com/api/GraphParameter/addGraphParameter'
        ##        PARAMS = {"stockName": symbol,
        ##                  "normalizedPrice":plot_normalized,
        ##                  "dates":date,
        ##                  "VXXAffected": plot_vxx,
        ##                  "newsAffected": plot_news1,
        ##                  "time": time
        ##                  }
        ##        headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
        ##        response = requests.post(sector_url, params = PARAMS, headers=headers)
        ##        data = response.json()
        ##        print(data)
        sector_url = 'https://apistest.tradetipsapp.com/api/GraphParameter/addGraphParameter'
        PARAMS = {"stockName": symbol,
                  "normalizedPrice": plot_normalized,
                  "dates": date,
                  "VXXAffected": plot_vxx,
                  "newsAffected": plot_news1,
                  "time": time
                  }
        headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
        response = requests.post(sector_url, params=PARAMS, headers=headers)
        data = response.json()
        print(data)
    except:
        pass
    return


Holiday = ['2019-11-28', '2019-12-25', '2020-01-01', '2020-01-20', '2020-02-17', '2020-04-20', '2020-04-10',
           '2020-05-25', '2020-07-03', '2020-09-07', '2020-11-26',
           '2020-12-25', '2021-01-01', '2021-01-18', '2021-02-15', '2021-04-02', '2021-05-31', '2021-07-05',
           '2021-09-06', '2021-11-25', '2021-12-24']
now = datetime.now()
time, Date = time_cal(now)
print(Date)
holiday1 = []
for j in Holiday:
    if j == Date:
        holiday1.append(j)
print(holiday1)
list_symbol = ["MSFT", "AAPL", "JPM", "GOOG", "SPY", "IWM", "QQQ", "DIA"]
# list_symbol = ["SPY"]
list_main = list_symbol[0:8]
for symbol in list_main:
    try:
        if len(holiday1) != 0:
            url = 'https://apistest.tradetipsapp.com/api/GraphParameter/getNormalPriceByStockName'
            AAA = token_generation()
            token = AAA
            PARAMS = {"stockName": symbol}
            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
            response = requests.post(url, params=PARAMS, headers=headers)
            data = response.json()
            print(data)
            list1 = []
            for i in data:
                A = i['time']
                print(A)
                if A == '09:30':
                    list1.append(i)
            print(list1)
            if len(list1) != 0:
                last_entry = list1[-1]
                plot_normalized = last_entry['normalizedPrice']
                print(plot_normalized)
                date = last_entry['dates']
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
                ##                          "time": '15:50'
                ##                          }
                ##                headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
                ##                response = requests.post(sector_url, params = PARAMS, headers=headers)
                ##                data = response.json()
                ##                print(data)
                sector_url = 'https://apistest.tradetipsapp.com/api/GraphParameter/addGraphParameter'
                PARAMS = {"stockName": symbol,
                          "normalizedPrice": plot_normalized,
                          "dates": date,
                          "VXXAffected": plot_vxx,
                          "newsAffected": plot_news1,
                          "time": '15:50'
                          }
                headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
                response = requests.post(sector_url, params=PARAMS, headers=headers)
                data = response.json()
                print(data)
            else:
                pass
        else:
            AAA = token_generation()
            url = "https://apistest.tradetipsapp.com/api/GraphParameter/getNormalPriceByStockName"
            token = AAA
            PARAMS = {"stockName": symbol}
            headers = {'Authorization': 'Bearer ' + token, "Content-Type": "json"}
            response = requests.post(url, params=PARAMS, headers=headers)
            data = response.json()
            print(data)
            print("--------------------------------------")
            list2 = []
            for i in data:
                A = i['time']
                print(A)
                if A == '09:30':
                    list2.append(i)
            print(list2)
            if len(list2) != 0:
                last_entry = list2[-1]
                print(last_entry)
                Normalized_value = last_entry['normalizedPrice']
                print(Normalized_value)
                Vxx_value = last_entry['vxxaffected']
                print(Vxx_value)
                News_value = last_entry['newsAffected']
                print(News_value)
                Normalized_value = Normalized_value.split(",")
                Normalized_value.remove(Normalized_value[0])
                Vxx_value = Vxx_value.split(",")
                Vxx_value.remove(Vxx_value[0])
                News_value = News_value.split(",")
                News_value.remove(News_value[0])
                Normal_price = []
                Vxx_price = []
                News_price = []
                for i in Normalized_value:
                    print(i)
                    print(type(i))
                    j = float(i)
                    print(j)
                    print(type(j))
                    Normal_price.append(j)
                print(Normal_price)
                for i in Vxx_value:
                    print(i)
                    print(type(i))
                    j = float(i)
                    print(j)
                    print(type(j))
                    Vxx_price.append(j)
                print(Vxx_price)
                for i in News_value:
                    print(i)
                    print(type(i))
                    j = float(i)
                    print(j)
                    print(type(j))
                    News_price.append(j)
                print(News_price)
                final_date = date_cal(symbol)
                print("final_date-------------------------------")
                print(final_date)
                aa = pd.DataFrame(data=final_date)
                aa['Normalized_Price'] = Normal_price
                aa.columns = ["Dates", "Normalized_Price"]
                aa['Vxx_price'] = Vxx_price
                aa['News_price'] = News_price
                # url_string =  ("https://cloud.iexapis.com/stable/stock/market/batch?types=quote&symbols="+symbol+"&token=pk_dd324da3fb5f4428a47b05ab12f23ce2".format(symbol))
                url_string = (
                            "https://sandbox.iexapis.com/stable/stock/market/batch?types=quote&symbols=" + symbol + "&token=Tpk_b5eb4116bd1542f99177767878ad0b55".format(
                        symbol))
                file_to_save = 'stock_market_data-%s.csv' % symbol
                url = urllib.request.urlopen(url_string)
                data = json.loads(url.read().decode())
                print("data--------------")
                print(data)
                current_price = (data[symbol]['quote']['latestPrice'])
                print("latest price----------------------")
                print(
                    current_price)  # close for 10 am ,10:30 nd so on..from current date(like if we have prediction for 11,12,13,14,15 so we take close for 11)
                aa['current_price'] = ''
                aa.loc[[0], 'current_price'] = current_price
                print(aa)
                print(type(aa['current_price'][0]))
                aa['minus1'] = ''
                aa.loc[[0], 'minus1'] = aa['current_price'][0] - aa['Normalized_Price'][0]
                print(aa)
                from datetime import datetime

                now = datetime.now()
                print("utc time-----", now)
                time, Date = time_cal(now)
                print("est time----", time)
                print(symbol)


                def is_between(time, time_range):
                    if time_range[1] < time_range[0]:
                        return time >= time_range[0] or time <= time_range[1]
                    return time_range[0] <= time <= time_range[1]


                print(time)
                first = (is_between(time, ("10:00", "10:29")))
                # print(first)

                if first == True:
                    print('yes')
                    time = '10:00'
                    zz = 10
                    number = 0.025  # according to 2.5% ~ 2.5/100
                    # number = 0.005
                    current_cal(aa, zz, symbol, time, number)
                second = (is_between(time, ("10:30", "10:59")))
                # print(second)
                if second == True:
                    print("Second yes")
                    time = '10:30'
                    zz = 15
                    number = 0.05  # double of 2.5*2 = 5% ~ 5/100
                    # number = 0.01
                    current_cal(aa, zz, symbol, time, number)
                Third = (is_between(time, ("11:00", "11:29")))
                # print(thired)
                if Third == True:
                    print("Third yes")
                    time = '11:00'
                    zz = 20
                    number = 0.075
                    # number = 0.015
                    current_cal(aa, zz, symbol, time, number)
                Fourth = (is_between(time, ("11:30", "11:59")))
                # print(Fourth)
                if Fourth == True:
                    print("Fourth yes")
                    time = '11:30'
                    zz = 35
                    number = 0.10
                    # number = 0.02
                    current_cal(aa, zz, symbol, time, number)
                fifth = (is_between(time, ("12:00", "12:29")))
                # print(fifth)
                if fifth == True:
                    print("fifth yes")
                    time = '12:00'
                    zz = 40
                    number = 0.125
                    # number = 0.025
                    current_cal(aa, zz, symbol, time, number)
                sixth = (is_between(time, ("12:30", "12:59")))
                # print(sixth)
                if sixth == True:
                    print("sixth yes")
                    time = '12:30'
                    zz = 55
                    number = 0.15
                    # number = 0.03
                    current_cal(aa, zz, symbol, time, number)
                seventh = (is_between(time, ("13:00", "13:29")))
                # print(seventh)
                if seventh == True:
                    print("seventh yes")
                    time = '13:00'
                    zz = 60
                    number = 0.175
                    # number = 0.035
                    current_cal(aa, zz, symbol, time, number)
                eighth = (is_between(time, ("13:30", "13:59")))
                # print(eighth)
                if eighth == True:
                    print("eighth yes")
                    time = '13:30'
                    zz = 75
                    number = 0.20
                    # number = 0.04
                    current_cal(aa, zz, symbol, time, number)
                ninth = (is_between(time, ("14:00", "14:29")))
                # print(ninth)
                if ninth == True:
                    print("ninth yes")
                    time = '14:00'
                    zz = 80
                    number = 0.225
                    # number = 0.045
                    current_cal(aa, zz, symbol, time, number)
                tenth = (is_between(time, ("14:30", "14:59")))
                # print(tenth)
                if tenth == True:
                    print("tenth yes")
                    time = '14:30'
                    zz = 85
                    number = 0.25
                    # number = 0.05
                    current_cal(aa, zz, symbol, time, number)
                eleventh = (is_between(time, ("15:00", "15:29")))
                # print(eleventh)
                if eleventh == True:
                    print("eleventh yes")
                    time = '15:00'
                    zz = 90
                    number = 0.275
                    # number = 0.055
                    current_cal(aa, zz, symbol, time, number)
                Twelfth = (is_between(time, ("15:30", "15:49")))
                # print(Twelfth)
                if Twelfth == True:
                    print("Twelfth yes")
                    time = '15:30'
                    zz = 95
                    number = 0.30
                    # number = 0.06
                    current_cal(aa, zz, symbol, time, number)
                Thirteenth = (is_between(time, ("15:50", "09:29")))
                # print(Thirteenth)
                if Thirteenth == True:
                    print("Thirteenth yes")
                    time = '15:50'
                    number = "0.96"  # at 15:50 generally added this number no use in the function
                    import random

                    zz = random.uniform(96.0, 102.0)
                    current_cal(aa, zz, symbol, time, number)
            else:
                pass
    except:
        pass