import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import deque
import random
import os
import datetime
import joblib
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# set seed, so we can get the same results after rerunning several times
np.random.seed(314)
tf.random.set_seed(314)
random.seed(314)

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def calculate_percentage(start_date, end_date, breakpoint_date):
    # Convert the date strings to datetime objects
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    breakpoint_date = datetime.datetime.strptime(breakpoint_date, '%Y-%m-%d')

    # Calculate the time differences
    time_difference_start_to_breakpoint = (breakpoint_date - start_date).days
    time_difference_start_to_end = (end_date - start_date).days
    # Calculate the percentage
    percentage = (time_difference_start_to_breakpoint / time_difference_start_to_end)
    return 1-percentage

def load_data_SARIMA(ticker, start_date, end_date, n_steps, scale, shuffle, store, k_days, split_by_date,
                     test_size, feature_columns, store_scale, rolling, breakpoint_date):
    global scaler
    # see if ticker is already a loaded stock from yahoo finance
    if isinstance(ticker, str):
        # load it from yahoo_fin library
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    elif isinstance(ticker, pd.DataFrame):
        # if already loaded, use it directly
        df = ticker
    else:
        #print error if there is
        raise TypeError("ticker can be either a str or a `pd.DataFrame` instances")

    print(df)
    # add date as a column
    if "Date" not in df.columns:
        df["Date"] = df.index

    df_series = df.set_index('Date').asfreq('D')
    df_series['Adj Close'].interpolate(method='linear', inplace=True)

    # what we want to return from this function
    result = {}
    # also return the original dataframe itself
    result['df'] = df.copy()

    series = pd.Series(df_series['Adj Close'], index=df_series.index)
    # Final Target Variable Dataset

    if rolling:
        series = rolling_func(series)
        print(series)

    target_final = series.iloc[3:, ]

    train = target_final[target_final.index <= breakpoint_date]
    test = target_final[target_final.index > breakpoint_date]
    true_test = df[df.index > breakpoint_date]['Adj Close']

    print(true_test)

    filename = ""

    # if store:
    #     filename = store_data(result, ticker)
    #
    # if store_scale:
    #     scaler_filename = "scaler.save"
    #     joblib.dump(scaler, scaler_filename)

    return [series, train, test, true_test, filename]

def store_data(data, ticker):
    # create these folders if they do not exist
    if not os.path.isdir("results"):
        os.mkdir("results")

    if not os.path.isdir("logs"):
        os.mkdir("logs")

    if not os.path.isdir("data"):
        os.mkdir("data")

    date_now = datetime.date.today().strftime('%Y-%m-%d')
    ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")
    # save the dataframe
    data['df'].to_csv(ticker_data_filename)
    filename = f"{ticker}_{date_now}.csv"
    return filename

# Load data from file saved in results folder
def load_data_file(ticker, date):
    df = pd.read_csv(f"data/{ticker}_{date}.csv")
    return df

def rolling_func(data):
    rolling_mean = data.rolling(window=12).mean()
    rolling_mean_diff = rolling_mean - rolling_mean.shift()
    ax1 = plt.subplot()
    rolling_mean_diff.plot(title='after rolling mean & differencing');
    ax2 = plt.subplot()
    # data.plot(title='original')
    plt.show()

    print(rolling_mean_diff)
    return rolling_mean_diff

def data_lookup(data):
    dftest = adfuller(data, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)


