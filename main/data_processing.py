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
    print(breakpoint_date)
    # Calculate the percentage
    percentage = (time_difference_start_to_breakpoint / time_difference_start_to_end)
    return 1-percentage

def load_data(ticker, start_date, end_date, n_steps, scale, shuffle, store, lookup_step, split_by_date,
                test_size, feature_columns, store_scale, breakpoint_date="2000-01-01"):
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

    # what we want to return from this function
    result = {}
    # also return the original dataframe itself
    result['df'] = df.copy()

    result['feature_columns'] = feature_columns

    # check if the featured columns exist
    for col in feature_columns:
        assert col in df.columns, f"'{col}' does not exist in the dataframe."

    # add date as a column
    if "Date" not in df.columns:
        df["Date"] = df.index

    #in case the user want to scale
    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler

        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler

    # add the target column (label) by shifting by `lookup_step`
    df['future'] = df['Adj Close'].shift(-lookup_step)

    # last `lookup_step` columns contains NaN in future column
    # get them before droping NaNs
    last_sequence = np.array(df[feature_columns].tail(lookup_step))

    # drop NaNs
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)

    for entry, target in zip(df[feature_columns + ["Date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    # get the last sequence by appending the last `n_step` sequence with `lookup_step` sequence
    # for instance, if n_steps=50 and lookup_step=10, last_sequence should be of 60 (that is 50+10) length
    # this last_sequence will be used to predict future stock prices that are not available in the dataset
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # add to result
    result['last_sequence'] = last_sequence

    # construct the X's and y's
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        test_size = calculate_percentage(start_date, end_date, breakpoint_date)
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)
    else:
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size,
                                                                                                    shuffle=shuffle)

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].loc[dates]
    # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)

    filename = ""

    if store:
        filename = store_data(result, ticker)

    if store_scale:
        scaler_filename = "scaler.save"
        joblib.dump(scaler, scaler_filename)

    return [result, filename]

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
