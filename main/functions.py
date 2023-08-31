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
import bokeh
from math import pi
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Whisker
from bokeh.io import show
from bokeh.resources import INLINE
from bokeh.layouts import column
import matplotlib.pyplot as plt

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

def load_data(ticker, start_date, end_date, n_steps=50, scale=True, shuffle=True, store=True, lookup_step=1, split_by_date=True,
                test_size=0.2, feature_columns=['Close','Open','High','Low','Adj Close', 'Volume']):

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
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
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

    if store:
        store_data(result, ticker)

    return result

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

def add_tooltip(ax, df):
    # This function adds tooltips to the candlestick chart
    for i, row in df.iterrows():
        ax.annotate(f'{row["Open"]:.2f}', xy=(i, row["Open"]), xytext=(i, row["Open"] + 2), textcoords='data', arrowprops=dict(arrowstyle='->'))

def delete_gaps(df, trading_days):
    index_lists = []
    i = trading_days - 1
    while i < len(df):
        index_lists.append(i)
        i += trading_days

    # Select rows with indexes in index_lists
    result_df = df.iloc[index_lists]

    return result_df

#Visualize the dataframe
def visualization(df, trading_days):
    dt_range = pd.date_range(start="2015-01-01", end="2015-03-01")
    df = df[df.index.isin(dt_range)]

    # Calculate the average values of 'trading_days' consecutive days
    df = df.rolling(window=trading_days, min_periods=1, step=trading_days).mean()

    # Delete the redundants rows after rolling
    # df = delete_gaps(df, trading_days)

    print(df)

    #Create median column for boxplot chart
    df['Median'] = df[['High', 'Low', 'Open', 'Close']].median(axis=1)

    #Hover tooltip box
    hover = HoverTool(
        tooltips=[
            ('Date', '@Date{%Y-%m-%d}'),  # Display the 'Date' value in 'yyyy-mm-dd' format
            ('Open', '@Open'),  # Display the 'Open' column value with formatting
            ('High', '@High'),  # Debugging: Print High value
            ('Low', '@Low'),  # Debugging: Print Low value
            ('Close', '@Close'),  # Debugging: Print Close value
            ('Volume', '@Volume')  # Debugging: Print Volume value
        ],
        formatters={'@Date': 'datetime'},
        mode='mouse'
    )

    source = ColumnDataSource(data=df)

    # Create ColumnDataSources for increasing and decreasing values
    inc_source = ColumnDataSource(data=df[df.Close > df.Open])
    dec_source = ColumnDataSource(data=df[df.Open > df.Close])

    #width
    w = 12 * 60 * 60 * 1000

    #Candlestick chart
    candle = figure(x_axis_type="datetime", width=800, height=500, title="Representation of the stock price")

    # Create the line
    candle.segment('Date', 'High', 'Date', 'Low', source=source, color="black")

    # Create vbar glyphs for increasing and decreasing values with different colors
    candle.vbar('Date', w, 'Open', 'Close', source=inc_source, fill_color="green", line_color="green",
             legend_label="Increasing")
    candle.vbar('Date', w, 'Open', 'Close', source=dec_source, fill_color="red", line_color="red",
             legend_label="Decreasing")

    # Add hovering function
    candle.add_tools(hover)

    # Boxplot Chart

    box = figure(x_axis_type="datetime", width=800, height=600)

    # Create whiskers
    whisker = Whisker(base="Date", lower="Low", upper="High", source=ColumnDataSource(df))

    box.add_layout(whisker)

    # Create increasing/decreasing boxes with different colors
    box.vbar("Date", w, "Median", "Open", color="green", source=inc_source, line_color="black", legend_label="Increasing")
    box.vbar("Date", w, "Close", "Median", color="green", source=inc_source, line_color="black")

    box.vbar("Date", w, "Median", "Open", color="red", source=dec_source, line_color="black", legend_label="Decreasing")
    box.vbar("Date", w, "Close", "Median", color="red", source=dec_source, line_color="black")

    # Add hovering function
    box.add_tools(hover)

    # Display charts
    show(column(candle, box))
