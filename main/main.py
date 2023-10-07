from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, SimpleRNN, GRU

from data_processing_LSTM import load_data
from visulization import visualization
from LSTM import create_model_LSTM, train_model_LSTM
from test_LSTM import test_model_LSTM
from SARIMA import calculate_step_wise_SARIMA, train_test_SARIMA
from data_processing_SARIMA import load_data_SARIMA
from ARIMA import calculate_step_wise_ARIMA, train_test_ARIMA, plot_graph
from RF import hyper_parameter_tuning_RF, train_modeL_RF, plot_graph_RF
from XGB import hyper_parameter_tuning_XGB, train_modeL_XGB, plot_graph_XGB
from data_processing_RF import load_data_RF
import pandas as pd


# Parameters

# Company name
COMPANY = "TSLA"

# start = '2015-01-01', end='2020-01-01'
TRAIN_START = '2015-01-01'
TRAIN_END = '2020-01-01'

# Window size or the sequence length
N_STEPS = 50
# Lookup step, 1 is the next day
K_DAYS = 3
STORE = True

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"
# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"
# whether to split the training/testing set by date
SPLIT_BY_DATE = True
BREAKPOINT_DATE = "2019-01-01"
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"
# test ratio size, 0.2 is 20%
TEST_SIZE = 0.2
# features to use
FEATURE_COLUMNS = ['Close','Open','High','Low','Adj Close', 'Volume']
STORE_SCALE = True
ROLLING = False

### Visulization parameters
TRADING_DAYS = 1

### model parameters

N_LAYERS = 2
# 256 LSTM neurons
UNITS = 150
# 40% dropout
DROPOUT = 0.4
# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### training parameters

# mean absolute error loss
# LOSS = "mae"
# huber loss
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 25

CELLS = [GRU, "ARIMA", "SARIMA", "RF"]
# CELLS = ["RF"]
forecast_results = []

for CELL in CELLS:
    ### Main code

    if CELL == LSTM or CELL == SimpleRNN or CELL == GRU:
        # load_data function
        data_loaded = load_data(COMPANY, TRAIN_START, TRAIN_END, N_STEPS, SCALE, SHUFFLE, STORE, K_DAYS, SPLIT_BY_DATE, TEST_SIZE, FEATURE_COLUMNS, STORE_SCALE, breakpoint_date=BREAKPOINT_DATE)

        # Assign dataframe
        data = data_loaded[0]
        # Filename
        filename = data_loaded[1]

        # Visulize candlestick and boxplot
        visualization(data['df'], TRADING_DAYS)

        # Create model
        model = create_model_LSTM(50, len(FEATURE_COLUMNS), CELL, UNITS, K_DAYS)

        train = False
        # Train model
        if train:
            train_model_LSTM(model, filename, data, 64, 25)

        result = test_model_LSTM(data, model, filename, SCALE, K_DAYS, LOSS, N_STEPS)[f"Adj Close_{K_DAYS}"]
        s1 = pd.Series(result)
        # Convert Series to DataFrame
        df1 = s1.to_frame()
        if CELL == LSTM:
            df1.rename(columns={f'Adj Close_{K_DAYS}': "LSTM"}, inplace=True)
        if CELL == SimpleRNN:
            df1.rename(columns={f'Adj Close_{K_DAYS}': "RNN"}, inplace=True)
        else:
            df1.rename(columns={f'Adj Close_{K_DAYS}': "GRU"}, inplace=True)
        print(df1)
        forecast_results.append(df1)

    elif CELL=="SARIMA":
        # load_data function
        series, train, test, true_test, filename = load_data_SARIMA(COMPANY, TRAIN_START, TRAIN_END, N_STEPS, SCALE, SHUFFLE, STORE, K_DAYS, SPLIT_BY_DATE, TEST_SIZE, FEATURE_COLUMNS, STORE_SCALE, ROLLING, breakpoint_date=BREAKPOINT_DATE)

        # Visulize candlestick and boxplot
        # visualization(data['df'], TRADING_DAYS)

        stepwise_fit = calculate_step_wise_SARIMA(series, train)
        predictions = train_test_SARIMA(stepwise_fit, K_DAYS, test, true_test)
        plot_graph(predictions, true_test)
        print(forecast_results)
        forecast_results.append(predictions)

    elif CELL=="ARIMA":
        # load_data function
        series, train, test, true_test, filename = load_data_SARIMA(COMPANY, TRAIN_START, TRAIN_END, N_STEPS, SCALE,
                                                                    SHUFFLE, STORE, K_DAYS, SPLIT_BY_DATE, TEST_SIZE,
                                                                    FEATURE_COLUMNS, STORE_SCALE, ROLLING,
                                                                    breakpoint_date=BREAKPOINT_DATE)
        # Visulize candlestick and boxplot
        # visualization(data['df'], TRADING_DAYS)

        stepwise_fit = calculate_step_wise_ARIMA(series, train)
        predictions = train_test_ARIMA(stepwise_fit, K_DAYS, test, true_test)
        plot_graph(predictions, true_test)

        forecast_results.append(predictions)

    elif CELL == "RF":
        # load_data function
        data_loaded = load_data_RF(COMPANY, TRAIN_START, TRAIN_END, N_STEPS, SCALE, SHUFFLE, STORE, K_DAYS, SPLIT_BY_DATE,
                                TEST_SIZE, FEATURE_COLUMNS, STORE_SCALE, breakpoint_date=BREAKPOINT_DATE)

        # Assign dataframe
        data = data_loaded[0]
        # Filename
        filename = data_loaded[1]

        # Visulize candlestick and boxplot
        # visualization(data['df'], TRADING_DAYS)

        # best_n_estimators, best_max_depth, best_min_samples_leaf, best_min_samples_split, best_train_accuracy, best_test_accuracy = hyper_parameter_tuning_RF(data)
        best_n_estimators = 1000
        best_max_depth = 50
        best_min_samples_leaf = 4
        best_min_samples_split = 100
        best_train_accuracy= 1
        best_test_accuracy=0

        final_df = train_modeL_RF(best_n_estimators, best_max_depth, best_min_samples_leaf, best_min_samples_split, best_train_accuracy, best_test_accuracy, data, SCALE)

        plot_graph_RF(final_df)

        result = final_df["RF"]

        s1 = pd.Series(result)
        # Convert Series to DataFrame
        df1 = s1.to_frame()
        forecast_results.append(df1)

    elif CELL == "XGB":
        # load_data function
        data_loaded = load_data_RF(COMPANY, TRAIN_START, TRAIN_END, N_STEPS, SCALE, SHUFFLE, STORE, K_DAYS, SPLIT_BY_DATE,
                                TEST_SIZE, FEATURE_COLUMNS, STORE_SCALE, breakpoint_date=BREAKPOINT_DATE)

        # Assign dataframe
        data = data_loaded[0]
        # Filename
        filename = data_loaded[1]

        # Visulize candlestick and boxplot
        # visualization(data['df'], TRADING_DAYS)

        # best_n_estimators, best_max_depth, best_min_child_weight, best_gamma, best_learning_rate, best_train_accuracy, best_test_accuracy = hyper_parameter_tuning_XGB(data)
        best_n_estimators = 2000
        best_max_depth = 100
        best_min_child_weight = 2
        best_gamma = 5
        best_learning_rate = 0.1
        best_train_accuracy = 1
        best_test_accuracy = 1

        final_df = train_modeL_XGB(best_n_estimators, best_max_depth, best_min_child_weight, best_gamma, best_learning_rate, best_train_accuracy, best_test_accuracy, data, SCALE)

        plot_graph_XGB(final_df)

        result = final_df["XGB"]

        s1 = pd.Series(result)
        # Convert Series to DataFrame
        df1 = s1.to_frame()
        forecast_results.append(df1)

print(forecast_results)

# Assuming forecast_results is your list of dataframes
merged_df = forecast_results[0]

for df in forecast_results[1:]:
    merged_df = pd.merge(merged_df, df, left_index=True, right_index=True)
merged_df['Final_Forecast'] = merged_df.mean(axis=1)
print(merged_df)



