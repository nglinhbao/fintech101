import pandas as pd
from  statsmodels.tsa.vector_ar.vecm import *
import matplotlib.pyplot as plt


import itertools
import math
import random
import xgboost as xgb
import tensorflow
import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

def hyper_parameter_tuning_XGB(data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # XG Boost
    # Number of trees
    n_estimators = [500, 1000, 2000]
    # Maximum number of levels in tree
    max_depth = [10, 50, 100]
    # minimum sum of weights of all observations required in a child
    min_child_weight = [1, 2]
    # Gamma specifies the minimum loss reduction required to make a split
    gamma = [1, 5]
    # boosting learning rate
    learning_rate = [.1, .05, .01]

    best_n_estimators = 0
    best_max_depth = 0
    best_min_child_weight = 0
    best_gamma = 0
    best_learning_rate = 0
    best_train_accuracy = 0
    best_test_accuracy = 0

    xgb_Test_Accuracy_Data = pd.DataFrame(
        columns=['n_estimators', 'max_depth', 'min_child_weight', 'gamma', 'learning_rate', 'Train Accurcay',
                 'Test Accurcay'])

    for x in list(itertools.product(n_estimators, max_depth, min_child_weight, gamma, learning_rate)):
        xgb_reg = xgb.XGBRegressor(n_estimators=x[0], max_depth=x[1], min_child_weight=x[2], gamma=x[3], learning_rate=x[4])

        # Train the model on training data
        xgb_reg.fit(X_train, y_train)

        # Train Data
        # Use the forest's predict method on the train data
        predictions_train = xgb_reg.predict(X_train)
        # Calculate the absolute errors
        errors_train = abs(predictions_train - y_train)
        # Calculate mean absolute percentage error (MAPE)
        mape_train = 100 * (errors_train / y_train)
        # Calculate and display accuracy
        accuracy_train = 100 - np.mean(mape_train)

        # Test Data
        # Use the forest's predict method on the test data
        predictions_test = xgb_reg.predict(X_test)
        # Calculate the absolute errors
        errors_test = abs(predictions_test - y_test)
        # Calculate mean absolute percentage error (MAPE)
        mape_test = 100 * (errors_test / y_test)
        # Calculate and display accuracy
        accuracy_test = 100 - np.mean(mape_test)

        xgb_Test_Accuracy_Data_One = pd.DataFrame(index=range(1),
                                                  columns=['n_estimators', 'max_depth', 'min_child_weight', 'gamma',
                                                           'learning_rate', 'Train Accurcay', 'Test Accurcay'])

        xgb_Test_Accuracy_Data_One.loc[:, 'n_estimators'] = x[0]
        xgb_Test_Accuracy_Data_One.loc[:, 'max_depth'] = x[1]
        xgb_Test_Accuracy_Data_One.loc[:, 'min_child_weight'] = x[2]
        xgb_Test_Accuracy_Data_One.loc[:, 'gamma'] = x[3]
        xgb_Test_Accuracy_Data_One.loc[:, 'learning_rate'] = x[4]
        xgb_Test_Accuracy_Data_One.loc[:, 'Train Accurcay'] = accuracy_train
        xgb_Test_Accuracy_Data_One.loc[:, 'Test Accurcay'] = accuracy_test

        xgb_Test_Accuracy_Data = pd.concat([xgb_Test_Accuracy_Data, xgb_Test_Accuracy_Data_One], ignore_index=True)

        if accuracy_test > best_test_accuracy:
            best_n_estimators = x[0]
            best_max_depth = x[1]
            best_min_child_weight = x[2]
            best_gamma = x[3]
            best_learning_rate = x[4]
            best_train_accuracy = accuracy_train
            best_test_accuracy = accuracy_test

        print(x)

    print(xgb_Test_Accuracy_Data)
    return best_n_estimators, best_max_depth, best_min_child_weight, best_gamma, best_learning_rate, best_train_accuracy, best_test_accuracy

def train_modeL_XGB(best_n_estimators, best_max_depth, best_min_child_weight, best_gamma, best_learning_rate, best_train_accuracy, best_test_accuracy, data, scale):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Random Forest

    # Initialize an empty list to store the values
    Best_Fit_XG_Boost = []

    # Append the values to the list
    Best_Fit_XG_Boost.append(best_n_estimators)
    Best_Fit_XG_Boost.append(best_max_depth)
    Best_Fit_XG_Boost.append(best_min_child_weight)
    Best_Fit_XG_Boost.append(best_gamma)
    Best_Fit_XG_Boost.append(best_learning_rate)
    Best_Fit_XG_Boost.append(best_train_accuracy)
    Best_Fit_XG_Boost.append(best_test_accuracy)

    # Fit  Model
    xgb_reg = xgb.XGBRegressor(n_estimators=Best_Fit_XG_Boost[0],max_depth =Best_Fit_XG_Boost[1],min_child_weight = Best_Fit_XG_Boost[2],gamma = Best_Fit_XG_Boost[3],learning_rate = Best_Fit_XG_Boost[4])
    xgb_reg.fit(X_train, y_train)

    # Use the forest's predict method on the test data
    y_pred = xgb_reg.predict(X_test).reshape(-1, 1)

    y_test = y_test.reshape(-1, 1)

    if scale:
        y_test = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_test))
        y_pred = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_pred))

    test_df = data["test_df"]
    # test_df = add_nan_rows(test_df, k_days-1)
    # add predicted future prices to the dataframe
    test_df["XGB"] = y_pred
    # add true future prices to the dataframe
    test_df["true_Adj Close"] = y_test

    # sort the dataframe by date
    test_df.sort_index(inplace=True)

    final_df = test_df

    return final_df

def plot_graph_XGB(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df['true_Adj Close'], c='b')
    plt.plot(test_df["XGB"], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()
