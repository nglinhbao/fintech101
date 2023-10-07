import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from  statsmodels.tsa.vector_ar.vecm import *


import itertools
import math
import random
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


def hyper_parameter_tuning_RF(data):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Random Forest
    # Number of trees in random forest
    n_estimators = [500, 1000, 2000]
    # Maximum number of levels in tree
    max_depth = [10, 50, 100]
    # Minimum number of samples required to split a node
    min_samples_split = [50, 100, 200]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 10]

    best_n_estimators = 0
    best_max_depth = 0
    best_min_samples_leaf = 0
    best_min_samples_split = 0
    best_train_accuracy = 0
    best_test_accuracy = 0


    RF_Test_Accuracy_Data = pd.DataFrame(
        columns=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'Train Accuracy', 'Test Accuracy'])

    for x in list(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf)):
        rf = RandomForestRegressor(n_estimators=x[0], max_depth=x[1], min_samples_split=x[2], min_samples_leaf=x[3],
                                   random_state=10, n_jobs=-1, max_features=6)
        # Train the model on training data
        rf.fit(X_train, y_train)

        # Train Data
        # Use the forest's predict method on the train data
        predictions_train = rf.predict(X_train)
        # Calculate the absolute errors
        errors_train = abs(predictions_train - y_train)
        # Calculate mean absolute percentage error (MAPE)
        mape_train = 100 * (errors_train / y_train)
        # Calculate and display accuracy
        accuracy_train = 100 - np.mean(mape_train)

        # Test Data
        # Use the forest's predict method on the test data
        predictions_test = rf.predict(X_test)
        # Calculate the absolute errors
        errors_test = abs(predictions_test - y_test)
        # Calculate mean absolute percentage error (MAPE)
        mape_test = 100 * (errors_test / y_test)
        # Calculate and display accuracy
        accuracy_test = 100 - np.mean(mape_test)

        RF_Test_Accuracy_Data_One = pd.DataFrame(index=range(1), columns=['n_estimators', 'max_depth', 'min_samples_split',
                                                                          'min_samples_leaf', 'Train Accurcay',
                                                                          'Test Accurcay'])

        RF_Test_Accuracy_Data_One.loc[:, 'n_estimators'] = x[0]
        RF_Test_Accuracy_Data_One.loc[:, 'max_depth'] = x[1]
        RF_Test_Accuracy_Data_One.loc[:, 'min_samples_split'] = x[2]
        RF_Test_Accuracy_Data_One.loc[:, 'min_samples_leaf'] = x[3]
        RF_Test_Accuracy_Data_One.loc[:, 'Train Accurcay'] = accuracy_train
        RF_Test_Accuracy_Data_One.loc[:, 'Test Accurcay'] = accuracy_test

        RF_Test_Accuracy_Data = pd.concat([RF_Test_Accuracy_Data, RF_Test_Accuracy_Data_One], ignore_index=True)

        if accuracy_test > best_test_accuracy:
            best_n_estimators =x[0]
            best_max_depth =x[1]
            best_min_samples_leaf = x[3]
            best_min_samples_split = x[2]
            best_train_accuracy = accuracy_train
            best_test_accuracy = accuracy_test

    print(RF_Test_Accuracy_Data)
    return best_n_estimators, best_max_depth, best_min_samples_leaf, best_min_samples_split, best_train_accuracy, best_test_accuracy

def train_modeL_RF(best_n_estimators, best_max_depth, best_min_samples_leaf, best_min_samples_split, best_train_accuracy, best_test_accuracy, data, scale):
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    # Random Forest

    # Initialize an empty list to store the values
    Best_Fit_Random_Forest = []

    # Append the values to the list
    Best_Fit_Random_Forest.append(best_n_estimators)
    Best_Fit_Random_Forest.append(best_max_depth)
    Best_Fit_Random_Forest.append(best_min_samples_leaf)
    Best_Fit_Random_Forest.append(best_min_samples_split)
    Best_Fit_Random_Forest.append(best_train_accuracy)
    Best_Fit_Random_Forest.append(best_test_accuracy)

    print(Best_Fit_Random_Forest)

    # Now, Best_Fit_Random_Forest contains the values in the desired order

    # Fit  Model
    rf = RandomForestRegressor(n_estimators=Best_Fit_Random_Forest[0], max_depth=Best_Fit_Random_Forest[1],
                               min_samples_split=Best_Fit_Random_Forest[2], min_samples_leaf=Best_Fit_Random_Forest[3],
                               random_state=10, n_jobs=-1, max_features=6)  # Train the model on training data
    rf.fit(X_train, y_train)

    # Use the forest's predict method on the test data
    y_pred = rf.predict(X_test).reshape(-1, 1)

    y_test = y_test.reshape(-1, 1)

    if scale:
        y_test = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_test))
        y_pred = np.squeeze(data["column_scaler"]["Adj Close"].inverse_transform(y_pred))

    test_df = data["test_df"]
    # test_df = add_nan_rows(test_df, k_days-1)
    # add predicted future prices to the dataframe
    test_df["RF"] = y_pred
    # add true future prices to the dataframe
    test_df["true_Adj Close"] = y_test

    # sort the dataframe by date
    test_df.sort_index(inplace=True)

    final_df = test_df

    return final_df

def plot_graph_RF(test_df):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test_df['true_Adj Close'], c='b')
    plt.plot(test_df["RF"], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


