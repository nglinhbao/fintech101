import pandas as pd
import numpy as np
import sys
import warnings
import itertools
warnings.filterwarnings("ignore")
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
import random
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def calculate_step_wise_SARIMA(series, train):
    # Holt Winter’s Exponential Smoothing (HWES)or Triple Smoothing
    # fit model

    random.seed(10)

    # Plot
    # HWES_Forecast.plot(marker='o', color='red', legend=True)
    # model_fit.fittedvalues.plot(marker='o',  color='blue')

    # fitting a stepwise model to find the best paramters for SARIMA:
    stepwise_fit = pm.auto_arima(train, start_p=1, start_q=1, max_p=3, max_d=2, max_q=3, m=7,
                                 start_P=0, start_Q=0, max_P=3, max_D=3, max_Q=3, seasonal=True, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)  # set to stepwise

    stepwise_fit.summary()
    return stepwise_fit

def train_test_SARIMA(stepwise_fit, k_days, test, true_test):
    # SARIMA

    # Using parameters automatically based on grid search
    SARIMA_Forecast = pd.DataFrame(stepwise_fit.predict(n_periods=len(test)))
    SARIMA_Forecast.columns = ['SARIMA_Forecast']
    # Reindex predictions to match the test DataFrame
    SARIMA_Forecast = SARIMA_Forecast.reindex(true_test.index)
    print(SARIMA_Forecast)

    # Manually fit the model
    # sarima_model = SARIMAX(series, order=(5, 2, 2), seasonal_order=(0, 0, 1, 7))
    # sarima_model_fit = sarima_model.fit(disp=False)
    # make prediction
    # SARIMA_Forecast = model_fit.predict(len(data), len(data))
    # print(yhat)

    return SARIMA_Forecast

def plot_graph(predictions, test):
    """
    This function plots true close price along with predicted close price
    with blue and red colors respectively
    """
    plt.plot(test, c='b')
    plt.plot(predictions, c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()

