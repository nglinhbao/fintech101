# Stock Price Prediction for Tesla

In this Final Report, I will discuss how I developed the Stock Price Prediction model for Tesla. This work presents a hybrid deep-learning model using both time series and sentiment analysis. The final product includes data processing and machine learning.

# Instruction

## Creating virtual environment

Firstly, you need to create a virtual environment.

`conda create --name myenv`

Next, as the project uses Python 3.10, you need to create a new environment with Python version 3.10:

`conda create --name myenv python=3.10`

The last step is installing the packages in requirements.txt. There are some version conflicts, however, we want to ignore them for some reasons. Please use the following command to install to packages regardless of version conflicts:

`pip install --no-deps -r requirements.txt`

## Get news

Change the current directory to main.

`cd main`

Finally, you can run the prediction model:

`python main.py`
