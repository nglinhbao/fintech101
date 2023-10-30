# Stock Price Prediction for Tesla

In this Final Report, I will discuss how I developed the Stock Price Prediction model for Tesla. This work presents a hybrid deep-learning model using both time series and sentiment analysis. The final product includes data processing and machine learning.

# Instruction

## Creating virtual environment

Firstly, you need to create a virtual environment.

`conda create --name myenv`

Next, as the project uses Python 3.10, you need to create a new environment with Python version 3.10:

`conda create --name myenv python=3.10`

The last step is installing the packages in requirements.txt. There are some version conflicts, however, we want to ignore them for some reasons:

`pip install --no-deps -r requirements.txt`

## Get news

Change the current directory to main.

`cd main`

Run get_news.py to get the headlines about Tesla:

`python get_news.py`

Then, sentiment scores for those headlines need calculating:

`python sentiment-analysis.py`

The result of this function will be stored in `tesla-sentiment-result/tesla-headlines-{start_year}-{end_year}-final.csv`.

After finishing all of these stages, you can run the prediction model:

`python main.py`
