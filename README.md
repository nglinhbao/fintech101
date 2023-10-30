# Stock Price Prediction for Tesla

During the semester, this program was built through 3 stages: data processing, machine learning, and extension. In the first stage, data processing includes importing, cleaning data and visualization. The second stage covers deep learning model deployment, multivariate, multistep and ensemble methods to combine all the models. For the last stage, I developed a sentiment analysis and integrated it to the model that was built in the previous stages. 

# Instruction

## Creating virtual environment

Firstly, you need to create a virtual environment.

`conda create --name myenv`

Next, as the project uses Python 3.10, you need to create a new environment with Python version 3.10:

`conda create --name myenv python=3.10`

The last step is installing the packages in requirements.txt. There are some version conflicts, however, we want to ignore this for some reasons:

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
