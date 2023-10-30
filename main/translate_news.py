import pandas as pd
from textblob import TextBlob
from translate import Translator
from langdetect import detect
from googletrans import Translator


def calculate_sentiment(article_list):
    sentiment_scores = [TextBlob(article).sentiment.polarity for article in article_list]
    return sentiment_scores


def translate_articles(article_list):
    translated_articles = []
    translator = Translator()
    for article in article_list:
        # Detect the language of the article
        lang = detect(article)
        print(f'Detected language: {lang}')

        # If the language is not English, translate it
        if lang != 'en':
            try:
                translation = translator.translate(article, dest='en').text
                translated_articles.append(translation)
                print(f'Translated article: {translation}')
            except Exception as e:
                print(f'Error translating article: {e}')
        else:
            translated_articles.append(article)

    return translated_articles


# Read the CSV file
df = pd.read_csv('./headlines/tesla_headlines_2015_2020.csv')

# Translate the articles and recalculate sentiment
for index, row in df.iterrows():
    # Convert string representation of list to list
    articles = eval(row['articles'])

    # Translate articles
    translated_articles = translate_articles(articles)

    # Recalculate sentiment
    sentiment = calculate_sentiment(translated_articles)
    average_sentiment = sum(sentiment) / len(sentiment) if sentiment else 0

    # Update the dataframe
    df.at[index, 'articles'] = str(translated_articles)
    df.at[index, 'sentiment'] = str(sentiment)
    df.at[index, 'average_sentiment'] = average_sentiment

# Write the updated dataframe to a new CSV file
df.to_csv('./headlines/translated_tesla_headlines_2015_2020.csv', index=False)

