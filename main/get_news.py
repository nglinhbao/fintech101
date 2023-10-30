import pandas as pd
from pygooglenews import GoogleNews
from datetime import datetime, timedelta
from datetime import date
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def get_news_df(ticker, start_date, end_date):
    gn = GoogleNews()

    if ticker == "TSLA": search = "Tesla"

    df = pd.DataFrame(get_news(gn, search, start_date, end_date))

    # Set 'date' as the index
    df.set_index('date', inplace=True)

    # Apply the function to the 'articles' column and create a new column 'sentiment'
    # df['textblob sentiment'] = df['articles'].apply(calculate_sentiment)

    print(df)

    # Get today's date
    today = date.today()

    # Format today's date as YYYY-MM-DD
    today_str = today.strftime("%Y-%m-%d")

    # Add today's date to the file name
    df.to_csv(f'./headlines/tesla_headlines_{start_date[:4]}_{end_date[:4]}.csv')

def get_news(gn, search, start_date, end_date):
    stories = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    delta = timedelta(days=1)

    while start_date <= end_date:
        result = gn.search(search, from_=start_date.strftime('%Y-%m-%d'), to_=(start_date + delta).strftime('%Y-%m-%d'))
        newsitem = result['entries']

        # Only take the first 5 articles
        # newsitem = newsitem[:5]

        # Group the articles of the same day into a single row
        articles = []
        for item in newsitem:
            try:
                if detect(item.title) == "en":
                    articles.append(item.title)
            except LangDetectException:
                # Ignore headlines that cause errors in language detection
                continue
        story = {
            'date': start_date,
            'articles': articles
        }
        print(start_date)
        stories.append(story)
        start_date += delta

    return stories

start_date = '2017-01-01'
end_date = '2020-01-01'
get_news_df("TSLA", start_date, end_date)




