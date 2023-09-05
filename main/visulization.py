import pandas as pd
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, Whisker
from bokeh.io import show
from bokeh.layouts import column
import matplotlib.pyplot as plt

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