import pandas as pd
import plotly.express as px

def plot_sentiment_trend(data_path='sample_data.csv'):
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.columns = df.columns.str.strip().str.lower()

    trend = df.groupby([df['timestamp'].dt.date, 'sentiment']).size().reset_index(name='count')

    fig = px.line(trend, x='timestamp', y='count', color='sentiment',
                  title='Sentiment Trend Over Time',
                  labels={'timestamp': 'Date', 'count': 'Number of Mentions'})
    return fig
