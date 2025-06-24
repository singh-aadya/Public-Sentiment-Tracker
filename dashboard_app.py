import streamlit as st
from sentiment_trend import plot_sentiment_trend
from map_visualizer import generate_heatmap

st.set_page_config(layout='wide')
st.title(" Public Sentiment Tracker Dashboard")

# Generate and display map
generate_heatmap('sample_data.csv', 'heatmap.html')
st.subheader(" Grievance Heatmap")
st.components.v1.html(open('heatmap.html', 'r').read(), height=500)

# Plot and display sentiment chart
st.subheader(" Sentiment Trend Over Time")
fig = plot_sentiment_trend('sample_data.csv')
st.plotly_chart(fig, use_container_width=True)
