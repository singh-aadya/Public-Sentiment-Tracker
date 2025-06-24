import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
from collections import Counter

def generate_heatmap(data_path='sample_data.csv', output_path='heatmap.html'):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip().str.lower()

    # Check required columns
    required = {'latitude', 'longitude', 'sentiment', 'urgency', 'category'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {required}")

    # Base map
    m = folium.Map(location=[22.9734, 78.6569], zoom_start=5)

    # Add heatmap
    heat_data = df[['latitude', 'longitude']].values.tolist()
    HeatMap(heat_data, radius=12, blur=15).add_to(m)

    # Add markers per unique (lat, long)
    marker_cluster = MarkerCluster().add_to(m)
    grouped = df.groupby(['latitude', 'longitude'])
    # Map coordinates to city name (hardcoded for demo)
    coord_to_city = {
        (28.7041, 77.1025): 'Delhi',
        (19.0760, 72.8777): 'Mumbai',
        (12.9716, 77.5946): 'Bangalore',
    }

    for (lat, lon), group in grouped:
        total_issues = len(group)
        sentiment_counts = group['sentiment'].value_counts().to_dict()
        urgency_counts = group['urgency'].value_counts().to_dict()
        category_counts = group['category'].value_counts()

        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        dominant_urgency = max(urgency_counts, key=urgency_counts.get)
        top_issues = ', '.join(category_counts.head(3).index)

        # Get city name from lat/lon (rounded for matching)
        city_name = coord_to_city.get((round(lat, 4), round(lon, 4)), f"({lat}, {lon})")

        popup_html = f"""
        <b>City:</b> {city_name}<br>
        <b>Total Issues:</b> {total_issues}<br>
        <b>Dominant Sentiment:</b> {dominant_sentiment}<br>
        <b>Dominant Urgency:</b> {dominant_urgency}<br>
        <b>Sentiment Distribution:</b><br>
        &nbsp;&nbsp;&nbsp;&nbsp;Positive: {sentiment_counts.get('positive', 0)}<br>
        &nbsp;&nbsp;&nbsp;&nbsp;Negative: {sentiment_counts.get('negative', 0)}<br>
        &nbsp;&nbsp;&nbsp;&nbsp;Neutral: {sentiment_counts.get('neutral', 0)}<br>
        <b>Top Issues:</b> {top_issues}
        """

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{city_name} - {total_issues} issues"
        ).add_to(marker_cluster)

    m.save(output_path)
