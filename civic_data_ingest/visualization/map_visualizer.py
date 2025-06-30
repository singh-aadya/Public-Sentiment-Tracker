"""
Map Visualizer using Folium
Creates interactive maps showing civic issue density and distribution
"""

import folium
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from collections import Counter, defaultdict

class CivicMapVisualizer:
    def __init__(self, default_location: Tuple[float, float] = (20.5937, 78.9629)):
        """
        Initialize map visualizer
        
        Args:
            default_location: Default center location (lat, lng) - India center by default
        """
        self.default_location = default_location
        self.city_coordinates = {
            'mumbai': (19.0760, 72.8777),
            'delhi': (28.6139, 77.2090),
            'bangalore': (12.9716, 77.5946),
            'chennai': (13.0827, 80.2707),
            'kolkata': (22.5726, 88.3639),
            'hyderabad': (17.3850, 78.4867),
            'pune': (18.5204, 73.8567),
            'ahmedabad': (23.0225, 72.5714),
            'jaipur': (26.9124, 75.7873),
            'surat': (21.1702, 72.8311)
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_city_coordinates(self, city_name: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a city name
        
        Args:
            city_name: Name of the city
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        city_lower = city_name.lower().strip()
        
        # Direct lookup
        if city_lower in self.city_coordinates:
            return self.city_coordinates[city_lower]
        
        # Partial matching
        for city, coords in self.city_coordinates.items():
            if city in city_lower or city_lower in city:
                return coords
        
        return None
    
    def aggregate_data_by_location(self, data: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate civic data by location
        
        Args:
            data: List of processed civic data records
            
        Returns:
            Dictionary with location-based aggregations
        """
        location_data = defaultdict(lambda: {
            'count': 0,
            'sentiments': [],
            'urgencies': [],
            'categories': [],
            'records': [],
            'coordinates': None
        })
        
        for record in data:
            locations = record.get('extracted_locations', [])
            if not locations:
                locations = ['Unknown']
            
            # Use the first valid location
            primary_location = locations[0] if locations else 'Unknown'
            
            # Clean up location name
            location_key = primary_location.split(',')[0].strip().lower()
            
            # Get coordinates
            if location_key != 'unknown':
                coords = self.get_city_coordinates(location_key)
                if coords:
                    location_data[location_key]['coordinates'] = coords
            
            # Aggregate data
            location_data[location_key]['count'] += 1
            location_data[location_key]['sentiments'].append(
                record.get('sentiment', 'neutral')
            )
            location_data[location_key]['urgencies'].append(
                record.get('urgency', 'low')
            )
            
            # Extract categories from keywords
            keywords = record.get('keywords', [])
            for kw in keywords:
                if isinstance(kw, dict):
                    location_data[location_key]['categories'].append(kw.get('keyword', ''))
                else:
                    location_data[location_key]['categories'].append(str(kw))
            
            location_data[location_key]['records'].append(record)
        
        # Calculate aggregated statistics
        for location, data_dict in location_data.items():
            # Sentiment distribution
            sentiment_counts = Counter(data_dict['sentiments'])
            data_dict['sentiment_distribution'] = dict(sentiment_counts)
            
            # Urgency distribution
            urgency_counts = Counter(data_dict['urgencies'])
            data_dict['urgency_distribution'] = dict(urgency_counts)
            
            # Top categories
            category_counts = Counter(data_dict['categories'])
            data_dict['top_categories'] = dict(category_counts.most_common(5))
            
            # Dominant sentiment
            if sentiment_counts:
                data_dict['dominant_sentiment'] = sentiment_counts.most_common(1)[0][0]
            else:
                data_dict['dominant_sentiment'] = 'neutral'
            
            # Dominant urgency
            if urgency_counts:
                data_dict['dominant_urgency'] = urgency_counts.most_common(1)[0][0]
            else:
                data_dict['dominant_urgency'] = 'low'
        
        return dict(location_data)
    
    def create_base_map(self, center_location: Tuple[float, float] = None, zoom_start: int = 6) -> folium.Map:
        """
        Create base map
        
        Args:
            center_location: Center coordinates for the map
            zoom_start: Initial zoom level
            
        Returns:
            Folium map object
        """
        if center_location is None:
            center_location = self.default_location
        
        # Create map
        m = folium.Map(
            location=center_location,
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('cartodbpositron', name='Light Map').add_to(m)
        folium.TileLayer('cartodbdark_matter', name='Dark Map').add_to(m)
        
        return m
    
    def get_marker_color(self, sentiment: str, urgency: str) -> str:
        """
        Get marker color based on sentiment and urgency
        
        Args:
            sentiment: Dominant sentiment
            urgency: Dominant urgency
            
        Returns:
            Color string for the marker
        """
        if urgency == 'high':
            return 'red'
        elif urgency == 'medium':
            return 'orange'
        elif sentiment == 'negative':
            return 'orange'
        elif sentiment == 'positive':
            return 'green'
        else:
            return 'blue'
    
    def get_marker_size(self, count: int, max_count: int) -> int:
        """
        Get marker size based on issue count
        
        Args:
            count: Number of issues in this location
            max_count: Maximum count across all locations
            
        Returns:
            Marker radius
        """
        if max_count == 0:
            return 5
        
        # Scale between 5 and 25
        normalized = count / max_count
        return int(5 + (normalized * 20))
    
    def create_heatmap(self, data: List[Dict], save_path: str = "civic_issues_heatmap.html") -> folium.Map:
        """
        Create heatmap of civic issues
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Folium map with heatmap
        """
        self.logger.info("Creating civic issues heatmap")
        
        # Aggregate data
        location_data = self.aggregate_data_by_location(data)
        
        # Create base map
        m = self.create_base_map()
        
        # Add markers for each location
        max_count = max([loc_data['count'] for loc_data in location_data.values()] + [1])
        
        for location, loc_data in location_data.items():
            if loc_data['coordinates'] is None:
                continue
            
            coords = loc_data['coordinates']
            count = loc_data['count']
            
            # Get marker properties
            color = self.get_marker_color(
                loc_data['dominant_sentiment'],
                loc_data['dominant_urgency']
            )
            radius = self.get_marker_size(count, max_count)
            
            # Create popup content
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 250px;">
                <h4>{location.title()}</h4>
                <p><strong>Total Issues:</strong> {count}</p>
                <p><strong>Dominant Sentiment:</strong> {loc_data['dominant_sentiment'].title()}</p>
                <p><strong>Dominant Urgency:</strong> {loc_data['dominant_urgency'].title()}</p>
                
                <h5>Sentiment Distribution:</h5>
                <ul>
                {chr(10).join([f"<li>{sentiment.title()}: {count}</li>" 
                              for sentiment, count in loc_data['sentiment_distribution'].items()])}
                </ul>
                
                <h5>Top Issues:</h5>
                <ul>
                {chr(10).join([f"<li>{category}: {count}</li>" 
                              for category, count in list(loc_data['top_categories'].items())[:3]])}
                </ul>
            </div>
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=coords,
                radius=radius,
                popup=folium.Popup(popup_content, max_width=300),
                color='black',
                weight=1,
                fillColor=color,
                fillOpacity=0.7,
                tooltip=f"{location.title()}: {count} issues"
            ).add_to(m)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Civic Issues Map</h4>
        <p><span style="color:red;">●</span> High Urgency</p>
        <p><span style="color:orange;">●</span> Medium Urgency/Negative</p>
        <p><span style="color:green;">●</span> Positive Issues</p>
        <p><span style="color:blue;">●</span> Neutral Issues</p>
        <p><small>Size indicates issue count</small></p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(save_path)
        self.logger.info(f"Heatmap saved to {save_path}")
        
        return m
    
    def create_category_map(self, data: List[Dict], save_path: str = "civic_categories_map.html") -> folium.Map:
        """
        Create map showing civic issue categories
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Folium map with categorized markers
        """
        self.logger.info("Creating civic categories map")
        
        # Aggregate data
        location_data = self.aggregate_data_by_location(data)
        
        # Create base map
        m = self.create_base_map()
        
        # Color mapping for categories
        category_colors = {
            'water': '#1f77b4',
            'power': '#ff7f0e', 
            'road': '#2ca02c',
            'traffic': '#d62728',
            'garbage': '#9467bd',
            'municipal': '#8c564b',
            'infrastructure': '#e377c2',
            'transport': '#7f7f7f',
            'other': '#bcbd22'
        }
        
        # Create feature groups for different categories
        feature_groups = {}
        for category in category_colors.keys():
            fg = folium.FeatureGroup(name=f"{category.title()} Issues")
            feature_groups[category] = fg
            m.add_child(fg)
        
        # Add markers for each location
        for location, loc_data in location_data.items():
            if loc_data['coordinates'] is None:
                continue
            
            coords = loc_data['coordinates']
            top_categories = loc_data['top_categories']
            
            if not top_categories:
                continue
            
            # Get the most common category
            primary_category = list(top_categories.keys())[0].lower()
            
            # Determine category group
            category_group = 'other'
            for cat in category_colors.keys():
                if cat in primary_category:
                    category_group = cat
                    break
            
            # Create marker
            marker = folium.Marker(
                location=coords,
                popup=f"""
                <div style="font-family: Arial, sans-serif;">
                    <h4>{location.title()}</h4>
                    <p><strong>Primary Category:</strong> {primary_category.title()}</p>
                    <p><strong>Total Issues:</strong> {loc_data['count']}</p>
                </div>
                """,
                tooltip=f"{location.title()}: {primary_category.title()}",
                icon=folium.Icon(
                    color='white',
                    icon_color=category_colors[category_group],
                    icon='info-sign',
                    prefix='glyphicon'
                )
            )
            
            # Add to appropriate feature group
            feature_groups[category_group].add_child(marker)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(save_path)
        self.logger.info(f"Category map saved to {save_path}")
        
        return m
    
    def create_sentiment_timeline_map(self, data: List[Dict], save_path: str = "sentiment_timeline_map.html") -> folium.Map:
        """
        Create map showing sentiment over time
        
        Args:
            data: List of processed civic data records
            save_path: Path to save the HTML file
            
        Returns:
            Folium map with timeline features
        """
        self.logger.info("Creating sentiment timeline map")
        
        # This would be more complex with time-based filtering
        # For now, create a basic sentiment map
        location_data = self.aggregate_data_by_location(data)
        
        # Create base map
        m = self.create_base_map()
        
        # Add sentiment-based markers
        for location, loc_data in location_data.items():
            if loc_data['coordinates'] is None:
                continue
            
            coords = loc_data['coordinates']
            sentiment_dist = loc_data['sentiment_distribution']
            
            # Create pie chart-like visualization (simplified)
            total = sum(sentiment_dist.values())
            if total == 0:
                continue
            
            # Calculate percentages
            percentages = {k: (v/total)*100 for k, v in sentiment_dist.items()}
            
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 200px;">
                <h4>{location.title()}</h4>
                <p><strong>Total Issues:</strong> {total}</p>
                <h5>Sentiment Distribution:</h5>
                {chr(10).join([f"<p>{sentiment.title()}: {perc:.1f}%</p>" 
                              for sentiment, perc in percentages.items()])}
            </div>
            """
            
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_content, max_width=250),
                tooltip=f"{location.title()}: {total} issues",
                icon=folium.Icon(
                    color=self.get_marker_color(loc_data['dominant_sentiment'], 'low'),
                    icon='time',
                    prefix='glyphicon'
                )
            ).add_to(m)
        
        # Save map
        m.save(save_path)
        self.logger.info(f"Sentiment timeline map saved to {save_path}")
        
        return m

def main():
    """Test the map visualizer"""
    # Create sample data
    sample_data = [
        {
            'extracted_locations': ['Mumbai'],
            'sentiment': 'negative',
            'urgency': 'high',
            'keywords': [{'keyword': 'water crisis'}, {'keyword': 'shortage'}]
        },
        {
            'extracted_locations': ['Mumbai'],
            'sentiment': 'negative', 
            'urgency': 'medium',
            'keywords': [{'keyword': 'power outage'}, {'keyword': 'electricity'}]
        },
        {
            'extracted_locations': ['Delhi'],
            'sentiment': 'negative',
            'urgency': 'high',
            'keywords': [{'keyword': 'road condition'}, {'keyword': 'pothole'}]
        },
        {
            'extracted_locations': ['Bangalore'],
            'sentiment': 'positive',
            'urgency': 'low',
            'keywords': [{'keyword': 'transport improved'}, {'keyword': 'bus service'}]
        }
    ]
    
    print("Testing Civic Map Visualizer:")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = CivicMapVisualizer()
    
    # Create heatmap
    print("Creating heatmap...")
    heatmap = visualizer.create_heatmap(sample_data, "test_heatmap.html")
    print("✓ Heatmap created: test_heatmap.html")
    
    # Create category map
    print("Creating category map...")
    category_map = visualizer.create_category_map(sample_data, "test_category_map.html")
    print("✓ Category map created: test_category_map.html")
    
    # Create sentiment timeline map
    print("Creating sentiment timeline map...")
    timeline_map = visualizer.create_sentiment_timeline_map(sample_data, "test_timeline_map.html")
    print("✓ Timeline map created: test_timeline_map.html")
    
    print(f"\n🗺️  Maps created successfully!")
    print("Open the HTML files in your browser to view the interactive maps.")

if __name__ == "__main__":
    main()