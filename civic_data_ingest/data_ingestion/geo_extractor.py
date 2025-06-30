"""
Geographic Location Extraction
Extracts location information from text data using various methods
"""

import re
import json
import logging
from typing import List, Dict, Optional, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time

class GeoExtractor:
    def __init__(self):
        """Initialize geo extractor with geocoder and location patterns"""
        self.geocoder = Nominatim(user_agent="civic_data_ingest")
        
        # Indian cities and states patterns
        self.indian_cities = [
            'mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'kolkata',
            'pune', 'ahmedabad', 'surat', 'jaipur', 'lucknow', 'kanpur',
            'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam', 'pimpri',
            'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik',
            'coimbatore', 'madurai', 'meerut', 'rajkot', 'varanasi', 'srinagar',
            'aurangabad', 'dhanbad', 'amritsar', 'navi mumbai', 'allahabad',
            'ranchi', 'howrah', 'jabalpur', 'gwalior', 'chandigarh'
        ]
        
        self.indian_states = [
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
            'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand',
            'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
            'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab', 'rajasthan',
            'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh',
            'uttarakhand', 'west bengal', 'delhi', 'puducherry', 'jammu and kashmir',
            'ladakh'
        ]
        
        # Location patterns
        self.location_patterns = [
            r'\b(?:in|at|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:city|district|area|zone|sector)\b',
            r'\b(?:located in|based in|happening in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_locations_from_text(self, text: str) -> List[str]:
        """
        Extract potential location names from text using regex patterns
        
        Args:
            text: Input text
            
        Returns:
            List of potential location names
        """
        if not text:
            return []
        
        locations = []
        
        # Apply regex patterns
        for pattern in self.location_patterns:
            matches = re.findall(pattern, text)
            locations.extend(matches)
        
        # Look for known Indian cities and states
        text_lower = text.lower()
        for city in self.indian_cities:
            if city in text_lower:
                locations.append(city.title())
        
        for state in self.indian_states:
            if state in text_lower:
                locations.append(state.title())
        
        # Remove duplicates and clean up
        locations = list(set(locations))
        locations = [loc.strip() for loc in locations if len(loc.strip()) > 2]
        
        return locations
    
    def geocode_location(self, location: str, country: str = 'India') -> Optional[Dict]:
        """
        Get coordinates for a location using geocoding
        
        Args:
            location: Location name
            country: Country context (default: India)
            
        Returns:
            Dictionary with location data or None
        """
        try:
            # Add country context for better results
            query = f"{location}, {country}"
            
            # Rate limiting
            time.sleep(1)
            
            location_obj = self.geocoder.geocode(query, timeout=10)
            
            if location_obj:
                return {
                    'name': location,
                    'full_address': location_obj.address,
                    'latitude': location_obj.latitude,
                    'longitude': location_obj.longitude,
                    'bbox': location_obj.raw.get('boundingbox', None)
                }
            else:
                return None
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            self.logger.warning(f"Geocoding failed for {location}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error geocoding {location}: {str(e)}")
            return None
    
    def extract_and_geocode(self, text: str) -> List[Dict]:
        """
        Extract locations from text and geocode them
        
        Args:
            text: Input text
            
        Returns:
            List of geocoded location dictionaries
        """
        # Extract potential locations
        potential_locations = self.extract_locations_from_text(text)
        
        geocoded_locations = []
        
        for location in potential_locations:
            geocoded = self.geocode_location(location)
            if geocoded:
                geocoded_locations.append(geocoded)
        
        return geocoded_locations
    
    def process_social_media_location(self, user_location: str, geo_data: Dict = None) -> Optional[Dict]:
        """
        Process location from social media posts (Twitter user location, geo data)
        
        Args:
            user_location: User's profile location
            geo_data: Tweet geo data
            
        Returns:
            Processed location data
        """
        location_info = {}
        
        # Process geo data if available
        if geo_data and 'coordinates' in geo_data:
            location_info['type'] = 'precise'
            location_info['coordinates'] = geo_data['coordinates']
            
            # Reverse geocode to get address
            try:
                coords = geo_data['coordinates']['coordinates']  # [lng, lat]
                lat, lng = coords[1], coords[0]
                
                time.sleep(1)  # Rate limiting
                address = self.geocoder.reverse(f"{lat}, {lng}", timeout=10)
                
                if address:
                    location_info['address'] = address.address
                    location_info['latitude'] = lat
                    location_info['longitude'] = lng
                    
            except Exception as e:
                self.logger.warning(f"Reverse geocoding failed: {str(e)}")
        
        # Process user location
        elif user_location:
            location_info['type'] = 'approximate'
            location_info['user_location'] = user_location
            
            # Try to geocode user location
            geocoded = self.geocode_location(user_location)
            if geocoded:
                location_info.update(geocoded)
        
        return location_info if location_info else None
    
    def aggregate_locations_by_region(self, locations: List[Dict]) -> Dict[str, int]:
        """
        Aggregate locations by region/city for visualization
        
        Args:
            locations: List of location dictionaries
            
        Returns:
            Dictionary with region counts
        """
        region_counts = {}
        
        for location in locations:
            # Extract city/region from full address
            address = location.get('full_address', '')
            
            # Simple region extraction (can be improved)
            region = None
            
            # Look for known cities in address
            for city in self.indian_cities:
                if city.lower() in address.lower():
                    region = city.title()
                    break
            
            # If no city found, use the location name
            if not region:
                region = location.get('name', 'Unknown')
            
            region_counts[region] = region_counts.get(region, 0) + 1
        
        return region_counts

def main():
    """Test the geo extractor"""
    extractor = GeoExtractor()
    
    # Test text samples
    test_texts = [
        "Water shortage in Mumbai affecting thousands of residents",
        "Potholes on the roads near Bangalore causing traffic problems",
        "Power outage in Delhi for 3 hours yesterday",
        "Garbage collection issues in Pune's Koregaon Park area"
    ]
    
    print("Testing location extraction:")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: {text}")
        
        # Extract locations
        locations = extractor.extract_locations_from_text(text)
        print(f"   Extracted locations: {locations}")
        
        # Geocode first location if available
        if locations:
            geocoded = extractor.geocode_location(locations[0])
            if geocoded:
                print(f"   Geocoded: {geocoded['name']} -> ({geocoded['latitude']}, {geocoded['longitude']})")
    
    # Test social media location processing
    print("\n\nTesting social media location processing:")
    user_location = "Mumbai, Maharashtra"
    geo_data = {
        'coordinates': {
            'coordinates': [72.8777, 19.0760],  # Mumbai coordinates [lng, lat]
            'type': 'Point'
        }
    }
    
    result = extractor.process_social_media_location(user_location, geo_data)
    print(f"Processed location: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    main()