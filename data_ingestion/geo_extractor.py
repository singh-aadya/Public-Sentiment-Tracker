from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="geo_locator")

def extract_city(text):
    cities = ["Delhi", "Mumbai", "Pune", "Hyderabad", "Bengaluru", "Chennai"]
    for city in cities:
        if city.lower() in text.lower():
            return city
    return None

def enrich_location(record):
    city = extract_city(record)
    if city:
        try:
            location = geolocator.geocode(city)
            if location:
                record["location"] = city
                record["latitude"] = location.latitude
                record["longitude"] = location.longitude
        except:
            pass
    return record
