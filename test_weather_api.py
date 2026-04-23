"""
Test weather API to verify it returns different data for different cities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.api.weather import WeatherAPI
from dotenv import load_dotenv

def test_weather_api():
    load_dotenv()
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("ERROR: No API key found")
        return
    
    weather_api = WeatherAPI(api_key)
    
    # Test different cities
    cities = [
        ("New Delhi", 28.61, 77.23),
        ("Mumbai", 19.07, 72.87),
        ("Chennai", 13.08, 80.27),
        ("Kolkata", 22.57, 88.36),
        ("Bangalore", 12.97, 77.59)
    ]
    
    print("=" * 70)
    print("TESTING WEATHER API FOR DIFFERENT CITIES")
    print("=" * 70)
    print()
    
    results = []
    for city, lat, lon in cities:
        try:
            weather = weather_api.get_weather(lat, lon, use_cache=False)
            results.append({
                'city': city,
                'lat': lat,
                'lon': lon,
                'temp': weather.get('temperature'),
                'humidity': weather.get('humidity'),
                'wind': weather.get('wind_speed'),
                'rain': weather.get('rainfall')
            })
            print(f"{city}: Temp={weather.get('temperature')}°C, Humidity={weather.get('humidity')}%, Wind={weather.get('wind_speed')} km/h")
        except Exception as e:
            print(f"{city}: ERROR - {str(e)}")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    temps = [r['temp'] for r in results]
    if len(set(temps)) > 1:
        print("✓ Different temperatures found for different cities")
    else:
        print("⚠️  Same temperature for all cities - API may be returning default/cached data")

if __name__ == "__main__":
    test_weather_api()
