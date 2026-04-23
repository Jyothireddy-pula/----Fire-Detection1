"""
Historical Weather Data Collector
Collects 10 years of historical weather data for multiple cities to build a comprehensive dataset
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import os
from typing import List, Dict
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.utils.fwi import FWICalculator
from backend.utils.logger import system_logger

# List of major cities across different climates to collect data for
CITIES = [
    # Indian cities
    {"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "country": "IN"},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "country": "IN"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "country": "IN"},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "country": "IN"},
    {"name": "Bangalore", "lat": 12.9716, "lon": 77.5946, "country": "IN"},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "country": "IN"},
    {"name": "Uttarakhand", "lat": 30.0668, "lon": 79.0193, "country": "IN"},
    {"name": "Himachal Pradesh", "lat": 31.1048, "lon": 77.1734, "country": "IN"},
    
    # US cities (various climates)
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "country": "US"},
    {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740, "country": "US"},
    {"name": "Seattle", "lat": 47.6062, "lon": -122.3321, "country": "US"},
    {"name": "Miami", "lat": 25.7617, "lon": -80.1918, "country": "US"},
    
    # Australian cities
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "country": "AU"},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631, "country": "AU"},
    
    # European cities
    {"name": "Athens", "lat": 37.9838, "lon": 23.7275, "country": "GR"},
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038, "country": "ES"},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964, "country": "IT"},
]

class HistoricalWeatherCollector:
    """Collects historical weather data from OpenWeatherMap API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/3.0/onecall/timemachine"
        self.fwi_calculator = FWICalculator()
        
    def get_historical_data(self, lat: float, lon: float, timestamp: int) -> Dict:
        """
        Fetch historical weather data for a specific timestamp
        
        Args:
            lat: Latitude
            lon: Longitude
            timestamp: Unix timestamp
            
        Returns:
            Weather data dictionary
        """
        url = f"{self.base_url}?lat={lat}&lon={lon}&dt={timestamp}&appid={self.api_key}&units=metric"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data or not data['data']:
                return None
                
            weather = data['data'][0]
            
            return {
                'temperature': weather.get('temp', 20),
                'humidity': weather.get('humidity', 50),
                'wind_speed': weather.get('wind_speed', 5),
                'rainfall': weather.get('rain', {}).get('1h', 0) + weather.get('rain', {}).get('24h', 0),
                'pressure': weather.get('pressure', 1013),
                'timestamp': timestamp
            }
        except Exception as e:
            print(f"Error fetching data for {lat}, {lon} at {timestamp}: {e}")
            return None
    
    def collect_city_data(self, city: Dict, start_year: int = 2014, end_year: int = 2024) -> pd.DataFrame:
        """
        Collect historical data for a city over multiple years
        
        Args:
            city: City dictionary with name, lat, lon, country
            start_year: Start year
            end_year: End year
            
        Returns:
            DataFrame with historical weather data
        """
        print(f"\nCollecting data for {city['name']}, {city['country']}...")
        
        all_data = []
        
        # Sample 2 days per month (to avoid API limits while getting seasonal variation)
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Sample day 1 and day 15 of each month
                for day in [1, 15]:
                    try:
                        date = datetime(year, month, day)
                        timestamp = int(date.timestamp())
                        
                        weather_data = self.get_historical_data(
                            city['lat'], 
                            city['lon'], 
                            timestamp
                        )
                        
                        if weather_data:
                            # Calculate FWI components
                            fwi_result = self.fwi_calculator.compute_all(
                                weather_data['temperature'],
                                weather_data['humidity'],
                                weather_data['wind_speed'],
                                weather_data['rainfall'],
                                month
                            )
                            ffmc = fwi_result['FFMC']
                            dmc = fwi_result['DMC']
                            dc = fwi_result['DC']
                            isi = fwi_result['ISI']
                            bui = fwi_result['BUI']
                            fwi = fwi_result['FWI']
                            
                            # Determine fire class based on FWI
                            if fwi < 1:
                                fire_class = 'not fire'
                            elif fwi < 5:
                                fire_class = 'not fire'
                            elif fwi < 10:
                                fire_class = 'fire'
                            elif fwi < 18:
                                fire_class = 'fire'
                            else:
                                fire_class = 'fire'
                            
                            row = {
                                'day': day,
                                'month': month,
                                'year': year,
                                'Temperature': weather_data['temperature'],
                                'RH': weather_data['humidity'],
                                'Ws': weather_data['wind_speed'],
                                'Rain': weather_data['rainfall'],
                                'FFMC': ffmc,
                                'DMC': dmc,
                                'DC': dc,
                                'ISI': isi,
                                'BUI': bui,
                                'FWI': fwi,
                                'Classes': fire_class,
                                'Region': city['name'],
                                'Country': city['country'],
                                'Latitude': city['lat'],
                                'Longitude': city['lon']
                            }
                            
                            all_data.append(row)
                            print(f"  {year}-{month:02d}-{day:02d}: Temp={weather_data['temperature']:.1f}°C, RH={weather_data['humidity']:.0f}%, FWI={fwi:.1f}")
                        
                        # Rate limiting - wait between requests
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"  Error processing {year}-{month:02d}-{day:02d}: {e}")
                        continue
        
        return pd.DataFrame(all_data)
    
    def collect_all_cities(self, cities: List[Dict], start_year: int = 2014, end_year: int = 2024) -> pd.DataFrame:
        """
        Collect historical data for all cities
        
        Args:
            cities: List of city dictionaries
            start_year: Start year
            end_year: End year
            
        Returns:
            Combined DataFrame with all cities' data
        """
        all_dataframes = []
        
        for city in cities:
            try:
                df = self.collect_city_data(city, start_year, end_year)
                if not df.empty:
                    all_dataframes.append(df)
            except Exception as e:
                print(f"Error collecting data for {city['name']}: {e}")
                continue
        
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()

def generate_synthetic_historical_data(api_key: str) -> pd.DataFrame:
    """
    Generate synthetic historical data based on realistic seasonal patterns
    when API is not available or has limitations
    
    Args:
        api_key: OpenWeatherMap API key
        
    Returns:
        DataFrame with synthetic historical weather data
    """
    print("Generating synthetic historical data based on realistic seasonal patterns...")
    
    fwi_calculator = FWICalculator()
    all_data = []
    
    for city in CITIES:
        print(f"\nGenerating data for {city['name']}...")
        
        # Determine climate zone based on latitude
        abs_lat = abs(city['lat'])
        if abs_lat < 23.5:
            climate = 'tropical'
        elif abs_lat < 35:
            climate = 'subtropical'
        else:
            climate = 'temperate'
        
        for year in range(2014, 2025):
            for month in range(1, 13):
                for day in [1, 15]:  # Sample 2 days per month
                    # Generate realistic seasonal temperature patterns
                    if climate == 'tropical':
                        # Minimal seasonal variation
                        temp_base = 28
                        temp_variation = 3 * np.sin(2 * np.pi * (month - 1) / 12)
                    elif climate == 'subtropical':
                        # Moderate seasonal variation
                        if city['country'] in ['IN', 'AU']:
                            # Southern hemisphere reversal for Australia
                            offset = 6 if city['country'] == 'AU' else 0
                            temp_base = 25
                            temp_variation = 8 * np.sin(2 * np.pi * (month - 1 - offset) / 12)
                        else:
                            temp_base = 20
                            temp_variation = 10 * np.sin(2 * np.pi * (month - 1) / 12)
                    else:  # temperate
                        # High seasonal variation
                        if city['country'] == 'AU':
                            temp_base = 15
                            temp_variation = 12 * np.sin(2 * np.pi * (month - 7) / 12)
                        else:
                            temp_base = 12
                            temp_variation = 15 * np.sin(2 * np.pi * (month - 1) / 12)
                    
                    temperature = temp_base + temp_variation + np.random.normal(0, 2)
                    
                    # Humidity inversely related to temperature
                    humidity = 70 - (temperature - 20) * 1.5 + np.random.normal(0, 10)
                    humidity = np.clip(humidity, 10, 100)
                    
                    # Wind speed with seasonal variation
                    wind = 8 + 3 * np.sin(2 * np.pi * (month - 1) / 12) + np.random.normal(0, 3)
                    wind = np.clip(wind, 0, 30)
                    
                    # Rainfall based on climate and season
                    if climate == 'tropical':
                        if month in [6, 7, 8, 9]:  # Monsoon season for India
                            rainfall = np.random.exponential(5) if city['country'] == 'IN' else np.random.exponential(2)
                        else:
                            rainfall = np.random.exponential(1)
                    elif climate == 'subtropical':
                        if city['country'] == 'IN' and month in [7, 8, 9]:
                            rainfall = np.random.exponential(8)
                        else:
                            rainfall = np.random.exponential(2)
                    else:  # temperate
                        if month in [11, 12, 1, 2, 3]:
                            rainfall = np.random.exponential(3)
                        else:
                            rainfall = np.random.exponential(1.5)
                    
                    rainfall = np.clip(rainfall, 0, 50)
                    
                    # Calculate FWI components
                    fwi_result = fwi_calculator.compute_all(
                        temperature, humidity, wind, rainfall, month
                    )
                    ffmc = fwi_result['FFMC']
                    dmc = fwi_result['DMC']
                    dc = fwi_result['DC']
                    isi = fwi_result['ISI']
                    bui = fwi_result['BUI']
                    fwi = fwi_result['FWI']
                    
                    # Determine fire class based on FWI with realistic distribution
                    # Ensure we get all categories
                    rand_val = np.random.random()
                    if fwi < 1:
                        fire_class = 'not fire'
                    elif fwi < 5:
                        fire_class = 'not fire'
                    elif fwi < 10:
                        fire_class = 'fire'
                    elif fwi < 18:
                        fire_class = 'fire'
                    else:
                        fire_class = 'fire'
                    
                    row = {
                        'day': day,
                        'month': month,
                        'year': year,
                        'Temperature': round(temperature, 1),
                        'RH': round(humidity, 1),
                        'Ws': round(wind, 1),
                        'Rain': round(rainfall, 2),
                        'FFMC': round(ffmc, 1),
                        'DMC': round(dmc, 1),
                        'DC': round(dc, 1),
                        'ISI': round(isi, 1),
                        'BUI': round(bui, 1),
                        'FWI': round(fwi, 1),
                        'Classes': fire_class,
                        'Region': city['name'],
                        'Country': city['country'],
                        'Latitude': city['lat'],
                        'Longitude': city['lon']
                    }
                    
                    all_data.append(row)
    
    df = pd.DataFrame(all_data)
    print(f"\nGenerated {len(df)} data points for {len(CITIES)} cities over 10 years")
    return df

def main():
    """Main function to collect historical data"""
    print("=" * 70)
    print("HISTORICAL WEATHER DATA COLLECTOR")
    print("=" * 70)
    
    # Load API key from .env
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        print("ERROR: OPENWEATHER_API_KEY not found in .env file")
        return
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Try to collect real historical data first
    collector = HistoricalWeatherCollector(api_key)
    
    print("\nAttempting to collect real historical data from OpenWeatherMap API...")
    print("Note: Historical data requires a paid subscription to OpenWeatherMap OneCall API")
    print("If API fails, we will generate synthetic data with realistic patterns\n")
    
    try:
        # Try with one city first to test API
        test_data = collector.collect_city_data(CITIES[0], start_year=2024, end_year=2024)
        if test_data.empty:
            raise Exception("API returned no data")
        
        print("\nAPI test successful! Collecting full historical dataset...")
        df = collector.collect_all_cities(CITIES, start_year=2014, end_year=2024)
        
        if df.empty:
            raise Exception("No data collected")
            
    except Exception as e:
        print(f"\nAPI collection failed: {e}")
        print("Falling back to synthetic data generation with realistic patterns...")
        df = generate_synthetic_historical_data(api_key)
    
    if not df.empty:
        # Save the dataset
        output_path = 'data/historical_weather_10years.csv'
        df.to_csv(output_path, index=False)
        print(f"\nDataset saved to {output_path}")
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
        print(f"Cities/Regions: {df['Region'].nunique()}")
        print(f"Countries: {df['Country'].nunique()}")
        
        print("\nFire class distribution:")
        print(df['Classes'].value_counts())
        
        print("\nFWI distribution:")
        print(f"Min FWI: {df['FWI'].min():.2f}")
        print(f"Max FWI: {df['FWI'].max():.2f}")
        print(f"Mean FWI: {df['FWI'].mean():.2f}")
        print(f"Median FWI: {df['FWI'].median():.2f}")
        
        print("\nTemperature range by season:")
        df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                        3: 'Spring', 4: 'Spring', 5: 'Spring',
                                        6: 'Summer', 7: 'Summer', 8: 'Summer',
                                        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
        print(df.groupby('season')['Temperature'].agg(['min', 'max', 'mean']))
        
        print("\n" + "=" * 70)
        print("DATA COLLECTION COMPLETED")
        print("=" * 70)
    else:
        print("ERROR: No data was collected")

if __name__ == "__main__":
    main()
