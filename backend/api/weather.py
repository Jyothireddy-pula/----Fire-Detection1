"""
Weather API Module
Integrates with OpenWeatherMap API for real-time weather data
"""

import requests
import time
from typing import Dict, Optional
from backend.utils.cache import weather_cache
from backend.utils.logger import system_logger


class WeatherAPI:
    """OpenWeatherMap API client with caching and error handling"""
    
    def __init__(self, api_key: str, timeout: int = 5, max_retries: int = 3):
        """
        Initialize Weather API client
        
        Args:
            api_key: OpenWeatherMap API key
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.api_failure_count = 0
        self.last_valid_cache = None
    
    def get_weather(self, lat: float, lon: float, use_cache: bool = True) -> Dict:
        """
        Get weather data from API
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with weather data
        """
        # Check cache first
        if use_cache:
            cached_data = weather_cache.get('weather', lat=lat, lon=lon)
            if cached_data is not None:
                system_logger.info(f"Using cached weather data for {lat}, {lon}")
                self.last_valid_cache = cached_data
                return cached_data
        
        # Try to fetch from API
        for attempt in range(self.max_retries):
            try:
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.api_key,
                    'units': 'metric'
                }
                
                start_time = time.time()
                response = requests.get(self.base_url, params=params, timeout=self.timeout)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    weather_data = self._parse_weather_data(data)
                    
                    # Cache the result
                    weather_cache.set('weather', weather_data, ttl=300, lat=lat, lon=lon)
                    
                    self.api_failure_count = 0
                    self.last_valid_cache = weather_data
                    
                    system_logger.log_api_call(
                        endpoint='weather',
                        params={'lat': lat, 'lon': lon},
                        status='success',
                        response_time=response_time
                    )
                    
                    return weather_data
                else:
                    system_logger.warning(f"API returned status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                system_logger.warning(f"API timeout on attempt {attempt + 1}")
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                system_logger.error(f"API request failed: {str(e)}")
                time.sleep(1)
                
            except Exception as e:
                system_logger.error(f"Unexpected error: {str(e)}")
                time.sleep(1)
        
        # API failed, return cached data if available
        self.api_failure_count += 1
        if self.last_valid_cache is not None:
            system_logger.warning("API failed, using last valid cached data")
            return self.last_valid_cache
        
        # Return default values if no cache
        system_logger.error("API failed and no cached data available, using defaults")
        return self._get_default_weather()
    
    def _parse_weather_data(self, data: Dict) -> Dict:
        """
        Parse API response to extract relevant weather data
        
        Args:
            data: Raw API response
            
        Returns:
            Parsed weather data
        """
        try:
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            
            # Handle rainfall (may be missing)
            rainfall = 0.0
            if 'rain' in data:
                if '1h' in data['rain']:
                    rainfall = data['rain']['1h']
                elif '3h' in data['rain']:
                    rainfall = data['rain']['3h'] / 3  # Convert to hourly
            
            return {
                'temperature': float(temperature),
                'humidity': float(humidity),
                'wind_speed': float(wind_speed),
                'rainfall': float(rainfall),
                'location': data.get('name', 'Unknown'),
                'country': data.get('sys', {}).get('country', 'Unknown')
            }
        except KeyError as e:
            system_logger.error(f"Failed to parse weather data: missing key {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self) -> Dict:
        """
        Get default weather values when API fails
        
        Returns:
            Default weather data
        """
        return {
            'temperature': 25.0,
            'humidity': 50.0,
            'wind_speed': 10.0,
            'rainfall': 0.0,
            'location': 'Unknown',
            'country': 'Unknown'
        }
    
    def get_api_status(self) -> Dict:
        """
        Get API status
        
        Returns:
            API status information
        """
        return {
            'status': 'working' if self.api_failure_count < 3 else 'failure',
            'failure_count': self.api_failure_count,
            'has_cached_data': self.last_valid_cache is not None
        }
    
    def reset_failure_count(self):
        """Reset API failure count"""
        self.api_failure_count = 0
