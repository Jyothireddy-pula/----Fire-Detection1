"""
Real-Time Weather API Integration for Wildfire Risk Prediction
Supports OpenWeatherMap API and other weather services
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import sqlite3
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import pytz
from dataclasses import dataclass
import joblib
import warnings
warnings.filterwarnings('ignore')

@dataclass
class WeatherData:
    """Weather data structure"""
    timestamp: datetime
    location: str
    latitude: float
    longitude: float
    temperature: float
    humidity: float
    wind_speed: float
    wind_direction: float
    pressure: float
    rainfall: float
    visibility: float
    weather_condition: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'wind_direction': self.wind_direction,
            'pressure': self.pressure,
            'rainfall': self.rainfall,
            'visibility': self.visibility,
            'weather_condition': self.weather_condition
        }

class WeatherAPIManager:
    """Advanced Weather API Manager with multiple providers"""
    
    def __init__(self, api_key: str, provider: str = 'openweathermap'):
        self.api_key = api_key
        self.provider = provider
        self.base_urls = {
            'openweathermap': 'https://api.openweathermap.org/data/2.5',
            'weatherapi': 'https://api.weatherapi.com/v1',
            'accuweather': 'https://dataservice.accuweather.com'
        }
        self.session = None
        self.cache = {}
        self.cache_duration = 600  # 10 minutes cache
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, location: str) -> str:
        """Generate cache key for location"""
        return f"{self.provider}_{location}_{int(time.time() // self.cache_duration)}"
    
    async def get_current_weather(self, location: str) -> Optional[WeatherData]:
        """Get current weather for a location"""
        cache_key = self._get_cache_key(location)
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.provider == 'openweathermap':
                weather_data = await self._get_openweathermap_current(location)
            elif self.provider == 'weatherapi':
                weather_data = await self._get_weatherapi_current(location)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Cache the result
            self.cache[cache_key] = weather_data
            return weather_data
            
        except Exception as e:
            print(f"Error fetching weather for {location}: {e}")
            return None
    
    async def get_forecast(self, location: str, days: int = 5) -> List[WeatherData]:
        """Get weather forecast for multiple days"""
        try:
            if self.provider == 'openweathermap':
                forecast_data = await self._get_openweathermap_forecast(location, days)
            elif self.provider == 'weatherapi':
                forecast_data = await self._get_weatherapi_forecast(location, days)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            return forecast_data
            
        except Exception as e:
            print(f"Error fetching forecast for {location}: {e}")
            return []
    
    async def _get_openweathermap_current(self, location: str) -> WeatherData:
        """Get current weather from OpenWeatherMap"""
        url = f"{self.base_urls['openweathermap']}/weather"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            
            if response.status != 200:
                raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            
            return WeatherData(
                timestamp=datetime.now(pytz.UTC),
                location=data['name'],
                latitude=data['coord']['lat'],
                longitude=data['coord']['lon'],
                temperature=data['main']['temp'],
                humidity=data['main']['humidity'],
                wind_speed=data['wind']['speed'],
                wind_direction=data['wind'].get('deg', 0),
                pressure=data['main']['pressure'],
                rainfall=data.get('rain', {}).get('1h', 0),
                visibility=data.get('visibility', 10000) / 1000,  # Convert to km
                weather_condition=data['weather'][0]['main']
            )
    
    async def _get_openweathermap_forecast(self, location: str, days: int) -> List[WeatherData]:
        """Get forecast from OpenWeatherMap"""
        url = f"{self.base_urls['openweathermap']}/forecast"
        params = {
            'q': location,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            
            if response.status != 200:
                raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
            
            forecast_list = []
            for item in data['list'][:days * 8]:  # 8 forecasts per day (3-hour intervals)
                forecast_time = datetime.fromtimestamp(item['dt'], pytz.UTC)
                
                weather_data = WeatherData(
                    timestamp=forecast_time,
                    location=data['city']['name'],
                    latitude=data['city']['coord']['lat'],
                    longitude=data['city']['coord']['lon'],
                    temperature=item['main']['temp'],
                    humidity=item['main']['humidity'],
                    wind_speed=item['wind']['speed'],
                    wind_direction=item['wind'].get('deg', 0),
                    pressure=item['main']['pressure'],
                    rainfall=item.get('rain', {}).get('3h', 0),
                    visibility=item.get('visibility', 10000) / 1000,
                    weather_condition=item['weather'][0]['main']
                )
                forecast_list.append(weather_data)
            
            return forecast_list
    
    async def _get_weatherapi_current(self, location: str) -> WeatherData:
        """Get current weather from WeatherAPI"""
        url = f"{self.base_urls['weatherapi']}/current.json"
        params = {
            'key': self.api_key,
            'q': location,
            'aqi': 'no'
        }
        
        async with self.session.get(url, params=params) as response:
            data = await response.json()
            
            if response.status != 200:
                raise Exception(f"API Error: {data.get('error', {}).get('message', 'Unknown error')}")
            
            return WeatherData(
                timestamp=datetime.now(pytz.UTC),
                location=data['location']['name'],
                latitude=data['location']['lat'],
                longitude=data['location']['lon'],
                temperature=data['current']['temp_c'],
                humidity=data['current']['humidity'],
                wind_speed=data['current']['wind_kph'] / 3.6,  # Convert to m/s
                wind_direction=data['current']['wind_degree'],
                pressure=data['current']['pressure_mb'],
                rainfall=data['current']['precip_mm'],
                visibility=data['current']['vis_km'],
                weather_condition=data['current']['condition']['text']
            )

class FireWeatherIndexCalculator:
    """Calculate fire weather indices from weather data"""
    
    @staticmethod
    def calculate_ffmc(temperature: float, humidity: float, wind_speed: float, rainfall: float) -> float:
        """Calculate Fine Fuel Moisture Code (FFMC)"""
        # Simplified FFMC calculation
        ffmc = 59.5 * np.exp(-0.1 * humidity) + 0.03 * wind_speed + 0.05 * temperature - 0.1 * rainfall
        return np.clip(ffmc, 0, 101)
    
    @staticmethod
    def calculate_dmc(temperature: float, humidity: float, rainfall: float, prev_dmc: float = 0) -> float:
        """Calculate Duff Moisture Code (DMC)"""
        # Simplified DMC calculation
        if rainfall > 1.5:
            dmc = max(0, prev_dmc - 1.5 * rainfall)
        else:
            dmc = prev_dmc + 0.05 * temperature - 0.1 * humidity
        return np.clip(dmc, 0, 100)
    
    @staticmethod
    def calculate_dc(temperature: float, rainfall: float, prev_dc: float = 0) -> float:
        """Calculate Drought Code (DC)"""
        # Simplified DC calculation
        if rainfall > 2.5:
            dc = max(0, prev_dc - 2.5 * rainfall)
        else:
            dc = prev_dc + 0.01 * temperature
        return np.clip(dc, 0, 500)
    
    @staticmethod
    def calculate_isi(wind_speed: float, ffmc: float) -> float:
        """Calculate Initial Spread Index (ISI)"""
        # Simplified ISI calculation
        isi = 0.208 * wind_speed * np.exp(0.05039 * ffmc)
        return np.clip(isi, 0, 50)
    
    @staticmethod
    def calculate_bui(dmc: float, dc: float) -> float:
        """Calculate Buildup Index (BUI)"""
        # Simplified BUI calculation
        if dmc <= 0.4 * dc:
            bui = 0.8 * dmc + 0.2 * dc
        else:
            bui = dmc + 0.3 * (dc - dmc)
        return np.clip(bui, 0, 100)
    
    @staticmethod
    def calculate_fwi(isi: float, bui: float) -> float:
        """Calculate Fire Weather Index (FWI)"""
        # Simplified FWI calculation
        if bui <= 50:
            fwi = 0.1 * isi * bui
        else:
            fwi = 0.1 * isi * (50 + 0.3 * (bui - 50))
        return np.clip(fwi, 0, 100)
    
    @classmethod
    def calculate_all_indices(cls, weather_data: WeatherData) -> Dict[str, float]:
        """Calculate all fire weather indices"""
        ffmc = cls.calculate_ffmc(weather_data.temperature, weather_data.humidity, 
                                 weather_data.wind_speed, weather_data.rainfall)
        dmc = cls.calculate_dmc(weather_data.temperature, weather_data.humidity, weather_data.rainfall)
        dc = cls.calculate_dc(weather_data.temperature, weather_data.rainfall)
        isi = cls.calculate_isi(weather_data.wind_speed, ffmc)
        bui = cls.calculate_bui(dmc, dc)
        fwi = cls.calculate_fwi(isi, bui)
        
        return {
            'FFMC': ffmc,
            'DMC': dmc,
            'DC': dc,
            'ISI': isi,
            'BUI': bui,
            'FWI': fwi
        }

class RealTimeWildfireMonitor:
    """Real-time wildfire risk monitoring system"""
    
    def __init__(self, api_key: str, db_path: str = 'wildfire_monitor.db'):
        self.api_key = api_key
        self.db_path = db_path
        self.weather_manager = WeatherAPIManager(api_key)
        self.index_calculator = FireWeatherIndexCalculator()
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for storing weather and risk data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                location TEXT,
                latitude REAL,
                longitude REAL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                wind_direction REAL,
                pressure REAL,
                rainfall REAL,
                visibility REAL,
                weather_condition TEXT,
                ffmc REAL,
                dmc REAL,
                dc REAL,
                isi REAL,
                bui REAL,
                fwi REAL,
                risk_score REAL,
                risk_level TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                location TEXT,
                risk_level TEXT,
                risk_score REAL,
                message TEXT,
                acknowledged BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def monitor_location(self, location: str, model_path: str = 'models/anfis_pso_optimized.h5') -> Dict:
        """Monitor a single location and calculate wildfire risk"""
        # Get current weather
        weather_data = await self.weather_manager.get_current_weather(location)
        if not weather_data:
            return {'error': 'Failed to fetch weather data'}
        
        # Calculate fire weather indices
        fire_indices = self.index_calculator.calculate_all_indices(weather_data)
        
        # Prepare features for ML model
        features = np.array([[
            weather_data.temperature,
            weather_data.humidity,
            weather_data.wind_speed,
            weather_data.rainfall,
            fire_indices['FFMC'],
            fire_indices['DMC'],
            fire_indices['DC'],
            fire_indices['ISI'],
            fire_indices['BUI'],
            fire_indices['FWI'],
            weather_data.temperature * (100 - weather_data.humidity),  # Temp_RH_Index
            weather_data.wind_speed * weather_data.rainfall  # Wind_Rain_Interaction
        ]])
        
        # Load model and predict risk
        try:
            try:
                from .anfis_system import AdvancedANFIS
            except ImportError:
                from anfis_system import AdvancedANFIS
            anfis = AdvancedANFIS()
            anfis.load_model(model_path)
            risk_score = float(anfis.predict(features)[0])
            risk_score = np.clip(risk_score, 0, 1)
        except Exception as e:
            print(f"Model loading error: {e}")
            # Fallback to simple risk calculation
            risk_score = min(1.0, fire_indices['FWI'] / 50)
        
        # Determine risk level
        if risk_score < 0.25:
            risk_level = 'No Risk'
        elif risk_score < 0.5:
            risk_level = 'Low'
        elif risk_score < 0.7:
            risk_level = 'Moderate'
        elif risk_score < 0.85:
            risk_level = 'High'
        else:
            risk_level = 'Extreme'
        
        # Store in database
        self.store_weather_data(weather_data, fire_indices, risk_score, risk_level)
        
        # Check for alerts
        if risk_score >= 0.7:
            self.create_alert(weather_data.location, risk_level, risk_score)
        
        return {
            'location': location,
            'timestamp': weather_data.timestamp.isoformat(),
            'weather': weather_data.to_dict(),
            'fire_indices': fire_indices,
            'risk_score': risk_score,
            'risk_level': risk_level
        }
    
    def store_weather_data(self, weather_data: WeatherData, fire_indices: Dict, 
                          risk_score: float, risk_level: str):
        """Store weather and risk data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO weather_data 
            (timestamp, location, latitude, longitude, temperature, humidity, 
             wind_speed, wind_direction, pressure, rainfall, visibility, 
             weather_condition, ffmc, dmc, dc, isi, bui, fwi, risk_score, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            weather_data.timestamp.isoformat(),
            weather_data.location,
            weather_data.latitude,
            weather_data.longitude,
            weather_data.temperature,
            weather_data.humidity,
            weather_data.wind_speed,
            weather_data.wind_direction,
            weather_data.pressure,
            weather_data.rainfall,
            weather_data.visibility,
            weather_data.weather_condition,
            fire_indices['FFMC'],
            fire_indices['DMC'],
            fire_indices['DC'],
            fire_indices['ISI'],
            fire_indices['BUI'],
            fire_indices['FWI'],
            risk_score,
            risk_level
        ))
        
        conn.commit()
        conn.close()
    
    def create_alert(self, location: str, risk_level: str, risk_score: float):
        """Create wildfire alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        message = f"Wildfire risk alert for {location}: {risk_level} (Risk Score: {risk_score:.2f})"
        
        cursor.execute('''
            INSERT INTO alerts (timestamp, location, risk_level, risk_score, message)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            location,
            risk_level,
            risk_score,
            message
        ))
        
        conn.commit()
        conn.close()
        
        print(f"ALERT: {message}")
    
    def get_historical_data(self, location: str, days: int = 7) -> pd.DataFrame:
        """Get historical weather and risk data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM weather_data 
            WHERE location = ? AND timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn, params=(location,))
        conn.close()
        
        return df
    
    def get_active_alerts(self) -> pd.DataFrame:
        """Get all active alerts"""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query('''
            SELECT * FROM alerts 
            WHERE acknowledged = FALSE 
            ORDER BY timestamp DESC
        ''', conn)
        
        conn.close()
        
        return df

async def demo_weather_monitoring():
    """Demonstration of real-time weather monitoring"""
    print("="*60)
    print("REAL-TIME WILDFIRE RISK MONITORING SYSTEM")
    print("="*60)
    
    # Note: You need to get a free API key from OpenWeatherMap
    API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"  # Replace with actual API key
    
    if API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        print("Please replace 'YOUR_OPENWEATHERMAP_API_KEY' with your actual OpenWeatherMap API key")
        return
    
    # Initialize monitor
    monitor = RealTimeWildfireMonitor(API_KEY)
    
    # Monitor some high-risk locations
    locations = [
        "California, USA",
        "Sydney, Australia",
        "Athens, Greece",
        "Lisbon, Portugal",
        "Vancouver, Canada"
    ]
    
    print("\nMonitoring wildfire risk for high-risk locations...")
    
    async with monitor.weather_manager:
        for location in locations:
            print(f"\nMonitoring {location}...")
            result = await monitor.monitor_location(location)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Risk Level: {result['risk_level']}")
                print(f"Risk Score: {result['risk_score']:.3f}")
                print(f"Temperature: {result['weather']['temperature']:.1f}°C")
                print(f"Humidity: {result['weather']['humidity']:.1f}%")
                print(f"Wind Speed: {result['weather']['wind_speed']:.1f} m/s")
                print(f"FWI: {result['fire_indices']['FWI']:.1f}")
            
            await asyncio.sleep(1)  # Rate limiting
    
    # Show active alerts
    print("\nActive Alerts:")
    alerts = monitor.get_active_alerts()
    if not alerts.empty:
        for _, alert in alerts.iterrows():
            print(f"- {alert['location']}: {alert['risk_level']} ({alert['risk_score']:.2f})")
    else:
        print("No active alerts")

if __name__ == "__main__":
    asyncio.run(demo_weather_monitoring())
