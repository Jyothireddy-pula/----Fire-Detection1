"""
Advanced Regional Grid Scanner for Wildfire Risk Monitoring
Covers 25+ forest locations with comprehensive risk assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
import joblib
import warnings
warnings.filterwarnings('ignore')

class ForestLocation:
    """Forest location data structure"""
    
    def __init__(self, name: str, latitude: float, longitude: float, 
                 forest_type: str, area_km2: float, elevation: float, 
                 region: str, country: str):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.forest_type = forest_type
        self.area_km2 = area_km2
        self.elevation = elevation
        self.region = region
        self.country = country
        self.current_weather = None
        self.fire_indices = None
        self.risk_score = 0.0
        self.risk_level = "Unknown"
        self.last_updated = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'forest_type': self.forest_type,
            'area_km2': self.area_km2,
            'elevation': self.elevation,
            'region': self.region,
            'country': self.country,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class RegionalGridScanner:
    """Advanced regional grid scanner for wildfire risk assessment"""
    
    def __init__(self, model_path: str = 'models/anfis_pso_optimized.h5'):
        self.model_path = model_path
        self.locations = []
        self.model = None
        self.scaler = None
        self.load_model()
        self.init_locations()
        self.init_database()
        
    def load_model(self):
        """Load the trained model"""
        try:
            try:
                from .anfis_system import AdvancedANFIS
            except ImportError:
                from anfis_system import AdvancedANFIS
            self.model = AdvancedANFIS()
            self.model.load_model(self.model_path)
            self.scaler = joblib.load('models/scaler.pkl')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to basic model
            try:
                self.model = joblib.load('models/mlp_pso.pkl')
                self.scaler = joblib.load('models/scaler.pkl')
                print("Fallback model loaded")
            except:
                print("No model available, using risk calculation only")
    
    def init_locations(self):
        """Initialize forest locations database"""
        # Major forest locations around the world
        locations_data = [
            # North America
            {"name": "Yellowstone National Park", "lat": 44.4280, "lon": -110.5885, "type": "Coniferous", "area": 8983, "elevation": 2400, "region": "Wyoming", "country": "USA"},
            {"name": "Yosemite National Park", "lat": 37.8651, "lon": -119.5383, "type": "Mixed", "area": 3027, "elevation": 3000, "region": "California", "country": "USA"},
            {"name": "Sequoia National Park", "lat": 36.4864, "lon": -118.5654, "type": "Coniferous", "area": 1631, "elevation": 2100, "region": "California", "country": "USA"},
            {"name": "Banff National Park", "lat": 51.4968, "lon": -115.9341, "type": "Coniferous", "area": 6641, "elevation": 1700, "region": "Alberta", "country": "Canada"},
            {"name": "Great Smoky Mountains", "lat": 35.6586, "lon": -83.5090, "type": "Deciduous", "area": 2116, "elevation": 1500, "region": "Tennessee/North Carolina", "country": "USA"},
            
            # South America
            {"name": "Amazon Rainforest", "lat": -3.4653, "lon": -62.2159, "type": "Tropical", "area": 5500000, "elevation": 200, "region": "Amazonas", "country": "Brazil"},
            {"name": "Patagonian Forest", "lat": -43.8333, "lon": -71.6667, "type": "Temperate", "area": 30000, "elevation": 800, "region": "Patagonia", "country": "Argentina"},
            {"name": "Mato Grosso", "lat": -14.5405, "lon": -55.8371, "type": "Savanna", "area": 903000, "elevation": 350, "region": "Mato Grosso", "country": "Brazil"},
            
            # Europe
            {"name": "Black Forest", "lat": 48.5333, "lon": 8.1667, "type": "Coniferous", "area": 6000, "elevation": 1000, "region": "Baden-Württemberg", "country": "Germany"},
            {"name": "Bialowieza Forest", "lat": 52.6883, "lon": 23.8667, "type": "Mixed", "area": 1500, "elevation": 160, "region": "Podlaskie", "country": "Poland/Belarus"},
            {"name": "Daintree Rainforest", "lat": -16.1167, "lon": 145.4333, "type": "Tropical", "area": 1200, "elevation": 400, "region": "Queensland", "country": "Australia"},
            {"name": "Sherwood Forest", "lat": 53.2167, "lon": -0.9833, "type": "Deciduous", "area": 423, "elevation": 50, "region": "Nottinghamshire", "country": "UK"},
            {"name": "Tatra Mountains", "lat": 49.2500, "lon": 20.0833, "type": "Coniferous", "area": 450, "elevation": 2000, "region": "Zakopane", "country": "Poland"},
            
            # Africa
            {"name": "Serengeti", "lat": -2.3333, "lon": 34.8333, "type": "Savanna", "area": 14763, "elevation": 1500, "region": "Tanzania", "country": "Tanzania"},
            {"name": "Drakensberg", "lat": -29.0000, "lon": 29.0000, "type": "Grassland", "area": 5000, "elevation": 3000, "region": "KwaZulu-Natal", "country": "South Africa"},
            {"name": "Mount Kenya Forest", "lat": -0.1500, "lon": 37.1833, "type": "Montane", "area": 2000, "elevation": 4000, "region": "Mount Kenya", "country": "Kenya"},
            
            # Asia
            {"name": "Sundarbans", "lat": 22.4667, "lon": 89.1833, "type": "Mangrove", "area": 10000, "elevation": 10, "region": "West Bengal", "country": "India/Bangladesh"},
            {"name": "Siberian Taiga", "lat": 60.0000, "lon": 100.0000, "type": "Boreal", "area": 12000000, "elevation": 500, "region": "Siberia", "country": "Russia"},
            {"name": "Himalayan Forest", "lat": 27.9881, "lon": 86.9250, "type": "Montane", "area": 50000, "elevation": 4000, "region": "Nepal", "country": "Nepal"},
            {"name": "Borneo Rainforest", "lat": 1.0000, "lon": 114.0000, "type": "Tropical", "area": 140000, "elevation": 1000, "region": "Kalimantan", "country": "Indonesia"},
            {"name": "Japanese Alps", "lat": 36.0000, "lon": 137.5000, "type": "Temperate", "area": 5000, "elevation": 3000, "region": "Honshu", "country": "Japan"},
            
            # Oceania
            {"name": "Blue Mountains", "lat": -33.7167, "lon": 150.3167, "type": "Temperate", "area": 11000, "elevation": 1100, "region": "New South Wales", "country": "Australia"},
            {"name": "Tasmania Forest", "lat": -42.0000, "lon": 147.0000, "type": "Temperate", "area": 15000, "elevation": 1000, "region": "Tasmania", "country": "Australia"},
            {"name": "New Zealand Forest", "lat": -40.9000, "lon": 175.3667, "type": "Temperate", "area": 8000, "elevation": 1200, "region": "North Island", "country": "New Zealand"},
            
            # India
            {"name": "Sundarbans West Bengal", "lat": 21.9442, "lon": 89.1833, "type": "Mangrove", "area": 4000, "elevation": 10, "region": "West Bengal", "country": "India"},
            {"name": "Western Ghats", "lat": 12.9716, "lon": 77.5946, "type": "Tropical", "area": 160000, "elevation": 1500, "region": "Karnataka", "country": "India"},
            {"name": "Himalayan Forest India", "lat": 32.2400, "lon": 77.1900, "type": "Montane", "area": 8000, "elevation": 3500, "region": "Himachal Pradesh", "country": "India"},
            {"name": "Jim Corbett National Park", "lat": 29.5333, "lon": 78.7833, "type": "Deciduous", "area": 1318, "elevation": 1200, "region": "Uttarakhand", "country": "India"},
            {"name": "Bandipur National Park", "lat": 12.3000, "lon": 76.6500, "type": "Deciduous", "area": 874, "elevation": 800, "region": "Karnataka", "country": "India"},
            {"name": "Srisailam Forest", "lat": 15.8333, "lon": 78.8667, "type": "Dry Deciduous", "area": 3568, "elevation": 300, "region": "Andhra Pradesh", "country": "India"},
        ]
        
        for loc_data in locations_data:
            location = ForestLocation(
                name=loc_data["name"],
                latitude=loc_data["lat"],
                longitude=loc_data["lon"],
                forest_type=loc_data["type"],
                area_km2=loc_data["area"],
                elevation=loc_data["elevation"],
                region=loc_data["region"],
                country=loc_data["country"]
            )
            self.locations.append(location)
        
        print(f"Initialized {len(self.locations)} forest locations")
    
    def init_database(self):
        """Initialize database for storing regional data"""
        conn = sqlite3.connect('regional_scanner.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regional_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                location_name TEXT,
                latitude REAL,
                longitude REAL,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                rainfall REAL,
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
        
        conn.commit()
        conn.close()
    
    def simulate_weather_data(self, location: ForestLocation) -> Dict:
        """Simulate realistic weather data based on location characteristics"""
        # Base weather patterns by forest type and region
        weather_patterns = {
            "Tropical": {"temp_range": (25, 35), "humidity_range": (70, 95), "rain_range": (0, 15)},
            "Temperate": {"temp_range": (10, 30), "humidity_range": (40, 80), "rain_range": (0, 8)},
            "Coniferous": {"temp_range": (-5, 25), "humidity_range": (30, 70), "rain_range": (0, 5)},
            "Boreal": {"temp_range": (-20, 15), "humidity_range": (20, 60), "rain_range": (0, 3)},
            "Mediterranean": {"temp_range": (15, 40), "humidity_range": (20, 60), "rain_range": (0, 2)},
            "Savanna": {"temp_range": (20, 40), "humidity_range": (30, 70), "rain_range": (0, 10)},
            "Montane": {"temp_range": (0, 25), "humidity_range": (40, 80), "rain_range": (0, 6)},
            "Mangrove": {"temp_range": (25, 35), "humidity_range": (80, 98), "rain_range": (0, 20)},
            "Dry Deciduous": {"temp_range": (20, 45), "humidity_range": (20, 60), "rain_range": (0, 3)},
            "Mixed": {"temp_range": (5, 30), "humidity_range": (35, 75), "rain_range": (0, 6)},
            "Deciduous": {"temp_range": (10, 30), "humidity_range": (40, 80), "rain_range": (0, 6)},
            "Grassland": {"temp_range": (15, 35), "humidity_range": (25, 65), "rain_range": (0, 4)}
        }
        
        pattern = weather_patterns.get(location.forest_type, weather_patterns["Temperate"])
        
        # Add seasonal variation
        month = datetime.now().month
        seasonal_factor = 1.0
        if location.country in ["USA", "Canada", "Russia"]:
            # Northern hemisphere seasonal variation
            if month in [12, 1, 2]:  # Winter
                seasonal_factor = 0.7
            elif month in [6, 7, 8]:  # Summer
                seasonal_factor = 1.3
        elif location.country in ["Australia", "New Zealand"]:
            # Southern hemisphere seasonal variation
            if month in [12, 1, 2]:  # Summer
                seasonal_factor = 1.3
            elif month in [6, 7, 8]:  # Winter
                seasonal_factor = 0.7
        
        # Elevation effect
        elevation_factor = 1.0 - (location.elevation / 10000)  # Temperature decreases with elevation
        
        # Generate weather data
        temp_range = pattern["temp_range"]
        base_temp = np.random.uniform(temp_range[0], temp_range[1]) * seasonal_factor * elevation_factor
        
        humidity_range = pattern["humidity_range"]
        humidity = np.random.uniform(humidity_range[0], humidity_range[1])
        
        rain_range = pattern["rain_range"]
        rainfall = np.random.uniform(rain_range[0], rain_range[1])
        
        wind_speed = np.random.uniform(0, 25)  # 0-25 m/s
        
        # Calculate fire weather indices
        try:
            from .weather_api import FireWeatherIndexCalculator
        except ImportError:
            from weather_api import FireWeatherIndexCalculator
        calculator = FireWeatherIndexCalculator()
        
        fire_indices = calculator.calculate_all_indices(
            type('WeatherData', (), {
                'temperature': base_temp,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'rainfall': rainfall
            })()
        )
        
        return {
            'temperature': base_temp,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'rainfall': rainfall,
            'fire_indices': fire_indices
        }
    
    def calculate_risk_score(self, location: ForestLocation, weather_data: Dict) -> Tuple[float, str]:
        """Calculate wildfire risk score for a location"""
        # Prepare features for ML model
        features = np.array([[
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['rainfall'],
            weather_data['fire_indices']['FFMC'],
            weather_data['fire_indices']['DMC'],
            weather_data['fire_indices']['DC'],
            weather_data['fire_indices']['ISI'],
            weather_data['fire_indices']['BUI'],
            weather_data['fire_indices']['FWI'],
            weather_data['temperature'] * (100 - weather_data['humidity']),  # Temp_RH_Index
            weather_data['wind_speed'] * weather_data['rainfall']  # Wind_Rain_Interaction
        ]])
        
        # Use model if available
        if self.model and self.scaler:
            try:
                risk_score = float(self.model.predict(features)[0])
                risk_score = np.clip(risk_score, 0, 1)
            except:
                # Fallback to FWI-based calculation
                risk_score = min(1.0, weather_data['fire_indices']['FWI'] / 50)
        else:
            # Simple risk calculation based on fire indices
            risk_score = min(1.0, weather_data['fire_indices']['FWI'] / 50)
        
        # Adjust for forest type
        forest_type_factors = {
            "Tropical": 0.8,  # Lower risk due to high humidity
            "Temperate": 1.0,
            "Coniferous": 1.2,  # Higher risk
            "Boreal": 1.1,
            "Mediterranean": 1.4,  # Highest risk
            "Savanna": 1.3,
            "Montane": 0.9,
            "Mangrove": 0.6,
            "Dry Deciduous": 1.5,  # Very high risk
            "Mixed": 1.0,
            "Deciduous": 1.0,
            "Grassland": 1.2
        }
        
        type_factor = forest_type_factors.get(location.forest_type, 1.0)
        risk_score = min(1.0, risk_score * type_factor)
        
        # Determine risk level
        if risk_score < 0.25:
            risk_level = "No Risk"
        elif risk_score < 0.5:
            risk_level = "Low"
        elif risk_score < 0.7:
            risk_level = "Moderate"
        elif risk_score < 0.85:
            risk_level = "High"
        else:
            risk_level = "Extreme"
        
        return risk_score, risk_level
    
    def scan_all_locations(self) -> pd.DataFrame:
        """Scan all locations for wildfire risk"""
        print("Scanning all forest locations...")
        
        results = []
        
        for location in self.locations:
            print(f"Scanning {location.name}...")
            
            # Get weather data
            weather_data = self.simulate_weather_data(location)
            
            # Calculate risk
            risk_score, risk_level = self.calculate_risk_score(location, weather_data)
            
            # Update location
            location.current_weather = weather_data
            location.fire_indices = weather_data['fire_indices']
            location.risk_score = risk_score
            location.risk_level = risk_level
            location.last_updated = datetime.now()
            
            # Store result
            result = {
                'Location': location.name,
                'Country': location.country,
                'Region': location.region,
                'Forest_Type': location.forest_type,
                'Latitude': location.latitude,
                'Longitude': location.longitude,
                'Temperature': weather_data['temperature'],
                'Humidity': weather_data['humidity'],
                'Wind_Speed': weather_data['wind_speed'],
                'Rainfall': weather_data['rainfall'],
                'FFMC': weather_data['fire_indices']['FFMC'],
                'DMC': weather_data['fire_indices']['DMC'],
                'DC': weather_data['fire_indices']['DC'],
                'ISI': weather_data['fire_indices']['ISI'],
                'BUI': weather_data['fire_indices']['BUI'],
                'FWI': weather_data['fire_indices']['FWI'],
                'Risk_Score': risk_score,
                'Risk_Level': risk_level,
                'Timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
            
            # Store in database
            self.store_scan_result(result)
        
        df = pd.DataFrame(results)
        df = df.sort_values('Risk_Score', ascending=False)
        
        print(f"Scan complete! Processed {len(df)} locations")
        return df
    
    def store_scan_result(self, result: Dict):
        """Store scan result in database"""
        conn = sqlite3.connect('regional_scanner.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO regional_scans 
            (timestamp, location_name, latitude, longitude, temperature, humidity, 
             wind_speed, rainfall, ffmc, dmc, dc, isi, bui, fwi, risk_score, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['Timestamp'],
            result['Location'],
            result['Latitude'],
            result['Longitude'],
            result['Temperature'],
            result['Humidity'],
            result['Wind_Speed'],
            result['Rainfall'],
            result['FFMC'],
            result['DMC'],
            result['DC'],
            result['ISI'],
            result['BUI'],
            result['FWI'],
            result['Risk_Score'],
            result['Risk_Level']
        ))
        
        conn.commit()
        conn.close()
    
    def create_risk_heatmap(self, df: pd.DataFrame, save_path=None):
        """Create interactive risk heatmap using Folium"""
        # Create base map
        center_lat = df['Latitude'].mean()
        center_lon = df['Longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=2)
        
        # Prepare data for heatmap
        heat_data = []
        for _, row in df.iterrows():
            # Use risk score as intensity
            heat_data.append([row['Latitude'], row['Longitude'], row['Risk_Score']])
        
        # Add heatmap layer
        HeatMap(heat_data, radius=25, blur=15, max_zoom=1).add_to(m)
        
        # Add markers for high-risk locations
        high_risk = df[df['Risk_Score'] >= 0.7]
        
        for _, row in high_risk.iterrows():
            color = 'red' if row['Risk_Score'] >= 0.85 else 'orange'
            
            popup_html = f"""
            <b>{row['Location']}</b><br>
            Risk Level: {row['Risk_Level']}<br>
            Risk Score: {row['Risk_Score']:.3f}<br>
            Temperature: {row['Temperature']:.1f}°C<br>
            Humidity: {row['Humidity']:.1f}%<br>
            FWI: {row['FWI']:.1f}
            """
            
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=popup_html,
                icon=folium.Icon(color=color, icon='warning')
            ).add_to(m)
        
        # Save map
        if save_path:
            m.save(save_path)
            print(f"Risk heatmap saved to {save_path}")
        
        return m
    
    def create_risk_dashboard(self, df: pd.DataFrame, save_path=None):
        """Create comprehensive risk dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Risk Score Distribution', 'Top 10 High Risk Locations',
                          'Risk by Forest Type', 'Risk by Country',
                          'Temperature vs Risk', 'FWI vs Risk'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Risk score distribution
        fig.add_trace(
            go.Histogram(x=df['Risk_Score'], nbinsx=20, name='Risk Distribution',
                        marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Top 10 high risk locations
        top_10 = df.nlargest(10, 'Risk_Score')
        fig.add_trace(
            go.Bar(x=top_10['Risk_Score'], y=top_10['Location'], 
                   orientation='h', name='Top 10 Risk',
                   marker_color='red'),
            row=1, col=2
        )
        
        # 3. Risk by forest type
        forest_risk = df.groupby('Forest_Type')['Risk_Score'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=forest_risk.values, y=forest_risk.index, 
                   orientation='h', name='Forest Type Risk',
                   marker_color='orange'),
            row=2, col=1
        )
        
        # 4. Risk by country
        country_risk = df.groupby('Country')['Risk_Score'].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=country_risk.values, y=country_risk.index, 
                   orientation='h', name='Country Risk',
                   marker_color='green'),
            row=2, col=2
        )
        
        # 5. Temperature vs Risk
        fig.add_trace(
            go.Scatter(x=df['Temperature'], y=df['Risk_Score'], 
                      mode='markers', name='Temperature vs Risk',
                      marker=dict(color=df['Risk_Score'], colorscale='Viridis', size=8)),
            row=3, col=1
        )
        
        # 6. FWI vs Risk
        fig.add_trace(
            go.Scatter(x=df['FWI'], y=df['Risk_Score'], 
                      mode='markers', name='FWI vs Risk',
                      marker=dict(color=df['Risk_Score'], colorscale='Plasma', size=8)),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Regional Wildfire Risk Dashboard",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Risk dashboard saved to {save_path}")
        
        return fig
    
    def generate_alert_report(self, df: pd.DataFrame, save_path='regional_alerts.html'):
        """Generate alert report for high-risk locations"""
        high_risk = df[df['Risk_Score'] >= 0.7]
        extreme_risk = df[df['Risk_Score'] >= 0.85]
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Regional Wildfire Risk Alert Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #d32f2f; color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .alert-box {{ background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .extreme-alert {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                .location-card {{ background-color: white; padding: 15px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .high-risk {{ border-left: 5px solid #ff9800; }}
                .extreme-risk {{ border-left: 5px solid #d32f2f; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ text-align: center; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .risk-score {{ font-size: 24px; font-weight: bold; }}
                .high {{ color: #ff9800; }}
                .extreme {{ color: #d32f2f; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Regional Wildfire Risk Alert Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>Total Locations Monitored</h3>
                    <div class="risk-score">{len(df)}</div>
                </div>
                <div class="stat-box">
                    <h3>High Risk Locations</h3>
                    <div class="risk-score high">{len(high_risk)}</div>
                </div>
                <div class="stat-box">
                    <h3>Extreme Risk Locations</h3>
                    <div class="risk-score extreme">{len(extreme_risk)}</div>
                </div>
            </div>
            
            {f'<div class="alert-box extreme-alert"><h2>EXTREME RISK ALERT</h2><p>{len(extreme_risk)} locations at extreme risk. Immediate action required!</p></div>' if len(extreme_risk) > 0 else ''}
            {f'<div class="alert-box"><h2>HIGH RISK ALERT</h2><p>{len(high_risk)} locations at high risk. Increased monitoring recommended.</p></div>' if len(high_risk) > 0 else ''}
            
            <h2>High Risk Locations</h2>
        """
        
        # Add high risk location cards
        for _, row in high_risk.iterrows():
            risk_class = 'extreme-risk' if row['Risk_Score'] >= 0.85 else 'high-risk'
            html_content += f"""
            <div class="location-card {risk_class}">
                <h3>{row['Location']} - {row['Country']}</h3>
                <p><strong>Risk Level:</strong> <span class="{'extreme' if row['Risk_Score'] >= 0.85 else 'high'}">{row['Risk_Level']}</span></p>
                <p><strong>Risk Score:</strong> {row['Risk_Score']:.3f}</p>
                <p><strong>Forest Type:</strong> {row['Forest_Type']}</p>
                <p><strong>Current Conditions:</strong> Temp: {row['Temperature']:.1f}°C, Humidity: {row['Humidity']:.1f}%, Wind: {row['Wind_Speed']:.1f} m/s</p>
                <p><strong>Fire Indices:</strong> FWI: {row['FWI']:.1f}, FFMC: {row['FFMC']:.1f}, DMC: {row['DMC']:.1f}</p>
            </div>
            """
        
        html_content += """
            </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Alert report saved to {save_path}")
        return save_path

def demo_regional_scanner():
    """Demonstration of regional scanner"""
    print("="*60)
    print("REGIONAL GRID SCANNER FOR WILDFIRE RISK")
    print("="*60)
    
    # Initialize scanner
    scanner = RegionalGridScanner()
    
    # Scan all locations
    print("Starting regional scan...")
    results_df = scanner.scan_all_locations()
    
    # Display summary
    print(f"\nRegional Scan Summary:")
    print(f"Total locations scanned: {len(results_df)}")
    print(f"High risk locations: {len(results_df[results_df['Risk_Score'] >= 0.7])}")
    print(f"Extreme risk locations: {len(results_df[results_df['Risk_Score'] >= 0.85])}")
    
    # Show top 10 high risk locations
    print("\nTop 10 High Risk Locations:")
    top_10 = results_df.nlargest(10, 'Risk_Score')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['Location']} ({row['Country']}) - Risk: {row['Risk_Score']:.3f} ({row['Risk_Level']})")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Risk heatmap
    heatmap = scanner.create_risk_heatmap(results_df, 'outputs/regional_risk_heatmap.html')
    
    # Risk dashboard
    dashboard = scanner.create_risk_dashboard(results_df, 'outputs/regional_risk_dashboard.html')
    
    # Alert report
    alert_report = scanner.generate_alert_report(results_df, 'outputs/regional_alerts.html')
    
    # Save results
    results_df.to_csv('outputs/regional_risk_table.csv', index=False)
    
    print("\nRegional scanner complete!")
    print("- Risk heatmap: outputs/regional_risk_heatmap.html")
    print("- Risk dashboard: outputs/regional_risk_dashboard.html")
    print("- Alert report: outputs/regional_alerts.html")
    print("- Data table: outputs/regional_risk_table.csv")

if __name__ == "__main__":
    demo_regional_scanner()
