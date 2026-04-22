"""
Regional Scanner Module
Scans multiple locations for wildfire risk
"""

import numpy as np
from typing import List, Dict
from backend.services.pipeline import DataPipeline
from backend.api.weather import WeatherAPI
from backend.services.decision import DecisionEngine
from backend.utils.logger import system_logger


class RegionalScanner:
    """Scanner for regional wildfire risk assessment"""
    
    def __init__(self, pipeline: DataPipeline, weather_api: WeatherAPI = None):
        """
        Initialize regional scanner
        
        Args:
            pipeline: Data pipeline instance
            weather_api: Weather API instance
        """
        self.pipeline = pipeline
        self.weather_api = weather_api
        self.decision_engine = DecisionEngine()
        
        # Define regions to scan (India-focused with wildfire-prone areas)
        self.regions = [
            {'name': 'Dehradun, Uttarakhand', 'lat': 30.33, 'lon': 78.06},
            {'name': 'Shimla, Himachal Pradesh', 'lat': 31.10, 'lon': 77.17},
            {'name': 'Srinagar, J&K', 'lat': 33.78, 'lon': 76.08},
            {'name': 'Gangtok, Sikkim', 'lat': 27.53, 'lon': 88.51},
            {'name': 'Itanagar, Arunachal Pradesh', 'lat': 28.21, 'lon': 94.72},
            {'name': 'Kohima, Nagaland', 'lat': 26.15, 'lon': 94.11},
            {'name': 'Imphal, Manipur', 'lat': 24.80, 'lon': 93.93},
            {'name': 'Aizawl, Mizoram', 'lat': 23.72, 'lon': 92.93},
            {'name': 'Shillong, Meghalaya', 'lat': 25.46, 'lon': 91.36},
            {'name': 'Guwahati, Assam', 'lat': 26.20, 'lon': 91.77},
            {'name': 'Agartala, Tripura', 'lat': 23.84, 'lon': 91.28},
            {'name': 'Kolkata, West Bengal', 'lat': 22.57, 'lon': 88.36},
            {'name': 'Bhubaneswar, Odisha', 'lat': 20.29, 'lon': 85.82},
            {'name': 'Raipur, Chhattisgarh', 'lat': 21.25, 'lon': 81.62},
            {'name': 'Bhopal, Madhya Pradesh', 'lat': 23.25, 'lon': 77.41},
            {'name': 'Nagpur, Maharashtra', 'lat': 21.14, 'lon': 79.08},
            {'name': 'Bengaluru, Karnataka', 'lat': 12.97, 'lon': 77.59},
            {'name': 'Thiruvananthapuram, Kerala', 'lat': 8.52, 'lon': 76.93},
            {'name': 'Chennai, Tamil Nadu', 'lat': 13.08, 'lon': 80.27},
            {'name': 'Amaravati, Andhra Pradesh', 'lat': 16.50, 'lon': 80.51},
            {'name': 'Hyderabad, Telangana', 'lat': 17.38, 'lon': 78.48},
            {'name': 'Panaji, Goa', 'lat': 15.49, 'lon': 73.82},
            {'name': 'Ahmedabad, Gujarat', 'lat': 23.02, 'lon': 72.57},
            {'name': 'Jaipur, Rajasthan', 'lat': 26.91, 'lon': 75.78},
            {'name': 'New Delhi', 'lat': 28.61, 'lon': 77.23}
        ]
    
    def scan_region(self, region: Dict, month: int = 6) -> Dict:
        """
        Scan a single region
        
        Args:
            region: Region dictionary with name, lat, lon
            month: Current month
            
        Returns:
            Region scan results
        """
        try:
            # Get weather data
            weather_data = self.weather_api.get_weather(region['lat'], region['lon'])
            
            # Make prediction
            prediction = self.pipeline.predict_pipeline(weather_data, month)
            
            # Make decision
            decision = self.decision_engine.make_decision(prediction)
            
            result = {
                'region_name': region['name'],
                'latitude': region['lat'],
                'longitude': region['lon'],
                'weather': weather_data,
                'prediction': prediction,
                'decision': decision
            }
            
            return result
            
        except Exception as e:
            system_logger.error(f"Failed to scan region {region['name']}: {str(e)}")
            return None
    
    def scan_all_regions(self, month: int = 6, max_regions: int = 25) -> List[Dict]:
        """
        Scan all regions
        
        Args:
            month: Current month
            max_regions: Maximum number of regions to scan
            
        Returns:
            List of region scan results
        """
        system_logger.info(f"Starting regional scan for {max_regions} regions")
        
        results = []
        regions_to_scan = self.regions[:max_regions]
        
        for region in regions_to_scan:
            result = self.scan_region(region, month)
            if result is not None:
                results.append(result)
        
        # Sort by risk score
        results.sort(key=lambda x: x['decision']['risk_score'], reverse=True)
        
        system_logger.info(f"Regional scan completed: {len(results)} regions scanned")
        
        return results
    
    def get_regional_summary(self, scan_results: List[Dict]) -> Dict:
        """
        Get summary of regional scan
        
        Args:
            scan_results: Results from scan_all_regions
            
        Returns:
            Summary statistics
        """
        if not scan_results:
            return {'error': 'No scan results available'}
        
        risk_scores = [r['decision']['risk_score'] for r in scan_results]
        
        # Count risk levels
        risk_level_counts = {}
        for result in scan_results:
            level = result['decision']['linguistic_risk_level']
            risk_level_counts[level] = risk_level_counts.get(level, 0) + 1
        
        summary = {
            'total_regions': len(scan_results),
            'average_risk': float(np.mean(risk_scores)),
            'max_risk': float(np.max(risk_scores)),
            'min_risk': float(np.min(risk_scores)),
            'risk_level_distribution': risk_level_counts,
            'high_risk_regions': [r['region_name'] for r in scan_results if r['decision']['risk_score'] > 0.7],
            'top_risk_region': scan_results[0]['region_name'] if scan_results else None
        }
        
        return summary
