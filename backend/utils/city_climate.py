"""
City Climate Database
Contains city-specific climate parameters and monthly patterns for accurate fire risk predictions
"""

import numpy as np

# City-specific climate database
# Each city has base parameters and monthly adjustments for realistic predictions
CITY_CLIMATE_DB = {
    "Delhi": {
        "country": "India",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "base_temp": 25.0,
        "base_humidity": 55.0,
        "base_wind": 8.0,
        "climate_zone": "subtropical",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -8, "humidity_adj": 10, "rain_prob": 0.15, "rain_intensity": 0.5},
            2: {"temp_adj": -6, "humidity_adj": 5, "rain_prob": 0.20, "rain_intensity": 0.8},
            3: {"temp_adj": 2, "humidity_adj": -5, "rain_prob": 0.25, "rain_intensity": 1.0},
            4: {"temp_adj": 8, "humidity_adj": -15, "rain_prob": 0.15, "rain_intensity": 0.5},
            5: {"temp_adj": 12, "humidity_adj": -20, "rain_prob": 0.10, "rain_intensity": 0.3},
            6: {"temp_adj": 15, "humidity_adj": -25, "rain_prob": 0.20, "rain_intensity": 2.0},
            7: {"temp_adj": 10, "humidity_adj": 15, "rain_prob": 0.70, "rain_intensity": 8.0},
            8: {"temp_adj": 8, "humidity_adj": 20, "rain_prob": 0.75, "rain_intensity": 10.0},
            9: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.50, "rain_intensity": 5.0},
            10: {"temp_adj": 2, "humidity_adj": -5, "rain_prob": 0.20, "rain_intensity": 1.0},
            11: {"temp_adj": -2, "humidity_adj": 0, "rain_prob": 0.10, "rain_intensity": 0.5},
            12: {"temp_adj": -6, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 0.3}
        }
    },
    "Mumbai": {
        "country": "India",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "base_temp": 27.0,
        "base_humidity": 75.0,
        "base_wind": 10.0,
        "climate_zone": "tropical",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -3, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 0.3},
            2: {"temp_adj": -2, "humidity_adj": 5, "rain_prob": 0.15, "rain_intensity": 0.5},
            3: {"temp_adj": 2, "humidity_adj": 0, "rain_prob": 0.20, "rain_intensity": 1.0},
            4: {"temp_adj": 4, "humidity_adj": -5, "rain_prob": 0.15, "rain_intensity": 0.8},
            5: {"temp_adj": 5, "humidity_adj": -10, "rain_prob": 0.20, "rain_intensity": 1.5},
            6: {"temp_adj": 6, "humidity_adj": 5, "rain_prob": 0.60, "rain_intensity": 5.0},
            7: {"temp_adj": 4, "humidity_adj": 15, "rain_prob": 0.85, "rain_intensity": 15.0},
            8: {"temp_adj": 3, "humidity_adj": 20, "rain_prob": 0.80, "rain_intensity": 12.0},
            9: {"temp_adj": 3, "humidity_adj": 10, "rain_prob": 0.65, "rain_intensity": 8.0},
            10: {"temp_adj": 2, "humidity_adj": 0, "rain_prob": 0.30, "rain_intensity": 2.0},
            11: {"temp_adj": 0, "humidity_adj": 0, "rain_prob": 0.15, "rain_intensity": 0.5},
            12: {"temp_adj": -2, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 0.3}
        }
    },
    "Chennai": {
        "country": "India",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "base_temp": 29.0,
        "base_humidity": 70.0,
        "base_wind": 12.0,
        "climate_zone": "tropical",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -2, "humidity_adj": 5, "rain_prob": 0.20, "rain_intensity": 1.0},
            2: {"temp_adj": -1, "humidity_adj": 5, "rain_prob": 0.15, "rain_intensity": 0.8},
            3: {"temp_adj": 2, "humidity_adj": 0, "rain_prob": 0.20, "rain_intensity": 1.0},
            4: {"temp_adj": 4, "humidity_adj": -5, "rain_prob": 0.15, "rain_intensity": 0.5},
            5: {"temp_adj": 5, "humidity_adj": -10, "rain_prob": 0.10, "rain_intensity": 0.3},
            6: {"temp_adj": 4, "humidity_adj": -5, "rain_prob": 0.20, "rain_intensity": 2.0},
            7: {"temp_adj": 2, "humidity_adj": 5, "rain_prob": 0.30, "rain_intensity": 3.0},
            8: {"temp_adj": 1, "humidity_adj": 10, "rain_prob": 0.40, "rain_intensity": 4.0},
            9: {"temp_adj": 2, "humidity_adj": 15, "rain_prob": 0.50, "rain_intensity": 5.0},
            10: {"temp_adj": 3, "humidity_adj": 10, "rain_prob": 0.60, "rain_intensity": 8.0},
            11: {"temp_adj": 2, "humidity_adj": 5, "rain_prob": 0.50, "rain_intensity": 6.0},
            12: {"temp_adj": 0, "humidity_adj": 5, "rain_prob": 0.30, "rain_intensity": 3.0}
        }
    },
    "Kolkata": {
        "country": "India",
        "latitude": 22.5726,
        "longitude": 88.3639,
        "base_temp": 26.0,
        "base_humidity": 70.0,
        "base_wind": 9.0,
        "climate_zone": "subtropical",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -6, "humidity_adj": 10, "rain_prob": 0.15, "rain_intensity": 0.5},
            2: {"temp_adj": -4, "humidity_adj": 5, "rain_prob": 0.20, "rain_intensity": 0.8},
            3: {"temp_adj": 2, "humidity_adj": -5, "rain_prob": 0.25, "rain_intensity": 1.5},
            4: {"temp_adj": 6, "humidity_adj": -15, "rain_prob": 0.20, "rain_intensity": 1.0},
            5: {"temp_adj": 10, "humidity_adj": -20, "rain_prob": 0.15, "rain_intensity": 0.5},
            6: {"temp_adj": 12, "humidity_adj": -15, "rain_prob": 0.40, "rain_intensity": 3.0},
            7: {"temp_adj": 8, "humidity_adj": 15, "rain_prob": 0.80, "rain_intensity": 12.0},
            8: {"temp_adj": 7, "humidity_adj": 20, "rain_prob": 0.85, "rain_intensity": 15.0},
            9: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.60, "rain_intensity": 8.0},
            10: {"temp_adj": 2, "humidity_adj": 0, "rain_prob": 0.30, "rain_intensity": 2.0},
            11: {"temp_adj": -2, "humidity_adj": 0, "rain_prob": 0.15, "rain_intensity": 0.8},
            12: {"temp_adj": -5, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 0.5}
        }
    },
    "Bangalore": {
        "country": "India",
        "latitude": 12.9716,
        "longitude": 77.5946,
        "base_temp": 24.0,
        "base_humidity": 60.0,
        "base_wind": 8.0,
        "climate_zone": "subtropical",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -2, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 0.3},
            2: {"temp_adj": -1, "humidity_adj": 5, "rain_prob": 0.15, "rain_intensity": 0.5},
            3: {"temp_adj": 2, "humidity_adj": 0, "rain_prob": 0.20, "rain_intensity": 1.0},
            4: {"temp_adj": 4, "humidity_adj": -5, "rain_prob": 0.25, "rain_intensity": 1.5},
            5: {"temp_adj": 5, "humidity_adj": -10, "rain_prob": 0.30, "rain_intensity": 2.0},
            6: {"temp_adj": 4, "humidity_adj": -5, "rain_prob": 0.40, "rain_intensity": 3.0},
            7: {"temp_adj": 3, "humidity_adj": 10, "rain_prob": 0.70, "rain_intensity": 8.0},
            8: {"temp_adj": 2, "humidity_adj": 15, "rain_prob": 0.65, "rain_intensity": 6.0},
            9: {"temp_adj": 3, "humidity_adj": 10, "rain_prob": 0.50, "rain_intensity": 4.0},
            10: {"temp_adj": 2, "humidity_adj": 0, "rain_prob": 0.30, "rain_intensity": 2.0},
            11: {"temp_adj": 0, "humidity_adj": 0, "rain_prob": 0.15, "rain_intensity": 0.8},
            12: {"temp_adj": -2, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 0.5}
        }
    },
    "Hyderabad": {
        "country": "India",
        "latitude": 17.3850,
        "longitude": 78.4867,
        "base_temp": 27.0,
        "base_humidity": 55.0,
        "base_wind": 9.0,
        "climate_zone": "subtropical",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -4, "humidity_adj": 10, "rain_prob": 0.10, "rain_intensity": 0.3},
            2: {"temp_adj": -2, "humidity_adj": 5, "rain_prob": 0.15, "rain_intensity": 0.5},
            3: {"temp_adj": 3, "humidity_adj": -5, "rain_prob": 0.20, "rain_intensity": 1.0},
            4: {"temp_adj": 7, "humidity_adj": -15, "rain_prob": 0.15, "rain_intensity": 0.5},
            5: {"temp_adj": 10, "humidity_adj": -20, "rain_prob": 0.10, "rain_intensity": 0.3},
            6: {"temp_adj": 12, "humidity_adj": -25, "rain_prob": 0.20, "rain_intensity": 2.0},
            7: {"temp_adj": 8, "humidity_adj": 10, "rain_prob": 0.60, "rain_intensity": 6.0},
            8: {"temp_adj": 6, "humidity_adj": 15, "rain_prob": 0.55, "rain_intensity": 5.0},
            9: {"temp_adj": 5, "humidity_adj": 5, "rain_prob": 0.40, "rain_intensity": 3.0},
            10: {"temp_adj": 3, "humidity_adj": -5, "rain_prob": 0.20, "rain_intensity": 1.0},
            11: {"temp_adj": 0, "humidity_adj": 0, "rain_prob": 0.10, "rain_intensity": 0.5},
            12: {"temp_adj": -3, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 0.3}
        }
    },
    "Uttarakhand": {
        "country": "India",
        "latitude": 30.0668,
        "longitude": 79.0193,
        "base_temp": 22.0,
        "base_humidity": 55.0,
        "base_wind": 12.0,
        "climate_zone": "temperate",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -8, "humidity_adj": 10, "rain_prob": 0.20, "rain_intensity": 1.0},
            2: {"temp_adj": -6, "humidity_adj": 5, "rain_prob": 0.25, "rain_intensity": 1.5},
            3: {"temp_adj": 2, "humidity_adj": -5, "rain_prob": 0.30, "rain_intensity": 2.0},
            4: {"temp_adj": 6, "humidity_adj": -10, "rain_prob": 0.25, "rain_intensity": 1.5},
            5: {"temp_adj": 12, "humidity_adj": -15, "rain_prob": 0.15, "rain_intensity": 0.8},
            6: {"temp_adj": 15, "humidity_adj": -20, "rain_prob": 0.20, "rain_intensity": 1.5},
            7: {"temp_adj": 8, "humidity_adj": 10, "rain_prob": 0.75, "rain_intensity": 10.0},
            8: {"temp_adj": 6, "humidity_adj": 15, "rain_prob": 0.70, "rain_intensity": 8.0},
            9: {"temp_adj": 4, "humidity_adj": 5, "rain_prob": 0.50, "rain_intensity": 4.0},
            10: {"temp_adj": 2, "humidity_adj": 0, "rain_prob": 0.25, "rain_intensity": 1.5},
            11: {"temp_adj": -4, "humidity_adj": 5, "rain_prob": 0.15, "rain_intensity": 0.8},
            12: {"temp_adj": -7, "humidity_adj": 10, "rain_prob": 0.15, "rain_intensity": 0.8}
        }
    },
    "Himachal Pradesh": {
        "country": "India",
        "latitude": 31.1048,
        "longitude": 77.1734,
        "base_temp": 15.0,
        "base_humidity": 60.0,
        "base_wind": 8.0,
        "climate_zone": "temperate",
        "monsoon_affected": True,
        "monthly_patterns": {
            1: {"temp_adj": -10, "humidity_adj": 15, "rain_prob": 0.25, "rain_intensity": 1.5},
            2: {"temp_adj": -8, "humidity_adj": 10, "rain_prob": 0.30, "rain_intensity": 2.0},
            3: {"temp_adj": 0, "humidity_adj": -5, "rain_prob": 0.35, "rain_intensity": 2.5},
            4: {"temp_adj": 5, "humidity_adj": -10, "rain_prob": 0.30, "rain_intensity": 2.0},
            5: {"temp_adj": 10, "humidity_adj": -15, "rain_prob": 0.25, "rain_intensity": 1.5},
            6: {"temp_adj": 12, "humidity_adj": -20, "rain_prob": 0.35, "rain_intensity": 3.0},
            7: {"temp_adj": 6, "humidity_adj": 15, "rain_prob": 0.80, "rain_intensity": 12.0},
            8: {"temp_adj": 4, "humidity_adj": 20, "rain_prob": 0.75, "rain_intensity": 10.0},
            9: {"temp_adj": 2, "humidity_adj": 10, "rain_prob": 0.55, "rain_intensity": 5.0},
            10: {"temp_adj": 0, "humidity_adj": 0, "rain_prob": 0.30, "rain_intensity": 2.0},
            11: {"temp_adj": -6, "humidity_adj": 5, "rain_prob": 0.20, "rain_intensity": 1.0},
            12: {"temp_adj": -9, "humidity_adj": 15, "rain_prob": 0.20, "rain_intensity": 1.0}
        }
    },
    "Rajasthan": {
        "country": "India",
        "latitude": 26.2389,
        "longitude": 73.0243,
        "base_temp": 30.0,
        "base_humidity": 40.0,
        "base_wind": 12.0,
        "climate_zone": "arid",
        "monsoon_affected": False,
        "monthly_patterns": {
            1: {"temp_adj": -5, "humidity_adj": 5, "rain_prob": 0.05, "rain_intensity": 0.2},
            2: {"temp_adj": -2, "humidity_adj": 0, "rain_prob": 0.05, "rain_intensity": 0.2},
            3: {"temp_adj": 5, "humidity_adj": -10, "rain_prob": 0.05, "rain_intensity": 0.2},
            4: {"temp_adj": 10, "humidity_adj": -15, "rain_prob": 0.05, "rain_intensity": 0.2},
            5: {"temp_adj": 15, "humidity_adj": -20, "rain_prob": 0.05, "rain_intensity": 0.2},
            6: {"temp_adj": 18, "humidity_adj": -25, "rain_prob": 0.10, "rain_intensity": 0.5},
            7: {"temp_adj": 12, "humidity_adj": -10, "rain_prob": 0.30, "rain_intensity": 2.0},
            8: {"temp_adj": 10, "humidity_adj": -5, "rain_prob": 0.25, "rain_intensity": 1.5},
            9: {"temp_adj": 8, "humidity_adj": -5, "rain_prob": 0.15, "rain_intensity": 1.0},
            10: {"temp_adj": 5, "humidity_adj": -5, "rain_prob": 0.05, "rain_intensity": 0.2},
            11: {"temp_adj": 0, "humidity_adj": 0, "rain_prob": 0.05, "rain_intensity": 0.2},
            12: {"temp_adj": -4, "humidity_adj": 5, "rain_prob": 0.05, "rain_intensity": 0.2}
        }
    },
    "Los Angeles": {
        "country": "USA",
        "latitude": 34.0522,
        "longitude": -118.2437,
        "base_temp": 20.0,
        "base_humidity": 55.0,
        "base_wind": 10.0,
        "climate_zone": "mediterranean",
        "monsoon_affected": False,
        "monthly_patterns": {
            1: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.25, "rain_intensity": 3.0},
            2: {"temp_adj": 6, "humidity_adj": 5, "rain_prob": 0.20, "rain_intensity": 2.5},
            3: {"temp_adj": 7, "humidity_adj": 0, "rain_prob": 0.15, "rain_intensity": 1.5},
            4: {"temp_adj": 8, "humidity_adj": -5, "rain_prob": 0.10, "rain_intensity": 0.8},
            5: {"temp_adj": 10, "humidity_adj": -10, "rain_prob": 0.05, "rain_intensity": 0.3},
            6: {"temp_adj": 12, "humidity_adj": -15, "rain_prob": 0.02, "rain_intensity": 0.1},
            7: {"temp_adj": 14, "humidity_adj": -20, "rain_prob": 0.01, "rain_intensity": 0.1},
            8: {"temp_adj": 13, "humidity_adj": -15, "rain_prob": 0.01, "rain_intensity": 0.1},
            9: {"temp_adj": 11, "humidity_adj": -10, "rain_prob": 0.05, "rain_intensity": 0.3},
            10: {"temp_adj": 8, "humidity_adj": -5, "rain_prob": 0.10, "rain_intensity": 0.8},
            11: {"temp_adj": 5, "humidity_adj": 5, "rain_prob": 0.15, "rain_intensity": 1.5},
            12: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.20, "rain_intensity": 2.0}
        }
    },
    "Phoenix": {
        "country": "USA",
        "latitude": 33.4484,
        "longitude": -112.0740,
        "base_temp": 25.0,
        "base_humidity": 35.0,
        "base_wind": 8.0,
        "climate_zone": "arid",
        "monsoon_affected": False,
        "monthly_patterns": {
            1: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.10, "rain_intensity": 1.0},
            2: {"temp_adj": 8, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 1.0},
            3: {"temp_adj": 10, "humidity_adj": -5, "rain_prob": 0.05, "rain_intensity": 0.5},
            4: {"temp_adj": 15, "humidity_adj": -10, "rain_prob": 0.02, "rain_intensity": 0.2},
            5: {"temp_adj": 20, "humidity_adj": -15, "rain_prob": 0.01, "rain_intensity": 0.1},
            6: {"temp_adj": 22, "humidity_adj": -20, "rain_prob": 0.01, "rain_intensity": 0.1},
            7: {"temp_adj": 20, "humidity_adj": -15, "rain_prob": 0.05, "rain_intensity": 0.5},
            8: {"temp_adj": 18, "humidity_adj": -10, "rain_prob": 0.10, "rain_intensity": 1.0},
            9: {"temp_adj": 15, "humidity_adj": -5, "rain_prob": 0.15, "rain_intensity": 1.5},
            10: {"temp_adj": 10, "humidity_adj": 0, "rain_prob": 0.10, "rain_intensity": 1.0},
            11: {"temp_adj": 5, "humidity_adj": 5, "rain_prob": 0.10, "rain_intensity": 1.0},
            12: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.10, "rain_intensity": 1.0}
        }
    },
    "Sydney": {
        "country": "Australia",
        "latitude": -33.8688,
        "longitude": 151.2093,
        "base_temp": 18.0,
        "base_humidity": 65.0,
        "base_wind": 11.0,
        "climate_zone": "subtropical",
        "monsoon_affected": False,
        "monthly_patterns": {
            1: {"temp_adj": 8, "humidity_adj": -10, "rain_prob": 0.20, "rain_intensity": 2.0},
            2: {"temp_adj": 7, "humidity_adj": -5, "rain_prob": 0.25, "rain_intensity": 2.5},
            3: {"temp_adj": 5, "humidity_adj": 0, "rain_prob": 0.30, "rain_intensity": 3.0},
            4: {"temp_adj": 3, "humidity_adj": 5, "rain_prob": 0.30, "rain_intensity": 3.0},
            5: {"temp_adj": 1, "humidity_adj": 5, "rain_prob": 0.25, "rain_intensity": 2.5},
            6: {"temp_adj": 0, "humidity_adj": 10, "rain_prob": 0.25, "rain_intensity": 2.5},
            7: {"temp_adj": -2, "humidity_adj": 10, "rain_prob": 0.20, "rain_intensity": 2.0},
            8: {"temp_adj": -3, "humidity_adj": 5, "rain_prob": 0.20, "rain_intensity": 2.0},
            9: {"temp_adj": -1, "humidity_adj": 0, "rain_prob": 0.25, "rain_intensity": 2.5},
            10: {"temp_adj": 2, "humidity_adj": -5, "rain_prob": 0.25, "rain_intensity": 2.5},
            11: {"temp_adj": 5, "humidity_adj": -10, "rain_prob": 0.20, "rain_intensity": 2.0},
            12: {"temp_adj": 7, "humidity_adj": -10, "rain_prob": 0.20, "rain_intensity": 2.0}
        }
    },
    "Athens": {
        "country": "Greece",
        "latitude": 37.9838,
        "longitude": 23.7275,
        "base_temp": 18.0,
        "base_humidity": 55.0,
        "base_wind": 12.0,
        "climate_zone": "mediterranean",
        "monsoon_affected": False,
        "monthly_patterns": {
            1: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.30, "rain_intensity": 3.0},
            2: {"temp_adj": 6, "humidity_adj": 5, "rain_prob": 0.25, "rain_intensity": 2.5},
            3: {"temp_adj": 8, "humidity_adj": 0, "rain_prob": 0.20, "rain_intensity": 2.0},
            4: {"temp_adj": 10, "humidity_adj": -5, "rain_prob": 0.15, "rain_intensity": 1.5},
            5: {"temp_adj": 12, "humidity_adj": -10, "rain_prob": 0.10, "rain_intensity": 1.0},
            6: {"temp_adj": 15, "humidity_adj": -15, "rain_prob": 0.05, "rain_intensity": 0.5},
            7: {"temp_adj": 18, "humidity_adj": -20, "rain_prob": 0.02, "rain_intensity": 0.2},
            8: {"temp_adj": 17, "humidity_adj": -15, "rain_prob": 0.03, "rain_intensity": 0.3},
            9: {"temp_adj": 14, "humidity_adj": -10, "rain_prob": 0.10, "rain_intensity": 1.0},
            10: {"temp_adj": 10, "humidity_adj": -5, "rain_prob": 0.20, "rain_intensity": 2.0},
            11: {"temp_adj": 6, "humidity_adj": 5, "rain_prob": 0.30, "rain_intensity": 3.0},
            12: {"temp_adj": 5, "humidity_adj": 10, "rain_prob": 0.35, "rain_intensity": 3.5}
        }
    }
}

def get_city_climate(city_name: str) -> dict:
    """Get climate data for a city"""
    city_name_normalized = city_name.strip().title()
    
    # Try exact match first
    if city_name_normalized in CITY_CLIMATE_DB:
        return CITY_CLIMATE_DB[city_name_normalized]
    
    # Try partial match
    for city, data in CITY_CLIMATE_DB.items():
        if city_name_normalized in city or city in city_name_normalized:
            return data
    
    # Return default (Delhi) if not found
    return CITY_CLIMATE_DB["Delhi"]

def get_monthly_adjustments(city_name: str, month: int) -> dict:
    """Get monthly weather adjustments for a city"""
    climate = get_city_climate(city_name)
    return climate["monthly_patterns"].get(month, {"temp_adj": 0, "humidity_adj": 0, "rain_prob": 0.2, "rain_intensity": 1.0})
