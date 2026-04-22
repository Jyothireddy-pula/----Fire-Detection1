"""
Multi-Zone Location Configuration
Defines cities with multiple zones for fine-grained wildfire monitoring
"""

LOCATIONS = [
    {
        "city": "Amaravati",
        "zones": [
            {"name": "Amaravati - Forest Edge", "lat": 16.506, "lon": 80.648},
            {"name": "Amaravati - Urban Center", "lat": 16.510, "lon": 80.650},
            {"name": "Amaravati - Dry Grassland", "lat": 16.502, "lon": 80.644},
            {"name": "Amaravati - River Bank", "lat": 16.514, "lon": 80.652},
            {"name": "Amaravati - Agricultural Zone", "lat": 16.498, "lon": 80.642}
        ]
    },
    {
        "city": "Hyderabad",
        "zones": [
            {"name": "Hyderabad - Northern Hills", "lat": 17.400, "lon": 78.480},
            {"name": "Hyderabad - Urban Center", "lat": 17.385, "lon": 78.486},
            {"name": "Hyderabad - Southern Forest", "lat": 17.365, "lon": 78.475},
            {"name": "Hyderabad - Eastern Drylands", "lat": 17.395, "lon": 78.505},
            {"name": "Hyderabad - Western Scrub", "lat": 17.375, "lon": 78.455}
        ]
    },
    {
        "city": "Dehradun",
        "zones": [
            {"name": "Dehradun - Rajpur Forest", "lat": 30.325, "lon": 78.055},
            {"name": "Dehradun - City Center", "lat": 30.327, "lon": 78.032},
            {"name": "Dehradun - Mussoorie Foothills", "lat": 30.340, "lon": 78.060},
            {"name": "Dehradun - Eastern Valley", "lat": 30.315, "lon": 78.045},
            {"name": "Dehradun - Western Timber", "lat": 30.335, "lon": 78.020}
        ]
    },
    {
        "city": "Bengaluru",
        "zones": [
            {"name": "Bengaluru - Bannerghatta Forest", "lat": 12.910, "lon": 77.580},
            {"name": "Bengaluru - City Center", "lat": 12.971, "lon": 77.594},
            {"name": "Bengaluru - Northern Outskirts", "lat": 13.020, "lon": 77.590},
            {"name": "Bengaluru - Eastern Dry Zone", "lat": 12.950, "lon": 77.650},
            {"name": "Bengaluru - Western Scrub", "lat": 12.930, "lon": 77.530}
        ]
    },
    {
        "city": "Srinagar",
        "zones": [
            {"name": "Srinagar - Dal Lake Area", "lat": 34.080, "lon": 74.800},
            {"name": "Srinagar - City Center", "lat": 34.085, "lon": 74.797},
            {"name": "Srinagar - Northern Forest", "lat": 34.100, "lon": 74.810},
            {"name": "Srinagar - Eastern Hills", "lat": 34.090, "lon": 74.820},
            {"name": "Srinagar - Western Valley", "lat": 34.070, "lon": 74.785}
        ]
    }
]

def get_all_zones():
    """Get all zones from all cities as a flat list"""
    all_zones = []
    for city_data in LOCATIONS:
        for zone in city_data["zones"]:
            all_zones.append({
                "city": city_data["city"],
                "zone_name": zone["name"],
                "lat": zone["lat"],
                "lon": zone["lon"]
            })
    return all_zones

def get_city_zones(city_name):
    """Get zones for a specific city"""
    for city_data in LOCATIONS:
        if city_data["city"] == city_name:
            return city_data["zones"]
    return []

def get_all_cities():
    """Get list of all city names"""
    return [city["city"] for city in LOCATIONS]
