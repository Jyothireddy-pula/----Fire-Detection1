"""
Test to verify all 5 risk categories appear for each city across different months
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline

def test_all_categories_per_city():
    """Test that all 5 risk categories appear for each city"""
    print("=" * 70)
    print("TESTING ALL 5 RISK CATEGORIES PER CITY")
    print("=" * 70)
    
    pipeline = DataPipeline(model_dir='models')
    pipeline.load_models()
    
    cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", 
              "Hyderabad", "Uttarakhand", "Himachal Pradesh", "Rajasthan"]
    
    # Different weather scenarios to trigger different risk levels
    weather_scenarios = [
        {'name': 'Cold Wet', 'weather': {'temperature': 10, 'humidity': 90, 'wind_speed': 5, 'rainfall': 10}},
        {'name': 'Mild Moderate', 'weather': {'temperature': 20, 'humidity': 60, 'wind_speed': 10, 'rainfall': 2}},
        {'name': 'Warm Dry', 'weather': {'temperature': 30, 'humidity': 40, 'wind_speed': 15, 'rainfall': 0}},
        {'name': 'Hot Dry', 'weather': {'temperature': 38, 'humidity': 25, 'wind_speed': 20, 'rainfall': 0}},
        {'name': 'Very Hot Very Dry Windy', 'weather': {'temperature': 45, 'humidity': 10, 'wind_speed': 35, 'rainfall': 0}},
        {'name': 'Extreme Heat', 'weather': {'temperature': 50, 'humidity': 5, 'wind_speed': 40, 'rainfall': 0}}
    ]
    
    print("\nTesting all cities across all months with different weather scenarios:\n")
    
    results = {}
    
    for city in cities:
        print(f"\n{'='*70}")
        print(f"CITY: {city}")
        print(f"{'='*70}")
        
        city_categories = set()
        city_results = []
        
        for month in range(1, 13):
            for scenario in weather_scenarios:
                result = pipeline.predict_pipeline(
                    scenario['weather'],
                    month,
                    location=city
                )
                
                risk_level = result['linguistic_risk_level']
                city_categories.add(risk_level)
                
                city_results.append({
                    'month': month,
                    'scenario': scenario['name'],
                    'risk_level': risk_level,
                    'weather': scenario['weather']
                })
        
        results[city] = {
            'categories': city_categories,
            'results': city_results
        }
        
        print(f"\nRisk categories for {city}: {sorted(city_categories)}")
        print(f"Total categories: {len(city_categories)}")
        
        if len(city_categories) < 5:
            missing = {'No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire'} - city_categories
            print(f"Missing categories: {missing}")
        else:
            print("✓ All 5 categories covered!")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for city, data in results.items():
        print(f"\n{city}: {len(data['categories'])}/5 categories - {sorted(data['categories'])}")
    
    cities_with_all = sum(1 for data in results.values() if len(data['categories']) == 5)
    print(f"\nCities with all 5 categories: {cities_with_all}/{len(cities)}")
    
    if cities_with_all < len(cities):
        print("\n⚠️  Some cities are missing categories. Adjusting prediction logic...")

if __name__ == "__main__":
    test_all_categories_per_city()
