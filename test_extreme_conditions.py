"""
Test script to verify extreme weather conditions trigger High and Extreme Fire
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline

def test_extreme_conditions():
    """Test predictions with extreme weather conditions"""
    print("=" * 70)
    print("TESTING EXTREME WEATHER CONDITIONS")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DataPipeline(model_dir='models')
    pipeline.load_models()
    
    # Test cities
    cities = ["Delhi", "Rajasthan", "Phoenix", "Athens", "Los Angeles"]
    
    # Extreme weather scenarios
    extreme_scenarios = [
        {
            'name': 'Extreme Heat - Very Dry',
            'weather': {'temperature': 48, 'humidity': 10, 'wind_speed': 25, 'rainfall': 0},
            'months': [5, 6, 7]
        },
        {
            'name': 'Very Hot - Dry - Windy',
            'weather': {'temperature': 45, 'humidity': 15, 'wind_speed': 30, 'rainfall': 0},
            'months': [4, 5, 6, 7, 8]
        },
        {
            'name': 'Hot - Extremely Dry',
            'weather': {'temperature': 42, 'humidity': 5, 'wind_speed': 20, 'rainfall': 0},
            'months': [5, 6, 7]
        },
        {
            'name': 'Moderate Heat - Very Dry - High Wind',
            'weather': {'temperature': 38, 'humidity': 12, 'wind_speed': 35, 'rainfall': 0},
            'months': [4, 5, 6, 7, 8]
        }
    ]
    
    print("\nTesting extreme conditions for high-risk cities:\n")
    
    all_results = []
    risk_categories = set()
    
    for city in cities:
        print(f"\n{'='*70}")
        print(f"CITY: {city}")
        print(f"{'='*70}")
        
        city_results = []
        for scenario in extreme_scenarios:
            print(f"\nScenario: {scenario['name']}")
            for month in scenario['months']:
                result = pipeline.predict_pipeline(scenario['weather'], month, location=city)
                
                risk_level = result['linguistic_risk_level']
                risk_score = result['risk_score']
                fwi = result['fwi_components']['FWI']
                temp = result['fwi_components']['Temperature']
                rh = result['fwi_components']['RH']
                wind = result['fwi_components']['Ws']
                
                city_results.append({
                    'month': month,
                    'scenario': scenario['name'],
                    'risk_level': risk_level,
                    'risk_score': risk_score,
                    'fwi': fwi,
                    'temp': temp,
                    'rh': rh,
                    'wind': wind
                })
                
                risk_categories.add(risk_level)
                
                print(f"  Month {month:2d}: {risk_level:15s} (Score: {risk_score:.3f}, FWI: {fwi:.1f}, Temp: {temp:.1f}°C, RH: {rh:.0f}%, Wind: {wind:.1f}km/h)")
        
        all_results.append({'city': city, 'results': city_results})
        
        # Show risk distribution for this city
        risk_dist = {}
        for r in city_results:
            risk_level = r['risk_level']
            risk_dist[risk_level] = risk_dist.get(risk_level, 0) + 1
        print(f"\nRisk distribution for {city} (extreme conditions):")
        for level, count in sorted(risk_dist.items()):
            print(f"  {level}: {count} scenarios")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal risk categories observed: {len(risk_categories)}")
    print(f"Risk categories: {sorted(risk_categories)}")
    
    # Check if all categories are covered
    expected_categories = {'No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire'}
    missing_categories = expected_categories - risk_categories
    
    if missing_categories:
        print(f"\n⚠️  Missing risk categories: {missing_categories}")
        print("\nFurther adjustments needed to trigger higher risk categories")
    else:
        print(f"\n✓ All risk categories covered!")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_extreme_conditions()
