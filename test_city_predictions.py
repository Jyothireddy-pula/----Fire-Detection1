"""
Test script to verify city-specific predictions show all risk categories
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline

def test_city_month_predictions():
    """Test predictions for multiple cities across all months"""
    print("=" * 70)
    print("TESTING CITY-SPECIFIC PREDICTIONS")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DataPipeline(model_dir='models')
    pipeline.load_models()
    
    # Test cities
    cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", 
              "Hyderabad", "Uttarakhand", "Himachal Pradesh", "Rajasthan",
              "Los Angeles", "Phoenix", "Sydney", "Athens"]
    
    # Base weather condition
    base_weather = {'temperature': 30, 'humidity': 50, 'wind_speed': 10, 'rainfall': 0}
    
    print("\nTesting predictions for all cities across all months:\n")
    
    all_results = []
    risk_categories = set()
    
    for city in cities:
        print(f"\n{'='*70}")
        print(f"CITY: {city}")
        print(f"{'='*70}")
        
        city_results = []
        for month in range(1, 13):
            result = pipeline.predict_pipeline(base_weather, month, location=city)
            
            risk_level = result['linguistic_risk_level']
            risk_score = result['risk_score']
            fwi = result['fwi_components']['FWI']
            temp = result['fwi_components']['Temperature']
            rh = result['fwi_components']['RH']
            
            city_results.append({
                'month': month,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'fwi': fwi,
                'temp': temp,
                'rh': rh
            })
            
            risk_categories.add(risk_level)
            
            print(f"Month {month:2d}: {risk_level:15s} (Score: {risk_score:.3f}, FWI: {fwi:.1f}, Temp: {temp:.1f}°C, RH: {rh:.0f}%)")
        
        all_results.append({'city': city, 'results': city_results})
        
        # Show risk distribution for this city
        risk_dist = {}
        for r in city_results:
            risk_level = r['risk_level']
            risk_dist[risk_level] = risk_dist.get(risk_level, 0) + 1
        print(f"\nRisk distribution for {city}:")
        for level, count in sorted(risk_dist.items()):
            print(f"  {level}: {count} months")
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal risk categories observed across all cities: {len(risk_categories)}")
    print(f"Risk categories: {sorted(risk_categories)}")
    
    # Check if all categories are covered
    expected_categories = {'No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire'}
    missing_categories = expected_categories - risk_categories
    
    if missing_categories:
        print(f"\n⚠️  Missing risk categories: {missing_categories}")
    else:
        print(f"\n✓ All risk categories covered!")
    
    # Show unique combinations
    print(f"\nUnique city-month risk combinations: {len(all_results) * 12}")
    
    # Show examples of each risk category
    print(f"\nExamples of each risk category:")
    for category in sorted(risk_categories):
        print(f"\n{category}:")
        examples = []
        for city_data in all_results:
            for r in city_data['results']:
                if r['risk_level'] == category and len(examples) < 3:
                    examples.append(f"  {city_data['city']}, Month {r['month']}: FWI={r['fwi']:.1f}, Temp={r['temp']:.1f}°C, RH={r['rh']:.0f}%")
        for ex in examples:
            print(ex)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_city_month_predictions()
