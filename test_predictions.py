"""
Test script to verify predictions show realistic risk variations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline

def test_predictions():
    """Test predictions with different cities and months"""
    print("=" * 70)
    print("TESTING PREDICTIONS WITH REALISTIC SCENARIOS")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DataPipeline(model_dir='models')
    pipeline.load_models()
    
    # Test scenarios with different weather conditions
    test_scenarios = [
        {
            'name': 'Winter - Cold & Humid (Low Risk)',
            'weather': {'temperature': 10, 'humidity': 80, 'wind_speed': 5, 'rainfall': 2},
            'month': 1
        },
        {
            'name': 'Summer - Hot & Dry (High Risk)',
            'weather': {'temperature': 40, 'humidity': 20, 'wind_speed': 15, 'rainfall': 0},
            'month': 5
        },
        {
            'name': 'Monsoon - Wet (Low Risk)',
            'weather': {'temperature': 28, 'humidity': 90, 'wind_speed': 8, 'rainfall': 15},
            'month': 7
        },
        {
            'name': 'Spring - Moderate (Medium Risk)',
            'weather': {'temperature': 30, 'humidity': 50, 'wind_speed': 10, 'rainfall': 0.5},
            'month': 3
        },
        {
            'name': 'Autumn - Cooling (Low-Medium Risk)',
            'weather': {'temperature': 25, 'humidity': 60, 'wind_speed': 8, 'rainfall': 1},
            'month': 10
        },
        {
            'name': 'Extreme Heat - Very Dry (Extreme Risk)',
            'weather': {'temperature': 45, 'humidity': 15, 'wind_speed': 20, 'rainfall': 0},
            'month': 6
        },
    ]
    
    print("\nTesting different scenarios:\n")
    
    for scenario in test_scenarios:
        result = pipeline.predict_pipeline(scenario['weather'], scenario['month'])
        
        print(f"Scenario: {scenario['name']}")
        print(f"  Month: {scenario['month']}")
        print(f"  Weather: Temp={scenario['weather']['temperature']}°C, RH={scenario['weather']['humidity']}%, Wind={scenario['weather']['wind_speed']}km/h, Rain={scenario['weather']['rainfall']}mm")
        print(f"  FWI: {result['fwi_components']['FWI']:.2f}")
        print(f"  Risk Level: {result['linguistic_risk_level']}")
        print(f"  Risk Score: {result['risk_score']:.3f}")
        print()
    
    # Test the same city in different months
    print("=" * 70)
    print("TESTING SEASONAL VARIATION FOR SAME CITY")
    print("=" * 70)
    
    base_weather = {'temperature': 30, 'humidity': 50, 'wind_speed': 10, 'rainfall': 0}
    
    print("\nBase weather condition: Temp=30°C, RH=50%, Wind=10km/h, Rain=0mm")
    print("\nRisk by month:\n")
    
    monthly_risks = []
    for month in range(1, 13):
        # Adjust temperature slightly by season
        if month in [12, 1, 2]:
            temp_adj = -10
        elif month in [3, 4, 5]:
            temp_adj = 5
        elif month in [6, 7, 8]:
            temp_adj = 8
        else:
            temp_adj = 0
        
        weather = {
            'temperature': base_weather['temperature'] + temp_adj,
            'humidity': base_weather['humidity'],
            'wind_speed': base_weather['wind_speed'],
            'rainfall': base_weather['rainfall']
        }
        
        result = pipeline.predict_pipeline(weather, month)
        monthly_risks.append({
            'month': month,
            'risk_level': result['linguistic_risk_level'],
            'risk_score': result['risk_score'],
            'fwi': result['fwi_components']['FWI']
        })
    
    for risk in monthly_risks:
        print(f"Month {risk['month']:2d}: {risk['risk_level']:15s} (Score: {risk['risk_score']:.3f}, FWI: {risk['fwi']:.2f})")
    
    # Count risk level distribution
    risk_levels = [r['risk_level'] for r in monthly_risks]
    print(f"\nRisk level distribution across months:")
    for level in ['No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire']:
        count = risk_levels.count(level)
        if count > 0:
            print(f"  {level}: {count} months")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_predictions()
