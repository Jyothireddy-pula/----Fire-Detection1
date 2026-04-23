"""
Test script to verify fuzzy logic system covers all 5 risk categories
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.models.fuzzy_wildfire import create_fuzzy_wildfire_system

def test_fuzzy_system():
    """Test fuzzy system with various input combinations"""
    print("=" * 70)
    print("TESTING FUZZY LOGIC WILDFIRE SYSTEM")
    print("=" * 70)
    
    fuzzy_system = create_fuzzy_wildfire_system()
    
    # Test cases covering all risk categories
    test_cases = [
        {
            'name': 'No Fire - Cold, Wet, Calm',
            'inputs': {'temperature': 10, 'humidity': 90, 'wind_speed': 5, 'rainfall': 10, 'vegetation_dryness': 50},
            'expected': 'No Fire'
        },
        {
            'name': 'No Fire - Heavy Rain',
            'inputs': {'temperature': 25, 'humidity': 70, 'wind_speed': 10, 'rainfall': 15, 'vegetation_dryness': 60},
            'expected': 'No Fire'
        },
        {
            'name': 'Low Fire - Mild Conditions',
            'inputs': {'temperature': 20, 'humidity': 50, 'wind_speed': 8, 'rainfall': 0.5, 'vegetation_dryness': 70},
            'expected': 'Low Fire'
        },
        {
            'name': 'Low Fire - Low Temp, Dry Veg',
            'inputs': {'temperature': 15, 'humidity': 40, 'wind_speed': 5, 'rainfall': 0, 'vegetation_dryness': 90},
            'expected': 'Low Fire'
        },
        {
            'name': 'Medium Fire - Moderate Conditions',
            'inputs': {'temperature': 30, 'humidity': 40, 'wind_speed': 15, 'rainfall': 0, 'vegetation_dryness': 80},
            'expected': 'Medium Fire'
        },
        {
            'name': 'Medium Fire - High Temp, Medium Humidity',
            'inputs': {'temperature': 38, 'humidity': 55, 'wind_speed': 8, 'rainfall': 0, 'vegetation_dryness': 75},
            'expected': 'Medium Fire'
        },
        {
            'name': 'High Fire - Multiple High Risk Factors',
            'inputs': {'temperature': 40, 'humidity': 25, 'wind_speed': 20, 'rainfall': 0, 'vegetation_dryness': 88},
            'expected': 'High Fire'
        },
        {
            'name': 'High Fire - High Wind, Dry',
            'inputs': {'temperature': 32, 'humidity': 20, 'wind_speed': 30, 'rainfall': 0, 'vegetation_dryness': 92},
            'expected': 'High Fire'
        },
        {
            'name': 'High Fire - High Temp, High Wind',
            'inputs': {'temperature': 42, 'humidity': 35, 'wind_speed': 28, 'rainfall': 0, 'vegetation_dryness': 85},
            'expected': 'High Fire'
        },
        {
            'name': 'Extreme Fire - All Critical Factors',
            'inputs': {'temperature': 48, 'humidity': 10, 'wind_speed': 35, 'rainfall': 0, 'vegetation_dryness': 98},
            'expected': 'Extreme Fire'
        },
        {
            'name': 'Extreme Fire - Very Hot, Very Dry, Very Windy',
            'inputs': {'temperature': 50, 'humidity': 5, 'wind_speed': 40, 'rainfall': 0, 'vegetation_dryness': 100},
            'expected': 'Extreme Fire'
        },
        {
            'name': 'Extreme Fire - Extreme Heat',
            'inputs': {'temperature': 55, 'humidity': 15, 'wind_speed': 30, 'rainfall': 0.2, 'vegetation_dryness': 95},
            'expected': 'Extreme Fire'
        }
    ]
    
    print("\nTesting fuzzy system with various input combinations:\n")
    
    results = []
    risk_categories = set()
    
    for test_case in test_cases:
        result = fuzzy_system.predict(
            temperature=test_case['inputs']['temperature'],
            humidity=test_case['inputs']['humidity'],
            wind_speed=test_case['inputs']['wind_speed'],
            rainfall=test_case['inputs']['rainfall'],
            vegetation_dryness=test_case['inputs']['vegetation_dryness']
        )
        
        risk_level = result['linguistic_risk_level']
        risk_score = result['risk_score']
        rule_firing = result['rule_firing_strength']
        reasoning = result['reasoning']
        
        risk_categories.add(risk_level)
        
        # Get linguistic terms for inputs
        temp_ling = fuzzy_system.get_input_linguistic(test_case['inputs']['temperature'], 'temperature')
        hum_ling = fuzzy_system.get_input_linguistic(test_case['inputs']['humidity'], 'humidity')
        wind_ling = fuzzy_system.get_input_linguistic(test_case['inputs']['wind_speed'], 'wind')
        rain_ling = fuzzy_system.get_input_linguistic(test_case['inputs']['rainfall'], 'rainfall')
        veg_ling = fuzzy_system.get_input_linguistic(test_case['inputs']['vegetation_dryness'], 'vegetation')
        
        print(f"Test: {test_case['name']}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Result: {risk_level} (Score: {risk_score:.3f}, Rule Firing: {rule_firing:.3f})")
        print(f"  Inputs: Temp={test_case['inputs']['temperature']}°C ({temp_ling}), "
              f"Humidity={test_case['inputs']['humidity']}% ({hum_ling}), "
              f"Wind={test_case['inputs']['wind_speed']}km/h ({wind_ling}), "
              f"Rain={test_case['inputs']['rainfall']}mm ({rain_ling}), "
              f"Veg={test_case['inputs']['vegetation_dryness']} ({veg_ling})")
        print(f"  Reasoning: {reasoning}")
        print()
        
        results.append({
            'test': test_case['name'],
            'expected': test_case['expected'],
            'result': risk_level,
            'match': risk_level == test_case['expected']
        })
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal risk categories observed: {len(risk_categories)}")
    print(f"Risk categories: {sorted(risk_categories)}")
    
    # Check if all categories are covered
    expected_categories = {'No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire'}
    missing_categories = expected_categories - risk_categories
    
    if missing_categories:
        print(f"\n⚠️  Missing risk categories: {missing_categories}")
    else:
        print(f"\n✓ All risk categories covered!")
    
    # Check test accuracy
    correct = sum(1 for r in results if r['match'])
    print(f"\nTest accuracy: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    if correct < len(results):
        print("\nMismatched tests:")
        for r in results:
            if not r['match']:
                print(f"  {r['test']}: Expected {r['expected']}, Got {r['result']}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_fuzzy_system()
