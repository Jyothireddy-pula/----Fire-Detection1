"""
Test fuzzy system with data from enhanced dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline

def test_with_dataset():
    """Test predictions with actual dataset samples"""
    print("=" * 70)
    print("TESTING FUZZY SYSTEM WITH ENHANCED DATASET SAMPLES")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DataPipeline(model_dir='models')
    pipeline.load_models()
    
    # Sample data from enhanced dataset
    test_cases = [
        {
            'name': 'Uttarakhand - April Day 3 (Fire)',
            'weather': {'temperature': 32, 'humidity': 55, 'wind_speed': 15, 'rainfall': 0},
            'month': 4,
            'location': 'Uttarakhand',
            'expected': 'fire'
        },
        {
            'name': 'Uttarakhand - April Day 12 (Fire)',
            'weather': {'temperature': 34, 'humidity': 48, 'wind_speed': 18, 'rainfall': 0},
            'month': 4,
            'location': 'Uttarakhand',
            'expected': 'fire'
        },
        {
            'name': 'Uttarakhand - April Day 1 (Not Fire)',
            'weather': {'temperature': 28, 'humidity': 65, 'wind_speed': 12, 'rainfall': 0.5},
            'month': 4,
            'location': 'Uttarakhand',
            'expected': 'not fire'
        },
        {
            'name': 'Himachal Pradesh - May Day 1 (Fire)',
            'weather': {'temperature': 35, 'humidity': 45, 'wind_speed': 18, 'rainfall': 0},
            'month': 5,
            'location': 'Himachal Pradesh',
            'expected': 'fire'
        },
        {
            'name': 'Himachal Pradesh - May Day 6 (Not Fire)',
            'weather': {'temperature': 28, 'humidity': 68, 'wind_speed': 10, 'rainfall': 2.8},
            'month': 5,
            'location': 'Himachal Pradesh',
            'expected': 'not fire'
        },
        {
            'name': 'Himachal Pradesh - May Day 8 (Not Fire)',
            'weather': {'temperature': 26, 'humidity': 75, 'wind_speed': 8, 'rainfall': 5.2},
            'month': 5,
            'location': 'Himachal Pradesh',
            'expected': 'not fire'
        }
    ]
    
    print("\nTesting with actual dataset samples:\n")
    
    for test_case in test_cases:
        result = pipeline.predict_pipeline(
            test_case['weather'],
            test_case['month'],
            location=test_case['location']
        )
        
        risk_level = result['linguistic_risk_level']
        risk_score = result['risk_score']
        reasoning = result['reasoning']
        
        print(f"Test: {test_case['name']}")
        print(f"  Expected: {test_case['expected']}")
        print(f"  Result: {risk_level} (Score: {risk_score:.3f})")
        print(f"  Reasoning: {reasoning}")
        print()
    
    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    test_with_dataset()
