"""
Simulation Module
Provides trend analysis, scenario simulation, and comparative analysis with explainability
"""

import numpy as np
from typing import Dict, List, Tuple
from backend.services.pipeline import DataPipeline
from backend.services.decision import DecisionEngine
from backend.services.explainability import ExplainabilityEngine
from backend.utils.logger import system_logger


class SimulationEngine:
    """Simulation engine for wildfire risk analysis"""
    
    def __init__(self, pipeline: DataPipeline):
        """
        Initialize simulation engine
        
        Args:
            pipeline: Data pipeline instance
        """
        self.pipeline = pipeline
        self.decision_engine = DecisionEngine()
    
    def trend_analysis(self, base_weather: Dict, month: int = 6, 
                      feature: str = 'temperature', 
                      n_points: int = 20, 
                      range_pct: float = 0.5) -> Dict:
        """
        Analyze how risk changes with a single feature
        
        Args:
            base_weather: Base weather conditions
            month: Current month
            feature: Feature to vary ('temperature', 'humidity', 'wind_speed', 'rainfall')
            n_points: Number of points in the analysis
            range_pct: Percentage range to vary (e.g., 0.5 = ±50%)
            
        Returns:
            Trend analysis results
        """
        base_value = base_weather[feature]
        value_range = base_value * range_pct
        
        # Generate values to test
        values = np.linspace(base_value - value_range, base_value + value_range, n_points)
        values = np.clip(values, 0, None)  # Ensure non-negative
        
        risk_scores = []
        risk_levels = []
        
        for value in values:
            # Create modified weather data
            modified_weather = base_weather.copy()
            modified_weather[feature] = float(value)
            
            # Make prediction
            prediction = self.pipeline.predict_pipeline(modified_weather, month)
            decision = self.decision_engine.make_decision(prediction)
            
            risk_scores.append(decision['risk_score'])
            risk_levels.append(decision['linguistic_risk_level'])
        
        return {
            'feature': feature,
            'base_value': base_value,
            'values': values.tolist(),
            'risk_scores': risk_scores,
            'risk_levels': risk_levels,
            'sensitivity': float(np.std(risk_scores))
        }
    
    def scenario_simulation(self, base_weather: Dict, month: int = 6,
                           scenarios: List[Dict] = None) -> List[Dict]:
        """
        Simulate different scenarios
        
        Args:
            base_weather: Base weather conditions
            month: Current month
            scenarios: List of scenario modifications
                      [{'name': 'Heat Wave', 'temperature': 1.5}, ...]
                      Multipliers are applied to base values
            
        Returns:
            Scenario simulation results
        """
        if scenarios is None:
            scenarios = [
                {'name': 'Heat Wave', 'temperature': 1.5, 'humidity': 0.7},
                {'name': 'Drought', 'rainfall': 0.0, 'humidity': 0.5},
                {'name': 'High Wind', 'wind_speed': 2.0},
                {'name': 'Storm', 'rainfall': 5.0, 'wind_speed': 1.5},
                {'name': 'Normal Conditions', 'temperature': 1.0, 'humidity': 1.0, 
                 'wind_speed': 1.0, 'rainfall': 1.0}
            ]
        
        results = []

        for scenario in scenarios:
            # Apply scenario modifications
            modified_weather = base_weather.copy()

            for feature, multiplier in scenario.items():
                if feature != 'name' and feature in modified_weather:
                    modified_weather[feature] = base_weather[feature] * multiplier

            # Ensure non-negative
            for key in modified_weather:
                if isinstance(modified_weather[key], (int, float)):
                    modified_weather[key] = max(0, modified_weather[key])

            # Make prediction
            try:
                prediction = self.pipeline.predict_pipeline(modified_weather, month)
                decision = self.decision_engine.make_decision(prediction)

                result = {
                    'scenario_name': scenario['name'],
                    'weather': modified_weather,
                    'prediction': prediction,
                    'decision': decision
                }

                # Add explainability
                try:
                    explainability = ExplainabilityEngine(
                        self.pipeline.fuzzy_wildfire_system,
                        self.pipeline.anfis_model
                    )
                    explanation = explainability.explain_prediction(
                        prediction,
                        modified_weather,
                        prediction.get('fwi_components', {})
                    )
                    result['explanation'] = explanation

                    # Explain delta from baseline if this isn't baseline
                    if scenario['name'] != 'Normal Conditions':
                        result['delta_explanation'] = explainability.explain_scenario_delta(
                            base_risk=1.0,  # placeholder
                            scenario_risk=decision['risk_score'],
                            base_explanation={},
                            scenario_explanation=explanation
                        )
                except Exception:
                    result['explanation'] = None

                results.append(result)
                
            except Exception as e:
                system_logger.error(f"Scenario {scenario['name']} failed: {str(e)}")
        
        # Sort by risk score
        results.sort(key=lambda x: x['decision']['risk_score'], reverse=True)
        
        return results
    
    def comparative_analysis(self, weather1: Dict, weather2: Dict, 
                            month: int = 6) -> Dict:
        """
        Compare two weather conditions
        
        Args:
            weather1: First weather condition
            weather2: Second weather condition
            month: Current month
            
        Returns:
            Comparative analysis
        """
        # Predict for both conditions
        pred1 = self.pipeline.predict_pipeline(weather1, month)
        dec1 = self.decision_engine.make_decision(pred1)
        
        pred2 = self.pipeline.predict_pipeline(weather2, month)
        dec2 = self.decision_engine.make_decision(pred2)
        
        # Calculate differences
        risk_diff = dec2['risk_score'] - dec1['risk_score']
        
        comparison = {
            'condition1': {
                'weather': weather1,
                'risk_score': dec1['risk_score'],
                'risk_level': dec1['linguistic_risk_level']
            },
            'condition2': {
                'weather': weather2,
                'risk_score': dec2['risk_score'],
                'risk_level': dec2['linguistic_risk_level']
            },
            'difference': {
                'risk_score': risk_diff,
                'higher_risk_condition': 2 if risk_diff > 0 else 1,
                'magnitude': abs(risk_diff)
            }
        }
        
        return comparison
