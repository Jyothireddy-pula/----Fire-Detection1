"""
Comprehensive Fuzzy Logic Wildfire Prediction System
Uses linguistic variables and weighted scoring for accurate wildfire risk assessment
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class FuzzyWildfireSystem:
    """
    Fuzzy Logic System for Wildfire Prediction
    Implements weighted scoring with linguistic variables and rule-based reasoning
    """
    
    def __init__(self):
        """Initialize the fuzzy system with membership functions and rules"""
        self._define_membership_functions()
        self._define_fuzzy_rules()
        
    def _define_membership_functions(self):
        """Define linguistic variables and membership functions for all inputs"""
        
        # Temperature (°C): Low (<20), Medium (20-38), High (>38)
        self.temp_ranges = {
            'low': (0, 0, 15, 25),
            'medium': (15, 25, 38, 42),
            'high': (38, 42, 60, 60)
        }
        
        # Humidity (%): High (>60), Medium (30-60), Low (<30)
        self.humidity_ranges = {
            'low': (0, 0, 20, 35),
            'medium': (20, 35, 55, 65),
            'high': (55, 65, 100, 100)
        }
        
        # Wind Speed (km/h): Low (<15), Medium (15-30), High (>30)
        self.wind_ranges = {
            'low': (0, 0, 10, 18),
            'medium': (10, 18, 28, 35),
            'high': (28, 35, 60, 60)
        }
        
        # Rainfall (mm): Heavy (>5), Medium (1-5), Light (<1)
        self.rain_ranges = {
            'light': (0, 0, 0.5, 2),
            'medium': (0.5, 2, 5, 10),
            'heavy': (5, 10, 50, 50)
        }
        
        # Vegetation Dryness (FFMC-based): Wet (<70), Medium (70-85), Dry (>85)
        self.vegetation_ranges = {
            'wet': (0, 0, 60, 75),
            'medium': (60, 75, 85, 95),
            'dry': (85, 95, 101, 101)
        }
    
    def _trapezoidal_mf(self, x: float, params: Tuple[float, float, float, float]) -> float:
        """
        Trapezoidal membership function
        params: (a, b, c, d) where a <= b <= c <= d
        """
        a, b, c, d = params
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - c) if d != c else 1.0
    
    def _get_fuzzified_input(self, value: float, ranges: Dict[str, Tuple]) -> Dict[str, float]:
        """Fuzzify a crisp input value into linguistic terms"""
        fuzzified = {}
        for term, params in ranges.items():
            fuzzified[term] = self._trapezoidal_mf(value, params)
        return fuzzified
    
    def _define_fuzzy_rules(self):
        """
        Define comprehensive fuzzy rules covering all input combinations
        Each rule: IF conditions THEN output WITH reasoning
        """
        self.fuzzy_rules = [
            # RULES FOR NO FIRE
            {
                'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'low', 'rain': 'heavy', 'vegetation': 'wet'},
                'output': 'no_fire',
                'weight': 2.0,
                'reasoning': 'Low temperature, high humidity, low wind, heavy rain, and wet vegetation create conditions where fire cannot start or spread.'
            },
            {
                'conditions': {'temp': 'low', 'humidity': 'high', 'vegetation': 'wet'},
                'output': 'no_fire',
                'weight': 1.5,
                'reasoning': 'Low temperature combined with high humidity and wet vegetation prevents fire ignition.'
            },
            {
                'conditions': {'rain': 'heavy', 'vegetation': 'wet'},
                'output': 'no_fire',
                'weight': 1.8,
                'reasoning': 'Heavy rainfall saturates vegetation, making fire ignition virtually impossible.'
            },
            {
                'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'low', 'rain': 'medium'},
                'output': 'no_fire',
                'weight': 1.2,
                'reasoning': 'Low temperature, high humidity, low wind, and moderate rainfall create very low fire risk.'
            },
            {
                'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'medium', 'rain': 'medium'},
                'output': 'no_fire',
                'weight': 1.0,
                'reasoning': 'Low temperature and high humidity with moderate rainfall prevent fire despite medium wind.'
            },
            {
                'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'low', 'rain': 'heavy'},
                'output': 'no_fire',
                'weight': 1.3,
                'reasoning': 'Low temperature and heavy rainfall prevent fire ignition even with medium humidity.'
            },
            
            # RULES FOR LOW FIRE
            {
                'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'low', 'vegetation': 'medium'},
                'output': 'low_fire',
                'weight': 0.8,
                'reasoning': 'Low temperature and low wind limit fire spread despite moderate conditions.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'low', 'vegetation': 'medium'},
                'output': 'low_fire',
                'weight': 0.8,
                'reasoning': 'High humidity and low wind suppress fire spread even with medium temperature.'
            },
            {
                'conditions': {'temp': 'low', 'vegetation': 'dry', 'wind': 'low'},
                'output': 'low_fire',
                'weight': 0.7,
                'reasoning': 'Dry vegetation poses some risk, but low temperature and wind prevent significant fire spread.'
            },
            {
                'conditions': {'temp': 'medium', 'rain': 'medium', 'wind': 'low'},
                'output': 'low_fire',
                'weight': 0.8,
                'reasoning': 'Moderate rainfall reduces fire risk despite medium temperature.'
            },
            {
                'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'medium', 'vegetation': 'wet'},
                'output': 'low_fire',
                'weight': 0.7,
                'reasoning': 'Low temperature with high humidity creates low fire risk even with medium wind.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'medium', 'vegetation': 'wet'},
                'output': 'low_fire',
                'weight': 0.7,
                'reasoning': 'High humidity and wet vegetation create low fire risk despite medium temperature and wind.'
            },
            {
                'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'medium', 'rain': 'medium'},
                'output': 'low_fire',
                'weight': 0.7,
                'reasoning': 'Low temperature with moderate rainfall creates low fire risk.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'low', 'rain': 'medium'},
                'output': 'low_fire',
                'weight': 0.7,
                'reasoning': 'Moderate conditions with rainfall and low wind create low fire risk.'
            },
            
            # RULES FOR MEDIUM FIRE
            {
                'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'medium', 'vegetation': 'medium'},
                'output': 'medium_fire',
                'weight': 0.9,
                'reasoning': 'Medium temperature, humidity, and wind create conditions where fire can start and spread moderately.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'low', 'vegetation': 'medium'},
                'output': 'medium_fire',
                'weight': 0.85,
                'reasoning': 'High temperature increases risk, but medium humidity and low wind limit spread to moderate levels.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'medium', 'vegetation': 'dry'},
                'output': 'medium_fire',
                'weight': 0.9,
                'reasoning': 'Low humidity and dry vegetation increase fire risk, resulting in medium fire conditions.'
            },
            {
                'conditions': {'temp': 'high', 'rain': 'light', 'wind': 'low'},
                'output': 'medium_fire',
                'weight': 0.8,
                'reasoning': 'High temperature is partially mitigated by light rain and low wind, resulting in medium risk.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'low', 'vegetation': 'medium'},
                'output': 'medium_fire',
                'weight': 0.8,
                'reasoning': 'Low humidity with medium temperature creates moderate fire risk.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'low', 'vegetation': 'dry'},
                'output': 'medium_fire',
                'weight': 0.85,
                'reasoning': 'Medium temperature and humidity with dry vegetation creates moderate fire risk.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'medium', 'rain': 'light'},
                'output': 'medium_fire',
                'weight': 0.75,
                'reasoning': 'Moderate conditions with light rainfall create medium fire risk.'
            },
            {
                'conditions': {'temp': 'low', 'humidity': 'low', 'wind': 'medium', 'vegetation': 'dry'},
                'output': 'medium_fire',
                'weight': 0.7,
                'reasoning': 'Low humidity and dry vegetation with medium wind create medium fire risk despite low temperature.'
            },
            
            # RULES FOR HIGH FIRE
            {
                'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'medium', 'vegetation': 'dry'},
                'output': 'high_fire',
                'weight': 0.9,
                'reasoning': 'High temperature, low humidity, medium wind, and dry vegetation create high fire danger with significant spread potential.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'vegetation': 'medium'},
                'output': 'high_fire',
                'weight': 0.9,
                'reasoning': 'High temperature and low humidity combined with high wind create high fire risk.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'high', 'vegetation': 'dry'},
                'output': 'high_fire',
                'weight': 0.9,
                'reasoning': 'Low humidity, high wind, and dry vegetation create high fire conditions even with medium temperature.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'high', 'vegetation': 'dry'},
                'output': 'high_fire',
                'weight': 0.9,
                'reasoning': 'High temperature and wind with dry vegetation create high fire risk despite medium humidity.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'medium', 'rain': 'light'},
                'output': 'high_fire',
                'weight': 0.8,
                'reasoning': 'High temperature and low humidity dominate over light rain, creating high fire risk.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'medium', 'vegetation': 'dry'},
                'output': 'high_fire',
                'weight': 0.8,
                'reasoning': 'High temperature with dry vegetation and moderate conditions creates high fire risk.'
            },
            {
                'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'high', 'vegetation': 'medium'},
                'output': 'high_fire',
                'weight': 0.8,
                'reasoning': 'Low humidity and high wind create high fire risk even with medium temperature and vegetation.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'low', 'vegetation': 'dry'},
                'output': 'high_fire',
                'weight': 0.85,
                'reasoning': 'High temperature with very low humidity and dry vegetation creates high fire risk even with low wind.'
            },
            
            # RULES FOR EXTREME FIRE
            {
                'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'vegetation': 'dry'},
                'output': 'extreme_fire',
                'weight': 2.0,
                'reasoning': 'CRITICAL: High temperature, very low humidity, high wind, and extremely dry vegetation create perfect conditions for extreme wildfire with rapid spread.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
                'output': 'extreme_fire',
                'weight': 1.8,
                'reasoning': 'Extreme conditions dominate over light rain; high temp, low humidity, high wind, and dry vegetation create extreme fire danger.'
            },
            {
                'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'vegetation': 'dry'},
                'output': 'extreme_fire',
                'weight': 2.0,
                'reasoning': 'All critical factors at extreme levels: maximum fire danger with explosive spread potential.'
            }
        ]
    
    def _calculate_rule_firing(self, fuzzified_inputs: Dict[str, Dict[str, float]], 
                                rule: Dict) -> Tuple[float, str]:
        """
        Calculate firing strength of a fuzzy rule using weighted average
        Returns: (firing_strength, output_term)
        """
        strengths = []
        
        for input_var, linguistic_term in rule['conditions'].items():
            if input_var in fuzzified_inputs:
                membership = fuzzified_inputs[input_var].get(linguistic_term, 0.0)
                strengths.append(membership)
        
        if not strengths:
            return 0.0, rule['output']
        
        # Use average for partial matching
        firing_strength = sum(strengths) / len(strengths)
        
        # Apply rule weight
        firing_strength = firing_strength * rule.get('weight', 1.0)
        
        return firing_strength, rule['output']
    
    def predict(self, temperature: float, humidity: float, wind_speed: float, 
                rainfall: float, vegetation_dryness: float) -> Dict[str, Any]:
        """
        Make fuzzy inference prediction using weighted scoring
        
        Args:
            temperature: Temperature in °C
            humidity: Relative humidity in %
            wind_speed: Wind speed in km/h
            rainfall: Rainfall in mm
            vegetation_dryness: Vegetation dryness index (0-101, similar to FFMC)
            
        Returns:
            Dictionary with prediction results and reasoning
        """
        # Fuzzify inputs
        fuzzified_inputs = {
            'temp': self._get_fuzzified_input(temperature, self.temp_ranges),
            'humidity': self._get_fuzzified_input(humidity, self.humidity_ranges),
            'wind': self._get_fuzzified_input(wind_speed, self.wind_ranges),
            'rain': self._get_fuzzified_input(rainfall, self.rain_ranges),
            'vegetation': self._get_fuzzified_input(vegetation_dryness, self.vegetation_ranges)
        }
        
        # Calculate scores for each output category
        output_scores = {
            'no_fire': 0.0,
            'low_fire': 0.0,
            'medium_fire': 0.0,
            'high_fire': 0.0,
            'extreme_fire': 0.0
        }
        
        fired_rules = []
        
        for rule in self.fuzzy_rules:
            firing_strength, output_term = self._calculate_rule_firing(fuzzified_inputs, rule)
            
            if firing_strength > 0:
                # Aggregate using weighted sum
                output_scores[output_term] += firing_strength
                fired_rules.append({
                    'rule': rule,
                    'firing_strength': firing_strength,
                    'reasoning': rule['reasoning']
                })
        
        # Store raw scores for comparison
        raw_scores = output_scores.copy()
        
        # Normalize scores to 0-1 range for final output
        max_score = max(output_scores.values()) if max(output_scores.values()) > 0 else 1.0
        for key in output_scores:
            output_scores[key] /= max_score
        
        # Determine final output with priority to more severe categories
        # Priority order: extreme_fire > high_fire > medium_fire > low_fire > no_fire
        priority_order = ['extreme_fire', 'high_fire', 'medium_fire', 'low_fire', 'no_fire']
        
        # Find the highest scoring category using raw scores, with priority to more severe ones in case of ties
        best_output = 'no_fire'
        best_score = 0.0
        
        for category in priority_order:
            if raw_scores[category] > best_score:
                best_score = raw_scores[category]
                best_output = category
            elif raw_scores[category] == best_score and priority_order.index(category) < priority_order.index(best_output):
                # Tie: choose the more severe category
                best_output = category
        
        final_output = best_output
        crisp_output = output_scores[final_output]
        
        # Map to linguistic output
        output_mapping = {
            'no_fire': 'No Fire',
            'low_fire': 'Low Fire',
            'medium_fire': 'Medium Fire',
            'high_fire': 'High Fire',
            'extreme_fire': 'Extreme Fire'
        }
        
        linguistic_output = output_mapping[final_output]
        
        # Get the most relevant reasoning (from highest firing rule)
        if fired_rules:
            top_rule = max(fired_rules, key=lambda x: x['firing_strength'])
            reasoning = top_rule['reasoning']
            rule_firing = top_rule['firing_strength']
        else:
            reasoning = "No fuzzy rules fired significantly. Using default low risk assessment."
            rule_firing = 0.0
        
        return {
            'risk_score': crisp_output,
            'linguistic_risk_level': linguistic_output,
            'fuzzy_output': crisp_output,
            'fuzzified_inputs': fuzzified_inputs,
            'fired_rules': fired_rules,
            'rule_firing_strength': rule_firing,
            'reasoning': reasoning,
            'output_scores': output_scores
        }
    
    def get_input_linguistic(self, value: float, input_type: str) -> str:
        """Get the linguistic term for a crisp input value"""
        if input_type == 'temperature':
            ranges = self.temp_ranges
        elif input_type == 'humidity':
            ranges = self.humidity_ranges
        elif input_type == 'wind':
            ranges = self.wind_ranges
        elif input_type == 'rainfall':
            ranges = self.rain_ranges
        elif input_type == 'vegetation':
            ranges = self.vegetation_ranges
        else:
            return 'unknown'
        
        fuzzified = self._get_fuzzified_input(value, ranges)
        # Return term with highest membership
        return max(fuzzified, key=fuzzified.get)


def create_fuzzy_wildfire_system():
    """Factory function to create fuzzy wildfire system"""
    return FuzzyWildfireSystem()
