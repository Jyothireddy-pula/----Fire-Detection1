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
        """Define the fuzzy rule base for wildfire prediction."""
        self.fuzzy_rules = [
            # Extreme fire conditions
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'extreme_fire', 'weight': 1.0,
             'reasoning': 'High temp, low humidity, high wind, no rain, and dry vegetation create extreme fire conditions'},
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'medium', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'extreme_fire', 'weight': 0.95,
             'reasoning': 'High temp with low humidity and dry vegetation lead to extreme fire risk even with moderate wind'},
            {'conditions': {'temp': 'high', 'humidity': 'low', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'extreme_fire', 'weight': 0.9,
             'reasoning': 'Combined high temperature and dry conditions create extreme fire danger'},
            {'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'high_fire', 'weight': 0.85,
             'reasoning': 'High temperature with strong winds significantly increase fire spread potential'},
            # High fire conditions
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'low', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'high_fire', 'weight': 0.8,
             'reasoning': 'High temp and low humidity with dry vegetation create high fire risk'},
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'high_fire', 'weight': 0.8,
             'reasoning': 'Low humidity and high wind with dry fuels create dangerous fire conditions'},
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'high_fire', 'weight': 0.85,
             'reasoning': 'Hot, dry, and windy conditions together produce high fire risk'},
            {'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.7,
             'reasoning': 'Moderate conditions with wind and dry vegetation elevate fire spread risk'},
            # Medium fire conditions
            {'conditions': {'temp': 'high', 'humidity': 'medium', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'medium_fire', 'weight': 0.65,
             'reasoning': 'Warm temperature with moderate humidity on semi-dry vegetation creates medium fire risk'},
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'medium', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'medium_fire', 'weight': 0.7,
             'reasoning': 'Low humidity and moderate wind on semi-dry fuels increase fire potential'},
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'low', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.7,
             'reasoning': 'Dry vegetation and low humidity on moderate temperature create medium fire conditions'},
            {'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'low', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.7,
             'reasoning': 'Hot and dry conditions with low wind create moderate fire risk'},
            {'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'high', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'low_fire', 'weight': 0.5,
             'reasoning': 'High humidity offsets some wind and temperature effects, reducing fire risk'},
            # Low fire conditions
            {'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'medium', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'low_fire', 'weight': 0.5,
             'reasoning': 'Balanced conditions with moderate humidity and wind keep fire risk low'},
            {'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'low', 'rain': 'light', 'vegetation': 'wet'},
             'output': 'low_fire', 'weight': 0.45,
             'reasoning': 'Wet vegetation and moderate humidity maintain low fire risk'},
            {'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'medium', 'rain': 'medium', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.3,
             'reasoning': 'Wet conditions and low temperature mean no fire risk'},
            {'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'low', 'rain': 'medium', 'vegetation': 'medium'},
             'output': 'no_fire', 'weight': 0.35,
             'reasoning': 'Cool, moist conditions with no extreme wind keep fire risk minimal'},
            # Rain-related rules
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'rain': 'medium', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.6,
             'reasoning': 'Recent rain partially offsets high temp and wind, reducing fire risk'},
            {'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'high', 'rain': 'heavy', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.2,
             'reasoning': 'Heavy rain on wet vegetation eliminates fire risk despite high temperature'},
            {'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'high', 'rain': 'heavy', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.2,
             'reasoning': 'Heavy rainfall completely suppresses any fire risk'},
            # Wind-driven fire spread
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'high_fire', 'weight': 0.8,
             'reasoning': 'Strong wind dramatically increases fire spread even with moderate temp'},
            {'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'high', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'medium_fire', 'weight': 0.6,
             'reasoning': 'High wind can spread fire even with lower temperature if fuels are dry'},
            # Default / fallback rules
            {'conditions': {'temp': 'high', 'humidity': 'high', 'wind': 'low', 'rain': 'light', 'vegetation': 'wet'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'High humidity and wet vegetation keep fire risk low despite high temperature'},
            {'conditions': {'temp': 'low', 'humidity': 'low', 'wind': 'low', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'low_fire', 'weight': 0.5,
             'reasoning': 'Low temperature limits fire risk even with dry conditions'},
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'low', 'rain': 'light', 'vegetation': 'wet'},
             'output': 'low_fire', 'weight': 0.45,
             'reasoning': 'Wet vegetation offsets low humidity, keeping fire risk manageable'},
            # Dry vegetation dominant rules
            {'conditions': {'temp': 'low', 'humidity': 'low', 'wind': 'medium', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.65,
             'reasoning': 'Dry vegetation is the dominant factor; moderate temp still creates fire risk'},
            {'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'medium', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'low_fire', 'weight': 0.5,
             'reasoning': 'Cool temperature limits fire despite dry vegetation and wind'},
            {'conditions': {'temp': 'high', 'humidity': 'high', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.65,
             'reasoning': 'Dry vegetation and wind offset some humidity effect, creating medium risk'},
            # No rain dominates
            {'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'low', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'low_fire', 'weight': 0.5,
             'reasoning': 'Dry vegetation with no rain creates some fire risk but moderate conditions limit it'},
            {'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'low', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'no_fire', 'weight': 0.35,
             'reasoning': 'Cool, moist conditions prevent fire despite lack of rain'},
            # Extreme combinations
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'high', 'rain': 'heavy', 'vegetation': 'wet'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'Heavy rain and wet vegetation suppress fire even in extreme heat and wind'},
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'low', 'rain': 'heavy', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.25,
             'reasoning': 'Heavy rainfall completely negates fire risk from high temperature'},
            # Additional coverage rules
            {'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'low', 'rain': 'medium', 'vegetation': 'medium'},
             'output': 'no_fire', 'weight': 0.3,
             'reasoning': 'Moist conditions with rain eliminate fire risk'},
            {'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'high', 'rain': 'medium', 'vegetation': 'medium'},
             'output': 'no_fire', 'weight': 0.25,
             'reasoning': 'High humidity and rain suppress fire despite wind'},
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'low', 'rain': 'heavy', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.2,
             'reasoning': 'Heavy rain and wet vegetation eliminate fire risk'},
            # Temperature-only dominant
            {'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'low', 'rain': 'medium', 'vegetation': 'wet'},
             'output': 'low_fire', 'weight': 0.35,
             'reasoning': 'Rain and wet vegetation dominate, limiting fire risk despite high temp'},
            {'conditions': {'temp': 'high', 'humidity': 'high', 'wind': 'medium', 'rain': 'medium', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.2,
             'reasoning': 'Rain and humidity together eliminate fire risk'},
            # Remaining combinations
            {'conditions': {'temp': 'low', 'humidity': 'low', 'wind': 'low', 'rain': 'medium', 'vegetation': 'dry'},
             'output': 'low_fire', 'weight': 0.45,
             'reasoning': 'Dry vegetation keeps some fire risk even with rain and low temp'},
            {'conditions': {'temp': 'low', 'humidity': 'low', 'wind': 'high', 'rain': 'medium', 'vegetation': 'wet'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'Wet vegetation limits fire risk despite wind and low humidity'},
            {'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'medium', 'rain': 'medium', 'vegetation': 'medium'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'Rain and medium humidity balance high temperature, limiting fire risk'},
            {'conditions': {'temp': 'medium', 'humidity': 'medium', 'wind': 'high', 'rain': 'heavy', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.15,
             'reasoning': 'Heavy rain and wet fuels eliminate fire risk even with strong wind'},
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'medium', 'rain': 'medium', 'vegetation': 'medium'},
             'output': 'medium_fire', 'weight': 0.6,
             'reasoning': 'Low humidity and high temp keep fire risk elevated despite some rain'},
            {'conditions': {'temp': 'high', 'humidity': 'low', 'wind': 'low', 'rain': 'medium', 'vegetation': 'dry'},
             'output': 'high_fire', 'weight': 0.75,
             'reasoning': 'Low humidity and dry vegetation dominate, creating high fire risk'},
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'low', 'rain': 'medium', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.6,
             'reasoning': 'Dry vegetation and low humidity create medium fire risk'},
            {'conditions': {'temp': 'high', 'humidity': 'high', 'wind': 'high', 'rain': 'medium', 'vegetation': 'medium'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'High humidity and rain offset wind and temperature effects'},
            {'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'medium', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'low_fire', 'weight': 0.5,
             'reasoning': 'High humidity limits fire despite dry vegetation and wind'},
            {'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'high', 'rain': 'light', 'vegetation': 'dry'},
             'output': 'medium_fire', 'weight': 0.6,
             'reasoning': 'Dry vegetation and wind can overcome cool temperature'},
            {'conditions': {'temp': 'low', 'humidity': 'low', 'wind': 'low', 'rain': 'heavy', 'vegetation': 'medium'},
             'output': 'no_fire', 'weight': 0.3,
             'reasoning': 'Heavy rain eliminates fire risk despite dry vegetation'},
            {'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'high', 'rain': 'light', 'vegetation': 'wet'},
             'output': 'low_fire', 'weight': 0.35,
             'reasoning': 'Wet vegetation and high humidity keep fire risk low even with wind'},
            {'conditions': {'temp': 'high', 'humidity': 'medium', 'wind': 'high', 'rain': 'heavy', 'vegetation': 'medium'},
             'output': 'low_fire', 'weight': 0.35,
             'reasoning': 'Heavy rain reduces fire risk significantly'},
            {'conditions': {'temp': 'medium', 'humidity': 'low', 'wind': 'high', 'rain': 'heavy', 'vegetation': 'medium'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'Heavy rain offsets low humidity and wind effects'},
            {'conditions': {'temp': 'high', 'humidity': 'high', 'wind': 'low', 'rain': 'light', 'vegetation': 'medium'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'High humidity and moisture in vegetation limit fire risk'},
            {'conditions': {'temp': 'low', 'humidity': 'medium', 'wind': 'low', 'rain': 'light', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.3,
             'reasoning': 'Wet vegetation and cool temperature eliminate fire risk'},
            {'conditions': {'temp': 'low', 'humidity': 'high', 'wind': 'high', 'rain': 'light', 'vegetation': 'wet'},
             'output': 'no_fire', 'weight': 0.2,
             'reasoning': 'Wet vegetation and high humidity eliminate fire risk despite wind'},
            {'conditions': {'temp': 'medium', 'humidity': 'high', 'wind': 'high', 'rain': 'medium', 'vegetation': 'dry'},
             'output': 'low_fire', 'weight': 0.4,
             'reasoning': 'Dry vegetation keeps some fire risk despite rain and humidity'},
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
        
        # Use average for partial matching, WITHOUT applying the severity weight
        # This ensures match confidence is unbiased
        firing_strength = sum(strengths) / len(strengths)
        
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
                # Aggregate using Max (standard fuzzy composition) instead of sum to prevent over-firing
                output_scores[output_term] = max(output_scores[output_term], firing_strength)
                fired_rules.append({
                    'rule': rule,
                    'firing_strength': firing_strength,
                    'reasoning': rule['reasoning']
                })
        
        # Store raw scores for comparison
        raw_scores = output_scores.copy()
        
        # Find the highest scoring category using raw scores, with priority to more severe ones in case of ties
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
            # Sort by firing strength first, then rule weight
            top_rule = max(fired_rules, key=lambda x: (x['firing_strength'], x['rule'].get('weight', 0.0)))
            reasoning = top_rule['reasoning']
            rule_firing = top_rule['firing_strength']
            crisp_output = rule_firing * top_rule['rule'].get('weight', 1.0) # True risk score
        else:
            reasoning = "No fuzzy rules fired significantly. Using default low risk assessment."
            rule_firing = 0.0
            crisp_output = 0.0
        
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
