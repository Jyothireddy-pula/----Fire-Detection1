"""
Decision Engine Module
Handles risk mapping, confidence scoring, and decision making
"""

import numpy as np
from typing import Dict, Tuple
from backend.utils.logger import system_logger


class DecisionEngine:
    """Decision engine for wildfire risk assessment"""
    
    def __init__(self):
        """Initialize decision engine"""
        self.risk_thresholds = {
            'no_risk': 0.25,
            'low': 0.5,
            'moderate': 0.7,
            'high': 0.85,
            'extreme': 1.0
        }
    
    def map_risk_level(self, risk_score: float) -> Tuple[str, str]:
        """
        Map risk score to linguistic fire risk level
        
        Args:
            risk_score: Risk score between 0 and 1
            
        Returns:
            Tuple of (linguistic_level, description)
        """
        if risk_score < 0.2:
            return 'No Fire', 'No Fire - Fire conditions not present'
        elif risk_score < 0.4:
            return 'Low Fire', 'Low Fire - Minimal fire danger'
        elif risk_score < 0.6:
            return 'Medium Fire', 'Medium Fire - Fire conditions possible'
        elif risk_score < 0.8:
            return 'High Fire', 'High Fire - Fire danger elevated'
        else:
            return 'Extreme Fire', 'Extreme Fire - Severe fire danger'
    
    def calculate_confidence(self, prediction_result: Dict, n_perturbations: int = 10) -> float:
        """
        Calculate confidence score using perturbation analysis
        
        Args:
            prediction_result: Prediction result dictionary
            n_perturbations: Number of perturbations for analysis
            
        Returns:
            Confidence score between 0 and 1
        """
        base_risk = prediction_result['risk_score']
        
        # Confidence based on agreement between ANFIS and Fuzzy outputs
        try:
            anfis_output = prediction_result.get('anfis_output', base_risk)
            fuzzy_output = prediction_result.get('fuzzy_output', base_risk)
            agreement = 1.0 - abs(float(anfis_output) - float(fuzzy_output))
        except (TypeError, ValueError):
            agreement = 0.7  # Default moderate confidence
        
        return float(np.clip(agreement, 0.0, 1.0))
    
    def calculate_early_warning_score(self, risk_score: float, confidence: float, 
                                     trend: float = 0.0) -> float:
        """
        Calculate Early Warning Score (EWS)
        
        Args:
            risk_score: Current risk score
            confidence: Prediction confidence
            trend: Risk trend (positive = increasing, negative = decreasing)
            
        Returns:
            Early warning score between 0 and 1
        """
        # Base score is risk score weighted by confidence
        base_ews = risk_score * confidence
        
        # Adjust for trend
        trend_factor = max(0.0, trend)  # Only consider increasing trend
        trend_boost = trend_factor * 0.2
        
        # Calculate final EWS
        ews = base_ews + trend_boost
        
        return float(np.clip(ews, 0.0, 1.0))
    
    def make_decision(self, prediction_result: Dict, trend: float = 0.0) -> Dict:
        """
        Make final decision based on prediction with linguistic output
        
        Args:
            prediction_result: Prediction result from pipeline
            trend: Risk trend
            
        Returns:
            Decision dictionary with linguistic outputs
        """
        risk_score = prediction_result['risk_score']
        
        # Use linguistic level from prediction result if available, otherwise map from score
        if 'linguistic_risk_level' in prediction_result:
            linguistic_level = prediction_result['linguistic_risk_level']
            descriptions = {
                'No Fire': 'No Fire - Fire conditions not present',
                'Low Fire': 'Low Fire - Minimal fire danger',
                'Medium Fire': 'Medium Fire - Fire conditions possible',
                'High Fire': 'High Fire - Fire danger elevated',
                'Extreme Fire': 'Extreme Fire - Severe fire danger'
            }
            description = descriptions.get(linguistic_level, linguistic_level)
        else:
            linguistic_level, description = self.map_risk_level(risk_score)
        
        confidence = self.calculate_confidence(prediction_result)
        ews = self.calculate_early_warning_score(risk_score, confidence, trend)
        
        # Determine action based on linguistic level
        if linguistic_level in ['High Fire', 'Extreme Fire']:
            action = 'immediate_alert'
            action_message = f'IMMEDIATE ACTION REQUIRED: {linguistic_level} detected'
        elif linguistic_level == 'Medium Fire':
            action = 'prepare'
            action_message = f'PREPARE: {linguistic_level} conditions present'
        elif linguistic_level == 'Low Fire':
            action = 'monitor'
            action_message = f'MONITOR: {linguistic_level} conditions'
        else:
            action = 'normal'
            action_message = f'NORMAL: {linguistic_level} conditions'
        
        decision = {
            'risk_score': risk_score,
            'linguistic_risk_level': linguistic_level,
            'risk_description': description,
            'confidence': confidence,
            'early_warning_score': ews,
            'action': action,
            'action_message': action_message,
            'trend': trend
        }
        
        system_logger.log_prediction(
            inputs=prediction_result.get('fwi_components', {}),
            output={'linguistic_risk_level': linguistic_level, 'action': action},
            confidence=confidence
        )
        
        return decision
    
    def get_feature_contribution(self, prediction_result: Dict) -> Dict:
        """
        Analyze feature contribution to risk
        
        Args:
            prediction_result: Prediction result
            
        Returns:
            Feature contribution breakdown
        """
        try:
            fwi = prediction_result.get('fwi_components', {})
            if not fwi:
                return {'Temperature': 0.25, 'Humidity': 0.20, 'Wind': 0.20, 'Rain': 0.15, 'FWI': 0.20}
            
            contributions = {
                'FFMC': (fwi.get('FFMC', 0) / 101.0) * 0.3,
                'DMC': (fwi.get('DMC', 0) / max(fwi.get('DMC', 1), 1)) * 0.25,
                'DC': (fwi.get('DC', 0) / max(fwi.get('DC', 1), 1)) * 0.2,
                'ISI': (fwi.get('ISI', 0) / max(fwi.get('ISI', 1), 1)) * 0.15,
                'BUI': (fwi.get('BUI', 0) / max(fwi.get('BUI', 1), 1)) * 0.1
            }
            contrib_sum = sum(contributions.values())
            if contrib_sum > 0:
                contributions = {k: v / contrib_sum for k, v in contributions.items()}
            
            return contributions
        except Exception as e:
            system_logger.warning(f"Feature contribution calculation failed: {str(e)}")
            return {'Temperature': 0.25, 'Humidity': 0.20, 'Wind': 0.20, 'Rain': 0.15, 'FWI': 0.20}
