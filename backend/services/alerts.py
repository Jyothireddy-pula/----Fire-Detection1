"""
Alert System Module
Handles alert generation and management with explainability
"""

import time
from typing import Dict, List, Optional
from datetime import datetime
from backend.services.database import HistoryDatabase
from backend.services.decision import DecisionEngine
from backend.services.explainability import ExplainabilityEngine
from backend.utils.logger import system_logger


class AlertSystem:
    """Alert system for wildfire risk notifications"""

    def __init__(self, database: HistoryDatabase, decision_engine: DecisionEngine,
                 pipeline=None):
        """
        Initialize alert system

        Args:
            database: History database instance
            decision_engine: Decision engine instance
            pipeline: Optional DataPipeline for explainability
        """
        self.database = database
        self.decision_engine = decision_engine
        self.pipeline = pipeline
        self.alert_threshold = 0.75
        self.active_alerts = {}
        self._explainability = None
    
    def check_and_generate_alert(self, prediction_result: Dict, location: str = 'Unknown') -> Dict:
        """
        Check if alert should be generated based on prediction
        
        Args:
            prediction_result: Prediction result from pipeline
            location: Location name
            
        Returns:
            Alert dictionary or None if no alert
        """
        risk_score = prediction_result['risk_score']
        
        if risk_score >= self.alert_threshold:
            # Generate alert
            alert = self._create_alert(prediction_result, location)
            
            # Save to database
            self.database.save_alert(alert)
            
            # Track active alert
            alert_key = f"{location}_{datetime.now().strftime('%Y%m%d')}"
            self.active_alerts[alert_key] = alert
            
            system_logger.warning(f"Alert generated for {location}: Risk {risk_score:.2f}")
            
            return alert
        
        return None
    
    def _create_alert(self, prediction_result: Dict, location: str) -> Dict:
        """
        Create alert dictionary with explainability.

        Args:
            prediction_result: Prediction result
            location: Location name

        Returns:
            Alert dictionary
        """
        risk_score = prediction_result['risk_score']
        risk_level, _ = self.decision_engine.map_risk_level(risk_score)

        # Determine severity
        if risk_score >= 0.9:
            severity = 'critical'
            message = f'CRITICAL: Extreme wildfire risk detected in {location}. Immediate action required.'
        elif risk_score >= 0.8:
            severity = 'high'
            message = f'HIGH: Severe wildfire risk detected in {location}. Emergency preparation recommended.'
        else:
            severity = 'medium'
            message = f'WARNING: Elevated wildfire risk detected in {location}. Monitor closely.'

        # Build explainability data
        weather_data = prediction_result.get('fwi_components', {})
        fwi_components = prediction_result.get('fwi_components', {})

        explanation = None
        if self.pipeline:
            try:
                explainability = ExplainabilityEngine(
                    self.pipeline.fuzzy_wildfire_system,
                    self.pipeline.anfis_model
                )
                explanation = explainability.explain_prediction(
                    prediction_result, weather_data, fwi_components
                )
                why_high_risk = explainability.explain_why_high_risk(
                    prediction_result, explanation
                )
            except Exception:
                explanation = None
                why_high_risk = []
        else:
            why_high_risk = []

        alert = {
            'location': location,
            'alert_type': 'wildfire_risk',
            'severity': severity,
            'message': message,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'fwi_components': fwi_components,
            'explanation': explanation,
            'why_high_risk': why_high_risk
        }

        return alert
    
    def get_active_alerts(self) -> List[Dict]:
        """
        Get all active alerts
        
        Returns:
            List of active alerts
        """
        return list(self.active_alerts.values())
    
    def clear_alert(self, location: str):
        """
        Clear alert for a location
        
        Args:
            location: Location name
        """
        alert_key = f"{location}_{datetime.now().strftime('%Y%m%d')}"
        if alert_key in self.active_alerts:
            del self.active_alerts[alert_key]
            system_logger.info(f"Alert cleared for {location}")
    
    def get_alert_summary(self) -> Dict:
        """
        Get summary of alerts
        
        Returns:
            Alert summary dictionary
        """
        active_alerts = self.get_active_alerts()
        
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for alert in active_alerts:
            severity = alert.get('severity', 'low')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        return {
            'total_active_alerts': len(active_alerts),
            'severity_distribution': severity_counts,
            'highest_severity': max(severity_counts.keys()) if severity_counts else 'none'
        }
    
    def set_alert_threshold(self, threshold: float):
        """
        Set alert threshold
        
        Args:
            threshold: Risk score threshold for alerts (0-1)
        """
        self.alert_threshold = max(0.0, min(1.0, threshold))
        system_logger.info(f"Alert threshold set to {self.alert_threshold}")
