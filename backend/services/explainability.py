"""
SHAP Explainability Module
Provides model explainability using SHAP values
"""

import numpy as np
import shap
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.services.pipeline import DataPipeline
from backend.utils.logger import system_logger


class ExplainabilityEngine:
    """SHAP-based explainability for predictions"""
    
    def __init__(self, pipeline: DataPipeline):
        """
        Initialize explainability engine
        
        Args:
            pipeline: Data pipeline instance
        """
        self.pipeline = pipeline
        self.explainer = None
        self.background_data = None
    
    def initialize_explainer(self, background_samples: int = 100):
        """
        Initialize SHAP explainer with background data
        
        Args:
            background_samples: Number of background samples
        """
        try:
            # Load training data for background
            X_train = np.load('models/X_train_bal.npy')
            
            # Use subset as background
            if len(X_train) > background_samples:
                indices = np.random.choice(len(X_train), background_samples, replace=False)
                self.background_data = X_train[indices]
            else:
                self.background_data = X_train
            
            # Create explainer using model wrapper
            self.explainer = shap.KernelExplainer(
                self._model_predict,
                self.background_data
            )
            
            system_logger.info("SHAP explainer initialized")
            
        except Exception as e:
            system_logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            self.explainer = None
    
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Wrapper function for SHAP explainer
        
        Args:
            X: Input features
            
        Returns:
            Model predictions
        """
        return self.pipeline.anfis_model.predict(X).flatten()
    
    def explain_prediction(self, features: np.ndarray) -> Dict:
        """
        Explain a single prediction using SHAP
        
        Args:
            features: Input features (1, n_features)
            
        Returns:
            SHAP explanation dictionary
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        if self.explainer is None:
            return {'error': 'SHAP explainer not available'}
        
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Get feature names
            feature_names = self.pipeline.preprocessor.feature_names
            
            # Create explanation
            explanation = {
                'shap_values': shap_values[0].tolist() if isinstance(shap_values, list) else shap_values.tolist(),
                'feature_names': feature_names,
                'base_value': float(self.explainer.expected_value[0]) if isinstance(self.explainer.expected_value, list) else float(self.explainer.expected_value),
                'feature_importance': {}
            }
            
            # Calculate feature importance (absolute SHAP values)
            abs_shap = np.abs(shap_values[0]) if isinstance(shap_values, list) else np.abs(shap_values)
            for i, name in enumerate(feature_names):
                explanation['feature_importance'][name] = float(abs_shap[i])
            
            # Sort by importance
            explanation['feature_importance'] = dict(
                sorted(explanation['feature_importance'].items(), 
                      key=lambda x: x[1], reverse=True)
            )
            
            return explanation
            
        except Exception as e:
            system_logger.error(f"SHAP explanation failed: {str(e)}")
            return {'error': str(e)}
    
    def get_feature_contribution_summary(self, explanation: Dict) -> List[Dict]:
        """
        Get feature contribution summary from explanation
        
        Args:
            explanation: SHAP explanation dictionary
            
        Returns:
            List of feature contributions
        """
        if 'error' in explanation:
            return []
        
        contributions = []
        for feature, importance in explanation['feature_importance'].items():
            contributions.append({
                'feature': feature,
                'importance': importance,
                'direction': 'positive' if explanation['shap_values'][list(explanation['feature_names']).index(feature)] > 0 else 'negative'
            })
        
        return contributions
    
    def get_global_feature_importance(self, X_test: np.ndarray, n_samples: int = 50) -> Dict:
        """
        Get global feature importance across multiple samples
        
        Args:
            X_test: Test dataset
            n_samples: Number of samples to analyze
            
        Returns:
            Global feature importance dictionary
        """
        if self.explainer is None:
            self.initialize_explainer()
        
        if self.explainer is None:
            return {'error': 'SHAP explainer not available'}
        
        try:
            # Sample from test data
            if len(X_test) > n_samples:
                indices = np.random.choice(len(X_test), n_samples, replace=False)
                X_sample = X_test[indices]
            else:
                X_sample = X_test
            
            # Calculate SHAP values for all samples
            shap_values = self.explainer.shap_values(X_sample)
            
            # Average absolute SHAP values
            if isinstance(shap_values, list):
                mean_shap = np.mean(np.abs(shap_values[0]), axis=0)
            else:
                mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Create importance dictionary
            feature_names = self.pipeline.preprocessor.feature_names
            importance_dict = {}
            for i, name in enumerate(feature_names):
                importance_dict[name] = float(mean_shap[i])
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            system_logger.error(f"Global feature importance failed: {str(e)}")
            return {'error': str(e)}
