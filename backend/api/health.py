"""
Health Check Endpoint
Provides system health monitoring and API failure handling
"""

import time
from typing import Dict
from backend.api.weather import WeatherAPI
from backend.services.pipeline import DataPipeline
from backend.services.database import HistoryDatabase
from backend.utils.logger import system_logger


class HealthCheckService:
    """Health check and monitoring service"""
    
    def __init__(self, weather_api: WeatherAPI = None, pipeline: DataPipeline = None):
        """
        Initialize health check service
        
        Args:
            weather_api: Weather API instance
            pipeline: Data pipeline instance
        """
        self.weather_api = weather_api
        self.pipeline = pipeline
        self.database = HistoryDatabase()
    
    def check_health(self) -> Dict:
        """
        Perform comprehensive health check
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'status': 'OK',
            'timestamp': time.time(),
            'checks': {}
        }
        
        # Check API status
        if self.weather_api:
            api_status = self.weather_api.get_api_status()
            health_status['checks']['api'] = api_status
            if api_status['status'] == 'failure':
                health_status['status'] = 'DEGRADED'
        else:
            health_status['checks']['api'] = {'status': 'not_configured'}
        
        # Check model status
        if self.pipeline and self.pipeline.anfis_model:
            model_status = {
                'status': 'loaded',
                'is_trained': self.pipeline.anfis_model.is_trained
            }
            health_status['checks']['model'] = model_status
        else:
            health_status['checks']['model'] = {'status': 'not_loaded'}
            health_status['status'] = 'DEGRADED'
        
        # Check database status
        try:
            db_stats = self.database.get_statistics()
            health_status['checks']['database'] = {
                'status': 'connected',
                'total_predictions': db_stats['total_predictions']
            }
        except Exception as e:
            health_status['checks']['database'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['status'] = 'DEGRADED'
        
        return health_status
    
    def handle_api_failure(self, error: Exception) -> Dict:
        """
        Handle API failure gracefully
        
        Args:
            error: The exception that occurred
            
        Returns:
            Fallback response
        """
        system_logger.error(f"API failure: {str(error)}")
        
        # Increment failure count
        if self.weather_api:
            self.weather_api.api_failure_count += 1
        
        # Return fallback data
        return {
            'status': 'error',
            'message': 'API request failed',
            'using_cached_data': self.weather_api.last_valid_cache is not None if self.weather_api else False,
            'fallback_available': True
        }
    
    def get_system_metrics(self) -> Dict:
        """
        Get system performance metrics
        
        Returns:
            System metrics dictionary
        """
        metrics = {
            'timestamp': time.time(),
            'cache_stats': {},
            'database_stats': {}
        }
        
        # Get cache statistics
        from backend.utils.cache import weather_cache, model_cache
        metrics['cache_stats']['weather'] = weather_cache.get_stats()
        metrics['cache_stats']['model'] = model_cache.get_stats()
        
        # Get database statistics
        metrics['database_stats'] = self.database.get_statistics()
        
        return metrics
    
    def reset_api_status(self):
        """Reset API failure count"""
        if self.weather_api:
            self.weather_api.reset_failure_count()
            system_logger.info("API status reset")
