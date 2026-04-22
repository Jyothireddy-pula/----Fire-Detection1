"""
Logging Module
Provides logging functionality for the system
"""

import logging
import os
from datetime import datetime
from typing import Optional


class SystemLogger:
    """System logger with file and console output"""
    
    def __init__(self, name='WildfireSystem', log_dir='logs', level=logging.INFO):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler for all logs
        log_file = os.path.join(log_dir, f'{name.lower()}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # File handler for errors only
        error_file = os.path.join(log_dir, f'{name.lower()}_error.log')
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_api_call(self, endpoint: str, params: dict, status: str, response_time: float):
        """
        Log API call details
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            status: Response status
            response_time: Response time in seconds
        """
        self.info(f"API Call - Endpoint: {endpoint}, Params: {params}, Status: {status}, Time: {response_time:.2f}s")
    
    def log_prediction(self, inputs: dict, output: dict, confidence: float):
        """
        Log prediction details
        
        Args:
            inputs: Input features
            output: Prediction output
            confidence: Confidence score
        """
        self.info(f"Prediction - Inputs: {inputs}, Output: {output}, Confidence: {confidence:.2f}")
    
    def log_model_training(self, model_name: str, metrics: dict, training_time: float):
        """
        Log model training details
        
        Args:
            model_name: Name of the model
            metrics: Training metrics
            training_time: Training time in seconds
        """
        self.info(f"Model Training - Model: {model_name}, Metrics: {metrics}, Time: {training_time:.2f}s")


# Global logger instance
system_logger = SystemLogger()
