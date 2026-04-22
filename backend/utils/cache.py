"""
Caching Module
Provides caching functionality for API responses and model predictions
"""

import time
import json
import hashlib
from typing import Any, Optional
from datetime import datetime, timedelta


class SimpleCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl=300):
        """
        Initialize cache
        
        Args:
            default_ttl: Default time-to-live in seconds (default: 5 minutes)
        """
        self.cache = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters"""
        key_dict = {'prefix': prefix, **kwargs}
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prefix: str, **kwargs) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            prefix: Key prefix
            **kwargs: Additional parameters for key generation
            
        Returns:
            Cached value or None if not found/expired
        """
        key = self._generate_key(prefix, **kwargs)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now() > entry['expiry']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, prefix: str, value: Any, ttl: Optional[int] = None, **kwargs):
        """
        Set value in cache
        
        Args:
            prefix: Key prefix
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            **kwargs: Additional parameters for key generation
        """
        key = self._generate_key(prefix, **kwargs)
        
        if ttl is None:
            ttl = self.default_ttl
        
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expiry': expiry,
            'created': datetime.now()
        }
    
    def delete(self, prefix: str, **kwargs):
        """
        Delete value from cache
        
        Args:
            prefix: Key prefix
            **kwargs: Additional parameters for key generation
        """
        key = self._generate_key(prefix, **kwargs)
        
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
    
    def cleanup_expired(self):
        """Remove all expired entries"""
        now = datetime.now()
        expired_keys = [k for k, v in self.cache.items() if now > v['expiry']]
        
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        self.cleanup_expired()
        
        total_entries = len(self.cache)
        total_size = sum(len(str(v['value'])) for v in self.cache.values())
        
        return {
            'total_entries': total_entries,
            'total_size_bytes': total_size,
            'default_ttl': self.default_ttl
        }


# Global cache instance
weather_cache = SimpleCache(default_ttl=300)  # 5 minutes for weather data
model_cache = SimpleCache(default_ttl=3600)    # 1 hour for model predictions
