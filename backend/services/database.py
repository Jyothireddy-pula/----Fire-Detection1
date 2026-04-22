"""
Database Module
Handles SQLite database for historical predictions
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
import os


class HistoryDatabase:
    """SQLite database for storing prediction history"""
    
    def __init__(self, db_path: str = 'database/history.db'):
        """
        Initialize database
        
        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                location TEXT,
                latitude REAL,
                longitude REAL,
                weather_data TEXT,
                risk_score REAL,
                risk_level TEXT,
                confidence REAL,
                action TEXT,
                fwi_components TEXT
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                location TEXT,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                acknowledged BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_prediction(self, prediction_data: Dict):
        """
        Save prediction to database
        
        Args:
            prediction_data: Dictionary with prediction data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (location, latitude, longitude, weather_data, risk_score, risk_level, 
             confidence, action, fwi_components)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction_data.get('location', 'Unknown'),
            prediction_data.get('latitude'),
            prediction_data.get('longitude'),
            json.dumps(prediction_data.get('weather', {})),
            prediction_data.get('risk_score'),
            prediction_data.get('risk_level', prediction_data.get('linguistic_risk_level', 'Unknown')),
            prediction_data.get('confidence'),
            prediction_data.get('action'),
            json.dumps(prediction_data.get('fwi_components', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Get recent predictions
        
        Args:
            limit: Maximum number of predictions to retrieve
            
        Returns:
            List of prediction dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        predictions = []
        for row in rows:
            prediction = dict(zip(columns, row))
            prediction['weather_data'] = json.loads(prediction['weather_data'])
            prediction['fwi_components'] = json.loads(prediction['fwi_components'])
            predictions.append(prediction)
        
        conn.close()
        return predictions
    
    def get_predictions_by_location(self, location: str, limit: int = 50) -> List[Dict]:
        """
        Get predictions for a specific location
        
        Args:
            location: Location name
            limit: Maximum number of predictions
            
        Returns:
            List of prediction dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE location = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (location, limit))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        predictions = []
        for row in rows:
            prediction = dict(zip(columns, row))
            prediction['weather_data'] = json.loads(prediction['weather_data'])
            prediction['fwi_components'] = json.loads(prediction['fwi_components'])
            predictions.append(prediction)
        
        conn.close()
        return predictions
    
    def save_alert(self, alert_data: Dict):
        """
        Save alert to database
        
        Args:
            alert_data: Dictionary with alert data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (location, alert_type, severity, message)
            VALUES (?, ?, ?, ?)
        ''', (
            alert_data.get('location', 'Unknown'),
            alert_data.get('alert_type', 'wildfire_risk'),
            alert_data.get('severity', 'high'),
            alert_data.get('message', '')
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_alerts(self, limit: int = 50, acknowledged: bool = None) -> List[Dict]:
        """
        Get recent alerts
        
        Args:
            limit: Maximum number of alerts
            acknowledged: Filter by acknowledgment status
            
        Returns:
            List of alert dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if acknowledged is not None:
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE acknowledged = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (acknowledged, limit))
        else:
            cursor.execute('''
                SELECT * FROM alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        alerts = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return alerts
    
    def acknowledge_alert(self, alert_id: int):
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID to acknowledge
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts 
            SET acknowledged = 1 
            WHERE id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        # Total alerts
        cursor.execute('SELECT COUNT(*) FROM alerts')
        total_alerts = cursor.fetchone()[0]
        
        # Unacknowledged alerts
        cursor.execute('SELECT COUNT(*) FROM alerts WHERE acknowledged = 0')
        unacknowledged_alerts = cursor.fetchone()[0]
        
        # Risk level distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) 
            FROM predictions 
            GROUP BY risk_level
        ''')
        risk_distribution = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'total_alerts': total_alerts,
            'unacknowledged_alerts': unacknowledged_alerts,
            'risk_distribution': risk_distribution
        }
    
    def clear_old_data(self, days: int = 30):
        """
        Clear data older than specified days
        
        Args:
            days: Number of days to keep
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM predictions 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        cursor.execute('''
            DELETE FROM alerts 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        conn.commit()
        conn.close()
