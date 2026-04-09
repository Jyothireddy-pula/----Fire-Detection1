"""
Advanced Alert System and Historical Analysis for Wildfire Risk Management
Provides real-time alerts, trend analysis, and historical reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import warnings
warnings.filterwarnings('ignore')

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class AlertStatus(Enum):
    """Alert status"""
    ACTIVE = "Active"
    ACKNOWLEDGED = "Acknowledged"
    RESOLVED = "Resolved"
    EXPIRED = "Expired"

@dataclass
class Alert:
    """Alert data structure"""
    id: int
    timestamp: datetime
    location: str
    severity: AlertSeverity
    status: AlertStatus
    risk_score: float
    risk_level: str
    message: str
    details: Dict
    notification_sent: bool
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'severity': self.severity.value,
            'status': self.status.value,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'message': self.message,
            'details': self.details,
            'notification_sent': self.notification_sent,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None
        }

@dataclass
class AlertThreshold:
    """Alert threshold configuration"""
    risk_score_threshold: float
    severity: AlertSeverity
    message_template: str
    notification_channels: List[str]
    auto_escalate: bool
    escalation_time_minutes: int

class AlertSystem:
    """Advanced alert system for wildfire risk management"""
    
    def __init__(self, db_path: str = 'alert_system.db'):
        self.db_path = db_path
        self.thresholds = self.init_thresholds()
        self.init_database()
        self.active_alerts = {}
        self.alert_history = []
        self.notification_settings = self.init_notification_settings()
        self.monitoring_thread = None
        self.is_monitoring = False
        
    def init_thresholds(self) -> Dict[str, AlertThreshold]:
        """Initialize alert thresholds"""
        return {
            'extreme': AlertThreshold(
                risk_score_threshold=0.85,
                severity=AlertSeverity.CRITICAL,
                message_template="EXTREME WILDFIRE RISK at {location}. Risk Score: {risk_score:.3f}. Immediate action required!",
                notification_channels=['email', 'sms', 'dashboard', 'webhook'],
                auto_escalate=True,
                escalation_time_minutes=15
            ),
            'high': AlertThreshold(
                risk_score_threshold=0.70,
                severity=AlertSeverity.HIGH,
                message_template="HIGH WILDFIRE RISK at {location}. Risk Score: {risk_score:.3f}. Increase monitoring and prepare resources.",
                notification_channels=['email', 'dashboard', 'webhook'],
                auto_escalate=True,
                escalation_time_minutes=60
            ),
            'moderate': AlertThreshold(
                risk_score_threshold=0.50,
                severity=AlertSeverity.MEDIUM,
                message_template="MODERATE WILDFIRE RISK at {location}. Risk Score: {risk_score:.3f}. Enhanced monitoring recommended.",
                notification_channels=['dashboard', 'webhook'],
                auto_escalate=False,
                escalation_time_minutes=0
            ),
            'low': AlertThreshold(
                risk_score_threshold=0.25,
                severity=AlertSeverity.LOW,
                message_template="Low wildfire risk at {location}. Risk Score: {risk_score:.3f}. Normal monitoring.",
                notification_channels=['dashboard'],
                auto_escalate=False,
                escalation_time_minutes=0
            )
        }
    
    def init_notification_settings(self) -> Dict:
        """Initialize notification settings"""
        return {
            'email': {
                'enabled': False,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': '',
                'password': '',
                'recipients': []
            },
            'sms': {
                'enabled': False,
                'api_key': '',
                'phone_numbers': []
            },
            'webhook': {
                'enabled': False,
                'url': '',
                'headers': {}
            },
            'dashboard': {
                'enabled': True
            }
        }
    
    def init_database(self):
        """Initialize alert database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                location TEXT,
                severity TEXT,
                status TEXT,
                risk_score REAL,
                risk_level TEXT,
                message TEXT,
                details TEXT,
                notification_sent BOOLEAN,
                acknowledged_by TEXT,
                resolved_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Alert history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id INTEGER,
                timestamp TEXT,
                action TEXT,
                user TEXT,
                details TEXT,
                FOREIGN KEY (alert_id) REFERENCES alerts (id)
            )
        ''')
        
        # Risk data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                location TEXT,
                risk_score REAL,
                risk_level TEXT,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                rainfall REAL,
                fwi REAL,
                ffmc REAL,
                dmc REAL,
                dc REAL,
                isi REAL,
                bui REAL,
                weather_condition TEXT,
                fire_indices TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def check_and_create_alert(self, location: str, risk_score: float, risk_level: str,
                              weather_data: Dict, fire_indices: Dict) -> Optional[Alert]:
        """Check risk score and create alert if threshold is exceeded"""
        
        # Check against thresholds
        for threshold_name, threshold in self.thresholds.items():
            if risk_score >= threshold.risk_score_threshold:
                # Check if alert already exists for this location
                existing_alert = self.get_active_alert_for_location(location)
                
                if existing_alert and existing_alert.risk_score == risk_score:
                    return None  # Alert already exists with same risk
                
                # Create new alert
                message = threshold.message_template.format(
                    location=location,
                    risk_score=risk_score
                )
                
                alert = Alert(
                    id=len(self.alert_history) + 1,
                    timestamp=datetime.now(),
                    location=location,
                    severity=threshold.severity,
                    status=AlertStatus.ACTIVE,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    message=message,
                    details={
                        'weather_data': weather_data,
                        'fire_indices': fire_indices,
                        'threshold': threshold_name
                    },
                    notification_sent=False
                )
                
                # Store alert
                self.store_alert(alert)
                self.active_alerts[location] = alert
                self.alert_history.append(alert)
                
                # Send notifications
                self.send_notifications(alert, threshold)
                
                return alert
        
        return None
    
    def store_alert(self, alert: Alert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts 
            (timestamp, location, severity, status, risk_score, risk_level, message, 
             details, notification_sent, acknowledged_by, resolved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.timestamp.isoformat(),
            alert.location,
            alert.severity.value,
            alert.status.value,
            alert.risk_score,
            alert.risk_level,
            alert.message,
            json.dumps(alert.details),
            alert.notification_sent,
            alert.acknowledged_by,
            alert.resolved_at.isoformat() if alert.resolved_at else None
        ))
        
        conn.commit()
        alert.id = cursor.lastrowid
        conn.close()
    
    def store_risk_data(self, location: str, risk_score: float, risk_level: str,
                       weather_data: Dict, fire_indices: Dict):
        """Store risk assessment data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO risk_data 
            (timestamp, location, risk_score, risk_level, temperature, humidity, 
             wind_speed, rainfall, fwi, ffmc, dmc, dc, isi, bui, weather_condition, fire_indices)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            location,
            risk_score,
            risk_level,
            weather_data.get('temperature', 0),
            weather_data.get('humidity', 0),
            weather_data.get('wind_speed', 0),
            weather_data.get('rainfall', 0),
            fire_indices.get('FWI', 0),
            fire_indices.get('FFMC', 0),
            fire_indices.get('DMC', 0),
            fire_indices.get('DC', 0),
            fire_indices.get('ISI', 0),
            fire_indices.get('BUI', 0),
            weather_data.get('weather_condition', 'Unknown'),
            json.dumps(fire_indices)
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_alert_for_location(self, location: str) -> Optional[Alert]:
        """Get active alert for a specific location"""
        return self.active_alerts.get(location)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM alerts 
            WHERE status = 'Active'
            ORDER BY timestamp DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        alerts = []
        for row in rows:
            alert = Alert(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                location=row[2],
                severity=AlertSeverity(row[3]),
                status=AlertStatus(row[4]),
                risk_score=row[5],
                risk_level=row[6],
                message=row[7],
                details=json.loads(row[8]) if row[8] else {},
                notification_sent=row[9],
                acknowledged_by=row[10],
                resolved_at=datetime.fromisoformat(row[11]) if row[11] else None
            )
            alerts.append(alert)
        
        return alerts
    
    def acknowledge_alert(self, alert_id: int, user: str) -> bool:
        """Acknowledge an alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts 
            SET status = ?, acknowledged_by = ?
            WHERE id = ?
        ''', (AlertStatus.ACKNOWLEDGED.value, user, alert_id))
        
        conn.commit()
        conn.close()
        
        # Update in memory
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                
                # Remove from active alerts
                if alert.location in self.active_alerts:
                    del self.active_alerts[alert.location]
                
                return True
        
        return False
    
    def resolve_alert(self, alert_id: int) -> bool:
        """Resolve an alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE alerts 
            SET status = ?, resolved_at = ?
            WHERE id = ?
        ''', (AlertStatus.RESOLVED.value, datetime.now().isoformat(), alert_id))
        
        conn.commit()
        conn.close()
        
        # Update in memory
        for alert in self.alert_history:
            if alert.id == alert_id:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                
                # Remove from active alerts
                if alert.location in self.active_alerts:
                    del self.active_alerts[alert.location]
                
                return True
        
        return False
    
    def send_notifications(self, alert: Alert, threshold: AlertThreshold):
        """Send notifications through configured channels"""
        for channel in threshold.notification_channels:
            if channel == 'email' and self.notification_settings['email']['enabled']:
                self.send_email_notification(alert)
            elif channel == 'sms' and self.notification_settings['sms']['enabled']:
                self.send_sms_notification(alert)
            elif channel == 'webhook' and self.notification_settings['webhook']['enabled']:
                self.send_webhook_notification(alert)
            elif channel == 'dashboard':
                # Dashboard notification is handled by the UI
                pass
        
        alert.notification_sent = True
    
    def send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            settings = self.notification_settings['email']
            
            msg = MIMEMultipart()
            msg['From'] = settings['username']
            msg['To'] = ', '.join(settings['recipients'])
            msg['Subject'] = f"Wildfire Alert: {alert.severity.value} Risk at {alert.location}"
            
            body = f"""
            WILDFIRE RISK ALERT
            
            Location: {alert.location}
            Severity: {alert.severity.value}
            Risk Score: {alert.risk_score:.3f}
            Risk Level: {alert.risk_level}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Message:
            {alert.message}
            
            Please take appropriate action immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(settings['smtp_server'], settings['smtp_port'])
            server.starttls()
            server.login(settings['username'], settings['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            print(f"Error sending email notification: {e}")
    
    def send_sms_notification(self, alert: Alert):
        """Send SMS notification"""
        # Implementation would depend on SMS gateway provider
        print(f"SMS notification would be sent for alert {alert.id}")
    
    def send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            import requests
            
            settings = self.notification_settings['webhook']
            
            payload = {
                'alert_id': alert.id,
                'location': alert.location,
                'severity': alert.severity.value,
                'risk_score': alert.risk_score,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat()
            }
            
            response = requests.post(settings['url'], json=payload, headers=settings['headers'])
            
            if response.status_code == 200:
                print(f"Webhook notification sent for alert {alert.id}")
            else:
                print(f"Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"Error sending webhook notification: {e}")

class HistoricalAnalysis:
    """Historical analysis system for trend detection"""
    
    def __init__(self, db_path: str = 'alert_system.db'):
        self.db_path = db_path
    
    def get_risk_trends(self, location: str = None, days: int = 30) -> pd.DataFrame:
        """Get risk trends over time"""
        conn = sqlite3.connect(self.db_path)
        
        if location:
            query = '''
                SELECT * FROM risk_data 
                WHERE location = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days)
            df = pd.read_sql_query(query, conn, params=(location,))
        else:
            query = '''
                SELECT * FROM risk_data 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days)
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze trends in the data"""
        if df.empty:
            return {'error': 'No data available for analysis'}
        
        results = {}
        
        # Risk score trend
        if len(df) > 1:
            risk_trend = np.polyfit(range(len(df)), df['risk_score'], 1)
            results['risk_trend_slope'] = risk_trend[0]
            results['risk_trend_direction'] = 'Increasing' if risk_trend[0] > 0 else 'Decreasing'
            
            # Calculate moving averages
            df['risk_ma_7d'] = df['risk_score'].rolling(window=7, min_periods=1).mean()
            results['current_7d_avg'] = df['risk_ma_7d'].iloc[-1] if not df['risk_ma_7d'].empty else 0
        
        # Temperature trend
        if 'temperature' in df.columns and len(df) > 1:
            temp_trend = np.polyfit(range(len(df)), df['temperature'], 1)
            results['temperature_trend'] = temp_trend[0]
        
        # Alert frequency
        high_risk_count = len(df[df['risk_score'] >= 0.7])
        results['high_risk_frequency'] = high_risk_count / len(df) if len(df) > 0 else 0
        
        # Peak risk analysis
        peak_risk = df['risk_score'].max()
        peak_risk_time = df.loc[df['risk_score'].idxmax(), 'timestamp'] if not df.empty else None
        results['peak_risk_score'] = peak_risk
        results['peak_risk_time'] = peak_risk_time
        
        # Seasonal patterns (if enough data)
        if len(df) >= 30:
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            seasonal_avg = df.groupby('day_of_year')['risk_score'].mean()
            results['seasonal_pattern'] = seasonal_avg.to_dict()
        
        return results
    
    def create_trend_visualization(self, df: pd.DataFrame, save_path=None):
        """Create trend visualization"""
        if df.empty:
            print("No data available for visualization")
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Risk Score Trend', 'Weather Parameters', 'Fire Weather Indices'),
            vertical_spacing=0.08
        )
        
        # Risk score trend
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['risk_score'],
                      mode='lines+markers', name='Risk Score',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Add moving average
        if 'risk_ma_7d' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['risk_ma_7d'],
                          mode='lines', name='7-day MA',
                          line=dict(color='orange', dash='dash')),
                row=1, col=1
            )
        
        # Weather parameters
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['temperature'],
                      mode='lines', name='Temperature (°C)',
                      line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['humidity'],
                      mode='lines', name='Humidity (%)',
                      line=dict(color='blue')),
            row=2, col=1
        )
        
        # Fire indices
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['fwi'],
                      mode='lines', name='FWI',
                      line=dict(color='purple')),
            row=3, col=1
        )
        
        fig.update_layout(
            height=900,
            title_text=f"Historical Trend Analysis - {df['location'].iloc[0] if 'location' in df.columns else 'All Locations'}",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Trend visualization saved to {save_path}")
        
        return fig
    
    def generate_historical_report(self, location: str = None, days: int = 30, 
                                 save_path='historical_report.html'):
        """Generate comprehensive historical report"""
        
        df = self.get_risk_trends(location, days)
        trends = self.analyze_trends(df)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Historical Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #2196f3; color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .metric {{ display: inline-block; text-align: center; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 8px; min-width: 150px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196f3; }}
                .metric-label {{ font-size: 12px; color: #666; }}
                .trend-up {{ color: #f44336; }}
                .trend-down {{ color: #4caf50; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Wildfire Risk Historical Analysis Report</h1>
                <p>Period: Last {days} days | Location: {location if location else 'All Locations'}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <div class="metric">
                    <div class="metric-value">{df['risk_score'].mean():.3f}</div>
                    <div class="metric-label">Average Risk</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{df['risk_score'].max():.3f}</div>
                    <div class="metric-label">Peak Risk</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(df)}</div>
                    <div class="metric-label">Data Points</div>
                </div>
                <div class="metric">
                    <div class="metric-value {('trend-up' if trends.get('risk_trend_slope', 0) > 0 else 'trend-down')}">
                        {trends.get('risk_trend_direction', 'Stable')}
                    </div>
                    <div class="metric-label">Risk Trend</div>
                </div>
            </div>
        """
        
        if 'peak_risk_score' in trends:
            html_content += f"""
            <div class="section">
                <h2>Risk Analysis</h2>
                <p><strong>Peak Risk Score:</strong> {trends['peak_risk_score']:.3f}</p>
                <p><strong>Peak Risk Time:</strong> {trends['peak_risk_time']}</p>
                <p><strong>High Risk Frequency:</strong> {trends.get('high_risk_frequency', 0):.1%}</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Historical report saved to {save_path}")
        return save_path

def demo_alert_system():
    """Demonstration of alert system"""
    print("="*60)
    print("ALERT SYSTEM AND HISTORICAL ANALYSIS")
    print("="*60)
    
    # Initialize alert system
    alert_system = AlertSystem()
    historical_analysis = HistoricalAnalysis()
    
    # Test locations with different risk levels
    test_locations = [
        {"location": "Yellowstone National Park", "risk_score": 0.92, "risk_level": "Extreme"},
        {"location": "Amazon Rainforest", "risk_score": 0.15, "risk_level": "No Risk"},
        {"location": "California Forest", "risk_score": 0.78, "risk_level": "High"},
        {"location": "Australian Bushland", "risk_score": 0.85, "risk_level": "Extreme"}
    ]
    
    print("\nTesting alert generation...")
    
    for test_case in test_locations:
        weather_data = {
            'temperature': 35,
            'humidity': 30,
            'wind_speed': 15,
            'rainfall': 0,
            'weather_condition': 'Sunny'
        }
        
        fire_indices = {
            'FWI': 35,
            'FFMC': 90,
            'DMC': 40,
            'DC': 100,
            'ISI': 10,
            'BUI': 50
        }
        
        # Store risk data
        alert_system.store_risk_data(
            test_case['location'],
            test_case['risk_score'],
            test_case['risk_level'],
            weather_data,
            fire_indices
        )
        
        # Check for alerts
        alert = alert_system.check_and_create_alert(
            test_case['location'],
            test_case['risk_score'],
            test_case['risk_level'],
            weather_data,
            fire_indices
        )
        
        if alert:
            print(f"✓ Alert created for {test_case['location']}: {alert.severity.value}")
        else:
            print(f"- No alert for {test_case['location']}: {test_case['risk_level']}")
    
    # Get active alerts
    active_alerts = alert_system.get_active_alerts()
    print(f"\nActive Alerts: {len(active_alerts)}")
    
    for alert in active_alerts:
        print(f"  - {alert.location}: {alert.risk_level} ({alert.risk_score:.3f})")
    
    # Historical analysis
    print("\nPerforming historical analysis...")
    
    # Get trends
    trends_df = historical_analysis.get_risk_trends(days=7)
    
    if not trends_df.empty:
        trends = historical_analysis.analyze_trends(trends_df)
        
        print(f"Data points: {len(trends_df)}")
        print(f"Average risk: {trends_df['risk_score'].mean():.3f}")
        print(f"Peak risk: {trends_df['risk_score'].max():.3f}")
        
        if 'risk_trend_direction' in trends:
            print(f"Risk trend: {trends['risk_trend_direction']}")
        
        # Create visualization
        historical_analysis.create_trend_visualization(trends_df, 'outputs/historical_trends.html')
        
        # Generate report
        historical_analysis.generate_historical_report(save_path='outputs/historical_report.html')
    
    print(f"\n{'='*60}")
    print("Alert system and historical analysis complete!")
    print("- Historical trends: outputs/historical_trends.html")
    print("- Historical report: outputs/historical_report.html")

if __name__ == "__main__":
    demo_alert_system()
