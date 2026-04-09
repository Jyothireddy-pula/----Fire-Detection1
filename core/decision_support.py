"""
Advanced Decision Support Engine for Wildfire Risk Management
Provides intelligent action recommendations and resource allocation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import sqlite3
import warnings
warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Risk level enumeration"""
    NO_RISK = "No Risk"
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    EXTREME = "Extreme"

class ActionPriority(Enum):
    """Action priority levels"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

@dataclass
class Resource:
    """Firefighting resource"""
    name: str
    type: str  # personnel, equipment, aircraft
    capacity: int
    availability: float  # 0-1
    cost_per_hour: float
    response_time: int  # minutes
    
@dataclass
class Action:
    """Recommended action"""
    title: str
    description: str
    priority: ActionPriority
    resources_required: List[str]
    estimated_time: int  # minutes
    effectiveness: float  # 0-1
    cost_estimate: float

@dataclass
class DecisionRecommendation:
    """Complete decision recommendation"""
    location: str
    risk_score: float
    risk_level: RiskLevel
    confidence: float
    recommended_actions: List[Action]
    resource_allocation: Dict[str, int]
    evacuation_recommendation: str
    communication_plan: List[str]
    monitoring_frequency: str
    estimated_impact: str

class DecisionSupportEngine:
    """Advanced decision support engine for wildfire management"""
    
    def __init__(self):
        self.resources = self.init_resources()
        self.action_templates = self.init_action_templates()
        self.decision_rules = self.init_decision_rules()
        self.init_database()
        
    def init_resources(self) -> Dict[str, Resource]:
        """Initialize available resources"""
        return {
            "firefighters": Resource(
                name="Firefighters",
                type="personnel",
                capacity=50,
                availability=0.8,
                cost_per_hour=50,
                response_time=30
            ),
            "fire_engines": Resource(
                name="Fire Engines",
                type="equipment",
                capacity=10,
                availability=0.9,
                cost_per_hour=200,
                response_time=15
            ),
            "water_tankers": Resource(
                name="Water Tankers",
                type="equipment",
                capacity=5,
                availability=0.85,
                cost_per_hour=150,
                response_time=20
            ),
            "helicopters": Resource(
                name="Firefighting Helicopters",
                type="aircraft",
                capacity=3,
                availability=0.7,
                cost_per_hour=2000,
                response_time=45
            ),
            "airplanes": Resource(
                name="Firefighting Airplanes",
                type="aircraft",
                capacity=2,
                availability=0.6,
                cost_per_hour=5000,
                response_time=60
            ),
            "drones": Resource(
                name="Monitoring Drones",
                type="equipment",
                capacity=15,
                availability=0.95,
                cost_per_hour=100,
                response_time=10
            ),
            "command_centers": Resource(
                name="Command Centers",
                type="facility",
                capacity=3,
                availability=0.9,
                cost_per_hour=500,
                response_time=5
            ),
            "evacuation_teams": Resource(
                name="Evacuation Teams",
                type="personnel",
                capacity=8,
                availability=0.75,
                cost_per_hour=75,
                response_time=25
            )
        }
    
    def init_action_templates(self) -> Dict[str, Dict]:
        """Initialize action templates based on risk levels"""
        return {
            "NO_RISK": {
                "actions": [
                    {
                        "title": "Routine Monitoring",
                        "description": "Continue normal monitoring schedule",
                        "priority": ActionPriority.LOW,
                        "resources": ["drones"],
                        "time": 60,
                        "effectiveness": 0.9,
                        "cost": 100
                    }
                ],
                "monitoring": "Daily",
                "evacuation": "No evacuation required"
            },
            "LOW": {
                "actions": [
                    {
                        "title": "Increased Patrol",
                        "description": "Increase ground patrol frequency",
                        "priority": ActionPriority.LOW,
                        "resources": ["firefighters"],
                        "time": 120,
                        "effectiveness": 0.8,
                        "cost": 200
                    },
                    {
                        "title": "Public Awareness",
                        "description": "Issue fire safety warnings",
                        "priority": ActionPriority.LOW,
                        "resources": [],
                        "time": 30,
                        "effectiveness": 0.7,
                        "cost": 50
                    }
                ],
                "monitoring": "Every 6 hours",
                "evacuation": "No evacuation required"
            },
            "MODERATE": {
                "actions": [
                    {
                        "title": "Pre-position Resources",
                        "description": "Move fire engines to strategic locations",
                        "priority": ActionPriority.MEDIUM,
                        "resources": ["fire_engines", "water_tankers"],
                        "time": 180,
                        "effectiveness": 0.85,
                        "cost": 1000
                    },
                    {
                        "title": "Aerial Surveillance",
                        "description": "Deploy drones for continuous monitoring",
                        "priority": ActionPriority.MEDIUM,
                        "resources": ["drones"],
                        "time": 90,
                        "effectiveness": 0.9,
                        "cost": 500
                    },
                    {
                        "title": "Community Alert",
                        "description": "Issue community preparedness alerts",
                        "priority": ActionPriority.MEDIUM,
                        "resources": [],
                        "time": 60,
                        "effectiveness": 0.75,
                        "cost": 100
                    }
                ],
                "monitoring": "Every 2 hours",
                "evacuation": "Prepare evacuation plans"
            },
            "HIGH": {
                "actions": [
                    {
                        "title": "Deploy Firefighters",
                        "description": "Deploy ground firefighting teams",
                        "priority": ActionPriority.HIGH,
                        "resources": ["firefighters", "fire_engines"],
                        "time": 60,
                        "effectiveness": 0.8,
                        "cost": 2000
                    },
                    {
                        "title": "Aerial Firefighting",
                        "description": "Deploy helicopters for water drops",
                        "priority": ActionPriority.HIGH,
                        "resources": ["helicopters"],
                        "time": 90,
                        "effectiveness": 0.85,
                        "cost": 5000
                    },
                    {
                        "title": "Establish Command Center",
                        "description": "Activate incident command center",
                        "priority": ActionPriority.HIGH,
                        "resources": ["command_centers"],
                        "time": 30,
                        "effectiveness": 0.9,
                        "cost": 1000
                    },
                    {
                        "title": "Partial Evacuation",
                        "description": "Evacuate high-risk areas",
                        "priority": ActionPriority.HIGH,
                        "resources": ["evacuation_teams"],
                        "time": 120,
                        "effectiveness": 0.8,
                        "cost": 1500
                    }
                ],
                "monitoring": "Every 30 minutes",
                "evacuation": "Voluntary evacuation recommended"
            },
            "EXTREME": {
                "actions": [
                    {
                        "title": "Full Emergency Response",
                        "description": "Deploy all available resources",
                        "priority": ActionPriority.CRITICAL,
                        "resources": ["firefighters", "fire_engines", "water_tankers", "helicopters", "airplanes"],
                        "time": 30,
                        "effectiveness": 0.9,
                        "cost": 15000
                    },
                    {
                        "title": "Mass Evacuation",
                        "description": "Immediate mandatory evacuation",
                        "priority": ActionPriority.CRITICAL,
                        "resources": ["evacuation_teams"],
                        "time": 60,
                        "effectiveness": 0.95,
                        "cost": 3000
                    },
                    {
                        "title": "Emergency Declaration",
                        "description": "Declare state of emergency",
                        "priority": ActionPriority.CRITICAL,
                        "resources": ["command_centers"],
                        "time": 15,
                        "effectiveness": 1.0,
                        "cost": 500
                    },
                    {
                        "title": "Request Mutual Aid",
                        "description": "Request assistance from neighboring regions",
                        "priority": ActionPriority.CRITICAL,
                        "resources": [],
                        "time": 45,
                        "effectiveness": 0.85,
                        "cost": 2000
                    }
                ],
                "monitoring": "Continuous",
                "evacuation": "Mandatory evacuation required"
            }
        }
    
    def init_decision_rules(self) -> Dict:
        """Initialize decision rules"""
        return {
            "weather_thresholds": {
                "high_wind": 15,  # m/s
                "low_humidity": 30,  # %
                "high_temperature": 35,  # °C
                "extreme_fwi": 30
            },
            "resource_allocation": {
                "NO_RISK": {"drones": 1},
                "LOW": {"firefighters": 2, "drones": 2},
                "MODERATE": {"firefighters": 5, "fire_engines": 2, "drones": 3},
                "HIGH": {"firefighters": 10, "fire_engines": 4, "water_tankers": 2, "helicopters": 1, "drones": 5},
                "EXTREME": {"firefighters": 20, "fire_engines": 8, "water_tankers": 4, "helicopters": 2, "airplanes": 1, "drones": 10}
            },
            "communication_channels": {
                "NO_RISK": ["Routine reports"],
                "LOW": ["Daily briefings", "Local media"],
                "MODERATE": ["6-hourly updates", "Social media", "Community alerts"],
                "HIGH": ["Hourly updates", "Emergency broadcast", "Press briefings"],
                "EXTREME": ["Continuous updates", "National emergency system", "International alerts"]
            }
        }
    
    def init_database(self):
        """Initialize database for storing decisions"""
        conn = sqlite3.connect('decision_support.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decision_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                location TEXT,
                risk_score REAL,
                risk_level TEXT,
                confidence REAL,
                recommended_actions TEXT,
                resource_allocation TEXT,
                evacuation_recommendation TEXT,
                estimated_cost REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def assess_situation(self, location: str, risk_score: float, 
                        weather_data: Dict, fire_indices: Dict) -> DecisionRecommendation:
        """Assess the situation and generate recommendations"""
        
        # Determine risk level
        if risk_score < 0.25:
            risk_level = RiskLevel.NO_RISK
        elif risk_score < 0.5:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.7:
            risk_level = RiskLevel.MODERATE
        elif risk_score < 0.85:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.EXTREME
        
        # Calculate confidence based on data quality and model consistency
        confidence = self.calculate_confidence(risk_score, weather_data, fire_indices)
        
        # Get action templates
        template = self.action_templates[risk_level.value]
        
        # Generate specific actions
        recommended_actions = []
        for action_template in template["actions"]:
            action = Action(
                title=action_template["title"],
                description=action_template["description"],
                priority=action_template["priority"],
                resources_required=action_template["resources"],
                estimated_time=action_template["time"],
                effectiveness=action_template["effectiveness"],
                cost_estimate=action_template["cost"]
            )
            
            # Adjust action based on specific conditions
            action = self.customize_action(action, weather_data, fire_indices)
            recommended_actions.append(action)
        
        # Calculate resource allocation
        resource_allocation = self.calculate_resource_allocation(risk_level, confidence)
        
        # Generate communication plan
        communication_plan = self.decision_rules["communication_channels"][risk_level.value]
        
        # Estimate overall impact
        estimated_impact = self.estimate_impact(risk_level, recommended_actions, confidence)
        
        # Create recommendation
        recommendation = DecisionRecommendation(
            location=location,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            recommended_actions=recommended_actions,
            resource_allocation=resource_allocation,
            evacuation_recommendation=template["evacuation"],
            communication_plan=communication_plan,
            monitoring_frequency=template["monitoring"],
            estimated_impact=estimated_impact
        )
        
        # Store in database
        self.store_decision(recommendation)
        
        return recommendation
    
    def calculate_confidence(self, risk_score: float, weather_data: Dict, fire_indices: Dict) -> float:
        """Calculate confidence in the prediction"""
        base_confidence = 0.8
        
        # Adjust based on extreme weather conditions
        if weather_data['temperature'] > 40:
            base_confidence += 0.1
        if weather_data['humidity'] < 20:
            base_confidence += 0.1
        if weather_data['wind_speed'] > 20:
            base_confidence += 0.1
        
        # Adjust based on fire indices consistency
        fwi = fire_indices['FWI']
        if fwi > 30 and risk_score > 0.8:
            base_confidence += 0.1
        elif fwi < 10 and risk_score < 0.3:
            base_confidence += 0.1
        
        # Cap at 0.95
        return min(base_confidence, 0.95)
    
    def customize_action(self, action: Action, weather_data: Dict, fire_indices: Dict) -> Action:
        """Customize action based on current conditions"""
        
        # Adjust based on wind speed
        if weather_data['wind_speed'] > 15:
            action.description += " (High wind conditions - increased priority)"
            action.effectiveness *= 0.9  # Reduced effectiveness in high wind
        
        # Adjust based on humidity
        if weather_data['humidity'] < 30:
            action.description += " (Low humidity - increased fire risk)"
            action.effectiveness *= 0.85
        
        # Adjust based on FWI
        if fire_indices['FWI'] > 30:
            action.estimated_time = int(action.estimated_time * 0.8)  # Faster response needed
            action.cost_estimate *= 1.2  # Higher cost for emergency response
        
        return action
    
    def calculate_resource_allocation(self, risk_level: RiskLevel, confidence: float) -> Dict[str, int]:
        """Calculate optimal resource allocation"""
        base_allocation = self.decision_rules["resource_allocation"][risk_level.value].copy()
        
        # Adjust based on confidence
        confidence_factor = confidence / 0.8  # Normalize to base confidence
        
        for resource in base_allocation:
            base_allocation[resource] = int(base_allocation[resource] * confidence_factor)
        
        # Check resource availability
        final_allocation = {}
        for resource, quantity in base_allocation.items():
            available = self.resources[resource].availability * self.resources[resource].capacity
            final_allocation[resource] = min(quantity, int(available))
        
        return final_allocation
    
    def estimate_impact(self, risk_level: RiskLevel, actions: List[Action], confidence: float) -> str:
        """Estimate the impact of recommended actions"""
        
        total_effectiveness = sum(action.effectiveness for action in actions) / len(actions)
        adjusted_effectiveness = total_effectiveness * confidence
        
        if risk_level == RiskLevel.EXTREME:
            if adjusted_effectiveness > 0.85:
                return "High likelihood of containing fire spread with minimal damage"
            elif adjusted_effectiveness > 0.7:
                return "Moderate chance of controlling fire, some damage expected"
            else:
                return "Low probability of effective control, significant damage likely"
        
        elif risk_level == RiskLevel.HIGH:
            if adjusted_effectiveness > 0.8:
                return "Good chance of preventing fire escalation"
            elif adjusted_effectiveness > 0.6:
                return "Moderate effectiveness in risk reduction"
            else:
                return "Limited impact on fire prevention"
        
        else:
            if adjusted_effectiveness > 0.7:
                return "Effective risk mitigation expected"
            else:
                return "Basic preventive measures in place"
    
    def store_decision(self, recommendation: DecisionRecommendation):
        """Store decision in database"""
        conn = sqlite3.connect('decision_support.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO decision_log 
            (timestamp, location, risk_score, risk_level, confidence, 
             recommended_actions, resource_allocation, evacuation_recommendation, estimated_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            recommendation.location,
            recommendation.risk_score,
            recommendation.risk_level.value,
            recommendation.confidence,
            json.dumps([action.__dict__ for action in recommendation.recommended_actions]),
            json.dumps(recommendation.resource_allocation),
            recommendation.evacuation_recommendation,
            sum(action.cost_estimate for action in recommendation.recommended_actions)
        ))
        
        conn.commit()
        conn.close()
    
    def generate_action_plan(self, recommendation: DecisionRecommendation, save_path='action_plan.html') -> str:
        """Generate detailed action plan HTML report"""
        
        total_cost = sum(action.cost_estimate for action in recommendation.recommended_actions)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wildfire Response Action Plan</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .header {{ background-color: #d32f2f; color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .risk-{recommendation.risk_level.value.lower().replace(' ', '-')} {{ 
                    background-color: {'#4caf50' if recommendation.risk_level == RiskLevel.NO_RISK else 
                                   '#2196f3' if recommendation.risk_level == RiskLevel.LOW else
                                   '#ff9800' if recommendation.risk_level == RiskLevel.MODERATE else
                                   '#ff5722' if recommendation.risk_level == RiskLevel.HIGH else
                                   '#d32f2f'}; 
                    color: white; padding: 15px; border-radius: 8px; margin: 20px 0; 
                }}
                .action-card {{ background-color: white; padding: 20px; margin: 15px 0; border-radius: 8px; 
                              box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-left: 5px solid 
                              {'#4caf50' if 'NO_RISK' else '#2196f3' if 'LOW' else '#ff9800' if 'MODERATE' else '#ff5722' if 'HIGH' else '#d32f2f'}; }}
                .priority-critical {{ border-left-color: #d32f2f; }}
                .priority-high {{ border-left-color: #ff5722; }}
                .priority-medium {{ border-left-color: #ff9800; }}
                .priority-low {{ border-left-color: #4caf50; }}
                .resource-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .resource-item {{ background-color: white; padding: 15px; border-radius: 8px; text-align: center; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ text-align: center; padding: 20px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .timeline {{ background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .confidence-bar {{ width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }}
                .confidence-fill {{ height: 100%; background-color: {'#4caf50' if recommendation.confidence > 0.8 else '#ff9800' if recommendation.confidence > 0.6 else '#f44336'}; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Wildfire Response Action Plan</h1>
                <p>Location: {recommendation.location}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="risk-{recommendation.risk_level.value.lower().replace(' ', '-')}">
                <h2>Risk Assessment</h2>
                <p><strong>Risk Level:</strong> {recommendation.risk_level.value}</p>
                <p><strong>Risk Score:</strong> {recommendation.risk_score:.3f}</p>
                <p><strong>Confidence:</strong> {recommendation.confidence:.2f}</p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {recommendation.confidence * 100}%"></div>
                </div>
            </div>
            
            <div class="stats">
                <div class="stat-box">
                    <h3>Total Actions</h3>
                    <div style="font-size: 24px; font-weight: bold;">{len(recommendation.recommended_actions)}</div>
                </div>
                <div class="stat-box">
                    <h3>Estimated Cost</h3>
                    <div style="font-size: 24px; font-weight: bold;">${total_cost:,.0f}</div>
                </div>
                <div class="stat-box">
                    <h3>Resources Deployed</h3>
                    <div style="font-size: 24px; font-weight: bold;">{sum(recommendation.resource_allocation.values())}</div>
                </div>
            </div>
            
            <h2>Recommended Actions</h2>
        """
        
        # Add action cards
        for i, action in enumerate(recommendation.recommended_actions, 1):
            priority_class = f"priority-{action.priority.value.lower()}"
            html_content += f"""
            <div class="action-card {priority_class}">
                <h3>Action {i}: {action.title}</h3>
                <p><strong>Description:</strong> {action.description}</p>
                <p><strong>Priority:</strong> {action.priority.value}</p>
                <p><strong>Resources Required:</strong> {', '.join(action.resources_required) if action.resources_required else 'None'}</p>
                <p><strong>Estimated Time:</strong> {action.estimated_time} minutes</p>
                <p><strong>Effectiveness:</strong> {action.effectiveness:.1%}</p>
                <p><strong>Cost Estimate:</strong> ${action.cost_estimate:,.0f}</p>
            </div>
            """
        
        html_content += f"""
            <h2>Resource Allocation</h2>
            <div class="resource-grid">
        """
        
        for resource, quantity in recommendation.resource_allocation.items():
            if quantity > 0:
                html_content += f"""
                <div class="resource-item">
                    <h4>{resource.replace('_', ' ').title()}</h4>
                    <div style="font-size: 24px; font-weight: bold;">{quantity}</div>
                    <div style="color: #666;">Available: {int(self.resources[resource].availability * self.resources[resource].capacity)}</div>
                </div>
                """
        
        html_content += f"""
            </div>
            
            <h2>Monitoring & Communication</h2>
            <div class="timeline">
                <p><strong>Monitoring Frequency:</strong> {recommendation.monitoring_frequency}</p>
                <p><strong>Evacuation Recommendation:</strong> {recommendation.evacuation_recommendation}</p>
                <p><strong>Communication Channels:</strong></p>
                <ul>
        """
        
        for channel in recommendation.communication_plan:
            html_content += f"<li>{channel}</li>"
        
        html_content += f"""
                </ul>
            </div>
            
            <h2>Expected Impact</h2>
            <div class="timeline">
                <p>{recommendation.estimated_impact}</p>
            </div>
            
            <h2>Action Timeline</h2>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Action</th>
                    <th>Priority</th>
                    <th>Duration</th>
                </tr>
        """
        
        # Create timeline
        current_time = 0
        for action in sorted(recommendation.recommended_actions, key=lambda x: (x.priority.value, x.estimated_time)):
            start_time = current_time
            end_time = start_time + action.estimated_time
            html_content += f"""
                <tr>
                    <td>{start_time} - {end_time} min</td>
                    <td>{action.title}</td>
                    <td>{action.priority.value}</td>
                    <td>{action.estimated_time} min</td>
                </tr>
            """
            current_time = end_time
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Action plan saved to {save_path}")
        return save_path
    
    def get_decision_history(self, location: str = None, days: int = 7) -> pd.DataFrame:
        """Get decision history"""
        conn = sqlite3.connect('decision_support.db')
        
        if location:
            query = '''
                SELECT * FROM decision_log 
                WHERE location = ? AND timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)
            df = pd.read_sql_query(query, conn, params=(location,))
        else:
            query = '''
                SELECT * FROM decision_log 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df

def demo_decision_support():
    """Demonstration of decision support engine"""
    print("="*60)
    print("DECISION SUPPORT ENGINE FOR WILDFIRE MANAGEMENT")
    print("="*60)
    
    # Initialize engine
    engine = DecisionSupportEngine()
    
    # Test scenarios
    test_scenarios = [
        {
            "location": "Yellowstone National Park",
            "risk_score": 0.92,
            "weather": {"temperature": 38, "humidity": 25, "wind_speed": 18, "rainfall": 0},
            "fire_indices": {"FWI": 35, "FFMC": 95, "DMC": 45, "DC": 120, "ISI": 12, "BUI": 55}
        },
        {
            "location": "Amazon Rainforest",
            "risk_score": 0.15,
            "weather": {"temperature": 28, "humidity": 85, "wind_speed": 5, "rainfall": 8},
            "fire_indices": {"FWI": 5, "FFMC": 65, "DMC": 15, "DC": 30, "ISI": 2, "BUI": 18}
        },
        {
            "location": "Siberian Taiga",
            "risk_score": 0.68,
            "weather": {"temperature": 22, "humidity": 35, "wind_speed": 12, "rainfall": 0.5},
            "fire_indices": {"FWI": 18, "FFMC": 82, "DMC": 28, "DC": 65, "ISI": 6, "BUI": 35}
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*40}")
        print(f"Analyzing: {scenario['location']}")
        print(f"Risk Score: {scenario['risk_score']:.3f}")
        
        # Generate recommendation
        recommendation = engine.assess_situation(
            scenario['location'],
            scenario['risk_score'],
            scenario['weather'],
            scenario['fire_indices']
        )
        
        print(f"Risk Level: {recommendation.risk_level.value}")
        print(f"Confidence: {recommendation.confidence:.2f}")
        print(f"Recommended Actions: {len(recommendation.recommended_actions)}")
        print(f"Total Resources: {sum(recommendation.resource_allocation.values())}")
        print(f"Evacuation: {recommendation.evacuation_recommendation}")
        
        # Generate action plan
        plan_file = f"outputs/action_plan_{scenario['location'].replace(' ', '_').replace(',', '')}.html"
        engine.generate_action_plan(recommendation, plan_file)
        
        print(f"Action plan: {plan_file}")
    
    print(f"\n{'='*60}")
    print("Decision support demonstration complete!")
    print("Check 'outputs/' directory for detailed action plans.")

if __name__ == "__main__":
    demo_decision_support()
