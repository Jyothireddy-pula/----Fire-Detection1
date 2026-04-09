"""
Advanced Scenario Simulator and Trend Analysis for Wildfire Risk Prediction
Provides what-if analysis, sensitivity analysis, and future risk projections
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class ScenarioSimulator:
    """Advanced scenario simulator for wildfire risk analysis"""
    
    def __init__(self, model_path: str = 'models/anfis_pso_optimized.h5'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.load_model()
        self.feature_names = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temp_RH_Index', 'Wind_Rain_Interaction']
        
    def load_model(self):
        """Load the trained model"""
        try:
            try:
                from .anfis_system import AdvancedANFIS
            except ImportError:
                from anfis_system import AdvancedANFIS
            self.model = AdvancedANFIS()
            self.model.load_model(self.model_path)
            self.scaler = joblib.load('models/scaler.pkl')
            print("ANFIS model loaded successfully")
        except Exception as e:
            print(f"Error loading ANFIS model: {e}")
            try:
                self.model = joblib.load('models/mlp_pso.pkl')
                self.scaler = joblib.load('models/scaler.pkl')
                print("Fallback MLP model loaded")
            except:
                print("No model available, using simulation only")
    
    def simulate_single_parameter(self, base_features: np.ndarray, 
                                 param_name: str, param_range: Tuple[float, float],
                                 n_points: int = 50) -> Dict:
        """Simulate risk changes by varying a single parameter"""
        
        param_idx = self.feature_names.index(param_name)
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        risk_scores = []
        
        for value in param_values:
            features = base_features.copy()
            features[param_idx] = value
            
            # Recalculate interaction features if needed
            if param_name == 'Temperature':
                features[self.feature_names.index('Temp_RH_Index')] = value * (100 - features[self.feature_names.index('RH')])
            elif param_name == 'RH':
                features[self.feature_names.index('Temp_RH_Index')] = features[self.feature_names.index('Temperature')] * (100 - value)
            elif param_name == 'Ws':
                features[self.feature_names.index('Wind_Rain_Interaction')] = value * features[self.feature_names.index('Rain')]
            elif param_name == 'Rain':
                features[self.feature_names.index('Wind_Rain_Interaction')] = features[self.feature_names.index('Ws')] * value
            
            # Calculate risk score
            if self.model and self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                risk = float(self.model.predict(features_scaled)[0])
                risk = np.clip(risk, 0, 1)
            else:
                # Fallback calculation
                risk = self.calculate_simple_risk(features)
            
            risk_scores.append(risk)
        
        return {
            'parameter': param_name,
            'values': param_values,
            'risk_scores': np.array(risk_scores),
            'sensitivity': self.calculate_sensitivity(param_values, risk_scores)
        }
    
    def simulate_multi_parameter(self, base_features: np.ndarray,
                                param_combinations: Dict[str, Tuple[float, float]],
                                n_points: int = 20) -> Dict:
        """Simulate risk changes by varying multiple parameters"""
        
        results = {}
        
        for param_name, param_range in param_combinations.items():
            print(f"Simulating {param_name}...")
            result = self.simulate_single_parameter(base_features, param_name, param_range, n_points)
            results[param_name] = result
        
        return results
    
    def calculate_simple_risk(self, features: np.ndarray) -> float:
        """Simple risk calculation when model is not available"""
        temp, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi = features[:10]
        
        # Simple risk formula based on fire indices
        risk = (ffmc / 100) * 0.3 + (fwi / 50) * 0.4 + (isi / 20) * 0.2 + (temp / 40) * 0.1
        
        # Reduce risk based on rain and humidity
        risk *= (1 - rain / 20) * (1 - rh / 200)
        
        return np.clip(risk, 0, 1)
    
    def calculate_sensitivity(self, param_values: np.ndarray, risk_scores: np.ndarray) -> float:
        """Calculate sensitivity coefficient"""
        if len(param_values) < 2:
            return 0.0
        
        # Calculate slope using linear regression
        coeffs = np.polyfit(param_values, risk_scores, 1)
        return float(coeffs[0])
    
    def create_what_if_scenarios(self, base_features: np.ndarray) -> Dict:
        """Create predefined what-if scenarios"""
        
        scenarios = {
            'Heat Wave': {
                'description': 'Extreme heat wave conditions',
                'changes': {'Temperature': (35, 45), 'RH': (15, 25)},
                'color': '#ff5722'
            },
            'Drought': {
                'description': 'Extended drought conditions',
                'changes': {'Rain': (0, 0.5), 'RH': (20, 35), 'Temperature': (30, 40)},
                'color': '#ff9800'
            },
            'Storm Front': {
                'description': 'Approaching storm with high winds',
                'changes': {'Ws': (15, 30), 'Rain': (0, 5), 'RH': (40, 60)},
                'color': '#2196f3'
            },
            'Humidity Drop': {
                'description': 'Rapid humidity decrease',
                'changes': {'RH': (10, 30), 'Temperature': (25, 35)},
                'color': '#9c27b0'
            },
            'Windy Conditions': {
                'description': 'High wind conditions',
                'changes': {'Ws': (20, 40), 'Rain': (0, 1)},
                'color': '#00bcd4'
            }
        }
        
        scenario_results = {}
        
        for scenario_name, scenario_config in scenarios.items():
            print(f"Analyzing scenario: {scenario_name}")
            
            # Simulate the scenario
            results = self.simulate_multi_parameter(base_features, scenario_config['changes'], n_points=30)
            
            scenario_results[scenario_name] = {
                'description': scenario_config['description'],
                'results': results,
                'color': scenario_config['color']
            }
        
        return scenario_results
    
    def simulate_time_series(self, base_features: np.ndarray, 
                           days: int = 7, weather_trend: str = 'normal') -> Dict:
        """Simulate risk over time with weather trends"""
        
        dates = pd.date_range(start=datetime.now(), periods=days * 24, freq='H')
        
        # Generate weather patterns based on trend
        if weather_trend == 'normal':
            temp_pattern = np.sin(np.linspace(0, 2*np.pi, days)) * 5 + 25
            rh_pattern = np.cos(np.linspace(0, 2*np.pi, days)) * 15 + 50
            wind_pattern = np.random.normal(10, 3, days)
            rain_pattern = np.random.exponential(2, days)
        elif weather_trend == 'warming':
            temp_pattern = np.linspace(25, 35, days) + np.random.normal(0, 2, days)
            rh_pattern = np.linspace(60, 30, days) + np.random.normal(0, 5, days)
            wind_pattern = np.random.normal(15, 5, days)
            rain_pattern = np.random.exponential(1, days)
        elif weather_trend == 'storm':
            temp_pattern = np.linspace(30, 20, days) + np.random.normal(0, 3, days)
            rh_pattern = np.linspace(40, 70, days) + np.random.normal(0, 8, days)
            wind_pattern = np.linspace(5, 25, days) + np.random.normal(0, 5, days)
            rain_pattern = np.concatenate([np.random.exponential(0.5, days//2), 
                                           np.random.exponential(5, days//2)])
        else:
            # Extreme conditions
            temp_pattern = np.linspace(35, 45, days) + np.random.normal(0, 2, days)
            rh_pattern = np.linspace(30, 15, days) + np.random.normal(0, 5, days)
            wind_pattern = np.linspace(10, 30, days) + np.random.normal(0, 5, days)
            rain_pattern = np.random.exponential(0.5, days)
        
        # Interpolate to hourly values
        temp_hourly = np.repeat(temp_pattern, 24) + np.random.normal(0, 1, len(dates))
        rh_hourly = np.repeat(rh_pattern, 24) + np.random.normal(0, 3, len(dates))
        wind_hourly = np.repeat(wind_pattern, 24) + np.random.normal(0, 2, len(dates))
        rain_hourly = np.repeat(rain_pattern, 24) + np.random.normal(0, 0.5, len(dates))
        
        # Ensure realistic bounds
        temp_hourly = np.clip(temp_hourly, 0, 50)
        rh_hourly = np.clip(rh_hourly, 10, 100)
        wind_hourly = np.clip(wind_hourly, 0, 40)
        rain_hourly = np.clip(rain_hourly, 0, 20)
        
        # Calculate risk over time
        risk_scores = []
        fire_indices_history = []
        
        for i in range(len(dates)):
            features = base_features.copy()
            features[0] = temp_hourly[i]  # Temperature
            features[1] = rh_hourly[i]    # RH
            features[2] = wind_hourly[i]  # Ws
            features[3] = rain_hourly[i]  # Rain
            
            # Update interaction features
            features[10] = temp_hourly[i] * (100 - rh_hourly[i])  # Temp_RH_Index
            features[11] = wind_hourly[i] * rain_hourly[i]        # Wind_Rain_Interaction
            
            # Calculate fire indices
            try:
                from .weather_api import FireWeatherIndexCalculator
            except ImportError:
                from weather_api import FireWeatherIndexCalculator
            calculator = FireWeatherIndexCalculator()
            
            fire_indices = calculator.calculate_all_indices(
                type('WeatherData', (), {
                    'temperature': temp_hourly[i],
                    'humidity': rh_hourly[i],
                    'wind_speed': wind_hourly[i],
                    'rainfall': rain_hourly[i]
                })()
            )
            
            # Update fire index features
            features[4] = fire_indices['FFMC']
            features[5] = fire_indices['DMC']
            features[6] = fire_indices['DC']
            features[7] = fire_indices['ISI']
            features[8] = fire_indices['BUI']
            features[9] = fire_indices['FWI']
            
            # Calculate risk
            if self.model and self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                risk = float(self.model.predict(features_scaled)[0])
                risk = np.clip(risk, 0, 1)
            else:
                risk = self.calculate_simple_risk(features)
            
            risk_scores.append(risk)
            fire_indices_history.append(fire_indices)
        
        return {
            'dates': dates,
            'temperature': temp_hourly,
            'humidity': rh_hourly,
            'wind_speed': wind_hourly,
            'rainfall': rain_hourly,
            'risk_scores': np.array(risk_scores),
            'fire_indices': fire_indices_history,
            'trend': weather_trend
        }
    
    def create_sensitivity_analysis(self, base_features: np.ndarray) -> Dict:
        """Create comprehensive sensitivity analysis"""
        
        # Define parameter ranges for each feature
        param_ranges = {
            'Temperature': (0, 50),
            'RH': (0, 100),
            'Ws': (0, 40),
            'Rain': (0, 20),
            'FFMC': (0, 101),
            'DMC': (0, 100),
            'DC': (0, 500),
            'ISI': (0, 50),
            'BUI': (0, 100),
            'FWI': (0, 100)
        }
        
        sensitivity_results = {}
        
        for param_name, param_range in param_ranges.items():
            print(f"Analyzing sensitivity for {param_name}...")
            result = self.simulate_single_parameter(base_features, param_name, param_range, n_points=30)
            sensitivity_results[param_name] = result
        
        # Rank parameters by sensitivity
        sensitivity_ranking = sorted(
            [(name, result['sensitivity']) for name, result in sensitivity_results.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return {
            'results': sensitivity_results,
            'ranking': sensitivity_ranking
        }
    
    def plot_scenario_comparison(self, scenario_results: Dict, save_path=None):
        """Create comprehensive scenario comparison plots"""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=tuple(scenario_results.keys())[:4],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"colspan": 2}]]
        )
        
        plot_idx = 1
        for scenario_name, scenario_data in list(scenario_results.items())[:4]:
            row = (plot_idx - 1) // 2 + 1
            col = (plot_idx - 1) % 2 + 1
            
            # Plot each parameter in the scenario
            for param_name, param_result in scenario_data['results'].items():
                fig.add_trace(
                    go.Scatter(
                        x=param_result['values'],
                        y=param_result['risk_scores'],
                        mode='lines',
                        name=f'{scenario_name} - {param_name}',
                        line=dict(color=scenario_data['color'])
                    ),
                    row=row, col=col
                )
            
            plot_idx += 1
        
        # Add sensitivity ranking
        row = 3
        col = 1
        sensitivities = []
        param_names = []
        
        for scenario_name, scenario_data in scenario_results.items():
            for param_name, param_result in scenario_data['results'].items():
                sensitivities.append(abs(param_result['sensitivity']))
                param_names.append(f'{scenario_name}: {param_name}')
        
        fig.add_trace(
            go.Bar(
                x=param_names[:10],  # Top 10
                y=sensitivities[:10],
                name='Sensitivity',
                marker_color='lightblue'
            ),
            row=row, col=1
        )
        
        fig.update_layout(
            height=1200,
            title_text="Scenario Comparison and Sensitivity Analysis",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Scenario comparison saved to {save_path}")
        
        return fig
    
    def plot_time_series_simulation(self, time_series_data: Dict, save_path=None):
        """Create time series simulation plot"""
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Weather Parameters', 'Fire Weather Indices', 'Risk Score Over Time'),
            vertical_spacing=0.05
        )
        
        # Weather parameters
        fig.add_trace(
            go.Scatter(x=time_series_data['dates'], y=time_series_data['temperature'],
                      mode='lines', name='Temperature (°C)', line=dict(color='red')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_series_data['dates'], y=time_series_data['humidity'],
                      mode='lines', name='Humidity (%)', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_series_data['dates'], y=time_series_data['wind_speed'],
                      mode='lines', name='Wind Speed (m/s)', line=dict(color='green')),
            row=1, col=1
        )
        
        # Fire indices
        fwi_values = [indices['FWI'] for indices in time_series_data['fire_indices']]
        ffmc_values = [indices['FFMC'] for indices in time_series_data['fire_indices']]
        
        fig.add_trace(
            go.Scatter(x=time_series_data['dates'], y=fwi_values,
                      mode='lines', name='FWI', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=time_series_data['dates'], y=ffmc_values,
                      mode='lines', name='FFMC', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Risk score
        fig.add_trace(
            go.Scatter(x=time_series_data['dates'], y=time_series_data['risk_scores'],
                      mode='lines', name='Risk Score', line=dict(color='red', width=2)),
            row=3, col=1
        )
        
        # Add risk level zones
        fig.add_hline(y=0.25, line_dash="dash", line_color="yellow", row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", row=3, col=1)
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=0.85, line_dash="dash", line_color="darkred", row=3, col=1)
        
        fig.update_layout(
            height=900,
            title_text=f"Time Series Simulation - {time_series_data['trend'].title()} Trend",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Time series simulation saved to {save_path}")
        
        return fig
    
    def create_risk_projection(self, base_features: np.ndarray, 
                            projection_days: int = 30) -> Dict:
        """Create risk projection based on historical patterns and climate trends"""
        
        # Generate projection dates
        dates = pd.date_range(start=datetime.now(), periods=projection_days, freq='D')
        
        # Simulate seasonal patterns and climate change effects
        day_of_year = [d.dayofyear for d in dates]
        
        # Base seasonal patterns
        temp_seasonal = 20 + 15 * np.sin(np.array(day_of_year) * 2 * np.pi / 365 - np.pi/2)
        humidity_seasonal = 60 - 20 * np.sin(np.array(day_of_year) * 2 * np.pi / 365 - np.pi/2)
        
        # Add climate change trend (warming)
        climate_trend = np.linspace(0, 2, projection_days)  # 2°C warming over period
        
        # Add random variations
        temp_variation = np.random.normal(0, 3, projection_days)
        humidity_variation = np.random.normal(0, 10, projection_days)
        
        # Final values
        projected_temp = temp_seasonal + climate_trend + temp_variation
        projected_humidity = humidity_seasonal - climate_trend + humidity_variation  # Inverse relationship
        projected_wind = np.random.normal(10, 5, projection_days)
        projected_rain = np.random.exponential(2, projection_days)
        
        # Ensure realistic bounds
        projected_temp = np.clip(projected_temp, 0, 50)
        projected_humidity = np.clip(projected_humidity, 10, 100)
        projected_wind = np.clip(projected_wind, 0, 40)
        projected_rain = np.clip(projected_rain, 0, 20)
        
        # Calculate projected risk
        projected_risk = []
        projected_fwi = []
        
        for i in range(projection_days):
            features = base_features.copy()
            features[0] = projected_temp[i]
            features[1] = projected_humidity[i]
            features[2] = projected_wind[i]
            features[3] = projected_rain[i]
            
            # Update interaction features
            features[10] = projected_temp[i] * (100 - projected_humidity[i])
            features[11] = projected_wind[i] * projected_rain[i]
            
            # Calculate fire indices
            from weather_api import FireWeatherIndexCalculator
            calculator = FireWeatherIndexCalculator()
            
            fire_indices = calculator.calculate_all_indices(
                type('WeatherData', (), {
                    'temperature': projected_temp[i],
                    'humidity': projected_humidity[i],
                    'wind_speed': projected_wind[i],
                    'rainfall': projected_rain[i]
                })()
            )
            
            features[4] = fire_indices['FFMC']
            features[5] = fire_indices['DMC']
            features[6] = fire_indices['DC']
            features[7] = fire_indices['ISI']
            features[8] = fire_indices['BUI']
            features[9] = fire_indices['FWI']
            
            # Calculate risk
            if self.model and self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
                risk = float(self.model.predict(features_scaled)[0])
                risk = np.clip(risk, 0, 1)
            else:
                risk = self.calculate_simple_risk(features)
            
            projected_risk.append(risk)
            projected_fwi.append(fire_indices['FWI'])
        
        return {
            'dates': dates,
            'temperature': projected_temp,
            'humidity': projected_humidity,
            'wind_speed': projected_wind,
            'rainfall': projected_rain,
            'risk_scores': np.array(projected_risk),
            'fwi_values': np.array(projected_fwi),
            'climate_trend': climate_trend
        }
    
    def generate_simulation_report(self, simulation_results: Dict, save_path='simulation_report.html'):
        """Generate comprehensive simulation report"""
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Wildfire Risk Simulation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                .header { background-color: #ff5722; color: white; padding: 20px; border-radius: 10px; text-align: center; }
                .section { background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .risk-high { border-left: 5px solid #d32f2f; }
                .risk-moderate { border-left: 5px solid #ff9800; }
                .risk-low { border-left: 5px solid #4caf50; }
                .metric { display: inline-block; text-align: center; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 8px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #ff5722; }
                .metric-label { font-size: 12px; color: #666; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .sensitivity-bar { width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; }
                .sensitivity-fill { height: 100%; background-color: #ff5722; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Wildfire Risk Simulation Report</h1>
                <p>Generated on """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <div class="metric-value">""" + f"{np.mean(simulation_results.get('risk_scores', [0])):.3f}" + """</div>
                    <div class="metric-label">Average Risk Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">""" + f"{np.max(simulation_results.get('risk_scores', [0])):.3f}" + """</div>
                    <div class="metric-label">Peak Risk Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">""" + f"{len(simulation_results.get('risk_scores', [0]))}" + """</div>
                    <div class="metric-label">Data Points Analyzed</div>
                </div>
            </div>
        """
        
        # Add scenario analysis if available
        if 'scenarios' in simulation_results:
            html_content += """
            <div class="section">
                <h2>Scenario Analysis</h2>
                <table>
                    <tr><th>Scenario</th><th>Description</th><th>Max Risk</th><th>Risk Increase</th></tr>
            """
            
            for scenario_name, scenario_data in simulation_results['scenarios'].items():
                max_risk = 0
                for param_result in scenario_data['results'].values():
                    max_risk = max(max_risk, np.max(param_result['risk_scores']))
                
                html_content += f"""
                <tr>
                    <td>{scenario_name}</td>
                    <td>{scenario_data['description']}</td>
                    <td>{max_risk:.3f}</td>
                    <td>{max_risk - 0.5:.3f}</td>
                </tr>
                """
            
            html_content += "</table></div>"
        
        # Add sensitivity analysis if available
        if 'sensitivity' in simulation_results:
            html_content += """
            <div class="section">
                <h2>Sensitivity Analysis</h2>
                <table>
                    <tr><th>Parameter</th><th>Sensitivity</th><th>Impact</th></tr>
            """
            
            for param_name, sensitivity in simulation_results['sensitivity']['ranking'][:10]:
                impact = "High" if abs(sensitivity) > 0.01 else "Medium" if abs(sensitivity) > 0.005 else "Low"
                html_content += f"""
                <tr>
                    <td>{param_name}</td>
                    <td>{sensitivity:.6f}</td>
                    <td>{impact}</td>
                </tr>
                """
            
            html_content += "</table></div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        print(f"Simulation report saved to {save_path}")
        return save_path

def demo_scenario_simulator():
    """Demonstration of scenario simulator"""
    print("="*60)
    print("SCENARIO SIMULATOR AND TREND ANALYSIS")
    print("="*60)
    
    # Initialize simulator
    simulator = ScenarioSimulator()
    
    # Create base features (current conditions)
    base_features = np.array([
        30,    # Temperature
        40,    # RH
        15,    # Ws
        0.5,   # Rain
        75,    # FFMC
        20,    # DMC
        60,    # DC
        6,     # ISI
        25,    # BUI
        12,    # FWI
        30 * 60,  # Temp_RH_Index
        15 * 0.5  # Wind_Rain_Interaction
    ])
    
    print("Base conditions:")
    print(f"Temperature: {base_features[0]}°C")
    print(f"Humidity: {base_features[1]}%")
    print(f"Wind Speed: {base_features[2]} m/s")
    print(f"Rain: {base_features[3]} mm")
    print(f"FWI: {base_features[9]}")
    
    # What-if scenarios
    print("\nGenerating what-if scenarios...")
    scenarios = simulator.create_what_if_scenarios(base_features)
    
    # Scenario comparison plot
    fig1 = simulator.plot_scenario_comparison(scenarios, 'outputs/scenario_comparison.html')
    
    # Time series simulation
    print("\nSimulating time series trends...")
    trends = ['normal', 'warming', 'storm', 'extreme']
    
    for trend in trends:
        print(f"Simulating {trend} trend...")
        time_data = simulator.simulate_time_series(base_features, days=5, weather_trend=trend)
        fig = simulator.plot_time_series_simulation(time_data, f'outputs/time_series_{trend}.html')
    
    # Sensitivity analysis
    print("\nPerforming sensitivity analysis...")
    sensitivity_results = simulator.create_sensitivity_analysis(base_features)
    
    # Risk projection
    print("\nGenerating 30-day risk projection...")
    projection = simulator.create_risk_projection(base_features, projection_days=30)
    
    # Create projection plot
    fig2 = simulator.plot_time_series_simulation(projection, 'outputs/risk_projection.html')
    
    # Generate comprehensive report
    simulation_results = {
        'scenarios': scenarios,
        'sensitivity': sensitivity_results,
        'projection': projection,
        'risk_scores': projection['risk_scores']
    }
    
    report_path = simulator.generate_simulation_report(simulation_results, 'outputs/simulation_report.html')
    
    print(f"\n{'='*60}")
    print("Scenario simulation complete!")
    print("- Scenario comparison: outputs/scenario_comparison.html")
    print("- Time series simulations: outputs/time_series_*.html")
    print("- Risk projection: outputs/risk_projection.html")
    print("- Full report: outputs/simulation_report.html")
    
    # Show key insights
    print(f"\nKey Insights:")
    print(f"- Most sensitive parameter: {sensitivity_results['ranking'][0][0]}")
    print(f"- Projected peak risk: {np.max(projection['risk_scores']):.3f}")
    print(f"- Average projected risk: {np.mean(projection['risk_scores']):.3f}")

if __name__ == "__main__":
    demo_scenario_simulator()
