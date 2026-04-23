"""
Streamlit Frontend
Professional UI for Wildfire Risk Prediction System
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.pipeline import DataPipeline
from backend.api.weather import WeatherAPI
from backend.services.decision import DecisionEngine
from backend.services.regional_scanner import RegionalScanner
from backend.services.simulation import SimulationEngine
from backend.services.database import HistoryDatabase
from backend.services.explainability import ExplainabilityEngine
from backend.services.alerts import AlertSystem
from backend.utils.logger import system_logger

# Page configuration
st.set_page_config(
    page_title="Wildfire Risk Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .risk-no_fire { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .risk-low_fire { background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%); }
    .risk-medium_fire { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .risk-high_fire { background: linear-gradient(135deg, #ff512f 0%, #dd2476 100%); }
    .risk-extreme_fire { background: linear-gradient(135deg, #c31432 0%, #240b36 100%); }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'weather_api' not in st.session_state:
    st.session_state.weather_api = None
if 'database' not in st.session_state:
    st.session_state.database = HistoryDatabase()
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'api_key_submitted' not in st.session_state:
    st.session_state.api_key_submitted = False
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

# Sidebar
with st.sidebar:
    st.title("🔥 Wildfire Risk System")
    
    # API Key input with submit and edit functionality
    if not st.session_state.api_key_submitted:
        api_key = st.text_input("OpenWeatherMap API Key", type="password", 
                               help="Enter your OpenWeatherMap API key for real-time data")
        
        if st.button("Submit API Key", use_container_width=True):
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.api_key_submitted = True
                st.session_state.weather_api = WeatherAPI(api_key)
                st.success("✅ API Key submitted and locked!")
                st.rerun()
            else:
                st.error("❌ Please enter an API key")
    else:
        # Show locked API key with edit option
        st.info("🔒 API Key is locked")
        masked_key = st.session_state.api_key[:8] + "..." + st.session_state.api_key[-4:] if len(st.session_state.api_key) > 12 else "****"
        st.text(f"Key: {masked_key}")
        
        if st.button("Edit API Key", use_container_width=True):
            st.session_state.api_key_submitted = False
            st.session_state.api_key = ''
            st.session_state.weather_api = None
            st.rerun()
    
    st.divider()
    
    # Model loading
    if not st.session_state.models_loaded:
        if st.button("Load Trained Models", use_container_width=True):
            with st.spinner("Loading models..."):
                try:
                    pipeline = DataPipeline(model_dir='models')
                    if pipeline.load_models():
                        st.session_state.pipeline = pipeline
                        st.session_state.models_loaded = True
                        st.success("✅ Models loaded successfully!")
                    else:
                        st.error("❌ Failed to load models. Please train first.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.models_loaded:
        st.success("✅ Models loaded")
    
    st.divider()
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["📍 Single Location", "🗺️ Regional Scanner", "📊 Simulation", "📈 History", "⚙️ Settings"],
        label_visibility="collapsed"
    )

# Main header
st.markdown('<h1 class="main-header">🔥 Intelligent Wildfire Risk Prediction System</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Single Location Page
if page == "📍 Single Location":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first from the sidebar")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📍 Location Input")
            
            # Input method selection
            input_method = st.radio("Input Method", ["Use Coordinates", "Use City Name"])
            
            if input_method == "Use Coordinates":
                lat = st.number_input("Latitude", value=28.61, min_value=-90.0, max_value=90.0, key="lat_input")
                lon = st.number_input("Longitude", value=77.23, min_value=-180.0, max_value=180.0, key="lon_input")
                
                # Auto-trigger analysis when coordinates change
                if 'last_coords' not in st.session_state:
                    st.session_state.last_coords = (lat, lon)
                elif st.session_state.last_coords != (lat, lon):
                    st.session_state.last_coords = (lat, lon)
                    st.session_state.analyze = True
                
                location_name = f"Lat: {lat:.2f}, Lon: {lon:.2f}"
            else:
                indian_cities = [
                    "New Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru",
                    "Hyderabad", "Ahmedabad", "Jaipur", "Lucknow", "Pune",
                    "Dehradun", "Shimla", "Srinagar", "Gangtok", "Guwahati",
                    "Bhubaneswar", "Raipur", "Bhopal", "Nagpur", "Thiruvananthapuram",
                    "Amaravati", "Panaji", "Agartala", "Kohima", "Imphal"
                ]
                city = st.selectbox("Select Indian City", indian_cities, index=0)
                
                # City coordinates
                city_coords = {
                    "New Delhi": (28.61, 77.23),
                    "Mumbai": (19.07, 72.87),
                    "Kolkata": (22.57, 88.36),
                    "Chennai": (13.08, 80.27),
                    "Bengaluru": (12.97, 77.59),
                    "Hyderabad": (17.38, 78.48),
                    "Ahmedabad": (23.02, 72.57),
                    "Jaipur": (26.91, 75.78),
                    "Lucknow": (26.84, 80.94),
                    "Pune": (18.52, 73.85),
                    "Dehradun": (30.33, 78.06),
                    "Shimla": (31.10, 77.17),
                    "Srinagar": (33.78, 76.08),
                    "Gangtok": (27.53, 88.51),
                    "Guwahati": (26.20, 91.77),
                    "Bhubaneswar": (20.29, 85.82),
                    "Raipur": (21.25, 81.62),
                    "Bhopal": (23.25, 77.41),
                    "Nagpur": (21.14, 79.08),
                    "Thiruvananthapuram": (8.52, 76.93),
                    "Amaravati": (16.50, 80.51),
                    "Panaji": (15.49, 73.82),
                    "Agartala": (23.84, 91.28),
                    "Kohima": (26.15, 94.11),
                    "Imphal": (24.80, 93.93)
                }
                
                lat, lon = city_coords.get(city, (28.61, 77.23))
                location_name = city
            
            month = st.selectbox("Month", list(range(1, 13)), index=5, key="month_select", format_func=lambda x: datetime(2024, x, 1).strftime('%B'))
        
        with col2:
            st.subheader("⚡ Quick Actions")
            if st.button("🔍 Analyze Risk", use_container_width=True, type="primary"):
                st.session_state.analyze = True
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.session_state.analyze = True
        
        # Analysis section
        if st.session_state.get('analyze', False):
            with st.spinner("Analyzing wildfire risk..."):
                try:
                    # Get weather data
                    if st.session_state.weather_api:
                        weather_data = st.session_state.weather_api.get_weather(lat, lon)
                    else:
                        weather_data = {
                            'temperature': 30.0,
                            'humidity': 40.0,
                            'wind_speed': 15.0,
                            'rainfall': 0.0
                        }
                    
                    # Make prediction
                    prediction = st.session_state.pipeline.predict_pipeline(weather_data, month, location=location_name)
                    
                    # Make decision
                    decision_engine = DecisionEngine()
                    decision = decision_engine.make_decision(prediction)
                    
                    # Save to database
                    st.session_state.database.save_prediction({
                        'location': location_name,
                        'latitude': lat,
                        'longitude': lon,
                        'weather': weather_data,
                        'risk_score': decision['risk_score'],
                        'risk_level': decision['linguistic_risk_level'],
                        'confidence': decision['confidence'],
                        'action': decision['action'],
                        'fwi_components': prediction['fwi_components']
                    })
                    
                    # Display results
                    st.markdown("---")
                    
                    # Risk score display with linguistic output
                    risk_class = f"risk-{decision['linguistic_risk_level'].lower().replace(' ', '_')}"
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h2 style="margin:0; font-size:3rem;">{decision['linguistic_risk_level']}</h2>
                        <p style="margin:0.5rem 0 0 0; font-size:1.2rem;">Risk Score: {decision['risk_score']:.2f}</p>
                        <p style="margin:0.5rem 0 0 0; opacity:0.9;">{decision['risk_description']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Confidence", f"{decision['confidence']:.1%}")
                    with col2:
                        st.metric("Early Warning Score", f"{decision['early_warning_score']:.2f}")
                    with col3:
                        st.metric("Temperature", f"{weather_data['temperature']:.1f}°C")
                    with col4:
                        st.metric("Humidity", f"{weather_data['humidity']:.1f}%")
                    
                    # FWI Components
                    st.subheader("📊 FWI Components")
                    fwi_data = prediction['fwi_components']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("FFMC", f"{fwi_data['FFMC']:.1f}")
                        st.metric("DMC", f"{fwi_data['DMC']:.1f}")
                    with col2:
                        st.metric("DC", f"{fwi_data['DC']:.1f}")
                        st.metric("ISI", f"{fwi_data['ISI']:.1f}")
                    with col3:
                        st.metric("BUI", f"{fwi_data['BUI']:.1f}")
                        st.metric("FWI", f"{fwi_data['FWI']:.1f}")
                    
                    # Action recommendation with dynamic, user-friendly expressions
                    st.subheader("🎯 Recommended Action")
                    
                    # Dynamic action based on risk level
                    risk_level = prediction['linguistic_risk_level']
                    risk_score = prediction['risk_score']
                    
                    # Create dynamic action message
                    if risk_level == 'No Fire':
                        action_color = "🟢"
                        action_text = "**Safe Conditions**"
                        action_detail = "Current weather conditions are safe. No fire risk detected. Continue normal operations and maintain regular monitoring."
                        icon = "✅"
                    elif risk_level == 'Low Fire':
                        action_color = "🟡"
                        action_text = "**Monitor Conditions**"
                        action_detail = "Low fire risk present. Maintain regular monitoring and ensure fire prevention measures are in place. Be prepared for changing conditions."
                        icon = "👁️"
                    elif risk_level == 'Medium Fire':
                        action_color = "🟠"
                        action_text = "**Prepare for Potential Fire**"
                        action_detail = "Medium fire risk detected. Review emergency procedures, ensure firefighting equipment is ready, and increase monitoring frequency. Prepare contingency plans."
                        icon = "⚠️"
                    elif risk_level == 'High Fire':
                        action_color = "🔴"
                        action_text = "**Take Immediate Action**"
                        action_detail = "High fire risk! Activate emergency protocols, alert authorities, evacuate vulnerable areas, and mobilize firefighting resources immediately."
                        icon = "🚨"
                    else:  # Extreme Fire
                        action_color = "🔴"
                        action_text = "**CRITICAL: Extreme Fire Danger**"
                        action_detail = "EXTREME fire risk! Maximum emergency response required. Immediate evacuation orders, full mobilization of all resources, and activation of regional emergency coordination."
                        icon = "💥"
                    
                    # Display dynamic action
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 20px; border-radius: 10px; color: white; margin: 10px 0;">
                        <h3 style="margin: 0 0 10px 0;">{action_color} {action_text}</h3>
                        <p style="margin: 0; font-size: 16px;">{action_detail}</p>
                        <p style="margin: 10px 0 0 0; font-size: 14px; opacity: 0.9;">
                            <strong>Risk Score:</strong> {risk_score:.2f} | <strong>Confidence:</strong> {(prediction.get('fuzzy_output', risk_score) * 100):.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Real-time status indicator
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 10px; margin: 15px 0;">
                        <div style="width: 12px; height: 12px; background: #00ff00; border-radius: 50%; 
                                    animation: pulse 2s infinite;"></div>
                        <span style="color: #666; font-size: 14px;">Live monitoring active • Last updated: Just now</span>
                    </div>
                    <style>
                        @keyframes pulse {{
                            0% {{ opacity: 1; }}
                            50% {{ opacity: 0.5; }}
                            100% {{ opacity: 1; }}
                        }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Feature contribution with user-friendly explanations
                    st.subheader("🔍 Feature Contribution Analysis")
                    
                    # Get contributions from fuzzy system
                    fuzzy_details = prediction.get('fuzzy_details', {})
                    output_scores = fuzzy_details.get('output_scores', {})
                    
                    # Create user-friendly feature analysis
                    st.markdown("""
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        <h4 style="margin: 0 0 10px 0;">📊 What's driving this risk level?</h4>
                        <p style="margin: 0; color: #666; font-size: 14px;">
                            The system analyzes multiple environmental factors to determine fire risk. 
                            Below shows how each factor contributed to today's risk assessment.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    contributions = decision_engine.get_feature_contribution(prediction)
                    
                    # Create real-time animated bar chart
                    contrib_df = pd.DataFrame([
                        {'Feature': k, 'Contribution': v * 100} for k, v in contributions.items()
                    ]).sort_values('Contribution', ascending=False)
                    
                    # Add color column based on contribution
                    colors = []
                    for contrib in contrib_df['Contribution']:
                        if contrib > 30:
                            colors.append('#ef4444')  # Red
                        elif contrib > 15:
                            colors.append('#f97316')  # Orange
                        else:
                            colors.append('#22c55e')  # Green
                    
                    contrib_df['Color'] = colors
                    
                    # Create animated bar chart
                    fig = px.bar(contrib_df, x='Feature', y='Contribution', 
                                color='Feature',
                                color_discrete_map={row['Feature']: row['Color'] for _, row in contrib_df.iterrows()},
                                animation_frame=None,
                                range_y=[0, max(contrib_df['Contribution']) * 1.2],
                                title='Feature Contribution to Fire Risk (%)')
                    
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        xaxis_title="",
                        yaxis_title="Contribution (%)",
                        font=dict(size=12),
                        margin=dict(l=20, r=20, t=40, b=80),
                        xaxis={'tickangle': -45}
                    )
                    
                    fig.update_traces(
                        marker_line_width=0,
                        opacity=0.9,
                        hovertemplate='<b>%{x}</b><br>Contribution: %{y:.1f}%<extra></extra>'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create user-friendly feature names and explanations
                    feature_explanations = {
                        'Temperature': {
                            'name': 'Temperature',
                            'icon': '🌡️',
                            'explanation': 'Higher temperatures increase fire risk by drying out vegetation and making it easier for fires to start and spread.'
                        },
                        'Humidity': {
                            'name': 'Humidity',
                            'icon': '💧',
                            'explanation': 'Lower humidity means drier air and vegetation, which significantly increases fire risk.'
                        },
                        'Wind Speed': {
                            'name': 'Wind Speed',
                            'icon': '💨',
                            'explanation': 'Strong winds can rapidly spread fires and make them harder to control.'
                        },
                        'Rainfall': {
                            'name': 'Rainfall',
                            'icon': '🌧️',
                            'explanation': 'Rain reduces fire risk by wetting vegetation and increasing humidity.'
                        },
                        'FFMC': {
                            'name': 'Fine Fuel Moisture Code',
                            'icon': '🌿',
                            'explanation': 'Indicates moisture in light fuels (leaves, needles). Higher values mean drier fuels and higher fire risk.'
                        },
                        'DMC': {
                            'name': 'Duff Moisture Code',
                            'icon': '🪵',
                            'explanation': 'Measures moisture in loosely compacted organic matter. Affects fire intensity and spread.'
                        },
                        'DC': {
                            'name': 'Drought Code',
                            'icon': '🏜️',
                            'explanation': 'Indicates long-term dryness. Higher values mean deeper drought conditions and higher fire risk.'
                        }
                    }
                    
                    # Display feature contributions with explanations in creative cards
                    for feature, contribution in sorted(contributions.items(), key=lambda x: x[1], reverse=True):
                        if feature in feature_explanations:
                            info = feature_explanations[feature]
                            contribution_pct = contribution * 100
                            
                            # Color based on contribution level
                            if contribution_pct > 30:
                                bar_color = "#ef4444"  # Red
                                bg_gradient = "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)"
                                icon_bg = "#dc2626"
                            elif contribution_pct > 15:
                                bar_color = "#f97316"  # Orange
                                bg_gradient = "linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%)"
                                icon_bg = "#ea580c"
                            else:
                                bar_color = "#22c55e"  # Green
                                bg_gradient = "linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%)"
                                icon_bg = "#16a34a"
                            
                            st.markdown(f"""
                            <div style="margin: 20px 0; padding: 0; background: {bg_gradient}; 
                                        border-radius: 16px; overflow: hidden; 
                                        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                                        transition: all 0.3s ease;">
                                <div style="padding: 20px;">
                                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px;">
                                        <div style="display: flex; align-items: center; gap: 15px;">
                                            <div style="width: 50px; height: 50px; background: {icon_bg}; 
                                                       border-radius: 12px; display: flex; align-items: center; 
                                                       justify-content: center; font-size: 24px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                                                {info['icon']}
                                            </div>
                                            <div>
                                                <h4 style="margin: 0; font-size: 18px; font-weight: 700; color: #1f2937;">{info['name']}</h4>
                                                <p style="margin: 5px 0 0 0; color: #6b7280; font-size: 13px;">Key Risk Factor</p>
                                            </div>
                                        </div>
                                        <div style="text-align: right;">
                                            <div style="font-size: 32px; font-weight: 800; color: {bar_color}; line-height: 1;">
                                                {contribution_pct:.1f}%
                                            </div>
                                            <p style="margin: 0; color: #6b7280; font-size: 12px;">Contribution</p>
                                        </div>
                                    </div>
                                    
                                    <div style="background: rgba(255,255,255,0.7); border-radius: 12px; padding: 12px; margin: 15px 0;">
                                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                            <span style="font-size: 12px; font-weight: 600; color: #374151;">Impact Level</span>
                                            <span style="font-size: 12px; font-weight: 600; color: {bar_color};">
                                                {'High Impact' if contribution_pct > 30 else 'Medium Impact' if contribution_pct > 15 else 'Low Impact'}
                                            </span>
                                        </div>
                                        <div style="background: #e5e7eb; border-radius: 6px; height: 12px; overflow: hidden;">
                                            <div style="background: {bar_color}; width: {min(contribution_pct, 100)}%; height: 100%; 
                                                       border-radius: 6px; position: relative;">
                                                <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                                                           background: linear-gradient(90deg, rgba(255,255,255,0.3) 0%, transparent 100%);"></div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <p style="margin: 0; color: #4b5563; font-size: 14px; line-height: 1.5;">
                                        <strong style="color: #1f2937;">What this means:</strong> {info['explanation']}
                                    </p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Enhanced fuzzy reasoning section with creative design
                    reasoning = prediction.get('reasoning', '')
                    if reasoning:
                        st.markdown("""
                        <div style="margin: 30px 0; padding: 0;">
                            <div style="background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); 
                                        padding: 25px; border-radius: 16px; color: white; 
                                        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);">
                                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
                                    <div style="width: 60px; height: 60px; background: rgba(255,255,255,0.2); 
                                               border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 32px;">
                                        🧠
                                    </div>
                                    <div>
                                        <h3 style="margin: 0; font-size: 24px; font-weight: 700;">Fuzzy Logic Reasoning</h3>
                                        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 14px;">AI-Powered Decision Making</p>
                                    </div>
                                </div>
                                
                                <div style="background: rgba(255,255,255,0.15); border-radius: 12px; padding: 20px; margin: 15px 0;">
                                    <h4 style="margin: 0 0 10px 0; font-size: 16px; font-weight: 600; opacity: 0.95;">
                                        🔍 Why this risk level?
                                    </h4>
                                    <p style="margin: 0; line-height: 1.6; font-size: 15px; opacity: 0.95;">
                                        {reasoning}
                                    </p>
                                </div>
                                
                                <div style="display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap;">
                                    <div style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 12px;">
                                        ✅ Real-time Analysis
                                    </div>
                                    <div style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 12px;">
                                        🎯 Rule-Based Logic
                                    </div>
                                    <div style="background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 20px; font-size: 12px;">
                                        📊 Weighted Scoring
                                    </div>
                                </div>
                            </div>
                        </div>
                        """.format(reasoning=reasoning), unsafe_allow_html=True)
                    
                    # Alert if high risk
                    if decision['risk_score'] > 0.75:
                        st.error(f"🚨 ALERT: {decision['action_message']}")
                    
                    # Clear analyze flag
                    st.session_state.analyze = False
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    system_logger.error(f"Frontend analysis error: {str(e)}")

# Regional Scanner Page
elif page == "🗺️ Regional Scanner":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first from the sidebar")
    else:
        st.subheader("🗺️ Regional Wildfire Risk Scanner")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            month = st.selectbox("Month", list(range(1, 13)), index=5, 
                               format_func=lambda x: datetime(2024, x, 1).strftime('%B'))
            max_regions = st.slider("Number of Regions", 5, 25, 25)
        
        with col2:
            if st.button("🔍 Scan Regions", use_container_width=True, type="primary"):
                st.session_state.scan = True
        
        if st.session_state.get('scan', False):
            with st.spinner("Scanning regions..."):
                try:
                    scanner = RegionalScanner(st.session_state.pipeline, st.session_state.weather_api)
                    results = scanner.scan_all_regions(month, max_regions)
                    summary = scanner.get_regional_summary(results)
                    
                    # Check for errors in summary
                    if 'error' in summary:
                        st.error(f"Error: {summary['error']}")
                    else:
                        # Display summary
                        st.subheader("📊 Regional Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Average Risk", f"{summary['average_risk']:.2f}")
                        col2.metric("Max Risk", f"{summary['max_risk']:.2f}")
                        col3.metric("High Risk Regions", len(summary['high_risk_regions']))
                        
                        # Display results table
                        st.subheader("📍 Regional Results")
                        
                        df_data = []
                        for r in results:
                            df_data.append({
                                'Region': r['region_name'],
                                'Risk Score': f"{r['decision']['risk_score']:.2f}",
                                'Fire Risk': r['decision']['linguistic_risk_level'],
                                'Temperature': f"{r['weather']['temperature']:.1f}°C",
                                'Humidity': f"{r['weather']['humidity']:.1f}%",
                                'Wind': f"{r['weather']['wind_speed']:.1f} km/h"
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Risk map visualization
                        st.subheader("🗺️ Risk Map")
                        
                        map_data = []
                        for r in results:
                            map_data.append({
                                'lat': r['latitude'],
                                'lon': r['longitude'],
                                'name': r['region_name'],
                                'risk': r['decision']['risk_score'],
                                'level': r['decision']['linguistic_risk_level']
                            })
                        
                        map_df = pd.DataFrame(map_data)
                        
                        fig = px.scatter_mapbox(
                            map_df,
                            lat='lat',
                            lon='lon',
                            color='risk',
                            size='risk',
                            hover_name='name',
                            color_continuous_scale='Viridis',
                            zoom=5,
                            mapbox_style="open-street-map"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error during scan: {str(e)}")
                    system_logger.error(f"Regional scan error: {str(e)}")
            
            # Clear scan flag
            st.session_state.scan = False

# Simulation Page
elif page == "📊 Simulation":
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first from the sidebar")
    else:
        st.subheader("📊 Risk Simulation & Analysis")
        
        sim_type = st.radio("Simulation Type", ["Trend Analysis", "Scenario Simulation", "Comparative Analysis"])
        
        if sim_type == "Trend Analysis":
            st.subheader("📈 Trend Analysis")
            
            feature = st.selectbox("Feature to Analyze", ['temperature', 'humidity', 'wind_speed', 'rainfall'])
            base_temp = st.slider("Base Temperature (°C)", 0, 50, 30)
            base_humidity = st.slider("Base Humidity (%)", 0, 100, 40)
            base_wind = st.slider("Base Wind Speed (km/h)", 0, 50, 15)
            base_rain = st.slider("Base Rainfall (mm)", 0, 50, 0)
            
            month = st.selectbox("Month", list(range(1, 13)), index=5)
            
            if st.button("🔍 Analyze Trend", type="primary"):
                with st.spinner("Analyzing trend..."):
                    try:
                        base_weather = {
                            'temperature': base_temp,
                            'humidity': base_humidity,
                            'wind_speed': base_wind,
                            'rainfall': base_rain
                        }
                        
                        sim_engine = SimulationEngine(st.session_state.pipeline)
                        trend = sim_engine.trend_analysis(base_weather, month, feature)
                        
                        # Plot trend
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=trend['values'],
                            y=trend['risk_scores'],
                            mode='lines+markers',
                            name='Risk Score',
                            line=dict(color='firebrick', width=3)
                        ))
                        
                        fig.update_layout(
                            title=f"Risk vs {feature.capitalize()}",
                            xaxis_title=feature.capitalize(),
                            yaxis_title="Risk Score",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.metric("Sensitivity", f"{trend['sensitivity']:.4f}")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        elif sim_type == "Scenario Simulation":
            st.subheader("🎬 Scenario Simulation")
            
            base_temp = st.slider("Base Temperature (°C)", 0, 50, 30)
            base_humidity = st.slider("Base Humidity (%)", 0, 100, 40)
            base_wind = st.slider("Base Wind Speed (km/h)", 0, 50, 15)
            base_rain = st.slider("Base Rainfall (mm)", 0, 50, 0)
            
            month = st.selectbox("Month", list(range(1, 13)), index=5, key='sim_month')
            
            if st.button("🔬 Run Scenarios", type="primary"):
                with st.spinner("Running simulations..."):
                    try:
                        base_weather = {
                            'temperature': base_temp,
                            'humidity': base_humidity,
                            'wind_speed': base_wind,
                            'rainfall': base_rain
                        }
                        
                        sim_engine = SimulationEngine(st.session_state.pipeline)
                        scenarios = sim_engine.scenario_simulation(base_weather, month)
                        
                        # Display results
                        for scenario in scenarios:
                            risk_class = f"risk-{scenario['decision']['linguistic_risk_level'].lower().replace(' ', '_')}"
                            st.markdown(f"""
                            <div class="metric-card {risk_class}" style="padding:1rem;">
                                <h3 style="margin:0;">{scenario['scenario_name']}</h3>
                                <p style="margin:0.5rem 0 0 0;">{scenario['decision']['linguistic_risk_level']} (Risk: {scenario['decision']['risk_score']:.2f})</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# History Page
elif page == "📈 History":
    st.subheader("📈 Prediction History")
    
    # Get recent predictions
    predictions = st.session_state.database.get_recent_predictions(50)
    
    if predictions:
        # Convert to DataFrame
        df_data = []
        for p in predictions:
            df_data.append({
                'Timestamp': p['timestamp'],
                'Location': p['location'],
                'Risk Score': f"{p['risk_score']:.2f}",
                'Fire Risk': p.get('risk_level', 'Unknown'),
                'Confidence': f"{p['confidence']:.1%}"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Statistics
        stats = st.session_state.database.get_statistics()
        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", stats['total_predictions'])
        col2.metric("Total Alerts", stats['total_alerts'])
        col3.metric("Risk Distribution", str(stats['risk_distribution']))
    else:
        st.info("No prediction history available")

# Settings Page
elif page == "⚙️ Settings":
    st.subheader("⚙️ System Settings")
    
    # Database settings
    st.subheader("🗄️ Database")
    if st.button("🧹 Clear Old Data (30+ days)"):
        st.session_state.database.clear_old_data(30)
        st.success("Old data cleared")
    
    # System info
    st.subheader("ℹ️ System Information")
    if st.session_state.weather_api:
        api_status = st.session_state.weather_api.get_api_status()
        st.json(api_status)
    else:
        st.info("Weather API not configured")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🔥 Intelligent Wildfire Risk Prediction System using PSO-ANFIS</p>
    <p>Powered by Soft Computing: Fuzzy Logic, ANFIS, PSO, SHAP</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")
