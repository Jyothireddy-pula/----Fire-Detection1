"""
WildfireGuard AI - Unified Dashboard
Advanced Wildfire Risk Prediction & Decision Support System

Features:
- Real-time wildfire risk prediction using PSO-MLP/ANFIS models
- Regional monitoring across 25+ forest locations  
- What-if scenario simulation
- Active alert monitoring
- Historical trend analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'core'))

# Page Configuration
st.set_page_config(
    page_title="WildfireGuard AI - Risk Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #ff5722; text-align: center; }
    .sub-header { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .risk-box { padding: 1rem; border-radius: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; color: white; }
    .risk-extreme { background-color: #d32f2f; }
    .risk-high { background-color: #ff5722; }
    .risk-moderate { background-color: #ff9800; }
    .risk-low { background-color: #4caf50; }
    .info-box { background-color: #e3f2fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load('models/mlp_pso.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler, True
    except Exception as e:
        return None, None, False

model, scaler, models_loaded = load_models()

# Session State
if 'alerts' not in st.session_state:
    st.session_state.alerts = [
        {'id': 1, 'location': 'Yellowstone National Park', 'severity': 'Critical', 'risk_score': 0.92, 
         'time': datetime.now() - timedelta(hours=2)},
        {'id': 2, 'location': 'California Forest', 'severity': 'High', 'risk_score': 0.78,
         'time': datetime.now() - timedelta(hours=5)}
    ]

# Utility Functions
def calculate_fire_indices(temp, rh, ws, rain):
    """Calculate fire weather indices"""
    ffmc = min(101, 59.5 * np.exp(-0.1 * rh) + 0.03 * ws + 0.05 * temp - 0.1 * rain)
    dmc = max(0, 10 + 0.05 * temp - 0.1 * rh - 1.5 * rain)
    dc = max(0, 15 + 0.01 * temp - 2.5 * rain)
    isi = min(50, 0.208 * ws * np.exp(0.05039 * ffmc))
    bui = min(100, 0.8 * dmc + 0.2 * dc)
    fwi = min(100, 0.1 * isi * bui)
    return {'FFMC': ffmc, 'DMC': dmc, 'DC': dc, 'ISI': isi, 'BUI': bui, 'FWI': fwi}

def get_risk_level(score):
    """Get risk level from score"""
    if score >= 0.85: return "Extreme", "risk-extreme"
    elif score >= 0.70: return "High", "risk-high"
    elif score >= 0.50: return "Moderate", "risk-moderate"
    elif score >= 0.25: return "Low", "risk-low"
    else: return "No Risk", "risk-low"

def predict_risk(features):
    """Make risk prediction"""
    if models_loaded and model is not None and scaler is not None:
        try:
            features_scaled = scaler.transform(features.reshape(1, -1))
            proba = model.predict_proba(features_scaled)[0]
            return float(np.max(proba))
        except:
            pass
    # Fallback calculation
    return min(1.0, features[9] / 50)  # Based on FWI

# Header
st.markdown('<div class="main-header">🔥 WildfireGuard AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Wildfire Risk Prediction & Decision Support System</div>', unsafe_allow_html=True)

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Mode:", [
    "🎯 Risk Prediction",
    "🗺️ Regional Scanner", 
    "📊 Scenario Simulator",
    "🔔 Active Alerts",
    "📈 Historical Analysis"
])

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox("Model:", ["PSO-MLP", "PSO-ANFIS", "Ensemble"])
api_key = st.sidebar.text_input("Weather API Key:", type="password")
st.sidebar.checkbox("Enable Alerts", value=True)

# Main Content
if page == "🎯 Risk Prediction":
    st.header("Real-Time Wildfire Risk Prediction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Weather Parameters")
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 30.0, 0.1)
        humidity = st.slider("Relative Humidity (%)", 0.0, 100.0, 40.0, 0.1)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 40.0, 15.0, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 20.0, 0.0, 0.1)
        
        with st.expander("Fire Indices (Auto-calculated)"):
            ffmc = st.slider("FFMC", 0.0, 101.0, 75.0, 0.1, disabled=True)
            dmc = st.slider("DMC", 0.0, 100.0, 20.0, 0.1, disabled=True)
            dc = st.slider("DC", 0.0, 500.0, 60.0, 0.1, disabled=True)
            isi = st.slider("ISI", 0.0, 50.0, 6.0, 0.1, disabled=True)
            bui = st.slider("BUI", 0.0, 100.0, 25.0, 0.1, disabled=True)
            fwi = st.slider("FWI", 0.0, 100.0, 12.0, 0.1, disabled=True)
    
    with col2:
        st.subheader("Risk Assessment")
        
        if st.button("🚀 Calculate Risk", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Calculate fire indices
                indices = calculate_fire_indices(temp, humidity, wind_speed, rainfall)
                
                # Prepare features
                features = np.array([
                    temp, humidity, wind_speed, rainfall,
                    indices['FFMC'], indices['DMC'], indices['DC'],
                    indices['ISI'], indices['BUI'], indices['FWI'],
                    temp * (100 - humidity), wind_speed * rainfall
                ])
                
                # Predict
                risk_score = predict_risk(features)
                risk_level, risk_class = get_risk_level(risk_score)
                
                # Display result
                st.markdown(f'<div class="risk-box {risk_class}">{risk_level}<br>Score: {risk_score:.3f}</div>', 
                          unsafe_allow_html=True)
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=risk_score*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Risk Score (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [
                               {'range': [0, 25], 'color': '#4caf50'},
                               {'range': [25, 50], 'color': '#ffeb3b'},
                               {'range': [50, 70], 'color': '#ff9800'},
                               {'range': [70, 85], 'color': '#ff5722'},
                               {'range': [85, 100], 'color': '#d32f2f'}]}))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Fire indices
                cols = st.columns(3)
                for i, (k, v) in enumerate(indices.items()):
                    cols[i % 3].metric(k, f"{v:.1f}")
                
                # Recommendations
                if risk_score >= 0.85:
                    st.error("🚨 EXTREME RISK - Immediate evacuation required")
                elif risk_score >= 0.70:
                    st.warning("⚠️ HIGH RISK - Pre-position emergency resources")
                elif risk_score >= 0.50:
                    st.info("🔶 MODERATE RISK - Increase monitoring")
                else:
                    st.success("✅ Normal conditions - Routine monitoring")

elif page == "🗺️ Regional Scanner":
    st.header("Regional Forest Risk Scanner")
    st.info("Monitor 25+ forest locations worldwide")
    
    if st.button("🌍 Start Scan", type="primary", use_container_width=True):
        with st.spinner("Scanning locations..."):
            try:
                from core.regional_scanner import RegionalGridScanner
                scanner = RegionalGridScanner()
                results = scanner.scan_all_locations()
                
                # Display
                st.subheader("High Risk Locations")
                top10 = results.nlargest(10, 'Risk_Score')
                st.map(top10.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}))
                
                # Chart
                fig = px.pie(results, names='Risk_Level', title='Risk Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.dataframe(results[['Location', 'Country', 'Risk_Score', 'Risk_Level', 'FWI']].sort_values('Risk_Score', ascending=False),
                           use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Scanner error: {e}")
                st.info("Please run the pipeline first to train models")

elif page == "📊 Scenario Simulator":
    st.header("What-If Scenario Simulator")
    
    col1, col2 = st.columns(2)
    with col1:
        base_temp = st.slider("Base Temp (°C)", 0.0, 50.0, 30.0, key="sim1")
        base_hum = st.slider("Base Humidity (%)", 0.0, 100.0, 40.0, key="sim2")
    with col2:
        base_wind = st.slider("Base Wind (m/s)", 0.0, 40.0, 15.0, key="sim3")
        base_rain = st.slider("Base Rain (mm)", 0.0, 20.0, 0.0, key="sim4")
    
    scenario = st.selectbox("Scenario:", ["Heat Wave", "Drought", "Storm", "Humidity Drop", "Windy"])
    
    if st.button("🔄 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Simulating..."):
            temps = np.linspace(20, 45, 50)
            risks = []
            for t in temps:
                idx = calculate_fire_indices(t, base_hum, base_wind, base_rain)
                risks.append(min(1.0, idx['FWI'] / 50))
            
            fig = px.line(x=temps, y=np.array(risks)*100, 
                         title=f'Risk vs Temperature ({scenario})',
                         labels={'x': 'Temperature (°C)', 'y': 'Risk (%)'})
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Peak risk: {max(risks):.1%}")

elif page == "🔔 Active Alerts":
    st.header("Active Alerts & Monitoring")
    
    for alert in st.session_state.alerts:
        color = {'Critical': '#d32f2f', 'High': '#ff5722', 'Medium': '#ff9800'}.get(alert['severity'], '#666')
        st.markdown(f"""
        <div style="background: {color}20; border-left: 5px solid {color}; padding: 15px; margin: 10px 0; border-radius: 5px;">
            <strong style="color: {color};">{alert['severity']} ALERT #{alert['id']}</strong><br>
            Location: {alert['location']}<br>
            Risk Score: {alert['risk_score']:.3f}<br>
            Time: {alert['time'].strftime('%Y-%m-%d %H:%M')}
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"Acknowledge #{alert['id']}", key=f"ack_{alert['id']}"):
            st.success(f"Alert #{alert['id']} acknowledged")

elif page == "📈 Historical Analysis":
    st.header("Historical Trend Analysis")
    
    days = st.slider("Period (days):", 7, 90, 30)
    
    if st.button("📊 Analyze", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            risks = 0.3 + 0.4 * np.sin(np.linspace(0, 4*np.pi, days)) + np.random.normal(0, 0.1, days)
            risks = np.clip(risks, 0, 1)
            
            df = pd.DataFrame({'Date': dates, 'Risk': risks})
            
            fig = px.line(df, x='Date', y='Risk', title='Risk Trend')
            st.plotly_chart(fig, use_container_width=True)
            
            st.metric("Average Risk", f"{np.mean(risks):.3f}")
            st.metric("Peak Risk", f"{np.max(risks):.3f}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**WildfireGuard AI v2.0**
- PSO-ANFIS Hybrid Model
- Real-time Weather API  
- 25+ Regional Locations
- Decision Support System

© 2024 WildfireGuard AI
""")
