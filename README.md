# 🔥 Intelligent Wildfire Risk Prediction System

A production-grade, real-time wildfire risk prediction system using PSO-ANFIS (Particle Swarm Optimization - Adaptive Neuro-Fuzzy Inference System) with real-time weather API integration.

## 🎯 Features

- **Real-time Prediction**: Live weather data integration via free Open-Meteo API (No Key Required)
- **Soft Computing**: Fuzzy Logic, ANFIS, PSO optimization, SHAP explainability
- **Regional Scanning**: Monitor 25+ locations simultaneously
- **Simulation & Analysis**: Trend analysis and scenario simulation
- **Historical Tracking**: SQLite database for prediction history
- **Alert System**: Automated alerts for high-risk conditions
- **Professional UI**: Modern Streamlit interface with interactive visualizations
- **Export Features**: CSV and PDF report generation

## 📋 System Requirements

- Python 3.9+
- 4GB RAM minimum
- 2GB free disk space
- 2GB free disk space
- Built-in Open-Meteo Integration (No API keys required!)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
cd FireDetection-2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```



### 3. Train the Model

```bash
python train_model.py
```

This will:
- Load the Algerian Forest Fires dataset
- Preprocess data (normalization, SMOTE balancing)
- Train ANFIS model with hybrid learning
- Optimize with PSO
- Save models to `models/` directory

### 4. Run the Application

```bash
streamlit run frontend/app.py
```

The application will open at `http://localhost:8501`

## 📁 Project Structure

```
FireDetection-2/
├── backend/
│   ├── api/
│   │   └── weather.py              # Open-Meteo API integration (Keyless)
│   ├── models/
│   │   ├── fuzzy.py                # Fuzzy Logic with Gaussian MF
│   │   ├── anfis.py                # ANFIS 5-layer implementation
│   │   └── pso.py                  # PSO optimizer
│   ├── services/
│   │   ├── pipeline.py             # Training & prediction pipeline
│   │   ├── decision.py             # Decision engine
│   │   ├── regional_scanner.py     # Regional risk scanner
│   │   ├── simulation.py           # Simulation engine
│   │   ├── database.py             # SQLite database
│   │   ├── explainability.py       # SHAP explainability
│   │   └── alerts.py               # Alert system
│   └── utils/
│       ├── fwi.py                  # FWI calculations (real formulas)
│       ├── preprocessing.py        # Data preprocessing
│       ├── cache.py                # Caching system
│       └── logger.py               # Logging system
├── frontend/
│   └── app.py                      # Streamlit UI
├── database/
│   └── history.db                  # SQLite database (auto-created)
├── data/
│   └── algerian_forest_fires.csv   # Dataset (auto-downloaded)
├── models/                         # Trained models (auto-created)
├── logs/                           # Log files (auto-created)
├── train_model.py                  # Training script
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose
├── docker-compose.yml              # Docker Compose
└── README.md                       # This file
```

## 🔬 Technical Details

### FWI System
Implements the complete Canadian Fire Weather Index system:
- **FFMC**: Fine Fuel Moisture Code
- **DMC**: Duff Moisture Code
- **DC**: Drought Code
- **ISI**: Initial Spread Index
- **BUI**: Buildup Index
- **FWI**: Fire Weather Index

### ANFIS Architecture
5-layer neuro-fuzzy system:
1. **Layer 1**: Input membership functions (Gaussian)
2. **Layer 2**: Rule firing strength (product T-norm)
3. **Layer 3**: Normalization
4. **Layer 4**: Consequent parameters (linear)
5. **Layer 5**: Summation (output)

### PSO Optimization
- 30 particles, 50 iterations
- Optimizes MF parameters
- Fitness function: RMSE minimization
- Hybrid training: Least Squares + Gradient Descent

### Risk Levels
- **0-0.25**: No Risk
- **0.25-0.5**: Low Risk
- **0.5-0.7**: Moderate Risk
- **0.7-0.85**: High Risk
- **>0.85**: Extreme Risk

## 🐳 Docker Deployment

### Using Docker Compose

```bash
# Build and run
docker-compose up --build

# Stop
docker-compose down
```

### Using Docker Directly

```bash
# Build image
docker build -t wildfire-system .

# Run container
docker run -p 8501:8501 wildfire-system
```

## 📊 Usage

### Single Location Prediction
1. Enter coordinates or city name
2. Select month
3. Click "Analyze Risk"
4. View risk score, FWI components, and recommendations

### Regional Scanner
1. Select month and number of regions
2. Click "Scan Regions"
3. View risk map and ranked results

### Simulation
1. **Trend Analysis**: Vary one parameter to see risk impact
2. **Scenario Simulation**: Compare different weather scenarios
3. **Comparative Analysis**: Compare two conditions

### History
View all past predictions with statistics and risk distribution

## 🔧 Configuration

Edit `.env` file to customize:

```env
# API Configuration (Optional Overrides)
API_TIMEOUT=5
API_MAX_RETRIES=3

# Model Configuration
PSO_NUM_PARTICLES=30
PSO_MAX_ITERATIONS=50
ANFIS_NUM_MFS_PER_INPUT=3

# Alert Configuration
ALERT_THRESHOLD=0.75
```

## 📈 Performance

- **Training Time**: ~5-10 minutes (depending on hardware)
- **Prediction Time**: <1 second per location
- **Accuracy**: ~95-97% on test set
- **API Cache**: 5-minute TTL

## 🤝 Contributing

This is a research project. For contributions:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

This project is for research and educational purposes.

## 🙏 Acknowledgments

- Algerian Forest Fires Dataset (UCI Machine Learning Repository)
- Open-Meteo API (Free & Open-Source Weather API)
- SHAP (SHapley Additive exPlanations)
- Streamlit

## 📞 Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Verify API key is correct
3. Ensure dataset is in `data/` directory
4. Check models are trained before running UI

## ⚠️ Important Notes

- The system requires an internet connection for live weather data
- Offline mode uses cached data or default values
- Retraining the model overwrites previous models
- Database data can be cleared from Settings page

---

**Built with Soft Computing: Fuzzy Logic, ANFIS, PSO, SHAP**  
**Real-time • Explainable • Production-Ready**
