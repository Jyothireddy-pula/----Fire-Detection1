# 🔥 WildfireGuard AI - Wildfire Risk Prediction System

A smart system that helps predict forest fire risk using artificial intelligence and real-time weather data.

---

## 📖 What This System Does

This system helps you:
- **Predict wildfire risk** using weather data
- **Monitor 25+ forest locations** around the world
- **Get alerts** when fire risk is high
- **Test scenarios** like "What if temperature increases?"
- **See explanations** for why risk is high
- **Get recommendations** on what actions to take

---

## 🚀 How to Get Started (Simple Steps)

### Step 1: Install Python
You need Python 3.10 or higher installed on your computer.
- Check if you have Python: Open terminal/command prompt and type `python --version`
- If you don't have it, download from [python.org](https://python.org)

### Step 2: Install Required Packages
Open your terminal/command prompt, go to the project folder, and run:

```bash
pip install -r requirements.txt
```

This will install all the needed software packages automatically.

### Step 3: Run the System
Simply run this command:

```bash
streamlit run app.py
```

The system will open in your web browser automatically (usually at http://localhost:8501)

---

## 🎮 How to Use the Dashboard

Once the dashboard opens, you will see 5 tabs:

### 1. Risk Prediction
- Enter weather values (temperature, humidity, wind speed, etc.)
- Click "Calculate Risk"
- See the risk level and recommendations

### 2. Regional Scanner
- Click "Start Regional Scan"
- See risk levels for 25+ forests worldwide
- View the map showing high-risk areas

### 3. Scenario Simulator
- Choose a scenario (like Heat Wave or Drought)
- Click "Run Simulation"
- See how different conditions affect fire risk

### 4. Active Alerts
- See current fire risk alerts
- Acknowledge alerts when resolved

### 5. Historical Analysis
- Choose how many days to analyze
- Click "Generate Analysis"
- See risk trends over time

---

## 📂 What's in Each Folder

```
FireDetection/
├── app.py              # Main application - run this to start
├── README.md           # This file
├── requirements.txt    # List of software packages needed
├── forestfires.csv     # Forest fire data
│
├── core/              # Main system files
│   ├── anfis_system.py         # AI model for prediction
│   ├── weather_api.py          # Gets real weather data
│   ├── regional_scanner.py     # Scans multiple locations
│   ├── decision_support.py     # Suggests actions
│   ├── scenario_simulator.py   # Tests different scenarios
│   ├── alert_system.py         # Sends alerts
│   ├── shap_explainability.py  # Explains predictions
│   └── model_comparison.py     # Compares different models
│
├── pipeline/          # Training files (optional)
│   ├── phase1_data.py
│   ├── phase2_fuzzy.py
│   └── ... (8 files total)
│
├── models/            # Trained AI models
├── outputs/           # Generated charts and reports
└── archive/           # Original data files
```

---

## 🎯 Risk Levels Explained

The system shows 5 risk levels:

| Risk Level | What It Means | What To Do |
|------------|---------------|-------------|
| **Extreme** (0.85-1.00) | Very high fire danger | Evacuate immediately, call emergency services |
| **High** (0.70-0.85) | High fire danger | Prepare to evacuate, have fire teams ready |
| **Moderate** (0.50-0.70) | Medium fire risk | Watch closely, prepare emergency plans |
| **Low** (0.25-0.50) | Low fire risk | Normal monitoring, issue warnings |
| **No Risk** (0.00-0.25) | Safe | Routine checks only |

---

## 🔧 Optional: Train Your Own Models

If you want to train the AI models from scratch (not required for basic use):

1. Go to the pipeline folder:
   ```bash
   cd pipeline
   ```

2. Run each training step in order:
   ```bash
   python phase1_data.py
   python phase2_fuzzy.py
   python phase3_mlp.py
   python phase4_pso.py
   python phase5_results.py
   python phase6_shap.py
   python phase7_grid.py
   python phase8_simulator.py
   ```

This takes about 10-20 minutes and trains the AI models.

---

## 🌐 Optional: Use Real Weather Data

To get live weather data instead of sample data:

1. Get a free API key from [OpenWeatherMap](https://openweathermap.org/api) (it's free)
2. In the dashboard sidebar, enter your API key
3. The system will fetch real weather data

---

## � Computer Requirements

**Minimum:**
- Windows, Mac, or Linux computer
- 4GB RAM
- 2 processor cores
- 500MB free space

**Recommended:**
- 8GB RAM
- 4 processor cores
- 1GB free space

---

## ❓ Common Problems

**Problem:** "Module not found" error
- **Solution:** Run `pip install -r requirements.txt` again

**Problem:** Dashboard won't open
- **Solution:** Make sure you're in the correct folder and run `streamlit run app.py`

**Problem:** Python not recognized
- **Solution:** Install Python from python.org and add it to your system PATH

---

## � How Accurate Is It?

| System | Accuracy |
|--------|----------|
| Basic Fuzzy System | 65% |
| Standard AI Model | 98% |
| **Our PSO-ANFIS System** | **98.5%** |

---

## 🆘 Need Help?

1. Check the README in each core module file
2. Look at the outputs folder for example results
3. Make sure all packages are installed

---

## 🔬 How It Works (Simple Explanation)

1. **Input:** You enter weather data (temperature, humidity, wind, etc.)
2. **Processing:** The AI calculates fire risk using trained models
3. **Output:** You get a risk score, level, and recommendations
4. **Explanation:** The system shows which factors contribute most to risk

---

## � What the AI Uses to Predict Risk

The AI looks at these factors:
- Temperature
- Humidity (moisture in air)
- Wind speed
- Rainfall
- Fire Weather Indices (FFMC, DMC, DC, ISI, BUI, FWI)
- These indices measure how dry and ready to burn the forest is

---

## 🎓 For Advanced Users

If you want to use the system in your own Python code:

```python
# Example: Check risk for a location
from core.regional_scanner import RegionalGridScanner

scanner = RegionalGridScanner()
results = scanner.scan_all_locations()
print(results)
```

---

## � License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- Algerian Forest Fires Dataset
- OpenWeatherMap for weather data
- SHAP for AI explanations

---

**Made with ❤️ to help prevent wildfires**
