# ============================================================
# PHASE 8 - TREND SIMULATOR (SENSITIVITY ANALYSIS)
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("  PHASE 8 - TREND SIMULATOR")
print("="*55)

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================
mlp_pso = joblib.load('models/mlp_pso.pkl')
scaler = joblib.load('models/scaler.pkl')

# We'll use a baseline sample from the test set
X_test = np.load('models/X_test.npy')
baseline_sample = X_test[0].reshape(1, -1)

# Feature names for reference
features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain', 'Temp_RH_Index', 'Wind_Rain_Interaction']

# ============================================================
# SENSITIVITY SIMULATION
# ============================================================
def simulate_trend(feature_idx, range_values):
    predictions = []
    for val in range_values:
        # Create a copy of the baseline
        sample = baseline_sample.copy()
        # Update the feature of interest (normalized value)
        sample[0, feature_idx] = val

        # We must also update the interaction features if the changed feature is part of one
        # (In a real system, we'd update raw values then re-scale, but for this
        #  sensitivity analysis we are perturbing the scaled latent space).

        prob = mlp_pso.predict_proba(sample)
        predictions.append(np.max(prob))
    return predictions

# Target features to simulate
targets = {
    'Temperature': 4,
    'Humidity': 5,
    'Wind': 6
}

plt.figure(figsize=(15, 5))

for i, (name, idx) in enumerate(targets.items()):
    plt.subplot(1, 3, i+1)
    x_axis = np.linspace(0, 1, 20)
    y_axis = simulate_trend(idx, x_axis)

    plt.plot(x_axis, y_axis, color='darkorange', linewidth=2)
    plt.title(f'Sensitivity: {name}', fontsize=12)
    plt.xlabel('Normalized Value')
    plt.ylabel('Max Prediction Probability')
    plt.grid(True, alpha=0.3)

plt.suptitle('Parameter Sensitivity Analysis (Trend Simulation)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/trend_simulator.png', dpi=150)
plt.close()
print("Saved: trend_simulator.png")

print("\n" + "="*55)
print("  PHASE 8 COMPLETE")
print("="*55)
