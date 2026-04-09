# ============================================================
# PHASE 7 - REGIONAL GRID SCANNER
# ============================================================
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("  PHASE 7 - REGIONAL GRID SCANNER")
print("="*55)

# ============================================================
# LOAD MODEL AND SCALER
# ============================================================
mlp_pso = joblib.load('models/mlp_pso.pkl')
scaler = joblib.load('models/scaler.pkl')
labels_map = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}

# ============================================================
# GENERATE SYNTHETIC REGIONAL GRID (25 Locations)
# ============================================================
# We simulate a 5x5 grid of locations with varying weather conditions
np.random.seed(42)
n_locations = 25
# Generate random weather samples based on the original feature ranges
# Features: ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
raw_data = np.random.uniform(low=[10, 20, 5, 0, 50, 0, 5, 0, 0, 0],
                             high=[45, 90, 30, 15, 100, 30, 200, 20, 100, 50],
                             size=(n_locations, 10))

df_grid = pd.DataFrame(raw_data, columns=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'])

# Add Interaction Features (must match Phase 1)
df_grid['Temp_RH_Index'] = df_grid['Temperature'] * (100 - df_grid['RH'])
df_grid['Wind_Rain_Interaction'] = df_grid['Ws'] * df_grid['Rain']


# Normalize using the original scaler
X_grid_scaled = scaler.transform(df_grid)

# ============================================================
# BATCH PREDICTIONS
# ============================================================
preds = mlp_pso.predict(X_grid_scaled)
probs = mlp_pso.predict_proba(X_grid_scaled)

# Create Results Table
results = []
for i in range(n_locations):
    results.append({
        'Location': f"Loc_{i+1}",
        'Risk_Class': labels_map[preds[i]],
        'Confidence': np.max(probs[i]) * 100
    })

df_results = pd.DataFrame(results).sort_values(by='Risk_Class', ascending=False)

print("\nRegional Risk Ranking (Top 10):")
print("-" * 40)
print(df_results.head(10).to_string(index=False))

# Save table
df_results.to_csv('outputs/regional_risk_table.csv', index=False)
print("\nSaved: regional_risk_table.csv")

# ============================================================
# GRID VISUALIZATION
# ============================================================
# Reshape predictions to 5x5 for heatmap
risk_values = preds.reshape(5, 5)
plt.figure(figsize=(8, 6))
plt.imshow(risk_values, cmap='YlOrRd', interpolation='nearest')
plt.colorbar(label='Risk Level (0:No Fire -> 3:High)')
plt.title('Regional Fire Risk Heatmap (5x5 Grid)', fontsize=13)
plt.xticks(range(5))
plt.yticks(range(5))
plt.savefig('outputs/regional_risk_heatmap.png', dpi=150)
plt.close()
print("Saved: regional_risk_heatmap.png")

print("\n" + "="*55)
print("  PHASE 7 COMPLETE")
print("="*55)
