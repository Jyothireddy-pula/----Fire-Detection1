# ============================================================
# PHASE 6 - SHAP EXPLAINABILITY
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("  PHASE 6 - SHAP EXPLAINABILITY")
print("="*55)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================
X_test = np.load('models/X_test.npy')
mlp_pso = joblib.load('models/mlp_pso.pkl')

# Feature names from Phase 1 (must match X_test shape)
features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temp_RH_Index', 'Wind_Rain_Interaction']
X_test_df = pd.DataFrame(X_test, columns=features)


print(f"\nAnalyzing model with {len(features)} features.")
print(f"Test samples: {X_test.shape[0]}")

# ============================================================
# SHAP EXPLAINER
# ============================================================
# Since MLP is a black box, we use KernelExplainer.
# We use a small summary of the data as background to speed up computation.
print("\nComputing SHAP values (this may take a minute)...")
X_summary = shap.kmeans(X_test, 10) # Use 10 representative samples as background
explainer = shap.KernelExplainer(mlp_pso.predict, X_summary)
shap_values = explainer.shap_values(X_test[:50], feature_names=features) # Analyze first 50 samples

# ============================================================
# VISUALIZATION 1 - Summary Plot (Feature Importance)
# ============================================================
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_df[:50], show=False)
plt.title('Global Feature Importance (SHAP)', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/shap_summary_plot.png', dpi=150)
plt.close()
print("Saved: shap_summary_plot.png")

# ============================================================
# VISUALIZATION 2 - Individual Prediction Explainability
# ============================================================
# Explain the first sample
sample_idx = 0
plt.figure(figsize=(12, 4))
shap.force_plot(explainer.expected_value, shap_values[sample_idx],
                X_test_df.iloc[sample_idx,:], matplotlib=True, show=False)
plt.title(f'Explanation for Sample {sample_idx+1}', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/shap_sample_explanation.png', dpi=150)
plt.close()
print("Saved: shap_sample_explanation.png")

print("\n" + "="*55)
print("  PHASE 6 COMPLETE")
print("="*55)
