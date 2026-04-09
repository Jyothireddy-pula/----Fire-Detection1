# ============================================================
# PHASE 2 - PLAIN FUZZY SUGENO BASELINE
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("  PHASE 2 - PLAIN FUZZY SUGENO BASELINE")
print("="*55)

# ============================================================
# LOAD DATA
# ============================================================
X_test = np.load('models/X_test.npy')
y_test = np.load('models/y_test.npy')

print(f"\nTest samples loaded : {X_test.shape[0]}")
print(f"Classes in test set : {np.unique(y_test)}")

# ============================================================
# MEMBERSHIP FUNCTIONS
# ============================================================
def gaussian_mf(x, center, sigma):
    return np.exp(-(x - center)**2 / (2 * sigma**2))

def get_mf(x_range):
    # Define centers and sigmas for Low, Med, High
    # Normalized range [0, 1]
    low  = gaussian_mf(x_range, 0.2, 0.15)
    med  = gaussian_mf(x_range, 0.5, 0.15)
    high = gaussian_mf(x_range, 0.8, 0.15)
    return low, med, high

# ============================================================
# FUZZY RULES - FULL 27-RULE MATRIX (3^3)
# Format: (Temp, RH, Wind) -> Output Risk Score
# 0=Low, 1=Med, 2=High
# ============================================================
fuzzy_rules = {}
for t in range(3):
    for r in range(3):
        for w in range(3):
            # Simple heuristic: High Temp, Low RH, High Wind = Max Risk
            # Low Temp, High RH, Low Wind = Min Risk
            score = (t * 0.5) + ((2-r) * 0.3) + (w * 0.2)
            # Normalize score to 0.0 - 1.0
            fuzzy_rules[(t, r, w)] = np.clip(score / 2.0, 0, 1)

print(f"\nFuzzy Rules defined : {len(fuzzy_rules)} rules (Full Matrix)")
set_names = ['Low', 'Med', 'High']
print(f"\n{'Rule':<5} {'Temp':>6} {'RH':>8} {'Wind':>8} {'Output':>8}")
print("-" * 40)
# Print a few sample rules to verify
for i, ((t, r, w), out) in enumerate(list(fuzzy_rules.items())[:10]):
    print(f"{i+1:<5} {set_names[t]:>6} {set_names[r]:>8} {set_names[w]:>8} {out:>8.2f}")
print("... (remaining rules generated automatically)")

# ============================================================
# INFERENCE ENGINE
# ============================================================
def fuzzy_sugeno_predict(sample):
    temp_val = sample[4]
    rh_val   = sample[5]
    wind_val = sample[6]

    sets = {}
    for val, name in zip([temp_val, rh_val, wind_val], ['temp', 'rh', 'wind']):
        r = np.array([val])
        low, med, high = get_mf(r)
        sets[name] = [low[0], med[0], high[0]]

    numerator, denominator = 0, 0
    for (ti, ri, wi), output in fuzzy_rules.items():
        firing = sets['temp'][ti] * sets['rh'][ri] * sets['wind'][wi]
        numerator   += firing * output
        denominator += firing

    if denominator < 1e-9:
        return 0.3
    return numerator / denominator

# ============================================================
# EVALUATE
# ============================================================
fuzzy_raw     = np.array([fuzzy_sugeno_predict(x) for x in X_test])

# Since the Algerian dataset is binary (0 and 3), we map the raw output:
# Raw output is 0.0 to 1.0. If > 0.5, we predict 3 (High Risk), else 0 (No Fire).
fuzzy_classes = np.where(fuzzy_raw > 0.5, 3, 0).astype(int)

acc_fuzzy = accuracy_score(y_test, fuzzy_classes)
f1_fuzzy  = f1_score(y_test, fuzzy_classes, average='weighted', zero_division=0)

print("\n" + "="*40)
print("  Plain Fuzzy Sugeno Results")
print("="*40)
print(f"  Accuracy : {acc_fuzzy*100:.2f}%")
print(f"  F1 Score : {f1_fuzzy:.4f}")
print("="*40)

labels_map     = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}
unique_classes = np.unique(np.concatenate([y_test, fuzzy_classes]))
target_names   = [labels_map[i] for i in unique_classes]

print("\nDetailed Classification Report:")
print(classification_report(y_test, fuzzy_classes,
      labels=unique_classes, target_names=target_names, zero_division=0))

# ============================================================
# SAVE
# ============================================================
np.save('models/fuzzy_preds.npy', fuzzy_classes)
np.save('models/fuzzy_raw.npy',   fuzzy_raw)
joblib.dump({'accuracy': acc_fuzzy, 'f1': f1_fuzzy}, 'models/fuzzy_results.pkl')
print("Results saved to models/ folder.")

# ============================================================
# PLOTS
# ============================================================
# Plot 1 - Membership Functions
x_range = np.linspace(0, 1, 200)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, label in zip(axes, ['Temperature', 'Relative Humidity', 'Wind Speed']):
    low, med, high = get_mf(x_range)
    ax.plot(x_range, low,  'b', label='Low',    linewidth=2)
    ax.plot(x_range, med,  'g', label='Medium',  linewidth=2)
    ax.plot(x_range, high, 'r', label='High',    linewidth=2)
    ax.set_title(f'{label} Gaussian MFs', fontsize=11)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Membership Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle('Fuzzy Sugeno --- Gaussian Membership Functions', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/membership_functions.png', dpi=150)
plt.close()
print("Saved: membership_functions.png")

# Plot 2 - Predicted vs Actual
x_pos = np.arange(len(X_test))
plt.figure(figsize=(12, 5))
plt.plot(x_pos, y_test,        'bo', alpha=0.5, markersize=5, label='Actual')
plt.plot(x_pos, fuzzy_classes, 'rx', alpha=0.5, markersize=5, label='Predicted')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Risk Class',   fontsize=12)
plt.yticks([0,1,2,3], ['No Fire','Low','Moderate','High'])
plt.title('Plain Fuzzy Sugeno --- Predicted vs Actual', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fuzzy_pred_vs_actual.png', dpi=150)
plt.close()
print("Saved: fuzzy_pred_vs_actual.png")

# Plot 3 - Raw Output Distribution
plt.figure(figsize=(8, 5))
plt.hist(fuzzy_raw, bins=20, color='tomato', edgecolor='black', alpha=0.85)
plt.xlabel('Raw Fuzzy Output Score', fontsize=12)
plt.ylabel('Frequency',              fontsize=12)
plt.title('Fuzzy Sugeno --- Raw Output Distribution', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fuzzy_output_distribution.png', dpi=150)
plt.close()
print("Saved: fuzzy_output_distribution.png")

print("\n" + "="*55)
print("  PHASE 2 COMPLETE")
print("="*55)