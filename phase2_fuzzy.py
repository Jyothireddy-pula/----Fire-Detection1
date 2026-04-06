# ============================================================
# PHASE 2 - PLAIN FUZZY SUGENO BASELINE
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

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
def trimf(x, a, b, c):
    return np.maximum(0, np.minimum(
        (x - a) / (b - a + 1e-9),
        (c - x) / (c - b + 1e-9)
    ))

def get_mf(x_range):
    low  = trimf(x_range, 0,   0,   0.5)
    med  = trimf(x_range, 0,   0.5, 1.0)
    high = trimf(x_range, 0.5, 1.0, 1.0)
    return low, med, high

# ============================================================
# FUZZY RULES
# ============================================================
fuzzy_rules = {
    (2, 0, 2): 0.95,
    (2, 0, 1): 0.80,
    (2, 0, 0): 0.65,
    (2, 1, 2): 0.70,
    (2, 1, 1): 0.55,
    (2, 1, 0): 0.40,
    (1, 0, 2): 0.60,
    (1, 0, 1): 0.45,
    (1, 1, 1): 0.30,
    (1, 2, 0): 0.15,
    (0, 2, 0): 0.05,
    (0, 2, 1): 0.10,
}

print(f"\nFuzzy Rules defined : {len(fuzzy_rules)} rules")
print(f"\n{'Rule':<5} {'Temp':>6} {'RH':>8} {'Wind':>8} {'Output':>8}")
print("-" * 40)
set_names = ['Low', 'Med', 'High']
for i, ((t, r, w), out) in enumerate(fuzzy_rules.items()):
    print(f"{i+1:<5} {set_names[t]:>6} {set_names[r]:>8} {set_names[w]:>8} {out:>8.2f}")

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

    numerator   = 0
    denominator = 0
    for (ti, ri, wi), output in fuzzy_rules.items():
        firing       = sets['temp'][ti] * sets['rh'][ri] * sets['wind'][wi]
        numerator   += firing * output
        denominator += firing

    if denominator < 1e-9:
        return 0.3
    return numerator / denominator

# ============================================================
# EVALUATE ON TEST SET
# ============================================================
fuzzy_raw     = np.array([fuzzy_sugeno_predict(x) for x in X_test])
fuzzy_classes = np.round(np.clip(fuzzy_raw, 0, 1) * 3).astype(int)

acc_fuzzy = accuracy_score(y_test, fuzzy_classes)
f1_fuzzy  = f1_score(y_test, fuzzy_classes,
                     average='weighted', zero_division=0)

print("\n" + "="*40)
print("  Plain Fuzzy Sugeno Results")
print("="*40)
print(f"  Accuracy : {acc_fuzzy*100:.2f}%")
print(f"  F1 Score : {f1_fuzzy:.4f}")
print("="*40)

# Dynamic class names based on what exists in data
labels_map    = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}
unique_classes = np.unique(np.concatenate([y_test, fuzzy_classes]))
target_names   = [labels_map[i] for i in unique_classes]

print("\nDetailed Classification Report:")
print(classification_report(y_test, fuzzy_classes,
      labels=unique_classes,
      target_names=target_names,
      zero_division=0))

# ============================================================
# SAVE RESULTS
# ============================================================
np.save('models/fuzzy_preds.npy', fuzzy_classes)
np.save('models/fuzzy_raw.npy',   fuzzy_raw)
joblib.dump({
    'accuracy':       acc_fuzzy,
    'f1':             f1_fuzzy,
    'unique_classes': unique_classes.tolist()
}, 'models/fuzzy_results.pkl')

print("Results saved to models/ folder.")

# ============================================================
# PLOT 1 - MEMBERSHIP FUNCTIONS
# ============================================================
x_range = np.linspace(0, 1, 200)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
input_labels = ['Temperature', 'Relative Humidity', 'Wind Speed']

for ax, label in zip(axes, input_labels):
    low, med, high = get_mf(x_range)
    ax.plot(x_range, low,  'b', label='Low',    linewidth=2)
    ax.plot(x_range, med,  'g', label='Medium', linewidth=2)
    ax.plot(x_range, high, 'r', label='High',   linewidth=2)
    ax.set_title(f'{label} Membership Functions', fontsize=11)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Membership Degree')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Fuzzy Sugeno — Membership Functions', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/membership_functions.png', dpi=150)
plt.close()
print("Saved: membership_functions.png")

# ============================================================
# PLOT 2 - PREDICTED VS ACTUAL
# ============================================================
x_pos = np.arange(len(X_test))
plt.figure(figsize=(12, 5))
plt.plot(x_pos, y_test,        'bo', alpha=0.5, markersize=5, label='Actual')
plt.plot(x_pos, fuzzy_classes, 'rx', alpha=0.5, markersize=5, label='Predicted')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Risk Class', fontsize=12)
plt.yticks([0, 1, 2, 3], ['No Fire', 'Low', 'Moderate', 'High'])
plt.title('Plain Fuzzy Sugeno — Predicted vs Actual', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fuzzy_pred_vs_actual.png', dpi=150)
plt.close()
print("Saved: fuzzy_pred_vs_actual.png")

# ============================================================
# PLOT 3 - RAW OUTPUT DISTRIBUTION
# ============================================================
plt.figure(figsize=(8, 5))
plt.hist(fuzzy_raw, bins=20, color='tomato', edgecolor='black', alpha=0.85)
plt.xlabel('Raw Fuzzy Output Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Fuzzy Sugeno — Raw Output Distribution', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/fuzzy_output_distribution.png', dpi=150)
plt.close()
print("Saved: fuzzy_output_distribution.png")

print("\n" + "="*55)
print("  PHASE 2 COMPLETE")
print("="*55)