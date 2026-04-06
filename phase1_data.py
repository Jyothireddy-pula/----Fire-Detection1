# ============================================================
# PHASE 1 - DATA LOADING, PREPARATION, AND VISUALIZATION
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import os

print("="*55)
print("  PHASE 1 - DATA PREPARATION")
print("="*55)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('forestfires.csv')
df = df.drop(columns=['month', 'day', 'X', 'Y'])
df['area_log'] = np.log1p(df['area'])

features = ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']

print(f"\nDataset shape     : {df.shape}")
print(f"Features          : {features}")
print(f"Null values       : {df.isnull().sum().sum()}")
print(f"\nStatistical Summary:")
print(df[features].describe().round(2))

# ============================================================
# CONVERT TARGET TO RISK CLASS - FIXED
# ============================================================
def area_to_risk(log_area):
    if log_area == 0.0:  return 0   # No Fire
    elif log_area < 1.0: return 1   # Low Risk
    elif log_area < 3.0: return 2   # Moderate Risk
    else:                return 3   # High Risk

y_class = np.array([area_to_risk(v) for v in df['area_log'].values])
X       = df[features].values

labels_map = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}

unique, counts = np.unique(y_class, return_counts=True)
print(f"\nClass Distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u} — {labels_map[u]:<12}: {c} samples ({100*c/len(y_class):.1f}%)")

# ============================================================
# NORMALIZE AND SPLIT
# ============================================================
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

# ============================================================
# SMOTE BALANCING
# ============================================================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print(f"\nBefore SMOTE : {Counter(y_train)}")
print(f"After SMOTE  : {Counter(y_train_bal)}")
print(f"\nTraining samples (balanced) : {X_train_bal.shape[0]}")
print(f"Testing samples             : {X_test.shape[0]}")
print(f"Features per sample         : {X_test.shape[1]}")

# ============================================================
# SAVE DATA AND SCALER
# ============================================================
np.save('models/X_train_bal.npy', X_train_bal)
np.save('models/y_train_bal.npy', y_train_bal)
np.save('models/X_test.npy',      X_test)
np.save('models/y_test.npy',      y_test)
joblib.dump(scaler, 'models/scaler.pkl')

print("\nData saved to models/ folder.")

# ============================================================
# PLOT 1 - CLASS DISTRIBUTION
# ============================================================
bar_colors = ['steelblue', 'green', 'orange', 'tomato']
plt.figure(figsize=(8, 5))
plt.bar([labels_map[u] for u in unique], counts,
        color=bar_colors[:len(unique)], edgecolor='black', alpha=0.85)
plt.title('Risk Class Distribution', fontsize=13)
plt.ylabel('Number of Samples')
plt.grid(True, alpha=0.3, axis='y')
for i, c in enumerate(counts):
    plt.text(i, c+2, str(c), ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/class_distribution.png', dpi=150)
plt.close()
print("Saved: class_distribution.png")

# ============================================================
# PLOT 2 - FEATURE HISTOGRAMS
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(features):
    axes[i].hist(df[col], bins=25, color='darkorange', edgecolor='black')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
plt.suptitle('Distribution of All Input Features', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/feature_histograms.png', dpi=150)
plt.close()
print("Saved: feature_histograms.png")

# ============================================================
# PLOT 3 - CORRELATION HEATMAP
# ============================================================
plt.figure(figsize=(10, 8))
corr = df[features + ['area_log']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5)
plt.title('Correlation Heatmap — Features vs Fire Risk', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# ============================================================
# PLOT 4 - TARGET DISTRIBUTION BEFORE AND AFTER LOG TRANSFORM
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['area'], bins=30, color='tomato', edgecolor='black')
axes[0].set_title('Burned Area - Original (Skewed)')
axes[0].set_xlabel('Area')
axes[1].hist(df['area_log'], bins=30, color='steelblue', edgecolor='black')
axes[1].set_title('Burned Area - After Log Transform')
axes[1].set_xlabel('log(area + 1)')
plt.tight_layout()
plt.savefig('outputs/target_distribution.png', dpi=150)
plt.close()
print("Saved: target_distribution.png")

# ============================================================
# PLOT 5 - SMOTE BEFORE AND AFTER
# ============================================================
before_counter = Counter(y_train)
after_counter  = Counter(y_train_bal)
all_classes    = sorted(after_counter.keys())

before_counts = [before_counter.get(i, 0) for i in all_classes]
after_counts  = [after_counter.get(i, 0)  for i in all_classes]
class_labels  = [labels_map[i] for i in all_classes]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(class_labels, before_counts,
            color=bar_colors[:len(all_classes)], edgecolor='black', alpha=0.85)
axes[0].set_title('Before SMOTE — Imbalanced', fontsize=12)
axes[0].set_ylabel('Samples')
axes[0].grid(True, alpha=0.3, axis='y')
for i, c in enumerate(before_counts):
    axes[0].text(i, c+1, str(c), ha='center', fontsize=10)

axes[1].bar(class_labels, after_counts,
            color=bar_colors[:len(all_classes)], edgecolor='black', alpha=0.85)
axes[1].set_title('After SMOTE — Balanced', fontsize=12)
axes[1].set_ylabel('Samples')
axes[1].grid(True, alpha=0.3, axis='y')
for i, c in enumerate(after_counts):
    axes[1].text(i, c+1, str(c), ha='center', fontsize=10)

plt.suptitle('SMOTE Class Balancing', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/smote_balancing.png', dpi=150)
plt.close()
print("Saved: smote_balancing.png")

print("\n" + "="*55)
print("  PHASE 1 COMPLETE")
print("="*55)