# ============================================================
# PHASE 1 - DATA LOADING, PREPARATION, AND VISUALIZATION
# Using the Algerian Forest Fires Dataset for higher accuracy
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("="*55)
print("  PHASE 1 - DATA PREPARATION (ALGERIAN)")
print("="*55)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('archive/Algerian_forest_fires_dataset.csv')
df.columns = df.columns.str.strip()

# Define relevant features for fire prediction
features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']

# IMPORTANT: Force numeric conversion for features to handle object/string types in CSV
for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle potential noise in the CSV (empty lines or non-data rows)
df = df.dropna(subset=['Temperature', 'Classes'] + features)

print(f"\nDataset shape      : {df.shape}")
print(f"Features           : {features}")
print(f"Null values        : {df.isnull().sum().sum()}")
print(f"\nStatistical Summary:")
print(df[features].describe().round(2))

# ============================================================
# FEATURE ENGINEERING
# ============================================================
# Interaction features for Algerian dataset
df['Temp_RH_Index'] = df['Temperature'] * (100 - df['RH'])
df['Wind_Rain_Interaction'] = df['Ws'] * df['Rain']
features = features + ['Temp_RH_Index', 'Wind_Rain_Interaction']

# ============================================================
# CONVERT TARGET TO RISK CLASS
# ============================================================
# The Algerian dataset has binary classes: 'fire' and 'not fire'
def algerian_to_risk(val):
    val = str(val).strip().lower()
    if 'not fire' in val:
        return 0 # No Fire
    elif 'fire' in val:
        return 3 # High Risk
    return 0 # Default

y_class = np.array([algerian_to_risk(v) for v in df['Classes'].values])
X = df[features].values

labels_map = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}
unique, counts = np.unique(y_class, return_counts=True)

print(f"\nClass Distribution:")
for u, c in zip(unique, counts):
    print(f"  Class {u} --- {labels_map[u]:<12}: {c} samples ({100*c/len(y_class):.1f}%)")

# ============================================================
# NORMALIZE AND SPLIT
# ============================================================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_class, test_size=0.2, random_state=42
)

# ============================================================
# SMOTE BALANCING -> RandomOverSampler
# ============================================================
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

print(f"\nBefore ROS : {Counter(y_train)}")
print(f"After ROS  : {Counter(y_train_bal)}")
print(f"\nTraining samples (balanced) : {X_train_bal.shape[0]}")
print(f"Testing samples             : {X_test.shape[0]}")
print(f"Features per sample         : {X_test.shape[1]}")

# ============================================================
# SAVE
# ============================================================
np.save('models/X_train_bal.npy', X_train_bal)
np.save('models/y_train_bal.npy', y_train_bal)
np.save('models/X_test.npy', X_test)
np.save('models/y_test.npy', y_test)
joblib.dump(scaler, 'models/scaler.pkl')
print("\nData saved to models/ folder.")

# ============================================================
# PLOTS
# ============================================================
bar_colors = ['steelblue', 'green', 'orange', 'tomato']

# Plot 1 - Class Distribution
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

# Plot 2 - Feature Histograms
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(features):
    axes[i].hist(df[col], bins=25, color='darkorange', edgecolor='black')
    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.suptitle('Distribution of All Input Features', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/feature_histograms.png', dpi=150)
plt.close()
print("Saved: feature_histograms.png")

# Plot 3 - Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5)
plt.title('Correlation Heatmap --- Features vs Fire Risk', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=150)
plt.close()
print("Saved: correlation_heatmap.png")

# Plot 4 - Target Distribution (Binary for Algerian)
plt.figure(figsize=(6, 4))
df['Classes'].value_counts().plot(kind='bar', color=['tomato', 'steelblue'], edgecolor='black')
plt.title('Algerian Dataset - Class Distribution', fontsize=13)
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outputs/target_distribution.png', dpi=150)
plt.close()
print("Saved: target_distribution.png")

# Plot 5 - RandomOverSampler Balancing
before_counter = Counter(y_train)
after_counter  = Counter(y_train_bal)
all_classes    = sorted(after_counter.keys())
before_counts  = [before_counter.get(i, 0) for i in all_classes]
after_counts   = [after_counter.get(i, 0)  for i in all_classes]
class_labels   = [labels_map[i] for i in all_classes]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(class_labels, before_counts,
            color=bar_colors[:len(all_classes)], edgecolor='black', alpha=0.85)
axes[0].set_title('Before ROS --- Imbalanced', fontsize=12)
axes[0].set_ylabel('Samples')
axes[0].grid(True, alpha=0.3, axis='y')
for i, c in enumerate(before_counts):
    axes[0].text(i, c+1, str(c), ha='center', fontsize=10)

axes[1].bar(class_labels, after_counts,
            color=bar_colors[:len(all_classes)], edgecolor='black', alpha=0.85)
axes[1].set_title('After ROS --- Balanced', fontsize=12)
axes[1].set_ylabel('Samples')
axes[1].grid(True, alpha=0.3, axis='y')
for i, c in enumerate(after_counts):
    axes[1].text(i, c+1, str(c), ha='center', fontsize=10)

plt.suptitle('RandomOverSampler Class Balancing', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/smote_balancing.png', dpi=150)
plt.close()
print("Saved: smote_balancing.png")

print("\n" + "="*55)
print("  PHASE 1 COMPLETE")
print("="*55)
