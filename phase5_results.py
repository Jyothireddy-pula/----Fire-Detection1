# ============================================================
# PHASE 5 - PSO-MLP & FINAL RESULTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import os

print("="*55)
print("  PHASE 5 - PSO-MLP & FINAL COMPARISON")
print("="*55)

# ============================================================
# LOAD DATA & MODELS
# ============================================================
X_train_bal = np.load('models/X_train_bal.npy')
y_train_bal = np.load('models/y_train_bal.npy')
X_test      = np.load('models/X_test.npy')
y_test      = np.load('models/y_test.npy')

# Load baseline results
fuzzy_results = joblib.load('models/fuzzy_results.pkl')
mlp_standard_results = joblib.load('models/mlp_standard_results.pkl')

acc_fuzzy = fuzzy_results['accuracy']
f1_fuzzy  = fuzzy_results['f1']
acc_standard = mlp_standard_results['accuracy']
f1_standard  = mlp_standard_results['f1']

# Load PSO optimal config
pso_config = joblib.load('models/pso_best_config.pkl')
n1 = pso_config['n1']
n2 = pso_config['n2']
lr = pso_config['lr']
al = pso_config['al']

print(f"\nLoaded Optimal PSO Config — Layers:({n1},{n2}) LR:{lr:.5f} Alpha:{al:.6f}")

# ============================================================
# TRAIN PSO-MLP
# ============================================================
print("\nTraining PSO-MLP with optimum parameters...")
mlp_pso = MLPClassifier(
    hidden_layer_sizes=(n1, n2),
    learning_rate_init=lr,
    alpha=al,
    max_iter=500,
    random_state=42
)
mlp_pso.fit(X_train_bal, y_train_bal)
preds_pso = mlp_pso.predict(X_test)

acc_pso = accuracy_score(y_test, preds_pso)
f1_pso  = f1_score(y_test, preds_pso, average='weighted', zero_division=0)

print(f"PSO-MLP — Accuracy: {acc_pso*100:.2f}% | F1: {f1_pso:.4f}")

# Save the final PSO-MLP model
joblib.dump(mlp_pso, 'models/mlp_pso.pkl')

# ============================================================
# FINAL RESULTS
# ============================================================
print("\n" + "="*55)
print(f"{'System':<25} {'Accuracy':>10} {'F1 Score':>10}")
print("="*55)
print(f"{'Plain Fuzzy Sugeno':<25} {acc_fuzzy*100:>9.2f}% {f1_fuzzy:>10.4f}")
print(f"{'Standard MLP':<25} {acc_standard*100:>9.2f}% {f1_standard:>10.4f}")
print(f"{'PSO-MLP':<25} {acc_pso*100:>9.2f}% {f1_pso:>10.4f}")
print("="*55)

labels_map     = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}
unique_classes = np.unique(np.concatenate([y_test, preds_pso]))
target_names   = [labels_map[i] for i in unique_classes]

print("\nPSO-MLP Classification Report:")
print(classification_report(y_test, preds_pso,
      labels=unique_classes,
      target_names=target_names,
      zero_division=0))

# ============================================================
# PLOTS - FINAL COMPARISON
# ============================================================
systems    = ['Plain Fuzzy\nSugeno', 'Standard\nMLP', 'PSO-MLP\n(Optimized)']
accuracies = [acc_fuzzy*100, acc_standard*100, acc_pso*100]
f1s        = [f1_fuzzy, f1_standard, f1_pso]
colors     = ['tomato', 'steelblue', 'darkorange']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Accuracy Bar
axes[0].bar(systems, accuracies, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Classification Accuracy (%)', fontsize=12)
axes[0].set_ylabel('Accuracy %')
axes[0].set_ylim(0, 100)
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(accuracies):
    axes[0].text(i, val+1, f'{val:.2f}%', ha='center', fontsize=11)

# F1 Bar
axes[1].bar(systems, f1s, color=colors, edgecolor='black', alpha=0.85)
axes[1].set_title('Weighted F1 Score', fontsize=12)
axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(f1s):
    axes[1].text(i, val+0.01, f'{val:.4f}', ha='center', fontsize=11)

for ax in axes:
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=10, ha='center', fontsize=11)

plt.suptitle('Three-Way Classification Comparison', fontsize=13)
plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/final_comparison.png', dpi=150)
plt.close()

print("\nFinal comparison plot saved to outputs folder.")
print("\nPIPELINE COMPLETE.")
