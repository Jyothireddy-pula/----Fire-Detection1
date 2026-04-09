# ============================================================
# PHASE 5 - PSO-MLP FULL SYSTEM + FUZZY SUGENO OUTPUT LAYER
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report,
                              ConfusionMatrixDisplay, confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*55)
print("  PHASE 5 - PSO-MLP FULL SYSTEM")
print("="*55)

# ============================================================
# LOAD DATA AND PSO RESULTS
# ============================================================
X_train_bal = np.load('models/X_train_bal.npy')
y_train_bal = np.load('models/y_train_bal.npy')
X_test      = np.load('models/X_test.npy')
y_test      = np.load('models/y_test.npy')

pso_config  = joblib.load('models/pso_best_config.pkl')
n1 = pso_config['n1']
n2 = pso_config['n2']
lr = pso_config['lr']
al = pso_config['al']

print(f"\nLoaded PSO optimal config:")
print(f"  Hidden layers : ({n1}, {n2})")
print(f"  Learning rate : {lr:.5f}")
print(f"  Alpha         : {al:.6f}")
print(f"  PSO Best CV   : {pso_config['best_cv_accuracy']*100:.2f}%")

# ============================================================
# TRAIN PSO-MLP WITH OPTIMAL CONFIG
# ============================================================
print("\nTraining PSO-MLP with optimal hyperparameters...")

mlp_pso = MLPClassifier(
    hidden_layer_sizes=(n1, n2),
    learning_rate_init=lr,
    alpha=al,
    max_iter=500,
    random_state=42
)
mlp_pso.fit(X_train_bal, y_train_bal)

preds_pso = mlp_pso.predict(X_test)
acc_pso   = accuracy_score(y_test, preds_pso)
f1_pso    = f1_score(y_test, preds_pso, average='weighted', zero_division=0)

print("\n" + "="*40)
print("  PSO-MLP Results")
print("="*40)
print(f"  Accuracy : {acc_pso*100:.2f}%")
print(f"  F1 Score : {f1_pso:.4f}")
print("="*40)

labels_map     = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}
unique_classes = np.unique(np.concatenate([y_test, preds_pso]))
target_names   = [labels_map[i] for i in unique_classes]

print("\nDetailed Classification Report:")
print(classification_report(y_test, preds_pso,
      labels=unique_classes, target_names=target_names, zero_division=0))

# ============================================================
# FINAL EVALUATION
# ============================================================
# We remove the manual 'fuzzy_sugeno_grade' wrapper and use raw model predictions.
# This removes arbitrary bias and shows the genuine performance of the system.
def get_final_prediction(model, X):
    return model.predict(X)

# Demo on 3 sample predictions
print("\nFinal Model Predictions --- Sample Results:")
print("-" * 50)
raw_scores = mlp_pso.predict_proba(X_test)
for i in range(min(3, len(X_test))):
    pred_class = preds_pso[i]
    prob = max(raw_scores[i])
    print(f"  Sample {i+1}: Predicted Class = {labels_map[pred_class]} | Confidence = {prob*100:.2f}%")

# ============================================================
# SAVE
# ============================================================
joblib.dump(mlp_pso, 'models/mlp_pso.pkl')
np.save('models/preds_pso.npy', preds_pso)
joblib.dump({'accuracy': acc_pso, 'f1': f1_pso}, 'models/mlp_pso_results.pkl')
print("\nPSO-MLP model saved to models/ folder.")

# ============================================================
# LOAD PREVIOUS RESULTS FOR COMPARISON
# ============================================================
fuzzy_results   = joblib.load('models/fuzzy_results.pkl')
standard_results= joblib.load('models/mlp_standard_results.pkl')

acc_fuzzy    = fuzzy_results['accuracy']
f1_fuzzy     = fuzzy_results['f1']
acc_standard = standard_results['accuracy']
f1_standard  = standard_results['f1']

# ============================================================
# THREE-WAY COMPARISON TABLE
# ============================================================
print("\n" + "="*55)
print(f"{'System':<25} {'Accuracy':>10} {'F1 Score':>10}")
print("="*55)
print(f"{'Plain Fuzzy Sugeno':<25} {acc_fuzzy*100:>9.2f}% {f1_fuzzy:>10.4f}")
print(f"{'Standard MLP':<25} {acc_standard*100:>9.2f}% {f1_standard:>10.4f}")
print(f"{'PSO-MLP':<25} {acc_pso*100:>9.2f}% {f1_pso:>10.4f}")
print("="*55)

# ============================================================
# PLOTS
# ============================================================
# Plot 1 - Three-Way Comparison Bar Chart
systems    = ['Plain Fuzzy\nSugeno', 'Standard\nMLP', 'PSO-MLP\n+ Fuzzy Output']
accuracies = [acc_fuzzy*100, acc_standard*100, acc_pso*100]
f1s        = [f1_fuzzy, f1_standard, f1_pso]
colors     = ['tomato', 'steelblue', 'darkorange']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(systems, accuracies, color=colors, edgecolor='black', alpha=0.85)
axes[0].set_title('Classification Accuracy (%)', fontsize=12)
axes[0].set_ylabel('Accuracy %')
axes[0].set_ylim(0, 100)
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(accuracies):
    axes[0].text(i, val+1, f'{val:.2f}%', ha='center', fontsize=11)

axes[1].bar(systems, f1s, color=colors, edgecolor='black', alpha=0.85)
axes[1].set_title('Weighted F1 Score', fontsize=12)
axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(f1s):
    axes[1].text(i, val+0.01, f'{val:.4f}', ha='center', fontsize=11)

for ax in axes:
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems, rotation=10, ha='right', fontsize=10)

plt.suptitle('Three-Way Classification Comparison --- Final Results', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/final_comparison.png', dpi=150)
plt.close()
print("Saved: final_comparison.png")

# Plot 2 - PSO-MLP Training Loss
plt.figure(figsize=(9, 5))
plt.plot(mlp_pso.loss_curve_, color='darkorange', linewidth=2)
plt.xlabel('Epoch',  fontsize=12)
plt.ylabel('Loss',   fontsize=12)
plt.title('PSO-MLP --- Training Loss Curve', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/pso_mlp_loss.png', dpi=150)
plt.close()
print("Saved: pso_mlp_loss.png")

# Plot 3 - PSO-MLP Predicted vs Actual
x_pos = np.arange(len(X_test))
plt.figure(figsize=(12, 5))
plt.plot(x_pos, y_test,    'bo', alpha=0.5, markersize=5, label='Actual')
plt.plot(x_pos, preds_pso, 'rx', alpha=0.5, markersize=5, label='Predicted')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Risk Class',   fontsize=12)
plt.yticks([0,1,2,3], ['No Fire','Low','Moderate','High'])
plt.title('PSO-MLP --- Predicted vs Actual', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/pso_mlp_pred_vs_actual.png', dpi=150)
plt.close()
print("Saved: pso_mlp_pred_vs_actual.png")

# Plot 4 - PSO-MLP Confusion Matrix
cm   = confusion_matrix(y_test, preds_pso, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap='Oranges', colorbar=False)
plt.title('PSO-MLP --- Confusion Matrix', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/pso_mlp_confusion.png', dpi=150)
plt.close()
print("Saved: pso_mlp_confusion.png")

print("\n" + "="*55)
print("  PHASE 5 COMPLETE")
print("="*55)
