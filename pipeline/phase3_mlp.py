# ============================================================
# PHASE 3 - STANDARD MLP BASELINE
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
print("  PHASE 3 - STANDARD MLP BASELINE")
print("="*55)

# ============================================================
# LOAD DATA
# ============================================================
X_train_bal = np.load('models/X_train_bal.npy')
y_train_bal = np.load('models/y_train_bal.npy')
X_test      = np.load('models/X_test.npy')
y_test      = np.load('models/y_test.npy')

print(f"\nTraining samples : {X_train_bal.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")
print(f"Classes          : {np.unique(y_train_bal)}")

# ============================================================
# TRAIN STANDARD MLP
# ============================================================
print("\nTraining Standard MLP...")

mlp_standard = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    learning_rate_init=0.001,
    alpha=0.01,
    max_iter=500,
    random_state=42
)
mlp_standard.fit(X_train_bal, y_train_bal)

preds_standard = mlp_standard.predict(X_test)
acc_standard   = accuracy_score(y_test, preds_standard)
f1_standard    = f1_score(y_test, preds_standard, average='weighted', zero_division=0)

print("\n" + "="*40)
print("  Standard MLP Results")
print("="*40)
print(f"  Accuracy : {acc_standard*100:.2f}%")
print(f"  F1 Score : {f1_standard:.4f}")
print("="*40)

labels_map     = {0:'No Fire', 1:'Low Risk', 2:'Moderate', 3:'High Risk'}
unique_classes = np.unique(np.concatenate([y_test, preds_standard]))
target_names   = [labels_map[i] for i in unique_classes]

print("\nDetailed Classification Report:")
print(classification_report(y_test, preds_standard,
      labels=unique_classes, target_names=target_names, zero_division=0))

# ============================================================
# SAVE
# ============================================================
joblib.dump(mlp_standard, 'models/mlp_standard.pkl')
np.save('models/preds_standard.npy', preds_standard)
joblib.dump({'accuracy': acc_standard, 'f1': f1_standard},
            'models/mlp_standard_results.pkl')
print("Model saved to models/ folder.")

# ============================================================
# PLOTS
# ============================================================
# Plot 1 - Training Loss Curve
plt.figure(figsize=(9, 5))
plt.plot(mlp_standard.loss_curve_, color='steelblue', linewidth=2)
plt.xlabel('Epoch',  fontsize=12)
plt.ylabel('Loss',   fontsize=12)
plt.title('Standard MLP --- Training Loss Curve', fontsize=13)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/mlp_standard_loss.png', dpi=150)
plt.close()
print("Saved: mlp_standard_loss.png")

# Plot 2 - Predicted vs Actual
x_pos = np.arange(len(X_test))
plt.figure(figsize=(12, 5))
plt.plot(x_pos, y_test,          'bo', alpha=0.5, markersize=5, label='Actual')
plt.plot(x_pos, preds_standard,  'rx', alpha=0.5, markersize=5, label='Predicted')
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Risk Class',   fontsize=12)
plt.yticks([0,1,2,3], ['No Fire','Low','Moderate','High'])
plt.title('Standard MLP --- Predicted vs Actual', fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/mlp_standard_pred_vs_actual.png', dpi=150)
plt.close()
print("Saved: mlp_standard_pred_vs_actual.png")

# Plot 3 - Confusion Matrix
cm   = confusion_matrix(y_test, preds_standard, labels=unique_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Standard MLP --- Confusion Matrix', fontsize=13)
plt.tight_layout()
plt.savefig('outputs/mlp_standard_confusion.png', dpi=150)
plt.close()
print("Saved: mlp_standard_confusion.png")

print("\n" + "="*55)
print("  PHASE 3 COMPLETE")
print("="*55)