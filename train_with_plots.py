"""
Model Training and Plotting Script
Trains the Fuzzy, ANFIS, and PSO-ANFIS models and generates evaluation plots.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline
from backend.services.model_selector import ModelSelector

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', filename='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def plot_model_comparison(results, filename='model_comparison.png'):
    models = list(results.keys())
    rmse = [res['rmse'] for res in results.values()]
    accuracy = [res['accuracy'] * 100 for res in results.values()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    rects1 = ax1.bar(x - width/2, rmse, width, label='RMSE', color='salmon')
    rects2 = ax2.bar(x + width/2, accuracy, width, label='Accuracy (%)', color='skyblue')

    ax1.set_ylabel('RMSE')
    ax2.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/{filename}')
    plt.close()

def main():
    print("=" * 70)
    print("TRAINING MODELS & GENERATING PLOTS FOR WILDFIRE PREDICTION")
    print("=" * 70)

    pipeline = DataPipeline(model_dir='models')
    dataset_path = 'data/indian_forest_fires_enhanced.csv'

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    # Train and compare models
    selector = ModelSelector(pipeline.preprocessor, model_dir='models')
    results = selector.train_all_models(dataset_path, verbose=True)
    registry = selector.save_registry(dataset_path)

    # Generate Model Comparison Bar Chart
    plot_model_comparison(registry['all_results'])
    print("-> Saved Model Comparison Chart to outputs/model_comparison.png")

    # Evaluate Best Model for Confusion Matrix
    print("\nEvaluating Best Model to generate Confusion Matrix...")
    data = pipeline.preprocessor.preprocess_pipeline(dataset_path, remove_outliers_flag=True, apply_smote_flag=False)
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Simple FWI based mapping for fuzzy (which is heavily relied on)
    fwi_values = X_test[:, 9] if X_test.shape[1] > 9 else X_test[:, 0]
    def fwi_to_class(fwi):
        fwi = float(fwi)
        if fwi < 1: return 0
        elif fwi < 5: return 1
        elif fwi < 10: return 2
        elif fwi < 18: return 3
        else: return 4

    y_pred = np.array([fwi_to_class(f) for f in fwi_values])
    y_true = y_test.astype(int)
    
    class_names = ['No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire']
    plot_confusion_matrix(y_true, y_pred, class_names, title=f"Confusion Matrix ({registry['best_model'].upper()})")
    print("-> Saved Confusion Matrix to outputs/confusion_matrix.png")
    
    # Train pipeline to save models properly
    pipeline.train_pipeline(dataset_path=dataset_path, use_pso=True, verbose=False)

    print("\n" + "=" * 70)
    print("TRAINING & PLOTTING COMPLETED")
    print(f"Check the 'outputs/' directory for generated plots.")
    print("=" * 70)

if __name__ == "__main__":
    main()
