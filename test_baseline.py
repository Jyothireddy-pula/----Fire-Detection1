"""
Test baseline models to assess feature quality
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('.')

from backend.utils.preprocessing import DataPreprocessor

def test_baseline():
    """Test baseline models to assess feature quality"""
    print("=" * 60)
    print("BASELINE MODEL TESTING")
    print("=" * 60)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_algerian_dataset('data/indian_forest_fires.csv')
    df = df.dropna()
    
    # Prepare features
    X, y = preprocessor.prepare_features(df)
    
    print(f"Features: {preprocessor.feature_names}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    print(f"\nTrain samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # Test Random Forest
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_norm, y_train)
    y_pred_rf = rf.predict(X_test_norm)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, 
                                target_names=['No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire']))
    
    # Feature importance
    print("\nFeature Importances:")
    for name, importance in zip(preprocessor.feature_names, rf.feature_importances_):
        print(f"  {name}: {importance:.4f}")
    
    # Test Gradient Boosting
    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING")
    print("=" * 60)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_norm, y_train)
    y_pred_gb = gb.predict(X_test_norm)
    gb_acc = accuracy_score(y_test, y_pred_gb)
    print(f"Gradient Boosting Accuracy: {gb_acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_gb,
                                target_names=['No Fire', 'Low Fire', 'Medium Fire', 'High Fire', 'Extreme Fire']))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")
    print(f"Gradient Boosting Accuracy: {gb_acc*100:.2f}%")
    
    if rf_acc > 0.5:
        print("\n✓ Features are good - issue is with ANFIS implementation")
    else:
        print("\n✗ Features are weak - need better feature engineering")

if __name__ == "__main__":
    test_baseline()
