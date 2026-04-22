"""
Data Preprocessing Module
Handles data cleaning, normalization, outlier removal, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Data preprocessing pipeline for wildfire prediction"""
    
    def __init__(self, scaler_type='minmax'):
        """
        Initialize preprocessor
        
        Args:
            scaler_type: 'minmax' or 'standard'
        """
        self.scaler_type = scaler_type
        if scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.feature_names = None
        self.target_encoder = None
    
    def load_algerian_dataset(self, filepath):
        """
        Load and clean Algerian Forest Fires dataset
        
        Args:
            filepath: Path to the dataset
            
        Returns:
            Cleaned DataFrame
        """
        # Indian dataset has clean headers
        df = pd.read_csv(filepath)
        
        # Clean column names (remove spaces)
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        numeric_cols = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 
                       'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with invalid numeric values
        df = df.dropna(subset=numeric_cols)
        
        # Encode target variable using linguistic fire risk levels based on FWI
        df['Classes'] = df['Classes'].str.lower()
        df['Classes'] = df['Classes'].str.replace(' ', '')  # Remove spaces
        
        # Create linguistic target based on FWI (Fire Weather Index) - adjusted for realistic Indian conditions
        def fwi_to_linguistic(fwi):
            if fwi < 1:
                return 'No Fire'
            elif fwi < 5:
                return 'Low Fire'
            elif fwi < 10:
                return 'Medium Fire'
            elif fwi < 18:
                return 'High Fire'
            else:
                return 'Extreme Fire'
        
        df['linguistic_target'] = df['FWI'].apply(fwi_to_linguistic)
        
        # Also keep binary target for compatibility
        df['target'] = (df['Classes'] == 'fire').astype(int)
        
        return df
    
    def remove_outliers(self, X, y, method='iqr', threshold=1.5):
        """
        Remove outliers from the dataset
        
        Args:
            X: Feature matrix
            y: Target vector
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            X, y with outliers removed
        """
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
            X_clean = X[mask]
            y_clean = y[mask]
        
        elif method == 'zscore':
            z_scores = np.abs((X - X.mean(axis=0)) / X.std(axis=0))
            mask = np.all(z_scores < threshold, axis=1)
            X_clean = X[mask]
            y_clean = y[mask]
        
        else:
            X_clean = X
            y_clean = y
        
        print(f"Outlier removal: {len(X)} -> {len(X_clean)} samples")
        return X_clean, y_clean
    
    def normalize_features(self, X, fit=True):
        """
        Normalize features using scaler
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler
            
        Returns:
            Normalized feature matrix
        """
        if fit:
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = self.scaler.transform(X)
        
        # Handle NaN values
        X_normalized = np.nan_to_num(X_normalized, nan=0.0)
        
        return X_normalized
    
    def apply_smote(self, X_train, y_train, sampling_strategy='auto', random_state=42):
        """
        Apply SMOTE to balance the dataset with handling for small classes
        
        Args:
            X_train: Training features
            y_train: Training labels
            sampling_strategy: SMOTE sampling strategy
            random_state: Random seed
            
        Returns:
            Balanced X_train, y_train
        """
        print(f"Before SMOTE: {Counter(y_train)}")
        
        # Get class counts
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        
        # Adjust k_neighbors based on minimum samples
        k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
        
        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy, 
                random_state=random_state,
                k_neighbors=k_neighbors
            )
            X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {Counter(y_train_bal)}")
        except Exception as e:
            print(f"SMOTE failed: {e}, using original data")
            X_train_bal, y_train_bal = X_train, y_train
        
        return X_train_bal, y_train_bal
    
    def prepare_features(self, df):
        """
        Prepare feature matrix from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Feature matrix and target vector
        """
        # Select relevant features - Algerian dataset uses specific column names
        feature_cols = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
        
        # Ensure all columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if len(available_cols) < len(feature_cols):
            print(f"Warning: Some features missing. Available: {available_cols}")
        
        X = df[available_cols].values
        
        # Use linguistic target encoding
        linguistic_map = {
            'No Fire': 0,
            'Low Fire': 1,
            'Medium Fire': 2,
            'High Fire': 3,
            'Extreme Fire': 4
        }
        y = df['linguistic_target'].map(linguistic_map).values
        
        self.feature_names = available_cols
        self.linguistic_labels = list(linguistic_map.keys())
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Remove stratify to handle small classes
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def preprocess_pipeline(self, filepath, remove_outliers_flag=True, apply_smote_flag=False):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to dataset
            remove_outliers_flag: Whether to remove outliers
            apply_smote_flag: Whether to apply SMOTE (disabled for multi-class)
            
        Returns:
            Dictionary with all processed data
        """
        # Load data
        df = self.load_algerian_dataset(filepath)
        print(f"Loaded dataset: {df.shape}")
        
        # Prepare features
        X, y = self.prepare_features(df)
        print(f"Features: {self.feature_names}")
        print(f"Target distribution: {Counter(y)}")
        
        # Remove outliers
        if remove_outliers_flag:
            X, y = self.remove_outliers(X, y, method='iqr')
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Normalize
        X_train_norm = self.normalize_features(X_train, fit=True)
        X_test_norm = self.normalize_features(X_test, fit=False)
        
        # Apply SMOTE (disabled for multi-class to avoid small class issues)
        if apply_smote_flag:
            X_train_bal, y_train_bal = self.apply_smote(X_train_norm, y_train)
        else:
            X_train_bal, y_train_bal = X_train_norm, y_train
        
        return {
            'X_train': X_train_bal,
            'y_train': y_train_bal,
            'X_test': X_test_norm,
            'y_test': y_test,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
    
    def save_preprocessor(self, filepath):
        """Save preprocessor to disk"""
        joblib.dump({
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'feature_names': self.feature_names
        }, filepath)
    
    def load_preprocessor(self, filepath):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        self.scaler = data['scaler']
        self.scaler_type = data['scaler_type']
        self.feature_names = data['feature_names']
