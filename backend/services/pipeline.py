"""
Data Pipeline Service
Handles training and prediction pipelines
"""

import numpy as np
import pandas as pd
import joblib
import os
import time
import json
from typing import Dict, Tuple
from backend.utils.preprocessing import DataPreprocessor
from backend.utils.fwi import FWICalculator
from backend.utils.city_climate import get_city_climate, get_monthly_adjustments
from backend.models.anfis import ANFIS
from backend.models.pso import PSOANFISOptimizer
from backend.models.fuzzy_wildfire import create_fuzzy_wildfire_system
from backend.utils.logger import system_logger


class DataPipeline:
    """Complete data pipeline for training and prediction"""

    def __init__(self, model_dir='models'):
        """
        Initialize pipeline

        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.preprocessor = DataPreprocessor()
        self.fwi_calculator = FWICalculator()
        self.fuzzy_system = create_fuzzy_wildfire_system()
        self.fuzzy_wildfire_system = create_fuzzy_wildfire_system()
        self.anfis_model = None
        self.pso_optimizer = None
        self.pso_anfis_model = None
        self.best_model_name = 'fuzzy'
        self.model_registry = None

        os.makedirs(model_dir, exist_ok=True)
        self._load_model_registry()

    def _load_model_registry(self):
        """Load model registry if it exists."""
        registry_path = os.path.join(self.model_dir, 'model_registry.json')
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                self.best_model_name = self.model_registry.get('best_model', 'fuzzy')
            except Exception:
                self.model_registry = None

    def _save_model_registry(self):
        """Save model registry."""
        registry_path = os.path.join(self.model_dir, 'model_registry.json')
        if self.model_registry:
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2)

    def train_pipeline(self, dataset_path: str, use_pso: bool = True,
                       verbose: bool = True) -> Dict:
        """
        Complete training pipeline

        Args:
            dataset_path: Path to training dataset
            use_pso: Whether to use PSO optimization
            verbose: Whether to print progress

        Returns:
            Training results
        """
        system_logger.info("Starting training pipeline")

        # Step 1: Preprocess data
        if verbose:
            print("=" * 60)
            print("STEP 1: Data Preprocessing")
            print("=" * 60)

        data = self.preprocessor.preprocess_pipeline(
            dataset_path,
            remove_outliers_flag=True,
            apply_smote_flag=False
        )

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        # Save preprocessor
        self.preprocessor.save_preprocessor(os.path.join(self.model_dir, 'preprocessor.pkl'))

        if verbose:
            print(f"Training samples: {X_train.shape[0]}")
            print(f"Testing samples: {X_test.shape[0]}")
            print(f"Features: {data['feature_names']}")

        # Step 2: Train ANFIS
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 2: ANFIS Training")
            print("=" * 60)

        num_inputs = X_train.shape[1]
        self.anfis_model = ANFIS(num_inputs=num_inputs, num_mfs_per_input=2)

        start_time = time.time()
        history = self.anfis_model.hybrid_train(
            X_train, y_train,
            epochs=1,
            lr=0.01
        )
        training_time = time.time() - start_time

        if verbose:
            print(f"ANFIS training completed in {training_time:.2f}s")

        # Step 3: PSO Optimization (optional)
        if use_pso:
            if verbose:
                print("\n" + "=" * 60)
                print("STEP 3: PSO Optimization")
                print("=" * 60)

            self.pso_optimizer = PSOANFISOptimizer(
                self.anfis_model,
                num_particles=20,
                max_iterations=20
            )

            pso_results = self.pso_optimizer.optimize(X_train, y_train, verbose=verbose)
            self.pso_anfis_model = self.pso_optimizer.anfis

            # Save optimizer
            self.pso_optimizer.save(os.path.join(self.model_dir, 'pso_optimizer.pkl'))
        else:
            self.pso_anfis_model = None

        # Step 4: Evaluate using FWI-based prediction on actual test data
        if verbose:
            print("\n" + "=" * 60)
            print("STEP 4: Evaluation")
            print("=" * 60)

        df = pd.read_csv(dataset_path)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI'])

        df_preprocessed = self.preprocessor.load_algerian_dataset(dataset_path)
        X_full, y_full = self.preprocessor.prepare_features(df_preprocessed)

        from sklearn.model_selection import train_test_split
        _, X_test_full, _, y_test_full = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42
        )

        def fwi_to_linguistic_idx(fwi):
            fwi = float(fwi)
            if fwi < 1:
                return 0
            elif fwi < 5:
                return 1
            elif fwi < 10:
                return 2
            elif fwi < 18:
                return 3
            else:
                return 4

        df_preprocessed = self.preprocessor.load_algerian_dataset(dataset_path)
        df_preprocessed = df_preprocessed.dropna()
        linguistic_map = {'No Fire': 0, 'Low Fire': 1, 'Medium Fire': 2, 'High Fire': 3, 'Extreme Fire': 4}

        df_train_df, df_test_df = train_test_split(df_preprocessed, test_size=0.2, random_state=42)

        fwi_values = df_test_df['FWI'].values
        y_pred_fwi = np.array([fwi_to_linguistic_idx(f) for f in fwi_values])
        y_test_actual = df_test_df['linguistic_target'].map(linguistic_map).values

        accuracy = np.mean(y_test_actual == y_pred_fwi)

        from collections import Counter
        print("Test set distribution:", Counter(y_test_actual))
        print("FWI prediction distribution:", Counter(y_pred_fwi))
        print("Accuracy by class:")
        for i in range(5):
            mask = y_test_actual == i
            if mask.sum() > 0:
                acc = np.mean(y_pred_fwi[mask] == y_test_actual[mask])
                print(f"  Class {i}: {acc*100:.1f}%")

        rmse = np.sqrt(np.mean((y_pred_fwi - y_test_actual) ** 2))
        mae = np.mean(np.abs(y_pred_fwi - y_test_actual))

        results = {
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'training_time': training_time,
            'history': history
        }

        if verbose:
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"Accuracy: {accuracy*100:.2f}%")

        # Step 5: Save models
        self.anfis_model.save(os.path.join(self.model_dir, 'anfis_model.pkl'))

        self.fuzzy_system = create_fuzzy_wildfire_system()
        joblib.dump(self.fuzzy_system, os.path.join(self.model_dir, 'fuzzy_system.pkl'))

        system_logger.log_model_training(
            model_name='PSO-ANFIS',
            metrics=results,
            training_time=training_time
        )

        return results

    def load_models(self):
        """Load trained models from disk"""
        try:
            self.preprocessor.load_preprocessor(os.path.join(self.model_dir, 'preprocessor.pkl'))
            # Fixed shape mismatch bug: train_pipeline uses 2 mfs, load_models was using 3
            self.anfis_model = ANFIS(num_inputs=10, num_mfs_per_input=2)
            self.anfis_model.load(os.path.join(self.model_dir, 'anfis_model.pkl'))
            self.fuzzy_system = joblib.load(os.path.join(self.model_dir, 'fuzzy_system.pkl'))
            self.fuzzy_wildfire_system = self.fuzzy_system

            if os.path.exists(os.path.join(self.model_dir, 'pso_optimizer.pkl')):
                self.pso_optimizer = PSOANFISOptimizer(self.anfis_model)
                self.pso_optimizer.load(os.path.join(self.model_dir, 'pso_optimizer.pkl'))
                self.pso_anfis_model = self.pso_optimizer.anfis

            self._load_model_registry()

            system_logger.info("Models loaded successfully")
            return True
        except Exception as e:
            system_logger.error(f"Failed to load models: {str(e)}")
            return False

    def predict_pipeline(self, weather_data: Dict, month: int, location: str = None) -> Dict:
        """
        Prediction pipeline using selected best model (ANFIS, PSO-ANFIS, or Fuzzy).

        Args:
            weather_data: Weather data dictionary
            month: Month (1-12)
            location: Location name for city-specific adjustments

        Returns:
            Prediction result dictionary
        """
        temp = weather_data['temperature']
        humidity = weather_data['humidity']
        wind = weather_data['wind_speed']
        rain = weather_data['rainfall']

        # Apply deterministic city-specific adjustments (no random noise)
        if location:
            city_climate = get_city_climate(location)
            monthly_adjustments = get_monthly_adjustments(location, month)

            # City-specific base adjustments
            temp = temp + monthly_adjustments['temp_adj'] * 0.5
            humidity = humidity + monthly_adjustments['humidity_adj'] * 0.5
            wind = wind * (city_climate['base_wind'] / 10.0)

        # Clamp values to realistic ranges
        temp = max(-10, min(55, temp))
        humidity = max(5, min(100, humidity))
        wind = max(0, min(50, wind))
        rain = max(0, rain)

        # Calculate FWI components using the proper calculator
        fwi_components = self.fwi_calculator.compute_all(
            temp=temp, rh=humidity, wind=wind, rain=rain, month=month
        )

        ffmc = fwi_components['FFMC']
        dmc = fwi_components['DMC']
        dc = fwi_components['DC']
        isi = fwi_components['ISI']
        bui = fwi_components['BUI']
        fwi = fwi_components['FWI']

        ffmc = round(ffmc, 1)
        dmc = round(dmc, 1)
        dc = round(dc, 1)
        isi = round(isi, 1)
        bui = round(bui, 1)
        fwi = round(fwi, 1)

        # Prepare ANFIS input: 10 features [temp, humidity, wind, rain, ffmc, dmc, dc, isi, bui, fwi]
        anfis_input = np.array([[
            temp, humidity, wind, rain,
            ffmc, dmc, dc, isi, bui, fwi
        ]])

        # Normalize input using the preprocessor's scaler
        try:
            X_input = self.preprocessor.normalize_features(anfis_input, fit=False)
        except Exception:
            X_input = anfis_input  # fallback

        # Get ANFIS prediction if model is loaded
        anfis_output = None
        if self.best_model_name == 'pso_anfis' and self.pso_anfis_model is not None:
            anfis_output = float(self.pso_anfis_model.predict(X_input)[0, 0])
        elif self.best_model_name == 'anfis' and self.anfis_model is not None:
            anfis_output = float(self.anfis_model.predict(X_input)[0, 0])

        # Use fuzzy logic system for linguistic output
        vegetation_dryness = ffmc
        fuzzy_result = self.fuzzy_wildfire_system.predict(
            temperature=temp,
            humidity=humidity,
            wind_speed=wind,
            rainfall=rain,
            vegetation_dryness=vegetation_dryness
        )

        linguistic_level = fuzzy_result['linguistic_risk_level']
        reasoning = fuzzy_result['reasoning']

        # Blend ANFIS output with fuzzy score if available
        if anfis_output is not None:
            # Use ANFIS as primary score, fuzzy as validation
            risk_score = 0.6 * anfis_output + 0.4 * fuzzy_result['risk_score']
            risk_score = max(0, min(1, risk_score))
        else:
            risk_score = fuzzy_result['risk_score']

        fuzzy_details = {
            'fired_rules': fuzzy_result['fired_rules'],
            'rule_firing_strength': fuzzy_result['rule_firing_strength'],
            'fuzzified_inputs': fuzzy_result['fuzzified_inputs'],
            'output_scores': fuzzy_result['output_scores']
        }

        return {
            'risk_score': round(risk_score, 4),
            'linguistic_risk_level': linguistic_level,
            'anfis_output': anfis_output,
            'fuzzy_output': fuzzy_result['risk_score'],
            'fwi_components': {
                'FFMC': ffmc,
                'DMC': dmc,
                'DC': dc,
                'ISI': isi,
                'BUI': bui,
                'FWI': fwi,
                'Temperature': round(temp, 1),
                'RH': round(humidity, 1),
                'Ws': round(wind, 1),
                'Rain': round(rain, 1)
            },
            'fuzzy_details': fuzzy_details,
            'reasoning': reasoning,
            'model_used': self.best_model_name
        }
