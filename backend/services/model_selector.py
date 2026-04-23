"""
Model Selector Module
Trains and compares Fuzzy, ANFIS, and PSO-ANFIS models,
then selects the best one based on weighted scoring.
"""

import numpy as np
import json
import os
import time
from typing import Dict, Tuple, Optional
from backend.models.anfis import ANFIS
from backend.models.pso import PSOANFISOptimizer
from backend.models.fuzzy_wildfire import create_fuzzy_wildfire_system
from backend.utils.preprocessing import DataPreprocessor
from backend.utils.logger import system_logger


class ModelSelector:
    """
    Trains and selects the best model from Fuzzy, ANFIS, and PSO-ANFIS.
    """

    def __init__(self, preprocessor: DataPreprocessor, model_dir: str = 'models'):
        self.preprocessor = preprocessor
        self.model_dir = model_dir
        self.results = {}
        self.best_model_name = None
        self.best_model = None
        self.best_metrics = None

        os.makedirs(model_dir, exist_ok=True)

    def train_all_models(self, dataset_path: str, verbose: bool = True) -> Dict:
        """
        Train all three model types and compare them.

        Args:
            dataset_path: Path to dataset
            verbose: Print progress

        Returns:
            Dictionary with comparison results
        """
        if verbose:
            print("=" * 70)
            print("MODEL TRAINING AND COMPARISON")
            print("=" * 70)

        # Preprocess data
        data = self.preprocessor.preprocess_pipeline(
            dataset_path,
            remove_outliers_flag=True,
            apply_smote_flag=False
        )

        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']

        if verbose:
            print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
            print()

        # ---- 1. Train Fuzzy System ----
        if verbose:
            print("-" * 50)
            print("Training FUZZY LOGIC SYSTEM...")
            print("-" * 50)

        fuzzy_model = create_fuzzy_wildfire_system()

        fuzzy_metrics = self._evaluate_fuzzy(
            fuzzy_model, X_train, y_train, X_test, y_test, verbose
        )
        self.results['fuzzy'] = fuzzy_metrics

        if verbose:
            print(f"  RMSE: {fuzzy_metrics['rmse']:.4f}")
            print(f"  MAE:  {fuzzy_metrics['mae']:.4f}")
            print(f"  Accuracy: {fuzzy_metrics['accuracy']*100:.2f}%")
            print()

        # ---- 2. Train ANFIS ----
        if verbose:
            print("-" * 50)
            print("Training ANFIS (Adaptive Neuro-Fuzzy Inference System)...")
            print("-" * 50)

        anfis_model = ANFIS(num_inputs=X_train.shape[1], num_mfs_per_input=2)

        start = time.time()
        anfis_model.hybrid_train(X_train, y_train, epochs=1, lr=0.01)
        anfis_train_time = time.time() - start

        anfis_metrics = self._evaluate_anfis(
            anfis_model, X_train, y_train, X_test, y_test, verbose
        )
        anfis_metrics['training_time'] = anfis_train_time
        self.results['anfis'] = anfis_metrics

        if verbose:
            print(f"  RMSE: {anfis_metrics['rmse']:.4f}")
            print(f"  MAE:  {anfis_metrics['mae']:.4f}")
            print(f"  Accuracy: {anfis_metrics['accuracy']*100:.2f}%")
            print(f"  Training time: {anfis_train_time:.2f}s")
            print()

        # ---- 3. Train PSO-ANFIS ----
        if verbose:
            print("-" * 50)
            print("Training PSO-ANFIS (PSO-optimized ANFIS)...")
            print("-" * 50)

        pso_anfis_model = ANFIS(num_inputs=X_train.shape[1], num_mfs_per_input=2)
        pso_optimizer = PSOANFISOptimizer(
            pso_anfis_model,
            num_particles=20,
            max_iterations=20
        )

        start = time.time()
        pso_optimizer.optimize(X_train, y_train, verbose=verbose)
        pso_train_time = time.time() - start

        pso_metrics = self._evaluate_anfis(
            pso_anfis_model, X_train, y_train, X_test, y_test, verbose
        )
        pso_metrics['training_time'] = pso_train_time
        self.results['pso_anfis'] = pso_metrics

        if verbose:
            print(f"  RMSE: {pso_metrics['rmse']:.4f}")
            print(f"  MAE:  {pso_metrics['mae']:.4f}")
            print(f"  Accuracy: {pso_metrics['accuracy']*100:.2f}%")
            print(f"  Training time: {pso_train_time:.2f}s")
            print()

        # ---- Select Best Model ----
        self._select_best_model()

        # ---- Summary ----
        if verbose:
            print("=" * 70)
            print("MODEL COMPARISON SUMMARY")
            print("=" * 70)
            print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'Accuracy':<12} {'Score':<10}")
            print("-" * 60)
            for name, res in self.results.items():
                score = self._compute_score(res)
                marker = "  <-- BEST" if name == self.best_model_name else ""
                print(f"{name:<15} {res['rmse']:<10.4f} {res['mae']:<10.4f} "
                      f"{res['accuracy']*100:<12.2f} {score:<10.4f}{marker}")
            print()
            print(f"Best model selected: {self.best_model_name.upper()}")
            print("=" * 70)

        return self.results

    def _evaluate_fuzzy(self, fuzzy_model, X_train, y_train,
                        X_test, y_test, verbose=False) -> Dict:
        """
        Evaluate Fuzzy system by mapping FWI values to risk classes.
        Since Fuzzy system uses weather inputs, we use the test set's
        FWI values to derive risk scores and compare with targets.
        """
        # Fuzzy evaluates weather features, so we approximate by using
        # the FWI column directly mapped to classes, then comparing with ground truth
        # This is the same evaluation approach used in the original pipeline

        fwi_values = X_test[:, 9] if X_test.shape[1] > 9 else X_test[:, 0]

        def fwi_to_class(fwi):
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

        y_pred = np.array([fwi_to_class(f) for f in fwi_values])
        y_true = y_test.astype(int)

        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        mae = np.mean(np.abs(y_pred - y_true))
        accuracy = np.mean(y_pred == y_true)

        return {'rmse': float(rmse), 'mae': float(mae), 'accuracy': float(accuracy)}

    def _evaluate_anfis(self, anfis_model, X_train, y_train,
                       X_test, y_test, verbose=False) -> Dict:
        """Evaluate ANFIS model."""
        y_pred_raw = anfis_model.predict(X_test).flatten()

        # Denormalize predictions to original scale
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-8
        y_pred_denorm = y_pred_raw * y_std + y_mean

        # Map to classes
        def to_class(v):
            v = float(v)
            if v < 1:
                return 0
            elif v < 2.5:
                return 1
            elif v < 3.5:
                return 2
            elif v < 4.5:
                return 3
            else:
                return 4

        y_pred_classes = np.array([to_class(v) for v in y_pred_denorm])
        y_true = y_test.astype(int)

        rmse = np.sqrt(np.mean((y_pred_classes - y_true) ** 2))
        mae = np.mean(np.abs(y_pred_classes - y_true))
        accuracy = np.mean(y_pred_classes == y_true)

        return {'rmse': float(rmse), 'mae': float(mae), 'accuracy': float(accuracy)}

    def _compute_score(self, metrics: Dict) -> float:
        """
        Compute weighted score: RMSE (40%), MAE (30%), Accuracy (30%).
        Lower is better.
        """
        # Normalize metrics to 0-1 range for fair comparison
        rmse_norm = min(metrics['rmse'] / 2.0, 1.0)
        mae_norm = min(metrics['mae'] / 2.0, 1.0)
        acc_norm = metrics['accuracy']

        score = 0.4 * rmse_norm + 0.3 * mae_norm + 0.3 * (1 - acc_norm)
        return round(score, 4)

    def _select_best_model(self):
        """Select best model based on weighted scoring."""
        scores = {}
        for name, metrics in self.results.items():
            scores[name] = self._compute_score(metrics)

        self.best_model_name = min(scores, key=scores.get)
        self.best_metrics = self.results[self.best_model_name]

    def get_best_model(self):
        """Return the best model name."""
        return self.best_model_name

    def save_registry(self, dataset_path: str):
        """Save comparison results to model registry JSON."""
        registry = {
            'best_model': self.best_model_name,
            'all_results': {
                name: {
                    'rmse': float(res['rmse']),
                    'mae': float(res['mae']),
                    'accuracy': float(res['accuracy']),
                    'score': self._compute_score(res)
                }
                for name, res in self.results.items()
            },
            'training_date': time.strftime('%Y-%m-%d'),
            'dataset': os.path.basename(dataset_path)
        }

        registry_path = os.path.join(self.model_dir, 'model_registry.json')
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        if self.best_model_name == 'pso_anfis':
            registry['best_model_path'] = os.path.join(self.model_dir, 'pso_optimizer.pkl')
        elif self.best_model_name == 'anfis':
            registry['best_model_path'] = os.path.join(self.model_dir, 'anfis_model.pkl')
        else:
            registry['best_model_path'] = os.path.join(self.model_dir, 'fuzzy_system.pkl')

        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)

        return registry
