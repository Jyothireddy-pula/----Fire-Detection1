"""
Model Training Script
Trains and compares Fuzzy, ANFIS, and PSO-ANFIS models,
then selects the best performer for the prediction pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline
from backend.services.model_selector import ModelSelector
from backend.utils.logger import system_logger


def main():
    """Main training function"""
    print("=" * 70)
    print("TRAINING SOFT COMPUTING MODELS FOR WILDFIRE PREDICTION")
    print("=" * 70)
    print("Comparing: Fuzzy Logic | ANFIS | PSO-ANFIS")
    print("=" * 70)

    # Initialize pipeline
    pipeline = DataPipeline(model_dir='models')

    dataset_path = 'data/indian_forest_fires_enhanced.csv'

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure the dataset is placed in the data/ directory")
        return

    print(f"\nLoading dataset from: {dataset_path}\n")

    # Use model selector to train and compare all three models
    selector = ModelSelector(pipeline.preprocessor, model_dir='models')
    results = selector.train_all_models(dataset_path, verbose=True)

    # Save model registry with best model info
    registry = selector.save_registry(dataset_path)

    # Train the full pipeline (for saving models)
    print("\n" + "=" * 70)
    print("TRAINING FINAL PIPELINE WITH BEST MODEL")
    print("=" * 70)

    pipeline_results = pipeline.train_pipeline(
        dataset_path=dataset_path,
        use_pso=True,  # Enable PSO
        verbose=True
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Best model: {registry['best_model'].upper()}")
    print(f"Best RMSE: {registry['all_results'][registry['best_model']]['rmse']:.4f}")
    print(f"Best Accuracy: {registry['all_results'][registry['best_model']]['accuracy']*100:.2f}%")
    print(f"\nModel registry saved to: models/model_registry.json")
    print("Run the app: streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
