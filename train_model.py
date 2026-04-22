"""
Model Training Script
Trains the PSO-ANFIS model on the Algerian Forest Fires dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.pipeline import DataPipeline
from backend.utils.logger import system_logger

def main():
    """Main training function"""
    print("=" * 70)
    print("TRAINING PSO-ANFIS WILDFIRE PREDICTION MODEL")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DataPipeline(model_dir='models')
    
    # Train the model
    dataset_path = 'data/indian_forest_fires.csv'
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please ensure the dataset is downloaded and placed in the data/ directory")
        return
    
    print(f"\nLoading dataset from: {dataset_path}")
    
    # Run training pipeline (PSO disabled for stability)
    results = pipeline.train_pipeline(
        dataset_path=dataset_path,
        use_pso=False,  # Disabled for stability
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Final RMSE: {results['rmse']:.4f}")
    print(f"Final MAE: {results['mae']:.4f}")
    print(f"Final Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Training Time: {results['training_time']:.2f}s")
    print("\nModels saved to models/ directory")
    print("You can now run the Streamlit app: streamlit run frontend/app.py")

if __name__ == "__main__":
    main()
