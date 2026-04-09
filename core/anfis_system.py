"""
Advanced ANFIS (Adaptive Neuro-Fuzzy Inference System) Implementation
for Wildfire Risk Prediction System
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.optimizers import Adam
import joblib
import warnings
warnings.filterwarnings('ignore')

class ANFISLayer(tf.keras.layers.Layer):
    """Custom ANFIS layer for fuzzy inference"""
    
    def __init__(self, n_inputs, n_mf, **kwargs):
        super(ANFISLayer, self).__init__(**kwargs)
        self.n_inputs = n_inputs
        self.n_mf = n_mf
        self.n_rules = n_mf ** n_inputs
        
    def build(self, input_shape):
        # Initialize membership function parameters (Gaussian)
        self.centers = self.add_weight(
            name='centers',
            shape=(self.n_inputs, self.n_mf),
            initializer='glorot_uniform',
            trainable=True
        )
        self.scales = self.add_weight(
            name='scales',
            shape=(self.n_inputs, self.n_mf),
            initializer='ones',
            trainable=True
        )
        
        # Initialize consequent parameters
        self.consequent = self.add_weight(
            name='consequent',
            shape=(self.n_rules, self.n_inputs + 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(ANFISLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Layer 1: Fuzzification
        inputs_expanded = tf.expand_dims(inputs, axis=2)  # (batch, n_inputs, 1)
        centers_expanded = tf.expand_dims(self.centers, axis=0)  # (1, n_inputs, n_mf)
        scales_expanded = tf.expand_dims(self.scales, axis=0)  # (1, n_inputs, n_mf)
        
        # Gaussian membership functions
        mf = tf.exp(-tf.square((inputs_expanded - centers_expanded) / scales_expanded))
        
        # Layer 2: Rule strength (product t-norm)
        batch_size = tf.shape(inputs)[0]
        rule_strength = tf.ones((batch_size, self.n_rules))
        
        rule_idx = 0
        for i in range(self.n_inputs):
            for j in range(self.n_mf):
                if rule_idx < self.n_rules:
                    rule_strength = rule_strength * tf.expand_dims(mf[:, i, j], axis=1)
                    rule_idx += 1
        
        # Layer 3: Normalization
        rule_strength_sum = tf.reduce_sum(rule_strength, axis=1, keepdims=True)
        normalized_strength = rule_strength / (rule_strength_sum + 1e-10)
        
        # Layer 4: Consequent
        inputs_with_bias = tf.concat([inputs, tf.ones((batch_size, 1))], axis=1)
        consequent_outputs = tf.matmul(inputs_with_bias, self.consequent, transpose_b=True)
        
        # Layer 5: Defuzzification (weighted sum)
        output = tf.reduce_sum(normalized_strength * consequent_outputs, axis=1)
        
        return output

class AdvancedANFIS:
    """Advanced ANFIS system with PSO optimization"""
    
    def __init__(self, n_inputs=12, n_mf=3, learning_rate=0.01):
        self.n_inputs = n_inputs
        self.n_mf = n_mf
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        """Build the ANFIS model using TensorFlow"""
        # Input layer
        input_layer = Input(shape=(self.n_inputs,), name='input')
        
        # ANFIS layer
        anfis_layer = ANFISLayer(self.n_inputs, self.n_mf, name='anfis')
        output = anfis_layer(input_layer)
        
        # Create model
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """Train the ANFIS model"""
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        else:
            validation_data = None
        
        # Build and train model
        self.build_model()
        
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        X_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_scaled)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        # For classification accuracy (if needed)
        y_pred_class = (predictions > 0.5).astype(int)
        y_test_class = (y_test > 0.5).astype(int)
        accuracy = accuracy_score(y_test_class, y_pred_class)
        
        return {
            'mse': mse,
            'mae': mae,
            'accuracy': accuracy,
            'predictions': predictions
        }
    
    def get_membership_functions(self, X_sample):
        """Get membership function values for visualization"""
        X_scaled = self.scaler.transform(X_sample.reshape(1, -1))
        
        # Get the ANFIS layer
        anfis_layer = self.model.get_layer('anfis')
        
        # Calculate membership functions
        inputs_expanded = tf.expand_dims(tf.constant(X_scaled, dtype=tf.float32), axis=2)
        centers_expanded = tf.expand_dims(anfis_layer.centers, axis=0)
        scales_expanded = tf.expand_dims(anfis_layer.scales, axis=0)
        
        mf_values = tf.exp(-tf.square((inputs_expanded - centers_expanded) / scales_expanded))
        
        return mf_values.numpy().squeeze()
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        joblib.dump(self.scaler, filepath.replace('.h5', '_scaler.pkl'))
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath, custom_objects={'ANFISLayer': ANFISLayer})
        self.scaler = joblib.load(filepath.replace('.h5', '_scaler.pkl'))

class PSO_ANFIS_Optimizer:
    """PSO optimizer for ANFIS parameters"""
    
    def __init__(self, n_particles=20, n_iterations=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w  # inertia weight
        self.c1 = c1  # cognitive parameter
        self.c2 = c2  # social parameter
        
        self.best_position = None
        self.best_score = float('inf')
        self.history = []
        
    def optimize(self, X_train, y_train, X_val, y_val):
        """Optimize ANFIS parameters using PSO"""
        n_inputs = X_train.shape[1]
        
        # Initialize particles (learning_rate, n_mf, batch_size, epochs)
        particles = np.random.uniform(0, 1, (self.n_particles, 4))
        velocities = np.random.uniform(-0.1, 0.1, (self.n_particles, 4))
        
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.n_particles, float('inf'))
        
        for iteration in range(self.n_iterations):
            iteration_scores = []
            
            for i in range(self.n_particles):
                # Decode particle position
                lr = 0.001 + particles[i, 0] * 0.099  # 0.001 to 0.1
                n_mf = int(2 + particles[i, 1] * 4)  # 2 to 6
                batch_size = int(16 + particles[i, 2] * 48)  # 16 to 64
                epochs = int(50 + particles[i, 3] * 150)  # 50 to 200
                
                # Train ANFIS with these parameters
                anfis = AdvancedANFIS(n_inputs=n_inputs, n_mf=n_mf, learning_rate=lr)
                history = anfis.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
                
                # Evaluate
                results = anfis.evaluate(X_val, y_val)
                score = results['mse']  # Minimize MSE
                
                iteration_scores.append(score)
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()
                
                # Update global best
                if score < self.best_score:
                    self.best_score = score
                    self.best_position = particles[i].copy()
                    self.best_anfis = anfis
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = np.random.random(2)
                velocities[i] = (self.w * velocities[i] + 
                                self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                self.c2 * r2 * (self.best_position - particles[i]))
                particles[i] += velocities[i]
                
                # Keep particles in bounds
                particles[i] = np.clip(particles[i], 0, 1)
            
            avg_score = np.mean(iteration_scores)
            self.history.append(avg_score)
            
            print(f"Iteration {iteration + 1}/{self.n_iterations} - Best MSE: {self.best_score:.6f}, Avg MSE: {avg_score:.6f}")
        
        return self.best_anfis, self.best_position, self.history

def demo_anfis():
    """Demonstration of ANFIS system"""
    print("="*60)
    print("ADVANCED ANFIS SYSTEM FOR WILDFIRE RISK PREDICTION")
    print("="*60)
    
    # Generate sample data (replace with real dataset)
    np.random.seed(42)
    n_samples = 1000
    n_features = 12
    
    X = np.random.randn(n_samples, n_features)
    # Create synthetic target (wildfire risk score 0-1)
    y = 1 / (1 + np.exp(-(0.3*X[:, 0] - 0.2*X[:, 1] + 0.1*X[:, 2] + np.random.randn(n_samples)*0.1)))
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # PSO-ANFIS Optimization
    print("\nStarting PSO-ANFIS Optimization...")
    pso_optimizer = PSO_ANFIS_Optimizer(n_particles=10, n_iterations=20)
    best_anfis, best_params, history = pso_optimizer.optimize(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    print("\nEvaluating on Test Set...")
    test_results = best_anfis.evaluate(X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"MSE: {test_results['mse']:.6f}")
    print(f"MAE: {test_results['mae']:.6f}")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    
    # Save the best model
    best_anfis.save_model('models/anfis_pso_optimized.h5')
    
    # Plot optimization history
    plt.figure(figsize=(10, 6))
    plt.plot(history, 'b-', linewidth=2)
    plt.title('PSO Optimization History')
    plt.xlabel('Iteration')
    plt.ylabel('Average MSE')
    plt.grid(True)
    plt.savefig('outputs/pso_anfis_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nModel saved as 'models/anfis_pso_optimized.h5'")
    print("Optimization plot saved as 'outputs/pso_anfis_optimization.png'")
    
    return best_anfis

if __name__ == "__main__":
    demo_anfis()
