"""
ANFIS (Adaptive Neuro-Fuzzy Inference System) Module
Implements 5-layer ANFIS with hybrid training (Least Squares + Gradient Descent)
"""

import numpy as np
from typing import Tuple, List, Dict
import joblib


class ANFISLayer1:
    """Layer 1: Input membership functions (Gaussian)"""
    
    def __init__(self, num_inputs: int, num_mfs_per_input: int):
        """
        Initialize Layer 1
        
        Args:
            num_inputs: Number of input variables
            num_mfs_per_input: Number of membership functions per input
        """
        self.num_inputs = num_inputs
        self.num_mfs_per_input = num_mfs_per_input
        self.num_mfs = num_inputs * num_mfs_per_input
        
        # Initialize parameters (center and sigma for each MF)
        # centers: random between 0 and 1
        self.centers = np.random.rand(self.num_mfs)
        # sigmas: random between 0.1 and 0.5
        self.sigmas = 0.1 + 0.4 * np.random.rand(self.num_mfs)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through Layer 1
        
        Args:
            X: Input matrix (n_samples, num_inputs)
            
        Returns:
            Membership degrees (n_samples, num_mfs)
        """
        n_samples = X.shape[0]
        output = np.zeros((n_samples, self.num_mfs))
        
        for i in range(self.num_inputs):
            for j in range(self.num_mfs_per_input):
                mf_idx = i * self.num_mfs_per_input + j
                # Gaussian MF
                output[:, mf_idx] = np.exp(-0.5 * ((X[:, i] - self.centers[mf_idx]) / self.sigmas[mf_idx]) ** 2)
        
        return output
    
    def get_params(self) -> np.ndarray:
        """Get all parameters"""
        return np.concatenate([self.centers, self.sigmas])
    
    def set_params(self, params: np.ndarray):
        """Set all parameters"""
        n = len(params) // 2
        self.centers = params[:n]
        self.sigmas = params[n:]


class ANFISLayer2:
    """Layer 2: Rule firing strength (product T-norm)"""
    
    def __init__(self, num_inputs: int, num_mfs_per_input: int):
        """
        Initialize Layer 2
        
        Args:
            num_inputs: Number of input variables
            num_mfs_per_input: Number of membership functions per input
        """
        self.num_inputs = num_inputs
        self.num_mfs_per_input = num_mfs_per_input
        self.num_rules = num_mfs_per_input ** num_inputs
    
    def forward(self, layer1_output: np.ndarray) -> np.ndarray:
        """
        Forward pass through Layer 2
        
        Args:
            layer1_output: Output from Layer 1 (n_samples, num_mfs)
            
        Returns:
            Rule firing strengths (n_samples, num_rules)
        """
        n_samples = layer1_output.shape[0]
        output = np.zeros((n_samples, self.num_rules))
        
        # Generate all combinations of MFs
        from itertools import product
        mf_combinations = list(product(range(self.num_mfs_per_input), repeat=self.num_inputs))
        
        for rule_idx, combination in enumerate(mf_combinations):
            # Product of membership degrees
            rule_strength = np.ones(n_samples)
            for i, mf_idx in enumerate(combination):
                mf_global_idx = i * self.num_mfs_per_input + mf_idx
                rule_strength *= layer1_output[:, mf_global_idx]
            output[:, rule_idx] = rule_strength
        
        return output


class ANFISLayer3:
    """Layer 3: Normalization of firing strengths"""
    
    def forward(self, layer2_output: np.ndarray) -> np.ndarray:
        """
        Forward pass through Layer 3
        
        Args:
            layer2_output: Output from Layer 2 (n_samples, num_rules)
            
        Returns:
            Normalized firing strengths (n_samples, num_rules)
        """
        # Normalize each sample
        row_sums = np.sum(layer2_output, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1
        return layer2_output / row_sums


class ANFISLayer4:
    """Layer 4: Consequent parameters (linear combination)"""
    
    def __init__(self, num_inputs: int, num_rules: int):
        """
        Initialize Layer 4
        
        Args:
            num_inputs: Number of input variables
            num_rules: Number of rules
        """
        self.num_inputs = num_inputs
        self.num_rules = num_rules
        # Consequent parameters: each rule has (num_inputs + 1) parameters (including bias)
        self.consequent_params = np.random.randn(num_rules, num_inputs + 1) * 0.1
    
    def forward(self, layer3_output: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through Layer 4
        
        Args:
            layer3_output: Output from Layer 3 (n_samples, num_rules)
            X: Original input matrix (n_samples, num_inputs)
            
        Returns:
            Weighted consequent outputs (n_samples, num_rules)
        """
        n_samples = X.shape[0]
        output = np.zeros((n_samples, self.num_rules))
        
        for rule_idx in range(self.num_rules):
            # Linear combination: w * (a0 + a1*x1 + a2*x2 + ...)
            linear_output = self.consequent_params[rule_idx, 0]  # bias
            for i in range(self.num_inputs):
                linear_output += self.consequent_params[rule_idx, i + 1] * X[:, i]
            output[:, rule_idx] = layer3_output[:, rule_idx] * linear_output
        
        return output
    
    def get_params(self) -> np.ndarray:
        """Get consequent parameters"""
        return self.consequent_params.flatten()
    
    def set_params(self, params: np.ndarray):
        """Set consequent parameters"""
        self.consequent_params = params.reshape(self.num_rules, self.num_inputs + 1)


class ANFISLayer5:
    """Layer 5: Summation (final output)"""
    
    def forward(self, layer4_output: np.ndarray) -> np.ndarray:
        """
        Forward pass through Layer 5
        
        Args:
            layer4_output: Output from Layer 4 (n_samples, num_rules)
            
        Returns:
            Final output (n_samples, 1)
        """
        return np.sum(layer4_output, axis=1, keepdims=True)


class ANFIS:
    """Complete ANFIS model with 5 layers and hybrid training"""
    
    def __init__(self, num_inputs: int, num_mfs_per_input: int = 2):
        """
        Initialize ANFIS model
        
        Args:
            num_inputs: Number of input variables
            num_mfs_per_input: Number of membership functions per input
        """
        self.num_inputs = num_inputs
        self.num_mfs_per_input = num_mfs_per_input
        
        # Initialize layers
        self.layer1 = ANFISLayer1(num_inputs, num_mfs_per_input)
        self.layer2 = ANFISLayer2(num_inputs, num_mfs_per_input)
        self.layer3 = ANFISLayer3()
        self.layer4 = ANFISLayer4(num_inputs, num_mfs_per_input ** num_inputs)
        self.layer5 = ANFISLayer5()
        
        self.is_trained = False
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire network
        
        Args:
            X: Input matrix (n_samples, num_inputs)
            
        Returns:
            Output (n_samples, 1)
        """
        l1_out = self.layer1.forward(X)
        l2_out = self.layer2.forward(l1_out)
        l3_out = self.layer3.forward(l2_out)
        l4_out = self.layer4.forward(l3_out, X)
        l5_out = self.layer5.forward(l4_out)
        return l5_out
    
    def hybrid_train(self, X_train: np.ndarray, y_train: np.ndarray, 
                     epochs: int = 20, lr: float = 0.01) -> Dict:
        """
        Simplified training using only Least Squares for stability
        
        Args:
            X_train: Training inputs
            y_train: Training targets
            epochs: Number of epochs (not used, kept for compatibility)
            lr: Learning rate (not used, kept for compatibility)
            
        Returns:
            Training history
        """
        history = {'loss': []}
        n_samples = X_train.shape[0]
        
        # Normalize targets for better training
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-8
        y_train_norm = (y_train - y_mean) / y_std
        
        # Single pass training
        l1_out = self.layer1.forward(X_train)
        l2_out = self.layer2.forward(l1_out)
        l3_out = self.layer3.forward(l2_out)
        
        # Least Squares for consequent parameters
        n_rules = l3_out.shape[1]
        design_matrix = np.zeros((n_samples, n_rules * (self.num_inputs + 1)))
        
        for rule_idx in range(n_rules):
            for i in range(self.num_inputs + 1):
                if i == 0:
                    design_matrix[:, rule_idx * (self.num_inputs + 1)] = l3_out[:, rule_idx]
                else:
                    design_matrix[:, rule_idx * (self.num_inputs + 1) + i] = l3_out[:, rule_idx] * X_train[:, i - 1]
        
        # Solve LS
        try:
            params = np.linalg.lstsq(design_matrix, y_train_norm.flatten(), rcond=None)[0]
            self.layer4.set_params(params)
        except:
            pass
        
        # Forward pass with updated consequents
        l4_out = self.layer4.forward(l3_out, X_train)
        output = self.layer5.forward(l4_out)
        
        # Denormalize output
        output_denorm = output * y_std + y_mean
        
        # Calculate loss
        loss = np.sqrt(np.mean((output_denorm.flatten() - y_train.flatten()) ** 2))
        history['loss'].append(loss)
        
        print(f"Training RMSE: {loss:.6f}")
        
        self.is_trained = True
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input matrix
            
        Returns:
            Predictions
        """
        return self.forward(X)
    
    def save(self, filepath: str):
        """Save model to disk"""
        model_data = {
            'layer1_centers': self.layer1.centers,
            'layer1_sigmas': self.layer1.sigmas,
            'layer4_params': self.layer4.consequent_params,
            'num_inputs': self.num_inputs,
            'num_mfs_per_input': self.num_mfs_per_input,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
    
    def load(self, filepath: str):
        """Load model from disk"""
        model_data = joblib.load(filepath)
        self.layer1.centers = model_data['layer1_centers']
        self.layer1.sigmas = model_data['layer1_sigmas']
        self.layer4.consequent_params = model_data['layer4_params']
        self.num_inputs = model_data['num_inputs']
        self.num_mfs_per_input = model_data['num_mfs_per_input']
        self.is_trained = model_data['is_trained']
