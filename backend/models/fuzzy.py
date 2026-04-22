"""
Fuzzy Logic Module
Implements Fuzzy Sugeno system with Gaussian membership functions
"""

import numpy as np
from typing import List, Dict, Tuple, Callable


class GaussianMF:
    """Gaussian Membership Function"""
    
    def __init__(self, center: float, sigma: float):
        """
        Initialize Gaussian membership function
        
        Args:
            center: Center of the Gaussian
            sigma: Standard deviation (width)
        """
        self.center = center
        self.sigma = sigma
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate membership function
        
        Args:
            x: Input values
            
        Returns:
            Membership degrees
        """
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)
    
    def get_params(self) -> Tuple[float, float]:
        """Get parameters"""
        return self.center, self.sigma
    
    def set_params(self, center: float, sigma: float):
        """Set parameters"""
        self.center = center
        self.sigma = sigma


class FuzzyVariable:
    """Fuzzy variable with multiple membership functions"""
    
    def __init__(self, name: str, mf_configs: List[Dict]):
        """
        Initialize fuzzy variable
        
        Args:
            name: Variable name
            mf_configs: List of membership function configurations
                       [{'name': 'low', 'center': 0.0, 'sigma': 0.5}, ...]
        """
        self.name = name
        self.mfs = {}
        for config in mf_configs:
            self.mfs[config['name']] = GaussianMF(config['center'], config['sigma'])
    
    def evaluate(self, x: np.ndarray, mf_name: str) -> np.ndarray:
        """Evaluate specific membership function"""
        return self.mfs[mf_name].evaluate(x)
    
    def evaluate_all(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate all membership functions"""
        return {name: mf.evaluate(x) for name, mf in self.mfs.items()}
    
    def get_mf_params(self, mf_name: str) -> Tuple[float, float]:
        """Get parameters of a membership function"""
        return self.mfs[mf_name].get_params()
    
    def set_mf_params(self, mf_name: str, center: float, sigma: float):
        """Set parameters of a membership function"""
        self.mfs[mf_name].set_params(center, sigma)
    
    def get_all_params(self) -> np.ndarray:
        """Get all parameters as a flat array"""
        params = []
        for mf in self.mfs.values():
            params.extend([mf.center, mf.sigma])
        return np.array(params)
    
    def set_all_params(self, params: np.ndarray):
        """Set all parameters from a flat array"""
        idx = 0
        for mf in self.mfs.values():
            mf.center = params[idx]
            mf.sigma = params[idx + 1]
            idx += 2


class FuzzyRule:
    """Fuzzy Sugeno rule"""
    
    def __init__(self, antecedents: List[str], consequent: float):
        """
        Initialize fuzzy rule
        
        Args:
            antecedents: List of antecedent conditions (e.g., ['temp:high', 'humidity:low'])
            consequent: Consequent value (for Sugeno-type)
        """
        self.antecedents = antecedents
        self.consequent = consequent
    
    def evaluate(self, membership_values: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate rule firing strength
        
        Args:
            membership_values: Dictionary of membership values
                              {'temp': {'low': 0.2, 'high': 0.8}, ...}
            
        Returns:
            Firing strength (minimum of antecedent memberships)
        """
        strengths = []
        
        for ant in self.antecedents:
            var_name, mf_name = ant.split(':')
            if var_name in membership_values and mf_name in membership_values[var_name]:
                strengths.append(membership_values[var_name][mf_name])
            else:
                strengths.append(0.0)
        
        # T-norm: minimum
        return min(strengths) if strengths else 0.0


class FuzzySugenoSystem:
    """Fuzzy Sugeno inference system"""
    
    def __init__(self, input_vars: Dict[str, List[Dict]], output_range: Tuple[float, float]):
        """
        Initialize fuzzy Sugeno system
        
        Args:
            input_vars: Dictionary of input variable configurations
                       {'temp': [{'name': 'low', 'center': 0, 'sigma': 0.5}, ...], ...}
            output_range: Range of output values (min, max)
        """
        self.input_vars = {}
        for name, configs in input_vars.items():
            self.input_vars[name] = FuzzyVariable(name, configs)
        
        self.rules = []
        self.output_range = output_range
    
    def add_rule(self, antecedents: List[str], consequent: float):
        """
        Add a fuzzy rule
        
        Args:
            antecedents: List of antecedent conditions
            consequent: Consequent value
        """
        self.rules.append(FuzzyRule(antecedents, consequent))
    
    def evaluate(self, inputs: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Evaluate the fuzzy system
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            Output value and detailed information
        """
        # Calculate membership values for all inputs
        membership_values = {}
        for var_name, value in inputs.items():
            if var_name in self.input_vars:
                membership_values[var_name] = self.input_vars[var_name].evaluate_all(np.array([value]))
                # Convert to scalar
                membership_values[var_name] = {k: float(v[0]) for k, v in membership_values[var_name].items()}
        
        # Evaluate rules
        rule_strengths = []
        weighted_outputs = []
        
        for rule in self.rules:
            strength = rule.evaluate(membership_values)
            if strength > 0:
                rule_strengths.append(strength)
                weighted_outputs.append(strength * rule.consequent)
        
        # Weighted average (defuzzification)
        if sum(rule_strengths) > 0:
            output = sum(weighted_outputs) / sum(rule_strengths)
        else:
            output = (self.output_range[0] + self.output_range[1]) / 2
        
        # Clamp to output range
        output = max(self.output_range[0], min(self.output_range[1], output))
        
        details = {
            'membership_values': membership_values,
            'rule_strengths': rule_strengths,
            'num_fired_rules': len(rule_strengths)
        }
        
        return output, details
    
    def get_all_params(self) -> np.ndarray:
        """Get all membership function parameters"""
        params = []
        for var in self.input_vars.values():
            params.extend(var.get_all_params())
        return np.array(params)
    
    def set_all_params(self, params: np.ndarray):
        """Set all membership function parameters"""
        idx = 0
        for var in self.input_vars.values():
            num_params = len(var.get_all_params())
            var.set_all_params(params[idx:idx + num_params])
            idx += num_params


def create_wildfire_fuzzy_system() -> FuzzySugenoSystem:
    """
    Create a fuzzy system for wildfire prediction
    
    Returns:
        Configured FuzzySugenoSystem
    """
    # Define input variables with Gaussian MFs
    input_vars = {
        'temperature': [
            {'name': 'low', 'center': 15.0, 'sigma': 5.0},
            {'name': 'medium', 'center': 25.0, 'sigma': 5.0},
            {'name': 'high', 'center': 35.0, 'sigma': 5.0}
        ],
        'humidity': [
            {'name': 'low', 'center': 20.0, 'sigma': 10.0},
            {'name': 'medium', 'center': 50.0, 'sigma': 10.0},
            {'name': 'high', 'center': 80.0, 'sigma': 10.0}
        ],
        'wind': [
            {'name': 'low', 'center': 5.0, 'sigma': 3.0},
            {'name': 'medium', 'center': 15.0, 'sigma': 5.0},
            {'name': 'high', 'center': 30.0, 'sigma': 10.0}
        ],
        'fwi': [
            {'name': 'low', 'center': 5.0, 'sigma': 3.0},
            {'name': 'medium', 'center': 15.0, 'sigma': 5.0},
            {'name': 'high', 'center': 30.0, 'sigma': 10.0}
        ]
    }
    
    system = FuzzySugenoSystem(input_vars, output_range=(0.0, 1.0))
    
    # Add rules based on expert knowledge
    # Format: ['variable:mf', ...], consequent
    
    # Low risk rules
    system.add_rule(['temperature:low', 'humidity:high', 'wind:low'], 0.1)
    system.add_rule(['temperature:low', 'humidity:high', 'fwi:low'], 0.1)
    system.add_rule(['temperature:medium', 'humidity:high', 'wind:low'], 0.2)
    
    # Moderate risk rules
    system.add_rule(['temperature:medium', 'humidity:medium', 'wind:medium'], 0.5)
    system.add_rule(['temperature:medium', 'humidity:low', 'wind:low'], 0.4)
    system.add_rule(['temperature:high', 'humidity:high', 'wind:low'], 0.4)
    system.add_rule(['fwi:medium'], 0.5)
    
    # High risk rules
    system.add_rule(['temperature:high', 'humidity:low', 'wind:high'], 0.9)
    system.add_rule(['temperature:high', 'humidity:medium', 'wind:high'], 0.8)
    system.add_rule(['temperature:high', 'humidity:low', 'fwi:high'], 0.95)
    system.add_rule(['fwi:high', 'wind:high'], 0.9)
    
    # Extreme risk rules
    system.add_rule(['temperature:high', 'humidity:low', 'wind:high', 'fwi:high'], 1.0)
    
    return system
