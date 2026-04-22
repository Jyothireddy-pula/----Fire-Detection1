"""
PSO (Particle Swarm Optimization) Module
Optimizes ANFIS membership function parameters
"""

import numpy as np
from typing import Callable, Dict, Tuple
import joblib


class Particle:
    """Single particle in PSO"""
    
    def __init__(self, dim: int, bounds: Tuple[float, float]):
        """
        Initialize particle
        
        Args:
            dim: Dimension of the search space
            bounds: Lower and upper bounds for positions
        """
        self.dim = dim
        self.bounds = bounds
        
        # Random position within bounds
        self.position = np.random.uniform(bounds[0], bounds[1], dim)
        
        # Random velocity
        self.velocity = np.random.uniform(-1, 1, dim)
        
        # Personal best
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
    
    def update_velocity(self, global_best_position: np.ndarray, 
                       w: float, c1: float, c2: float):
        """
        Update particle velocity
        
        Args:
            global_best_position: Best position found by swarm
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
        """
        r1, r2 = np.random.rand(2, self.dim)
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
        # Clamp velocity
        max_vel = (self.bounds[1] - self.bounds[0]) * 0.2
        self.velocity = np.clip(self.velocity, -max_vel, max_vel)
    
    def update_position(self):
        """Update particle position"""
        self.position += self.velocity
        self.position = np.clip(self.position, self.bounds[0], self.bounds[1])
    
    def evaluate(self, fitness_func: Callable) -> float:
        """
        Evaluate particle fitness
        
        Args:
            fitness_func: Fitness function to evaluate
            
        Returns:
            Fitness value
        """
        fitness = fitness_func(self.position)
        
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
        
        return fitness


class PSO:
    """Particle Swarm Optimization"""
    
    def __init__(self, num_particles: int = 20, max_iterations: int = 20,
                 w: float = 0.7, c1: float = 1.5, c2: float = 1.5,
                 bounds: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize PSO
        
        Args:
            num_particles: Number of particles in swarm
            max_iterations: Maximum number of iterations
            w: Inertia weight
            c1: Cognitive coefficient
            c2: Social coefficient
            bounds: Search space bounds
        """
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds
        
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.history = []
    
    def initialize(self, dim: int):
        """
        Initialize swarm
        
        Args:
            dim: Dimension of search space
        """
        self.particles = [Particle(dim, self.bounds) for _ in range(self.num_particles)]
        
        # Initialize global best
        self.global_best_position = self.particles[0].best_position.copy()
        self.global_best_fitness = self.particles[0].best_fitness
    
    def optimize(self, fitness_func: Callable, dim: int, verbose: bool = True) -> Dict:
        """
        Run PSO optimization
        
        Args:
            fitness_func: Function to minimize (takes position array, returns fitness)
            dim: Dimension of search space
            verbose: Whether to print progress
            
        Returns:
            Optimization results
        """
        self.initialize(dim)
        
        for iteration in range(self.max_iterations):
            # Evaluate all particles
            for particle in self.particles:
                fitness = particle.evaluate(fitness_func)
                
                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position.copy()
            
            # Update particles
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
            
            # Record history
            self.history.append(self.global_best_fitness)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Fitness: {self.global_best_fitness:.6f}")
        
        return {
            'best_position': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'history': self.history
        }


class PSOANFISOptimizer:
    """PSO optimizer specifically for ANFIS"""
    
    def __init__(self, anfis_model, num_particles: int = 20, 
                 max_iterations: int = 20):
        """
        Initialize PSO-ANFIS optimizer
        
        Args:
            anfis_model: ANFIS model to optimize
            num_particles: Number of particles
            max_iterations: Maximum iterations
        """
        self.anfis = anfis_model
        self.pso = PSO(num_particles, max_iterations)
        self.X_train = None
        self.y_train = None
    
    def fitness_function(self, params: np.ndarray) -> float:
        """
        Fitness function for PSO (RMSE)
        
        Args:
            params: ANFIS parameters to evaluate
            
        Returns:
            RMSE
        """
        # Set ANFIS parameters
        self.anfis.layer1.set_params(params)
        
        # Make predictions
        predictions = self.anfis.predict(self.X_train)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((predictions.flatten() - self.y_train.flatten()) ** 2))
        
        return rmse
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, 
                verbose: bool = True) -> Dict:
        """
        Optimize ANFIS using PSO
        
        Args:
            X_train: Training data
            y_train: Training targets
            verbose: Whether to print progress
            
        Returns:
            Optimization results
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Get dimension of ANFIS premise parameters
        dim = len(self.anfis.layer1.get_params())
        
        # Run PSO
        results = self.pso.optimize(self.fitness_function, dim, verbose)
        
        # Set best parameters to ANFIS
        self.anfis.layer1.set_params(results['best_position'])
        
        # Train consequent parameters with LS
        self.anfis.hybrid_train(X_train, y_train, epochs=10, lr=0.01)
        
        return results
    
    def save(self, filepath: str):
        """Save optimizer state"""
        optimizer_data = {
            'pso_config': {
                'num_particles': self.pso.num_particles,
                'max_iterations': self.pso.max_iterations,
                'w': self.pso.w,
                'c1': self.pso.c1,
                'c2': self.pso.c2
            },
            'best_position': self.pso.global_best_position,
            'best_fitness': self.pso.global_best_fitness,
            'history': self.pso.history
        }
        joblib.dump(optimizer_data, filepath)
    
    def load(self, filepath: str):
        """Load optimizer state"""
        optimizer_data = joblib.load(filepath)
        self.pso.global_best_position = optimizer_data['best_position']
        self.pso.global_best_fitness = optimizer_data['best_fitness']
        self.pso.history = optimizer_data['history']
