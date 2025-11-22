"""
Natural Evolution Strategies (NES) Implementation for Autonomous Driving

This module implements the Natural Evolution Strategies optimization algorithm 
for training neural network-based autonomous driving agents. NES is a gradient-free
optimization method that evolves a population of neural network parameters to
maximize driving performance in the CARLA simulator.

Key Features:
- Population-based parameter optimization
- Gradient estimation through fitness-weighted parameter perturbations  
- Adaptive parameter updates for neural network weights
- Integration with CARLA ML agents for autonomous driving

Author: Manuel Di Lullo
Date: 2025
"""

import numpy as np
from typing import List, Any, Optional


class NES:
    """
    Natural Evolution Strategies optimizer for autonomous driving neural networks.
    
    This class implements the NES algorithm to evolve neural network parameters
    for autonomous driving agents. It maintains a population of parameter vectors,
    evaluates their fitness through driving performance, and updates the population
    to improve driving behavior over generations.
    
    The algorithm works by:
    1. Maintaining a population of neural network parameter sets
    2. Evaluating each parameter set through driving simulation
    3. Computing gradients based on fitness-weighted parameter differences
    4. Updating the base parameters in the direction of better performance
    5. Generating new population around updated parameters
    
    Attributes:
        agent: The ML agent whose neural network will be optimized
        population_size (int): Number of individuals in the population
        sigma (float): Standard deviation for parameter perturbation
        learning_rate (float): Step size for parameter updates
        base_params (np.ndarray): Current best parameters (population mean)
        population (List[np.ndarray]): Current population of parameter vectors
    """
    
    def __init__(self, agent: Any, population_size: int = 10, sigma: float = 0.1, learning_rate: float = 0.1):
        """
        Initialize the NES optimizer with the given hyperparameters.
        
        Args:
            agent: ML agent with a neural network model to optimize
            population_size (int): Size of the population for evolution (default: 10)
            sigma (float): Standard deviation for Gaussian noise in parameter perturbation (default: 0.1)
            learning_rate (float): Learning rate for parameter updates (default: 0.1)
            
        Note:
            Larger population_size provides better gradient estimates but requires more evaluations.
            Higher sigma increases exploration but may destabilize learning.
            Higher learning_rate speeds up learning but may cause instability.
        """
        self.agent = agent
        self.population_size = population_size
        self.sigma = sigma
        self.initial_sigma = sigma  # Store initial value for reset
        self.min_sigma = 0.05  # Minimum sigma to maintain exploration
        self.learning_rate = learning_rate
        
        # Initialize population around current agent parameters
        self.base_params = agent.model.parameters.copy()
        self.population = self._generate_population()
        
        # Track best solution found
        self.best_params = self.base_params.copy()
        self.best_fitness = -float('inf')
        
        # Generation counter for adaptive behavior
        self.generation = 0
        
        print(f"✓ NES initialized with adaptive sigma: pop={population_size}, sigma={sigma:.3f}, lr={learning_rate}")
    
    def _generate_population(self) -> List[np.ndarray]:
        """
        Generate a new population of parameter vectors around the current base parameters.
        
        Each individual in the population is created by adding Gaussian noise to the
        base parameters, providing exploration around the current best solution.
        
        Returns:
            List[np.ndarray]: List of parameter vectors for the population
        """
        population = []
        for _ in range(self.population_size):
            # Add Gaussian noise to base parameters for exploration
            individual = self.base_params + self.sigma * np.random.randn(len(self.base_params))
            population.append(individual)
        return population

    def step(self, fitnesses: List[float]) -> dict:
        """
        Perform one evolutionary step using the fitness scores from population evaluation.
        
        This method implements the core NES algorithm:
        1. Standardize fitness scores to reduce variance
        2. Compute gradient estimate using fitness-weighted parameter differences
        3. Update base parameters in direction of gradient
        4. Generate new population around updated parameters
        5. Update the agent's model with new best parameters
        
        Args:
            fitnesses (List[float]): Fitness scores for each individual in current population.
                                   Higher values indicate better performance.
                                   
        Returns:
            dict: Statistics about the optimization step including:
                - mean_fitness: Average fitness of current population
                - max_fitness: Best fitness in current population  
                - min_fitness: Worst fitness in current population
                - std_fitness: Standard deviation of fitness scores
                - gradient_norm: Magnitude of computed gradient
        
        Raises:
            ValueError: If number of fitness scores doesn't match population size
        """
        if len(fitnesses) != self.population_size:
            raise ValueError(f"Expected {self.population_size} fitness values, got {len(fitnesses)}")
        
        fitnesses = np.array(fitnesses)
        
        # Standardize fitness scores to reduce impact of outliers
        mean_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)
        
        # Avoid division by zero if all fitnesses are identical
        if std_fitness > 1e-8:
            standardized_fitness = (fitnesses - mean_fitness) / std_fitness
        else:
            standardized_fitness = np.zeros_like(fitnesses)
        
        # Compute gradient estimate using fitness-weighted parameter differences
        gradient = np.zeros_like(self.base_params)
        for i in range(self.population_size):
            # Weight parameter difference by standardized fitness
            parameter_diff = (self.population[i] - self.base_params)
            gradient += standardized_fitness[i] * parameter_diff
        
        # Scale gradient by population size and noise standard deviation
        gradient /= (self.population_size * self.sigma)
        
        # Update base parameters in direction of positive gradient
        self.base_params += self.learning_rate * gradient
        
        # Track best individual
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_params = self.population[best_idx].copy()
        
        # Adaptive sigma: prevent premature convergence
        self.generation += 1
        
        # Check for stagnation (low fitness variance = converged)
        if std_fitness < 1.0 and std_fitness > 1e-8:
            # Population has converged - increase sigma to encourage exploration
            self.sigma = min(self.sigma * 1.5, self.initial_sigma)
            print(f"  ⚠️ Gen {self.generation}: Low variance ({std_fitness:.2f}) - Increasing sigma to {self.sigma:.3f}")
        else:
            # Gradually decay sigma during normal operation
            self.sigma = max(self.sigma * 0.995, self.min_sigma)
        
        # Every 50 generations, inject diversity if parameters are too similar
        if self.generation % 50 == 0:
            param_diversity = np.std(self.base_params)
            if param_diversity < 0.1:
                print(f"  🔄 Gen {self.generation}: Low diversity ({param_diversity:.3f}) - Adding noise")
                self.base_params += np.random.randn(len(self.base_params)) * 0.3
        
        # Generate new population around updated base parameters
        self.population = self._generate_population()
        
        # Update the agent's neural network with new best parameters
        self.agent.model.set_parameters(self.base_params)
        
        # Return optimization statistics for monitoring
        return {
            'mean_fitness': mean_fitness,
            'max_fitness': np.max(fitnesses),
            'min_fitness': np.min(fitnesses), 
            'std_fitness': std_fitness,
            'gradient_norm': np.linalg.norm(gradient),
            'best_fitness_ever': self.best_fitness,
            'current_sigma': self.sigma
        }
    
    def get_best_parameters(self) -> np.ndarray:
        """
        Get the best parameters found so far across all generations.
        
        Returns:
            np.ndarray: Copy of best parameter vector
        """
        return self.best_params.copy()
    
    def save_parameters(self, filepath: str) -> None:
        """
        Save the current best parameters to a file.
        
        Args:
            filepath (str): Path where to save the parameters
        """
        try:
            np.save(filepath, self.base_params)
            print(f"NES parameters saved to {filepath}")
        except Exception as e:
            print(f"Error saving parameters: {e}")
    
    def load_parameters(self, filepath: str) -> None:
        """
        Load parameters from a file and update the population.
        
        Args:
            filepath (str): Path to load parameters from
        """
        try:
            self.base_params = np.load(filepath)
            self.population = self._generate_population()
            self.agent.model.set_parameters(self.base_params)
            print(f"NES parameters loaded from {filepath}")
        except Exception as e:
            print(f"Error loading parameters: {e}")
