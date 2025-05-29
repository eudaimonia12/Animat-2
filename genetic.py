import numpy as np
import random
from config import *

class GeneticAlgorithm:
    def __init__(self):
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.min_fitness_history = []
    
    def initialize_population(self):
        """Initialize a new population of animats"""
        self.population = []
        for _ in range(POPULATION_SIZE):
            genome = [random.randint(0, 99) for _ in range(GENOME_LENGTH)]
            self.population.append(genome)
    
    def select_parent(self):
        """Select a parent using tournament selection"""
        tournament_size = 7
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: self.evaluate_fitness(x))
    
    def crossover(self, parent1, parent2):
        """Perform crossover between two parents"""
        if random.random() < CROSSOVER_RATE:
            point = random.randint(0, GENOME_LENGTH - 1)
            child1 = parent1[:point] + parent2[point:]
            child2 = parent2[:point] + parent1[point:]
            return child1, child2
        return parent1, parent2
    
    def mutate(self, genome):
        """Apply mutation to a genome"""
        mutated = list(genome)
        for i in range(len(mutated)):
            if random.random() < MUTATION_RATE:
                mutated[i] = random.randint(0, 99)
        return mutated
    
    def evaluate_fitness(self, genome):
        """Evaluate the fitness of a genome by running a simulation"""
        from animat import Animat
        from environment import Environment
        
        env = Environment()
        animat = Animat(genome=genome)
        total_fitness = 0
        steps = 0
        
        while animat.alive and steps < ANIMAT_MAX_LIFESPAN:
            animat.update(env)
            total_fitness += animat.get_fitness()
            steps += 1
        
        return total_fitness / steps if steps > 0 else 0
    
    def evolve(self):
        """Evolve the population for one generation"""
        # Evaluate current population
        fitnesses = [self.evaluate_fitness(genome) for genome in self.population]
        
        # Record statistics
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        self.min_fitness_history.append(min(fitnesses))
        
        # Create new population
        new_population = []
        
        # Elitism: keep the best 5 individuals
        elite_indices = np.argsort(fitnesses)[-5:]
        for idx in elite_indices:
            new_population.append(self.population[idx])
        
        # Create rest of new population
        while len(new_population) < POPULATION_SIZE:
            # Select parents
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            
            # Crossover
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutate
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to population size if we have too many
        self.population = new_population[:POPULATION_SIZE]
        self.generation += 1
    
    def get_best_individual(self):
        """Get the best individual from the current population"""
        fitnesses = [self.evaluate_fitness(genome) for genome in self.population]
        best_idx = np.argmax(fitnesses)
        return self.population[best_idx]
    
    def get_statistics(self):
        """Get the current statistics of the population"""
        return {
            'generation': self.generation,
            'best_fitness': self.best_fitness_history[-1],
            'avg_fitness': self.avg_fitness_history[-1],
            'min_fitness': self.min_fitness_history[-1]
        } 