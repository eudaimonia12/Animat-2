import pygame
import sys
from environment import Environment
from animat import Animat
from genetic import GeneticAlgorithm
from visualization import Visualizer
from config import *

def get_num_animats():
    """Get the number of animats from user input"""
    while True:
        try:
            num = int(input("Enter number of animats (1-10): "))
            if 1 <= num <= 10:
                return num
            print("Please enter a number between 1 and 10")
        except ValueError:
            print("Please enter a valid number")

def main():
    # Get number of animats
    num_animats = get_num_animats()
    
    # Initialize environment and genetic algorithm
    env = Environment(num_animats)
    ga = GeneticAlgorithm()
    ga.initialize_population()
    
    # Initialize visualization
    visualizer = Visualizer(env.size)
    
    # Main simulation loop
    running = True
    generation = 0
    max_generations = 200
    
    while running and generation < max_generations:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Evolve population
        ga.evolve()
        stats = ga.get_statistics()
        print(f"Generation {stats['generation']}: "
              f"Best={stats['best_fitness']:.3f}, "
              f"Avg={stats['avg_fitness']:.3f}, "
              f"Min={stats['min_fitness']:.3f}")
        
        # Create animats from current population
        animats = [Animat(genome=genome) for genome in ga.population[:num_animats]]
        
        # Simulation loop for current generation
        step = 0
        while step < ANIMAT_MAX_LIFESPAN and any(animat.alive for animat in animats):
            # Update animats multiple times per frame for faster simulation
            for _ in range(SIMULATION_SPEED):
                # Update animats
                for animat in animats:
                    if animat.alive:
                        animat.update(env, animats)
                
                # Check for animat collisions
                for i in range(len(animats)):
                    for j in range(i + 1, len(animats)):
                        if (animats[i].alive and animats[j].alive and 
                            env.check_animat_collision(animats[i].position, animats[j].position)):
                            animats[i].battery1 = max(0, animats[i].battery1 - COLLISION_DAMAGE)
                            animats[i].battery2 = max(0, animats[i].battery2 - COLLISION_DAMAGE)
                            animats[j].battery1 = max(0, animats[j].battery1 - COLLISION_DAMAGE)
                            animats[j].battery2 = max(0, animats[j].battery2 - COLLISION_DAMAGE)
                
                step += 1
                if step >= ANIMAT_MAX_LIFESPAN or not any(animat.alive for animat in animats):
                    break
            
            # Draw environment only every FRAME_SKIP frames
            if step % FRAME_SKIP == 0:
                visualizer.draw_environment(env, animats)
        
        generation += 1
    
    # Plot final statistics
    visualizer.plot_statistics(ga)
    visualizer.close()

if __name__ == "__main__":
    main()