import pygame
import numpy as np
import matplotlib.pyplot as plt
from config import *

class Visualizer:
    def __init__(self, env_size):
        pygame.init()
        self.screen = pygame.display.set_mode((env_size, env_size))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
    
    def draw_environment(self, env, animats):
        """Draw the environment and all objects"""
        self.screen.fill(COLORS['BLACK'])
        
        # Draw food sources
        for pos in env.food_sources:
            pygame.draw.circle(self.screen, COLORS['GREEN'], 
                             (int(pos[0]), int(pos[1])), SOURCE_SIZE)
        
        # Draw water sources
        for pos in env.water_sources:
            pygame.draw.circle(self.screen, COLORS['BLUE'], 
                             (int(pos[0]), int(pos[1])), SOURCE_SIZE)
        
        # Draw traps
        for pos in env.traps:
            pygame.draw.circle(self.screen, COLORS['RED'], 
                             (int(pos[0]), int(pos[1])), SOURCE_SIZE)
        
        # Draw animats
        for animat in animats:
            if animat.alive:
                # Draw animat body
                pygame.draw.circle(self.screen, COLORS['WHITE'], 
                                 (int(animat.position[0]), int(animat.position[1])), 
                                 ANIMAT_SIZE)
                
                # Draw direction indicator
                end_x = animat.position[0] + np.cos(animat.angle) * ANIMAT_SIZE * 1.5
                end_y = animat.position[1] + np.sin(animat.angle) * ANIMAT_SIZE * 1.5
                pygame.draw.line(self.screen, COLORS['YELLOW'],
                               (int(animat.position[0]), int(animat.position[1])),
                               (int(end_x), int(end_y)), 2)
                
                # Draw battery levels
                battery_width = 20
                battery_height = 4
                x = animat.position[0] - battery_width/2
                y = animat.position[1] - ANIMAT_SIZE - 10
                
                # Food battery
                pygame.draw.rect(self.screen, COLORS['GREEN'],
                               (x, y, battery_width * (animat.battery1/BATTERY_MAX), battery_height))
                pygame.draw.rect(self.screen, COLORS['WHITE'],
                               (x, y, battery_width, battery_height), 1)
                
                # Water battery
                y += battery_height + 2
                pygame.draw.rect(self.screen, COLORS['BLUE'],
                               (x, y, battery_width * (animat.battery2/BATTERY_MAX), battery_height))
                pygame.draw.rect(self.screen, COLORS['WHITE'],
                               (x, y, battery_width, battery_height), 1)
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def draw_trajectory(self, animat):
        """Draw the trajectory of an animat"""
        if len(animat.trajectory) < 2:
            return
        
        points = [(int(x), int(y)) for x, y in animat.trajectory]
        pygame.draw.lines(self.screen, COLORS['CYAN'], False, points, 1)
        pygame.display.flip()
    
    def plot_statistics(self, ga):
        """Plot fitness statistics using matplotlib"""
        plt.figure(figsize=(12, 6))
        
        # Plot fitness history
        plt.subplot(1, 2, 1)
        plt.plot(ga.best_fitness_history, label='Best')
        plt.plot(ga.avg_fitness_history, label='Average')
        plt.plot(ga.min_fitness_history, label='Minimum')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        
        # Plot trajectory of best individual
        plt.subplot(1, 2, 2)
        best_genome = ga.get_best_individual()
        from animat import Animat
        from environment import Environment
        
        env = Environment()
        animat = Animat(genome=best_genome)
        trajectory = []
        
        while animat.alive and len(trajectory) < ANIMAT_MAX_LIFESPAN:
            animat.update(env)
            trajectory.append(animat.position)
        
        trajectory = np.array(trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5)
        plt.scatter(trajectory[0, 0], trajectory[0, 1], c='g', label='Start')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c='r', label='End')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Best Individual Trajectory')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close the pygame window"""
        pygame.quit() 