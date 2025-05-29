import numpy as np
import random
from config import *

class Environment:
    def __init__(self, num_animats=1):
        self.num_animats = num_animats
        self.size = BASE_ENV_SIZE * num_animats
        self.food_count = BASE_FOOD_COUNT * num_animats
        self.water_count = BASE_WATER_COUNT * num_animats
        self.trap_count = BASE_TRAP_COUNT * num_animats
        
        # Initialize objects
        self.food_sources = []
        self.water_sources = []
        self.traps = []
        self.reset_objects()
    
    def reset_objects(self):
        """Reset all objects to random positions"""
        self.food_sources = self._generate_objects(self.food_count)
        self.water_sources = self._generate_objects(self.water_count)
        self.traps = self._generate_objects(self.trap_count)
    
    def _generate_objects(self, count):
        """Generate random positions for objects"""
        objects = []
        for _ in range(count):
            x = random.uniform(OBJECT_PLACEMENT_PADDING, self.size - OBJECT_PLACEMENT_PADDING)
            y = random.uniform(OBJECT_PLACEMENT_PADDING, self.size - OBJECT_PLACEMENT_PADDING)
            objects.append((x, y))
        return objects
    
    def replace_object(self, object_type, position):
        """Replace a consumed object with a new one at a random position"""
        new_pos = self._generate_objects(1)[0]
        if object_type == 'food':
            self.food_sources.remove(position)
            self.food_sources.append(new_pos)
        elif object_type == 'water':
            self.water_sources.remove(position)
            self.water_sources.append(new_pos)
    
    def get_nearest_object(self, position, object_type):
        """Get the nearest object of specified type and its distance"""
        if object_type == 'food':
            objects = self.food_sources
        elif object_type == 'water':
            objects = self.water_sources
        elif object_type == 'trap':
            objects = self.traps
        else:
            return None, float('inf')
        
        if not objects:
            return None, float('inf')
        
        distances = [np.sqrt((x - position[0])**2 + (y - position[1])**2) for x, y in objects]
        min_idx = np.argmin(distances)
        return objects[min_idx], distances[min_idx]
    
    def check_collision(self, position, object_type):
        """Check if animat collides with any object of specified type"""
        if object_type == 'food':
            objects = self.food_sources
        elif object_type == 'water':
            objects = self.water_sources
        elif object_type == 'trap':
            objects = self.traps
        else:
            return False, None
        
        for obj_pos in objects:
            distance = np.sqrt((obj_pos[0] - position[0])**2 + (obj_pos[1] - position[1])**2)
            if distance < SOURCE_SIZE + ANIMAT_SIZE:
                return True, obj_pos
        return False, None
    
    def check_animat_collision(self, position1, position2):
        """Check if two animats collide"""
        distance = np.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2)
        return distance < ANIMAT_SIZE * 2 