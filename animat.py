import numpy as np
import random
from config import *

class Animat:
    def __init__(self, genome=None, position=None):
        self.position = position or (random.uniform(0, BASE_ENV_SIZE), random.uniform(0, BASE_ENV_SIZE))
        self.angle = random.uniform(0, 2 * np.pi)
        self.battery1 = BATTERY_MAX  # Food battery
        self.battery2 = BATTERY_MAX  # Water battery
        self.alive = True
        self.stuck_counter = 0
        self.last_position = self.position
        self.genome = genome or [random.randint(0, 99) for _ in range(GENOME_LENGTH)]
        self.trajectory = [self.position]
        
        # Initialize sensor values
        self.sensors = {
            'food_left': 0, 'food_right': 0,
            'water_left': 0, 'water_right': 0,
            'trap_left': 0, 'trap_right': 0,
            'other_left': 0, 'other_right': 0
        }
    
    def update_sensors(self, env, other_animats=None):
        """Update sensor values based on environment and other animats"""
        sensor_range = BASE_SENSOR_RANGE * env.num_animats
        
        # Update food sensors
        nearest_food, food_dist = env.get_nearest_object(self.position, 'food')
        if nearest_food:
            food_angle = np.arctan2(nearest_food[1] - self.position[1], nearest_food[0] - self.position[0])
            relative_angle = (food_angle - self.angle) % (2 * np.pi)
            sensor_value = max(0, 100 * (1 - food_dist / sensor_range))
            
            if relative_angle < np.pi:
                self.sensors['food_left'] = sensor_value * 1.2
                self.sensors['food_right'] = sensor_value
            else:
                self.sensors['food_left'] = sensor_value
                self.sensors['food_right'] = sensor_value * 1.2
        
        # Update water sensors (similar to food)
        nearest_water, water_dist = env.get_nearest_object(self.position, 'water')
        if nearest_water:
            water_angle = np.arctan2(nearest_water[1] - self.position[1], nearest_water[0] - self.position[0])
            relative_angle = (water_angle - self.angle) % (2 * np.pi)
            sensor_value = max(0, 100 * (1 - water_dist / sensor_range))
            
            if relative_angle < np.pi:
                self.sensors['water_left'] = sensor_value * 1.2
                self.sensors['water_right'] = sensor_value
            else:
                self.sensors['water_left'] = sensor_value
                self.sensors['water_right'] = sensor_value * 1.2
        
        # Update trap sensors
        nearest_trap, trap_dist = env.get_nearest_object(self.position, 'trap')
        if nearest_trap:
            trap_angle = np.arctan2(nearest_trap[1] - self.position[1], nearest_trap[0] - self.position[0])
            relative_angle = (trap_angle - self.angle) % (2 * np.pi)
            sensor_value = max(0, 100 * (1 - trap_dist / sensor_range))
            
            if relative_angle < np.pi:
                self.sensors['trap_left'] = sensor_value * 1.2
                self.sensors['trap_right'] = sensor_value
            else:
                self.sensors['trap_left'] = sensor_value
                self.sensors['trap_right'] = sensor_value * 1.2
        
        # Update other animat sensors
        if other_animats:
            for other in other_animats:
                if other != self and other.alive:
                    dist = np.sqrt((other.position[0] - self.position[0])**2 + 
                                 (other.position[1] - self.position[1])**2)
                    if dist < sensor_range:
                        other_angle = np.arctan2(other.position[1] - self.position[1],
                                               other.position[0] - self.position[0])
                        relative_angle = (other_angle - self.angle) % (2 * np.pi)
                        sensor_value = max(0, 100 * (1 - dist / sensor_range))
                        
                        if relative_angle < np.pi:
                            self.sensors['other_left'] = max(self.sensors['other_left'], sensor_value * 1.2)
                            self.sensors['other_right'] = max(self.sensors['other_right'], sensor_value)
                        else:
                            self.sensors['other_left'] = max(self.sensors['other_left'], sensor_value)
                            self.sensors['other_right'] = max(self.sensors['other_right'], sensor_value * 1.2)
    
    def process_sensorimotor_links(self):
        """Process sensor inputs through the neural network to determine wheel speeds"""
        left_wheel_sum = 0
        right_wheel_sum = 0
        
        # Process each sensor through its links
        for i, (sensor_name, sensor_value) in enumerate(self.sensors.items()):
            # Get the 9 parameters for this sensor's link
            start_idx = i * 9
            params = self.genome[start_idx:start_idx + 9]
            
            # Calculate the output based on the transfer function
            output = self._calculate_link_output(sensor_value, params)
            
            # Add to appropriate wheel sum
            if i < 4:  # First 4 sensors go to left wheel
                left_wheel_sum += output
            else:  # Last 4 sensors go to right wheel
                right_wheel_sum += output
        
        # Apply sigmoid function and scale to wheel speeds
        left_speed = self._sigmoid(left_wheel_sum, self.genome[-2])
        right_speed = self._sigmoid(right_wheel_sum, self.genome[-1])
        
        return left_speed, right_speed
    
    def _calculate_link_output(self, sensor_value, params):
        """Calculate the output of a sensorimotor link"""
        # Extract parameters
        threshold1 = (params[0] / 99.0) * 200 - 100
        threshold2 = (params[1] / 99.0) * 200 - 100
        gradient1 = np.tan((params[2] / 99.0) * np.pi - np.pi/2)
        gradient2 = np.tan((params[3] / 99.0) * np.pi - np.pi/2)
        gradient3 = np.tan((params[4] / 99.0) * np.pi - np.pi/2)
        gradient4 = np.tan((params[5] / 99.0) * np.pi - np.pi/2)
        
        # Battery influence parameters
        slope_mod = params[6] / 99.0
        offset_mod = params[7] / 99.0
        battery_choice = params[8]
        
        # Calculate base output
        if sensor_value < threshold1:
            output = gradient1 * (sensor_value - threshold1)
        elif sensor_value < threshold2:
            output = gradient2 * (sensor_value - threshold1)
        else:
            output = gradient3 * (sensor_value - threshold2) + gradient4
        
        # Apply battery influence
        battery = self.battery1 if battery_choice % 2 == 0 else self.battery2
        battery_factor = (battery - 100) / 100.0
        
        output = output + output * (battery_factor * slope_mod)
        output = output + ((battery / 200.0) * offset_mod)
        
        return output
    
    def _sigmoid(self, x, threshold):
        """Apply sigmoid function with threshold"""
        threshold = (threshold / 99.0) * 6 - 3  # Scale threshold to [-3, 3]
        return 2.0 / (1.0 + np.exp(-(x - threshold))) - 1.0
    
    def update(self, env, other_animats=None):
        """Update animat state"""
        if not self.alive:
            return
        
        # Update sensors
        self.update_sensors(env, other_animats)
        
        # Get wheel speeds
        left_speed, right_speed = self.process_sensorimotor_links()
        
        # Calculate movement
        speed = (left_speed + right_speed) / 2 * ANIMAT_MAX_SPEED
        rotation = (right_speed - left_speed) * np.pi / 4
        
        # Update position and angle
        self.angle = (self.angle + rotation) % (2 * np.pi)
        new_x = self.position[0] + speed * np.cos(self.angle)
        new_y = self.position[1] + speed * np.sin(self.angle)
        
        # Keep within bounds
        new_x = max(0, min(env.size, new_x))
        new_y = max(0, min(env.size, new_y))
        
        self.position = (new_x, new_y)
        self.trajectory.append(self.position)
        
        # Check if stuck
        if np.sqrt((self.position[0] - self.last_position[0])**2 + 
                  (self.position[1] - self.last_position[1])**2) < 0.1:
            self.stuck_counter += 1
            if self.stuck_counter > STUCK_THRESHOLD:
                self.angle = random.uniform(0, 2 * np.pi)
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0
        
        self.last_position = self.position
        
        # Update batteries
        self.battery1 = max(0, self.battery1 - BATTERY_DECAY_RATE)
        self.battery2 = max(0, self.battery2 - BATTERY_DECAY_RATE)
        
        # Check if dead
        if self.battery1 <= 0 and self.battery2 <= 0:
            self.alive = False
        
        # Check collisions
        for obj_type in ['food', 'water', 'trap']:
            collided, pos = env.check_collision(self.position, obj_type)
            if collided:
                if obj_type == 'trap':
                    self.alive = False
                    self.battery1 = 0
                    self.battery2 = 0
                elif obj_type == 'food':
                    self.battery1 = BATTERY_MAX
                    env.replace_object('food', pos)
                elif obj_type == 'water':
                    self.battery2 = BATTERY_MAX
                    env.replace_object('water', pos)
    
    def get_fitness(self):
        """Calculate fitness based on average battery levels"""
        return (self.battery1 + self.battery2) / (2 * BATTERY_MAX) 