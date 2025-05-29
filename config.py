# Environment constants
BASE_ENV_SIZE = 200
OBJECT_PLACEMENT_PADDING = 10
BASE_FOOD_COUNT = 3
BASE_WATER_COUNT = 3
BASE_TRAP_COUNT = 3

# Animat constants
ANIMAT_SIZE = 5
SOURCE_SIZE = 16
ANIMAT_MAX_SPEED = 2.8
BASE_SENSOR_RANGE = 100
BATTERY_MAX = 200
BATTERY_DECAY_RATE = 1.0
ANIMAT_MAX_LIFESPAN = 800
STUCK_THRESHOLD = 30  # Time steps before random movement
COLLISION_DAMAGE = 10  # Battery damage when animats collide

# Simulation speed control
SIMULATION_SPEED = 100  # Speed multiplier
BASE_FPS = 60
FPS = BASE_FPS * SIMULATION_SPEED  # Adjusted FPS for faster simulation
FRAME_SKIP = 10  # Number of frames to skip in visualization

# Genetic Algorithm constants
POPULATION_SIZE = 100
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.04
GENOME_LENGTH = 83  # 9 links * 9 parameters + 2 sigmoid thresholds

# Visualization constants
WINDOW_TITLE = "Animat Simulation"
COLORS = {
    'BLACK': (0, 0, 0),
    'WHITE': (255, 255, 255),
    'RED': (255, 0, 0),
    'GREEN': (0, 255, 0),
    'BLUE': (0, 0, 255),
    'YELLOW': (255, 255, 0),
    'CYAN': (0, 255, 255),
    'MAGENTA': (255, 0, 255)
} 