# Animat World Simulation

## Overview
This project simulates the behavior-selection and evolution of minimalist animats (artificial agents) in a dynamic, multi-agent world. The animats must survive by seeking food and water, avoiding traps, and evolving their behavior over generations using a genetic algorithm.

## World Model
- **Environment:**
  - The world is a 2D square area (default 200x200, scales with animat count).
  - Contains three types of objects: food, water, and traps (all are circles with radius 16).
  - The number of each object type scales with the number of animats.
  - Objects are randomly placed and respawn at new locations when consumed.

- **Objects:**
  - **Food:** Refills the animat's food battery when reached.
  - **Water:** Refills the animat's water battery when reached.
  - **Trap:** Instantly kills the animat and sets both batteries to zero.

## Animat Model
- **Body:**
  - Represented as a circle (radius 5) with two wheels for movement.
  - Has two internal batteries: one for food, one for water (max 200 each).
  - Batteries deplete over time; if both reach zero, the animat dies.

- **Sensors:**
  - Each animat has six sensors (left/right for food, water, trap) in single-agent mode.
  - In multi-agent mode, two additional sensors detect other animats.
  - Sensors respond to the distance and direction of the nearest object of each type, with higher activation for closer and more lateral objects.

- **Behavior:**
  - Sensor signals are transformed into wheel speeds via a set of genetically-encoded transfer functions (sensorimotor links).
  - The animat moves according to the difference in wheel speeds.
  - If stuck (not moving for 30 steps), the animat will reorient randomly.
  - If two animats collide, both lose battery.

## Genetic Algorithm
- **Encoding:**
  - Each animat's behavior is encoded as a genome of integers (0-99), specifying the parameters of the sensorimotor links and wheel thresholds.
  - In multi-agent mode, the genome expands to accommodate extra sensors.

- **Evolution:**
  - Population size: 100 (configurable).
  - Tournament selection (size 7), crossover, and mutation (rate 0.01).
  - Elitism: top 5 individuals are preserved each generation.
  - Fitness is the average battery level over the animat's lifespan, rewarding survival and resource gathering.
  - Each genome is evaluated by simulating an animat in the world.

## Visualization
- **Pygame** is used to visualize the world, animats, and objects in real time.
- **Matplotlib** plots:
  - Fitness evolution (best, average, minimum) over generations.
  - Trajectory of the best animat, with food, water, and trap locations marked.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the simulation:
   ```bash
   python main.py
   ```
3. Enter the number of animats (1-10) when prompted.
4. Observe the evolution and behavior of animats in the visualizations.

## File Structure
- `main.py` — Entry point, runs the simulation and handles user input.
- `config.py` — All configuration constants.
- `environment.py` — World and object management.
- `animat.py` — Animat class, sensors, movement, and behavior logic.
- `genetic.py` — Genetic algorithm implementation.
- `visualization.py` — Visualization and plotting.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Files to ignore in version control.

## Customization
- Adjust parameters in `config.py` to change world size, population, mutation rate, etc.
- Modify the animat's sensorimotor logic or genetic encoding for experiments.

## License
MIT License. 
