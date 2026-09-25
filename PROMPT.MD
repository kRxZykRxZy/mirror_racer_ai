Build a complete, runnable Python project named `mirror_racer_ai`.

The project must be a top-down 2D car-racing AI trainer using Pygame. It must train neural-network-controlled cars through neuroevolution, not TensorFlow, PyTorch, NEAT, or any machine-learning framework. Implement the neural network, genome representation, selection, crossover, mutation, simulation, sensor raycasts, scoring, saving, and Pygame rendering manually using only Python standard library, Pygame, and NumPy if needed.

I want clean, readable, well-structured production-quality code with type hints, docstrings on major classes/functions, deterministic seeded runs, config constants, useful keyboard controls, error handling, and comments explaining non-obvious math. Do not leave pseudocode, TODOs, fake functions, placeholders, or incomplete files. The project must run after installing requirements and generating/loading a track.

PROJECT GOAL
Create a top-down autonomous racing-car simulator. Train 100 cars simultaneously. Each car has its own neural-network genome. Cars use 12 raycast/state inputs and produce 3 `tanh` control outputs. At the end of every generation, select the best cars into a breeding pool, preserve elites, generate children through crossover and Gaussian mutation, inject a small number of random genomes, reset all 100 cars, and repeat.

The design must be distinctly mirror-symmetry-aware:
- Evaluate each genome on both a normal track and a horizontally mirrored version of the track, OR use a mathematically identical mirrored start/track transform.
- Fitness must be the average of normal and mirrored evaluation fitness.
- When the world is mirrored, swap left/right sensor values and negate left/right signed state features.
- Steering output must negate under mirroring.
- Throttle and brake outputs must remain unchanged under mirroring.
- Make this mirror behavior explicit in code with `mirror_inputs()` and `mirror_outputs()` functions.
- Do not reduce the network to a one-hidden-layer network. It must use exactly two hidden layers.

EXACT NEURAL NETWORK
Use this fully connected network, including biases:

    12 inputs -> 25 tanh neurons -> 24 tanh neurons -> 3 tanh outputs

All neurons in both hidden layers and the output layer must use `math.tanh`.

The exact parameter count must be exactly 1,024:

    Input to hidden layer 1:
        (12 + 1) * 25 = 325 parameters

    Hidden layer 1 to hidden layer 2:
        (25 + 1) * 24 = 624 parameters

    Hidden layer 2 to output:
        (24 + 1) * 3 = 75 parameters

    Total:
        325 + 624 + 75 = 1,024 parameters

Store every car’s genome as one flat `numpy.float32` array of shape `(1024,)`.

Create a `NeuralNetwork` class that:
- Accepts a genome.
- Decodes it into matrices and biases without copying unnecessarily.
- Performs a forward pass manually.
- Validates exactly 12 inputs and exactly 1,024 genes.
- Uses tanh after all three dense layers.
- Returns three float outputs in `[-1.0, 1.0]`.
- Has no external ML dependency.

INPUTS
Use exactly 12 normalized inputs, all preferably scaled to roughly `[-1, 1]` or `[0, 1]` consistently.

Use this ordering:

0. Far front-left ray distance
1. Near front-left ray distance
2. Left ray distance
3. Rear-left ray distance
4. Front ray distance
5. Rear ray distance
6. Far front-right ray distance
7. Near front-right ray distance
8. Right ray distance
9. Rear-right ray distance
10. Normalized forward speed
11. Signed heading error relative to the next checkpoint or local track direction

All ray sensors must return normalized distance from 0.0 to 1.0:
- 1.0 means no wall detected until maximum range.
- 0.0 means wall immediately adjacent.
- Sensor maximum range must be configurable.

The sensor angles relative to the car forward direction should be:

    -135, -90, -45, -20, 0, 180, +135, +90, +45, +20

If needed, order the input list exactly as requested above even if raycast implementation stores angles differently. Document clearly which angle corresponds to each input.

The heading-error input must be:
- Normalized into `[-1, 1]`.
- Negative when the next target direction is to the car’s left.
- Positive when the next target direction is to the car’s right.
- Computed from the signed shortest angular difference between the car heading and the target direction.

OUTPUTS
Use exactly three tanh outputs:

0. Steering:
   -1.0 is full left
   +1.0 is full right

1. Throttle:
   Convert from tanh to a non-negative acceleration request:
   throttle = max(0.0, output[1])

2. Brake/reverse:
   Convert from tanh to a non-negative brake/reverse request:
   brake = max(0.0, output[2])

Add logic preventing full throttle and full brake simultaneously:
- If both are non-zero, reduce the weaker one or prefer the larger command.
- Document the chosen approach.
- Use throttle for forward acceleration.
- Use brake for braking; at very low speed it may become reverse acceleration if configured.

CAR PHYSICS
Implement a simple but stable arcade top-down car model:
- Position x/y.
- Heading angle.
- Velocity vector or scalar forward speed.
- Steering rate / turn response.
- Acceleration.
- Braking.
- Drag.
- Maximum forward speed.
- Optional limited reverse speed.
- Collision response.
- Car rectangle/polygon collision footprint.
- Configurable values in a central configuration file.

The car should turn more naturally when moving faster, but keep the physics simple enough to run 100 cars efficiently. The model does not need realistic tire physics. It needs predictable, learnable movement.

TRACK AND PYGAME MAP
Use Pygame for the map and visualization.

Create an `assets/` directory and support:
- `assets/track.png`: a top-down track image.
- `assets/track_mirror.png`: optional generated mirrored image.
- `assets/checkpoints.json`: ordered checkpoints.

Track requirements:
- Read a PNG map.
- Use a clearly defined drivable-road color and a clearly defined wall/off-track color.
- Create an off-road collision mask from pixels.
- Support transparent background or an explicitly configured wall color.
- Generate a horizontally mirrored version of the track surface/mask in code with `pygame.transform.flip(surface, True, False)`.
- Mirror checkpoint positions and checkpoint segment endpoints correctly.
- Include a fallback procedural test track generator if `assets/track.png` does not exist.
- The fallback track must generate a closed loop with wide roads, walls, a start line, and ordered checkpoints, so the project runs immediately with no downloaded assets.

The procedural fallback track should:
- Use a dark grass/background.
- Draw a thick road loop.
- Draw walls or off-road regions.
- Create at least 12 ordered checkpoints around the loop.
- Save generated assets if appropriate.
- Be mirrored correctly.

CHECKPOINTS AND LAP VALIDATION
Implement ordered checkpoints. Each checkpoint is a line segment with an index.

Rules:
- A car can only gain progress by crossing the expected next checkpoint.
- A checkpoint crossed out of order does not grant progress.
- A car that crosses the final checkpoint and then the start/finish checkpoint in order completes a lap.
- Prevent repeated scoring from hovering around a checkpoint.
- Track `next_checkpoint_index`, `checkpoints_passed`, `laps_completed`, and `lap_time`.
- Track `best_lap_time` globally only for valid completed laps.
- Support an optional anti-cheat condition: cars must cross checkpoints in sequence and cannot get lap credit by driving backwards.

MIRROR SYMMETRY
Implement explicit symmetry helpers.

The exact input mirror mapping must be:

Original input index -> mirrored source and sign:

0 <- 6,  sign +1
1 <- 7,  sign +1
2 <- 8,  sign +1
3 <- 9,  sign +1
4 <- 4,  sign +1
5 <- 5,  sign +1
6 <- 0,  sign +1
7 <- 1,  sign +1
8 <- 2,  sign +1
9 <- 3,  sign +1
10 <- 10, sign +1
11 <- 11, sign -1

So implement:

def mirror_inputs(x):
    return np.array([
        x[6], x[7], x[8], x[9],
        x[4], x[5],
        x[0], x[1], x[2], x[3],
        x[10], -x[11],
    ], dtype=np.float32)

The output mirror must be:

def mirror_outputs(y):
    return np.array([
        -y[0],
         y[1],
         y[2],
    ], dtype=np.float32)

Do not claim hard weight tying unless it is actually implemented. The required symmetry mechanism for the first version is:
- Evaluate the same genome on normal and mirrored track layouts.
- Fitness equals the average of those runs.
- Add a configurable symmetry-consistency diagnostic:
  compare `network.forward(mirror_inputs(x))` with `mirror_outputs(network.forward(x))`.
- Display/log the mean symmetry error, but do not use it as a required training loss because this is neuroevolution, not backpropagation.
- Optionally allow a configurable small fitness penalty for large symmetry error, but default it to zero.

POPULATION AND GENERATIONS
Use exactly 100 cars per training generation by default.

The lifecycle must be:

1. Start generation with 100 cars.
2. Every car receives a genome.
3. Simulate all cars concurrently on the current evaluation track.
4. Individual cars die or become inactive when they:
   - collide with a wall/off-road area,
   - do not make forward checkpoint progress for a configurable timeout,
   - travel backward for too long,
   - exceed the generation time limit,
   - or finish the configured number of laps.
5. Do not end the generation just because one car gets a record.
6. End an episode only when all cars are inactive or the episode time limit has elapsed.
7. Evaluate every genome on both normal and mirrored versions of the map.
8. Combine the normal and mirrored score into one final fitness.
9. Rank genomes by final fitness.
10. Update all-time best controller and global valid lap record.
11. Create the next population of 100 genomes.
12. Reset all cars at the starting line.
13. Continue indefinitely or until configured maximum generations.

To avoid waiting for a first completed lap, scoring must be continuous and based primarily on valid ordered checkpoint progress, not merely full-lap time.

FITNESS FUNCTION
Implement a detailed, configurable fitness score.

Suggested form:

progress_score =
    laps_completed * 1_000_000
  + checkpoints_passed * 100_000
  + normalized_progress_to_next_checkpoint * 10_000

finish_bonus =
    500_000 - lap_time * 1_000
    only if a valid lap is completed

penalties =
    wall_hits * 2_000
  + stopped_seconds * 300
  + reverse_seconds * 2_000
  + unnecessary_time_penalty

fitness =
    progress_score
  + finish_bonus
  - penalties

Requirements:
- Use the furthest legitimate ordered progress reached in the episode, not current position only.
- Make it impossible for a crashed car to earn high score from a brief backward crossing.
- Include speed reward only if it does not encourage reckless wall collisions.
- Ensure a car making more checkpoint progress outranks a car that merely survives at the start.
- When valid laps begin occurring, fast clean laps must be more valuable than slow laps.
- Save per-car normal score, mirrored score, final score, progress, checkpoint count, laps, and time.

BREEDING POOL
Use a configurable breeding pool. Defaults:

POPULATION_SIZE = 100
BREEDING_POOL_SIZE = 20
ELITE_COUNT = 5
RANDOM_INJECTION_COUNT = 5

Rules:
- Sort the finished population by final fitness.
- The top 20 become the breeding pool.
- Preserve the top 5 current-generation genomes unchanged.
- Preserve the all-time best genome unchanged.
- Preserve the current valid lap-record genome unchanged when it differs from all-time best.
- Keep exact copies only once if genomes are identical.
- Do not only breed cars that beat the all-time lap record; use the top 20 by continuous fitness every generation.
- Use tournament selection from the breeding pool with tournament size 4.
- Include diversity in the next generation.

The intended next-generation composition is:
- 5 exact elite copies from the current generation.
- 1 exact all-time-best copy if not already included.
- 1 exact lap-record copy if not already included.
- Enough low-mutation descendants of strong parents to bring the protected/near-elite portion to approximately 15.
- Most remaining children from two-parent crossover followed by mutation.
- 5 fresh random genomes at the end.

Implement this safely so the population total is always exactly 100.

CROSSOVER
Genome crossover:
- For each of 1,024 genes, select either parent A or parent B with 50% probability.
- Also provide an optional arithmetic-blend crossover mode:
  child_gene = alpha * parent_a_gene + (1 - alpha) * parent_b_gene
  where alpha is random in [0, 1].
- Default to uniform per-gene crossover.
- Preserve float32 dtype.

MUTATION
Use Gaussian mutation:
- Default mutation rate: 0.08. This means each gene independently has an 8% chance of mutation.
- Default Gaussian sigma: 0.12.
- Use smaller mutation for near-elite children, for example:
  mutation rate 0.05
  sigma 0.05
- Clip every mutated gene into configurable bounds, default [-3.0, +3.0].
- Implement a `gaussian_random()` helper or use NumPy seeded generator.
- Record mutation statistics each generation.
- Add optional adaptive mutation:
  if best fitness has not improved for N generations, slowly increase mutation sigma;
  after meaningful improvement, return it toward the default.
- Keep adaptive mutation disabled by default but implement it cleanly.

INITIALIZATION
For generation zero:
- Initialize all 100 genomes randomly.
- Sample weights/biases from a configurable distribution, such as uniform [-1, 1] or normal mean 0, standard deviation 0.5.
- Make random seed configurable.
- Save the seed and settings in training logs.

SAVING AND RESUMING
Implement save/load support.

Save into `runs/<run_name>/`:
- `config.json`
- `generation_metrics.csv`
- `best_ever_genome.npy`
- `lap_record_genome.npy` when one exists
- `checkpoint_latest.npz` containing current generation, full population, RNG state if practical, best genome, lap record, and settings
- optional generation snapshots, configurable interval such as every 25 generations
- `best_car_replay.json` or equivalent metrics metadata

Must support:
- Press S to save.
- Press L to load latest checkpoint.
- Command-line `--resume path/to/checkpoint_latest.npz`.
- Save on exit if configured.
- Never overwrite a good best-ever genome accidentally.
- Validate genome length when loading.

HEADLESS/TURBO TRAINING
The trainer must support accelerated training:
- Render every N physics ticks rather than every tick.
- Toggle rendering off completely while continuing physics and evolution.
- Use a `SIMULATION_STEPS_PER_FRAME` configuration.
- In headless mode, run many physics steps per display update.
- Avoid unnecessary per-tick Python allocations.
- Preallocate sensor and network buffers where practical.
- Use NumPy arrays where helpful.
- Draw only the current best car in detailed mode if rendering becomes slow.
- Optionally draw all cars as small colored circles/rectangles.

KEYBOARD CONTROLS
Implement:

SPACE: pause/unpause
R: toggle normal rendering / fast rendering / headless mode
UP/DOWN or ]/[ : increase/decrease simulation steps per frame
N: advance one simulation tick while paused
G: force end current generation and breed next generation
S: save checkpoint now
L: load latest checkpoint
B: toggle drawing sensor rays
C: toggle drawing checkpoints
M: toggle normal versus mirrored preview map
H: show/hide help overlay
ESC: quit cleanly

DISPLAY/HUD
Show a clear HUD with:
- Current generation number.
- Current map orientation: normal or mirrored.
- Number of active cars / 100.
- Current simulation speed / steps per frame.
- Best current-generation fitness.
- All-time best fitness.
- Furthest checkpoint reached.
- Best valid lap record, if one exists.
- Current mutation rate and sigma.
- Breeding pool size.
- Headless/rendering state.
- Symmetry consistency diagnostic.
- FPS and simulated ticks per second.
- Status message when a new best-ever fitness or lap record occurs.

Color conventions:
- Current generation elite/best car: bright green.
- Record holder: gold/yellow.
- Ordinary active cars: semitransparent or muted blue/red.
- Inactive/crashed cars: hidden or dark gray.
- Sensor rays: configurable bright lines.
- Checkpoints: visible translucent lines and index labels in debug mode.

PROJECT STRUCTURE
Create a sensible file tree such as:

mirror_racer_ai/
  README.md
  requirements.txt
  main.py
  config.py
  simulation.py
  car.py
  track.py
  sensors.py
  neural_net.py
  evolution.py
  persistence.py
  visualization.py
  assets/
    .gitkeep
  runs/
    .gitkeep
  tests/
    test_network.py
    test_mirror.py
    test_evolution.py

You may alter names if the structure remains modular and clear.

REQUIRED CLASSES / DATA MODELS
Implement practical equivalents of:

- Config
- Genome / flat numpy genome utilities
- NeuralNetwork
- CarState
- Car
- SensorSystem
- Checkpoint
- Track
- EpisodeResult
- GenerationResult
- EvolutionManager
- PersistenceManager
- GameRenderer or Visualization

TESTS
Include small runnable tests for:
1. The network has exactly 1,024 parameters.
2. Genome decoding produces correct shapes:
   W1 shape (25, 12), b1 shape (25,)
   W2 shape (24, 25), b2 shape (24,)
   W3 shape (3, 24), b3 shape (3,)
3. Network output is exactly three finite values in [-1, 1].
4. `mirror_inputs()` has the exact requested index/sign mapping.
5. `mirror_outputs()` negates only steering.
6. A generated next generation always has exactly 100 genomes of shape (1024,).
7. Mutation remains within configured clipping bounds.
8. Mirroring a checkpoint twice returns it to the original position.
9. A car cannot score the same checkpoint repeatedly without passing the next one.
10. Save/load preserves genomes and generation metadata.

README
Write a complete README with:
- Project description.
- Why neuroevolution is used.
- Exact network architecture and 1,024-parameter derivation.
- Explanation of the normal/mirrored evaluation scheme.
- Installation commands:
  python -m venv .venv
  source .venv/bin/activate on Linux/macOS
  .venvScriptsactivate on Windows
  pip install -r requirements.txt
  python main.py
- How to add a custom PNG track.
- Pixel-color/mask configuration for roads/walls.
- How to define checkpoints.
- How the fitness function works.
- How the breeding pool, elitism, crossover, random injection, and mutation work.
- Keyboard controls.
- How to resume training.
- Troubleshooting for cars that do not learn.
- Performance advice for 100 cars.
- A note that this is an evolutionary simulator and not a guarantee of real-world vehicle safety or autonomous-driving capability.

DELIVERY REQUIREMENTS
Return:
1. The full file tree.
2. Complete contents of every source file.
3. Complete requirements.txt.
4. Complete README.md.
5. Instructions to run it.
6. Do not omit code due to length.
7. Ensure imports and filenames are consistent.
8. Verify logically that the parameter count is exactly 1,024 and the population is exactly 100.
9. Make the default fallback track runnable immediately.
10. Do not use external assets, models, APIs, or ML frameworks.
11. Use a very modular layout to be readable.
