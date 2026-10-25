# RL Container Management Using PPO and Collision Prediction

**Reference paper:** A. Pendyala and T. Glasmachers, *Curriculum RL meets Monte Carlo Planning: Optimization of a Real World Container Management Problem* (2025). [Read the paper](https://arxiv.org/abs/2503.17194).

This project is based on the paper above. The main idea is to use reinforcement learning to decide when a container should be emptied, while trying to avoid overflow and make better use of the processing unit. I implemented a smaller version of the problem to understand the approach and test it in a working simulation. It is inspired by the paper, but it is not the original ContainerGym environment or an exact reproduction of the reported experiments.

## What the project does

The simulation has multiple containers whose fill levels increase over time. A shared processing unit empties one container at a time and remains busy for a certain number of steps after an operation. The agent has to decide whether to wait or empty a particular container.

The project has a few main parts:

- **Simulation environment (`waste_sorting_env.py`):** Handles container filling, available actions, processing-unit status, rewards and overflow.
- **PPO agent (`ppo_numpy.py`):** Uses an actor–critic network to learn which action to take. Training uses the clipped PPO objective, advantage estimation, entropy regularization and gradient updates through PyTorch.
- **Reward curriculum:** The reward changes over three phases. It starts with one preferred fill level, then introduces two target levels, and finally rewards emptying within a defined tolerance of either target.
- **Collision prediction (`collision_model.py`):** Monte Carlo simulations generate pairwise examples for training a classifier. XGBoost is used when available, with a scikit-learn fallback. During inference, the model can override a no-op action when the predicted collision risk is high and the processing unit is idle.
- **Experiments (`run_experiments.py`):** Trains and compares the PPO approaches with a vanilla PPO agent, a collision-aware agent, a safety heuristic and a random policy.

The action `0` means no-op, while actions `1` through `N` select the corresponding container for emptying. If the processing unit is busy, an emptying request cannot be processed. An overflow ends the episode.

## Running the simulation

Python 3.10 or later is recommended. A CPU is sufficient for this version.

First, install the required packages:

```bash
python -m pip install -r requirements.txt
```

To play the simulation yourself in the terminal:

```bash
python play_game.py --mode manual
```

Enter `0` to wait or a number from `1` to `N` to empty a container. Enter `q` to stop.

You can also run a random policy:

```bash
python play_game.py --mode random
```

To watch the trained PPO policy, use the saved checkpoint:

```bash
python play_game.py --mode ppo --checkpoint curriculum_ppo_seed7.pt
```

The checkpoint needs to be present in the project directory.

## Training and testing

Run the tests with:

```bash
pytest -q
```

To train the models and run the experiments:

```bash
python run_experiments.py
```

The experiment script uses three training seeds (`7`, `19` and `43`), trains the PPO variants, generates Monte Carlo data for the collision model and evaluates the different approaches.

The generated files include:

- `results_by_seed.csv` – results for individual seeds
- `summary.json` – average evaluation metrics
- `training_history.csv` – training statistics
- `collision_model.pkl` and `collision_model_backend.txt` – saved collision model and backend information
- `curriculum_ppo_seed*.pt` and `vanilla_ppo_seed*.pt` – PPO checkpoints
- `throughput_comparison.png` and `safety_comparison.png` – comparison charts

## Results from the current run

The following values are averages over three training seeds and five evaluation episodes per seed. Throughput is the total normalized volume emptied per episode. Overflow violations and collision timesteps are reported as counts, not percentages.

| Method | Mean return | Throughput | Overflow violations | Collision timesteps | PU idle fraction | Overrides |
|---|---:|---:|---:|---:|---:|---:|
| Curriculum PPO | -0.8333 | 6.9896 | 0.3333 | 0.3333 | 0.2277 | 0.0000 |
| Vanilla PPO | -1.0000 | 6.2397 | 0.4000 | 0.0000 | 0.4780 | 0.0000 |
| Curriculum PPO + collision model | -0.9000 | 6.9459 | 0.3333 | 0.2667 | 0.2240 | 0.1333 |
| Safety heuristic | 5.7333 | 9.7407 | 0.0000 | 0.0000 | 0.1900 | 0.0000 |
| Random policy | -8.9333 | 4.1519 | 0.9333 | 0.1333 | 0.1265 | 0.0000 |

In this run, curriculum PPO achieved slightly higher throughput than vanilla PPO. Adding the collision model reduced the average collision timesteps from `0.3333` to `0.2667`, with little change in throughput. However, it did not reduce overflow violations. The safety heuristic performed best on these particular metrics, so the results do not show that the learned policy is better than every baseline.

These numbers come from this implementation and its selected settings. They are not the results reported in the research paper, and they may change with different random seeds or training settings.

## How this differs from the paper

The original paper evaluates configurations with 7–12 containers, episodes of 600 timesteps, 15 training seeds and five evaluation rollouts per seed. It also describes a collision dataset based on two million repetitions.

This implementation is smaller: it supports 1–3 containers, uses shorter episodes and three training seeds, and generates a much smaller Monte Carlo dataset of 10,000 pairwise samples. The container dynamics, reward setup and collision model are simplified as well.

The aim was to build and test the main ideas in a manageable simulation, rather than reproduce the original environment or its numerical results exactly.

## Main files

- `waste_sorting_env.py` – simulation environment and reward logic
- `ppo_numpy.py` – PPO actor–critic agent
- `collision_model.py` – Monte Carlo data generation and collision prediction
- `run_experiments.py` – training, evaluation and baseline comparisons
- `play_game.py` – terminal simulation
- `test_project.py` – automated tests
- `requirements.txt` – package dependencies

## Reference

Pendyala, A. and Glasmachers, T. (2025). *Curriculum RL meets Monte Carlo Planning: Optimization of a Real World Container Management Problem*. arXiv:2503.17194v1. https://arxiv.org/abs/2503.17194
