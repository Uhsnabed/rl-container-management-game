# Curriculum PPO + Monte Carlo Collision Model for Container Management

**Paper basis:** A. Pendyala and T. Glasmachers, *Curriculum RL meets Monte Carlo Planning: Optimization of a Real World Container Management Problem*, arXiv:2503.17194v1 (21 March 2025), https://arxiv.org/abs/2503.17194.

This project implements a **small educational approximation of the paper's core approach**. It is not the authors' original ContainerGym environment and it does not reproduce their published numerical results.

## What is implemented

1. **Container MDP** (`waste_sorting_env.py`): discrete actions `0=no-op`, `i+1=empty container i`; stochastic filling; a shared processing-unit busy counter; overflow penalty/termination; target peaks at 0.35 and 0.68 after normalizing the paper's 0–40 volume scale to 0–1.
2. **PPO actor–critic** (`ppo_numpy.py`): separate neural actor and critic, PyTorch backpropagation, Adam, clipped PPO policy objective, clipped value objective, entropy regularization, GAE, normalized advantages, minibatches, and gradient clipping. `loss.backward()` computes gradients and `optimizer.step()` updates weights.
3. **Three-phase reward curriculum**: phase 1 rewards a Gaussian centered at the higher peak; phase 2 uses Gaussian rewards around both lower and higher peaks; phase 3 gives reward 1 only within the paper's ±1 physical-unit tolerance around either peak, and 0 otherwise. No-op reward is 0; invalid empty attempts are penalized.
4. **Offline Monte Carlo collision model** (`collision_model.py`): simulated pairwise fill trajectories produce binary collision labels; an XGBoost classifier is trained offline (scikit-learn histogram-gradient-boosting fallback if XGBoost is unavailable). At inference, when PPO proposes no-op and the PU is idle, predicted pairwise risk can override no-op to empty the most at-risk near-peak container. An empty attempt while the PU is busy is never used as an override because the paper's MDP says such requests are lost.
5. **Evaluation**: PPO with curriculum, vanilla PPO, collision-augmented PPO, a safety heuristic, and random policy are compared using the same evaluation seed families. Metrics include return, throughput, overflow violations, collision timesteps, energy/action count, press idle fraction, peak deviation, and number of overrides.

## Play the simulation as a terminal game

The environment is a step-by-step RL game/simulation; graphics are not required. You can play it yourself or watch an agent:

```bash
# Choose each action yourself
python play_game.py --mode manual

# Watch the saved PPO agent play
python play_game.py --mode ppo --checkpoint curriculum_ppo_seed7.pt

# Watch a random baseline
python play_game.py --mode random
```

Each turn displays container fill levels, processing-unit status, the chosen action, reward, throughput, overflow events, and collision timesteps. In manual mode, enter `0` to wait or `1..N` to empty a container; enter `q` to stop.

## Run the project

Python 3.10+ is recommended. CPU is sufficient.

```bash
python -m pip install -r requirements.txt
pytest -q
python run_experiments.py
```

The script trains PPO variants using seeds `7, 19, 43`, fits the collision model on offline Monte Carlo data, saves checkpoints and model, and writes:

- `results_by_seed.csv` — per-seed evaluation metrics
- `summary.json` — averages across the three seeds
- `training_history.csv` — training returns, throughput, violations and PPO losses
- `collision_model.pkl` and `collision_model_backend.txt`
- `curriculum_ppo_seed*.pt` and `vanilla_ppo_seed*.pt`
- `throughput_comparison.png` and `safety_comparison.png` — summary charts

## Current run results (our simulator, not the paper's results)

Results are averages across 3 training seeds and 5 evaluation episodes per seed. Throughput is total normalized volume emptied in one episode. Violations and collisions are counts, not percentages.

| Method | Mean return | Throughput | Overflow violations | Collision timesteps | Press idle fraction | Overrides |
|---|---:|---:|---:|---:|---:|---:|
| Curriculum PPO | -0.8333 | 6.9896 | 0.3333 | 0.3333 | 0.2277 | 0.0000 |
| Vanilla PPO | -1.0000 | 6.2397 | 0.4000 | 0.0000 | 0.4780 | 0.0000 |
| Curriculum PPO + collision model | -0.9000 | 6.9459 | 0.3333 | 0.2667 | 0.2240 | 0.1333 |
| Safety heuristic | 5.7333 | 9.7407 | 0.0000 | 0.0000 | 0.1900 | 0.0000 |
| Random | -8.9333 | 4.1519 | 0.9333 | 0.1333 | 0.1265 | 0.0000 |

**Interpretation:** In this run, curriculum PPO has somewhat higher throughput than vanilla PPO. The collision model lowers mean collision timesteps from 0.3333 to 0.2667 without reducing throughput materially, but does not reduce overflow violations. The safety heuristic is the strongest baseline in these short experiments. Do not claim the learned policy is superior or that the paper's results were reproduced.

## Differences from the paper (be transparent in report/viva)

The paper evaluates 7–12 containers per processing unit, 600 timesteps per episode, 15 independent training seeds and 5 evaluation rollouts per seed, and reports a pairwise collision dataset generated using two million repetitions. This teaching implementation is limited to 1–3 containers, uses shorter episodes, three training seeds, a much smaller Monte Carlo dataset (10,000 pairwise samples), and simplified normalized fill/processing dynamics. The reward curriculum and PPO-CL-CM architecture are implemented in simplified form; the exact original environment, distributions, hyperparameter schedule, full configuration sweep, and experimental scale are not reproduced.

The reported numbers above are generated by the included code in this environment. They are not copied from the paper. RL metrics vary by seed and hyperparameters.

## Files

- `waste_sorting_env.py`: MDP, stochastic fill dynamics, reward curriculum and safety metrics.
- `ppo_numpy.py`: PyTorch PPO actor–critic implementation (legacy filename retained for compatibility).
- `collision_model.py`: offline Monte Carlo pairwise collision dataset, classifier and inference override.
- `run_experiments.py`: training/evaluation pipeline and baselines.
- `test_project.py`: automated tests for curriculum, backpropagation, action semantics, safety and collision override.
- `CODE_WALKTHROUGH.md`: beginner-friendly code explanation.

## Reference

Pendyala, A. & Glasmachers, T. (2025). *Curriculum RL meets Monte Carlo Planning: Optimization of a Real World Container Management Problem*. arXiv:2503.17194v1. https://arxiv.org/abs/2503.17194
