"""Train/evaluate PPO variants and simple baselines with reproducible seeds."""
from pathlib import Path
import csv, json
import numpy as np
from waste_sorting_env import WasteSortingEnv
from ppo_numpy import LinearPPO
from collision_model import fit_collision_model, pairwise_override
import pickle

ROOT = Path(__file__).parent
SEEDS = (7, 19, 43)


def train(seed, curriculum=True, total_episodes=180, save_path=None):
    agent = LinearPPO(seed=seed)
    if curriculum:
        first = total_episodes // 3
        second = total_episodes // 3
        stages = [(1, first), (2, second), (3, total_episodes - first - second)]
    else:
        stages = [(3, total_episodes)]
    history = []
    episode_num = 0
    for stage, episodes in stages:
        env = WasteSortingEnv(stage=stage, seed=seed + 101 * stage)
        for _ in range(episodes):
            obs, done, trajectory = env.reset(), False, []
            while not done:
                action, logp, value = agent.act(obs, valid_actions=env.n + 1)
                nxt, reward, done, stats = env.step(action)
                trajectory.append(dict(obs=obs.copy(), action=action, logp=logp, value=value,
                                       reward=reward, done=done,
                                       action_mask=np.arange(4) < (env.n + 1)))
                obs = nxt
            # GAE uses the stored critic estimates and episode terminal flags.
            returns, advantages = agent._returns_and_advantages(
                [x['reward'] for x in trajectory], [x['value'] for x in trajectory],
                [x['done'] for x in trajectory], last_value=0.0)
            for i, step in enumerate(trajectory):
                step['return'], step['advantage'] = returns[i], advantages[i]
            update_stats = agent.update(trajectory, epochs=4, minibatch_size=128)
            episode_num += 1
            history.append(dict(seed=seed, method='curriculum_ppo' if curriculum else 'vanilla_ppo',
                                stage=stage, episode=episode_num,
                                return_=float(sum(x['reward'] for x in trajectory)),
                                steps=len(trajectory), throughput=stats['throughput'],
                                violations=stats['violations'], collisions=stats['collisions'],
                                policy_loss=update_stats.get('policy_loss', 0.0),
                                value_loss=update_stats.get('value_loss', 0.0),
                                entropy=update_stats.get('entropy', 0.0)))
    if save_path:
        agent.save(save_path)
    return agent, history


def evaluate(policy, seed, episodes=10):
    rows = []
    for ep in range(episodes):
        env = WasteSortingEnv(stage=3, seed=seed * 1000 + ep)
        obs, done, total = env.reset(), False, 0.0
        while not done:
            if policy == 'random':
                action = int(env.rng.integers(env.n + 1))
            elif policy == 'greedy':
                action = 0
                if env.busy == 0:
                    candidates = np.flatnonzero(env.levels >= env.LOW_PEAK)
                    if len(candidates):
                        # Prefer containers near either target, with higher peak preferred.
                        scores = np.maximum(np.exp(-((env.levels[candidates]-env.HIGH_PEAK)/.10)**2),
                                            .55*np.exp(-((env.levels[candidates]-env.LOW_PEAK)/.08)**2))
                        action = int(candidates[np.argmax(scores)]) + 1
            else:
                action, _, _ = policy.act(obs, deterministic=True, valid_actions=env.n + 1)
            obs, reward, done, stats = env.step(action)
            total += reward
        rows.append(dict(return_=total, **stats))
    return rows


def evaluate_collision_model(policy, collision_model, seed, episodes=10, threshold=0.35):
    rows = []
    for ep in range(episodes):
        env = WasteSortingEnv(stage=3, seed=seed * 1000 + ep)
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action, _, _ = policy.act(obs, deterministic=True, valid_actions=env.n + 1)
            final_action, risk = pairwise_override(env, action, collision_model, threshold=threshold)
            if final_action != action:
                env.stats['overrides'] += 1
            obs, reward, done, stats = env.step(final_action)
            total += reward
        rows.append(dict(return_=total, **stats))
    return rows


def mean(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    all_history, summary = [], []
    trained_curriculum = {}
    # Reuse each trained curriculum policy for the collision-model comparison;
    # avoid redundant training that would waste time and compute.
    for seed in SEEDS:
        print(f"Training curriculum PPO, seed={seed} ...", flush=True)
        curriculum, hist = train(seed, True, total_episodes=180,
                                 save_path=ROOT / f'curriculum_ppo_seed{seed}.pt')
        trained_curriculum[seed] = curriculum
        all_history.extend(hist)
        print(f"Training vanilla PPO, seed={seed} ...", flush=True)
        vanilla, hist = train(seed, False, total_episodes=180,
                              save_path=ROOT / f'vanilla_ppo_seed{seed}.pt')
        all_history.extend(hist)
        for name, policy in [('Curriculum PPO', curriculum), ('Vanilla PPO', vanilla),
                             ('Safety heuristic', 'greedy'), ('Random', 'random')]:
            metrics = mean(evaluate(policy, seed, episodes=5))
            metrics.update(method=name, seed=seed)
            summary.append(metrics)
    print('Generating offline Monte Carlo collision data and fitting classifier ...', flush=True)
    collision_model, backend = fit_collision_model(n_samples=10000, seed=123)
    with (ROOT / 'collision_model.pkl').open('wb') as f:
        pickle.dump(collision_model, f)
    for seed in SEEDS:
        metrics = mean(evaluate_collision_model(trained_curriculum[seed], collision_model,
                                                seed, episodes=5))
        metrics.update(method='Curriculum PPO + collision model', seed=seed)
        summary.append(metrics)
    (ROOT / 'collision_model_backend.txt').write_text(backend + '\n', encoding='utf-8')
    for filename, rows in [('training_history.csv', all_history), ('results_by_seed.csv', summary)]:
        with (ROOT / filename).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    metrics_to_report = ['return_', 'throughput', 'violations', 'collisions', 'energy',
                         'press_idle_fraction', 'mean_peak_deviation', 'overrides']
    grouped = {}
    for row in summary:
        grouped.setdefault(row['method'], []).append(row)
    report = {name: {k: round(float(np.mean([x[k] for x in rows])), 4)
                              for k in metrics_to_report} for name, rows in grouped.items()}
    (ROOT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
