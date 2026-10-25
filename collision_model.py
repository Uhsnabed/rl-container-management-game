"""Offline Monte Carlo pairwise collision predictor inspired by paper Section 5.

This is a lightweight educational approximation, not the authors' original data
pipeline. It generates pairwise stochastic fill trajectories, labels imminent
collisions while the PU is busy, fits XGBoost when installed (HistGradientBoosting
fallback), and exposes an inference-time no-op override.
"""
from __future__ import annotations
import numpy as np

FEATURE_NAMES = ["volume_i", "volume_j", "gap_high_i", "gap_high_j",
                 "fill_rate_i", "fill_rate_j", "busy_steps"]


def generate_pairwise_data(n_samples=30000, seed=123, horizon=12):
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.int8)
    # Monte Carlo rollouts: independently sampled initial volumes/fill rates,
    # then flag if both containers approach their high peak while PU is busy.
    for k in range(n_samples):
        vi, vj = rng.uniform(0.15, 0.95, 2)
        ri, rj = rng.uniform(0.025, 0.075, 2)
        busy = int(rng.integers(1, 5))
        X[k] = [vi, vj, max(0., .68-vi), max(0., .68-vj), ri, rj, busy]
        a, b = vi, vj
        collision = False
        for t in range(1, horizon + 1):
            a = max(0., a + ri + rng.normal(0, .018))
            b = max(0., b + rj + rng.normal(0, .018))
            if t <= busy and a >= .62 and b >= .62:
                collision = True
            if a > 1.0 or b > 1.0:
                collision = collision or (a >= .62 and b >= .62)
                break
        y[k] = int(collision)
    return X, y


def fit_collision_model(n_samples=30000, seed=123):
    X, y = generate_pairwise_data(n_samples=n_samples, seed=seed)
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.08,
                              subsample=.85, colsample_bytree=.9, eval_metric='logloss',
                              random_state=seed, n_jobs=1)
        backend = 'xgboost'
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                               learning_rate=.08, random_state=seed)
        backend = 'sklearn HistGradientBoosting fallback'
    model.fit(X, y)
    return model, backend


def pairwise_override(env, proposed_action, model, threshold=.35):
    """Override only no-op, choosing the most at-risk near-high container."""
    # Overrides are useful only while the PU is idle: requests made while it
    # is busy are lost in the paper's MDP. Predict whether waiting now would
    # expose a pair of near-peak containers during the next processing window.
    if proposed_action != 0 or env.busy > 0:
        return proposed_action, 0.0
    levels = env.levels
    near = np.flatnonzero(levels >= env.HIGH_PEAK - .10)
    if len(near) < 2:
        return 0, 0.0
    risks = np.zeros(env.n, dtype=float)
    for pos, i in enumerate(near):
        for j in near[pos + 1:]:
            ri = .05
            rj = .05
            features = np.array([[levels[i], levels[j], max(0, env.HIGH_PEAK-levels[i]),
                                  max(0, env.HIGH_PEAK-levels[j]), ri, rj,
                                  max(env.busy, int(np.ceil(1.0 + 2.0 * max(levels[i], levels[j]))))]], dtype=np.float32)
            prob = float(model.predict_proba(features)[0, 1])
            risks[i] = max(risks[i], prob)
            risks[j] = max(risks[j], prob)
    candidate = int(np.argmax(risks))
    if risks[candidate] >= threshold:
        return candidate + 1, float(risks[candidate])
    return 0, float(risks[candidate])
