"""Tabular Q-learning and SARSA on FrozenLake."""

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

ACTION_NAMES = ("LEFT", "DOWN", "RIGHT", "UP")
ACTION_DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))  # (row, col) per action

DEFAULTS = {
    "n_episodes": 30_000,
    "alpha": 0.1,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "snapshot_every": 1_000,
}


def make_env(map_name="8x8", is_slippery=True, success_rate=1 / 3, max_episode_steps=200):
    # FrozenLake-v1 registers only 100 steps, too few for the 8x8 map
    return gym.make(
        "FrozenLake-v1",
        map_name=map_name,
        is_slippery=is_slippery,
        success_rate=success_rate,
        max_episode_steps=max_episode_steps,
    )


def greedy_action(Q, state, rng):
    q = Q[state]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def epsilon_greedy(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    return greedy_action(Q, state, rng)


def _td_update(Q, s, a, target, alpha):
    td_error = target - Q[s, a]
    Q[s, a] += alpha * td_error
    return float(td_error)


def q_learning_update(Q, s, a, r, s_next, terminated, alpha, gamma):
    bootstrap = 0.0 if terminated else gamma * Q[s_next].max()
    return _td_update(Q, s, a, r + bootstrap, alpha)


def sarsa_update(Q, s, a, r, s_next, a_next, terminated, alpha, gamma):
    bootstrap = 0.0 if terminated else gamma * Q[s_next, a_next]
    return _td_update(Q, s, a, r + bootstrap, alpha)


def linear_epsilon(episode, n_episodes, eps_start, eps_end):
    if n_episodes <= 1:
        return eps_end
    frac = min(episode / (n_episodes - 1), 1.0)
    return eps_start + frac * (eps_end - eps_start)


@dataclass
class TrainResult:
    Q: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)


def _run_episode(env, Q, algo, epsilon, alpha, gamma, rng, seed):
    s, _ = env.reset(seed=seed)
    a = epsilon_greedy(Q, s, epsilon, rng)
    total, length = 0.0, 0
    while True:
        s_next, r, terminated, truncated, _ = env.step(a)
        total += r
        length += 1
        a_next = epsilon_greedy(Q, s_next, epsilon, rng)
        if algo == "q_learning":
            q_learning_update(Q, s, a, r, s_next, terminated, alpha, gamma)
        else:
            sarsa_update(Q, s, a, r, s_next, a_next, terminated, alpha, gamma)
        if terminated or truncated:
            return total, length
        s, a = s_next, a_next


def train(env, algo, n_episodes, alpha, gamma, eps_start, eps_end, seed, snapshot_every):
    if algo not in ("q_learning", "sarsa"):
        raise ValueError(f"unknown algo {algo!r}")
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = np.zeros(n_episodes)
    lengths = np.zeros(n_episodes, dtype=int)
    snapshots = {}
    for episode in range(n_episodes):
        epsilon = linear_epsilon(episode, n_episodes, eps_start, eps_end)
        reset_seed = seed if episode == 0 else None
        returns[episode], lengths[episode] = _run_episode(
            env, Q, algo, epsilon, alpha, gamma, rng, reset_seed
        )
        if episode % snapshot_every == 0 or episode == n_episodes - 1:
            snapshots[episode] = Q.copy()
    return TrainResult(Q, returns, lengths, snapshots)


def success_rate(returns, last_n=1000):
    return float(np.mean(np.asarray(returns)[-last_n:] > 0))


def evaluate(env, Q, n_episodes=1000, seed=0):
    """Success rate of the greedy policy (no exploration, no learning)."""
    rng = np.random.default_rng(seed)
    wins = 0
    env.reset(seed=seed)
    for _ in range(n_episodes):
        s, _ = env.reset()
        while True:
            s, r, terminated, truncated, _ = env.step(greedy_action(Q, s, rng))
            if terminated or truncated:
                wins += r > 0
                break
    return wins / n_episodes


if __name__ == "__main__":
    result = train(make_env(), "q_learning", seed=0, **DEFAULTS)
    print(f"seed 0 success rate (last 1000): {success_rate(result.returns):.3f}")
