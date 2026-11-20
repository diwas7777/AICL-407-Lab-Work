"""Deep Q-Network on CartPole."""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

DEFAULTS = {
    "n_episodes": 500,
    "gamma": 0.99,
    "lr": 5e-4,
    "lr_end": 2.5e-5,
    "lr_decay_start": 50_000,
    "lr_decay_steps": 100_000,
    "buffer_size": 200_000,
    "batch_size": 64,
    "warmup_steps": 1_000,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 10_000,
    "target_sync": 500,
    "eval_every": 20,
    "eval_episodes": 5,
    "hidden": 256,
    "max_grad_norm": None,
}


def make_env():
    return gym.make("CartPole-v1")


class ReplayBuffer:
    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = 1.0 if done else 0.0
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.actions[idx]),
            torch.from_numpy(self.rewards[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.dones[idx]),
        )

    def __len__(self):
        return self.size


class QNetwork(nn.Module):
    def __init__(self, obs_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


def dqn_loss(online, target, batch, gamma):
    s, a, r, s_next, done = batch
    q = online(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        y = r + gamma * (1.0 - done) * target(s_next).max(1).values
    return F.mse_loss(q, y)


def sync_target(online, target):
    target.load_state_dict(online.state_dict())
    target.eval()


def linear_epsilon(step, warmup_steps, eps_start, eps_end, eps_decay_steps):
    if step < warmup_steps:
        return eps_start
    frac = min(step / eps_decay_steps, 1.0)
    return eps_start + frac * (eps_end - eps_start)


def linear_lr(step, lr, lr_end, lr_decay_start, lr_decay_steps):
    """Learning rate at a global step: lr until lr_decay_start, then linear to lr_end over lr_decay_steps."""
    if lr_end is None or lr_decay_steps is None:
        return lr
    frac = min(max(step - lr_decay_start, 0) / lr_decay_steps, 1.0)
    return lr + frac * (lr_end - lr)


def greedy_action(net, obs):
    with torch.no_grad():
        return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax())


def select_action(net, obs, epsilon, rng, n_actions):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    return greedy_action(net, obs)


def evaluate(env, net, n_episodes, seed):
    total = 0.0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        while True:
            obs, r, terminated, truncated, _ = env.step(greedy_action(net, obs))
            total += r
            if terminated or truncated:
                break
    return total / n_episodes


def moving_average(x, window=100):
    window = min(window, len(x))
    return np.convolve(x, np.ones(window) / window, mode="valid")


@dataclass
class TrainResult:
    returns: np.ndarray
    losses: np.ndarray
    mean_max_q: np.ndarray
    epsilons: np.ndarray
    eval_episodes: np.ndarray
    eval_returns: np.ndarray
    online: QNetwork
    target: QNetwork


def train_dqn(env, seed, n_episodes, gamma, lr, buffer_size, batch_size, warmup_steps,
              eps_start, eps_end, eps_decay_steps, target_sync, eval_every, eval_episodes, hidden,
              max_grad_norm=None, lr_end=None, lr_decay_start=0, lr_decay_steps=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    online, target = QNetwork(obs_dim, n_actions, hidden), QNetwork(obs_dim, n_actions, hidden)
    sync_target(online, target)
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, obs_dim)
    eval_env = make_env()

    returns, losses, mean_max_q, epsilons = [], [], [], []
    eval_episodes_done, eval_returns = [], []
    step = 0
    obs, _ = env.reset(seed=seed)
    for episode in range(n_episodes):
        if episode > 0:
            obs, _ = env.reset()
        episode_return = 0.0
        while True:
            epsilon = linear_epsilon(step, warmup_steps, eps_start, eps_end, eps_decay_steps)
            action = select_action(online, obs, epsilon, rng, n_actions)
            next_obs, r, terminated, truncated, _ = env.step(action)
            buffer.push(obs, action, r, next_obs, terminated)
            epsilons.append(epsilon)
            episode_return += r
            obs = next_obs
            step += 1

            if step > warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = linear_lr(step, lr, lr_end, lr_decay_start, lr_decay_steps)
                batch = buffer.sample(batch_size, rng)
                loss = dqn_loss(online, target, batch, gamma)
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(online.parameters(), max_grad_norm)
                optimizer.step()
                losses.append(loss.item())
                with torch.no_grad():
                    mean_max_q.append(online(batch[0]).max(1).values.mean().item())
            if step % target_sync == 0:
                sync_target(online, target)
            if terminated or truncated:
                break
        returns.append(episode_return)
        if (episode + 1) % eval_every == 0:
            eval_episodes_done.append(episode + 1)
            eval_returns.append(evaluate(eval_env, online, eval_episodes, seed=10_000 + seed))

    return TrainResult(
        np.array(returns), np.array(losses), np.array(mean_max_q), np.array(epsilons),
        np.array(eval_episodes_done), np.array(eval_returns), online, target,
    )


if __name__ == "__main__":
    result = train_dqn(make_env(), seed=0, **DEFAULTS)
    print(f"seed 0 best 100-episode moving average: {moving_average(result.returns).max():.1f}")
