# AICL 407 — RL Labs

One shared [uv](https://docs.astral.sh/uv/) project (Python 3.11) for all labs. Each `LabAssignmentN/` folder contains only that assignment's code, tests, and outputs.

| Lab | Topic | Environment | Main module |
| --- | --- | --- | --- |
| [`LabAssignment1/`](LabAssignment1) | Tabular Q-learning and SARSA | FrozenLake 8x8 (slippery) | `qlearning.py` |
| [`LabAssignment2/`](LabAssignment2) | Deep Q-Network, with a no-target-network ablation | CartPole-v1 | `dqn.py` |

## Setup

```sh
uv sync                                  # once: creates .venv with numpy, matplotlib, gymnasium, torch
```

## Lab 1 — Q-learning and SARSA on FrozenLake

```sh
cd LabAssignment1
uv run python qlearning.py               # train seed 0 and print the last-1000 success rate
uv run python experiments.py             # train everything -> results/ + figures/
```

## Lab 2 — DQN on CartPole

```sh
cd LabAssignment2
uv run python dqn.py                     # train seed 0 and print the best 100-episode moving average
uv run python experiments.py             # train baseline + no-target ablation -> results/ + figures/
```

## Experiment options

Both labs' `experiments.py` take the same flags:

| Flag | Effect |
| --- | --- |
| `--quick` | Tiny run to check the pipeline; writes to `quick_run/` instead of `results/` and `figures/` |
| `--force` | Retrain even if a cached result exists (finished runs are otherwise reused) |
| `--workers N` | Number of parallel training processes (default 6) |
