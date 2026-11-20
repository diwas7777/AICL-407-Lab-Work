"""Run the Lab 2 experiments (baseline DQN and no-target ablation), saving results/, figures/ and stats.json."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import dqn

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
VARIANTS = {"baseline": {}, "no_target": {"target_sync": 1}}
LABELS = {"baseline": "DQN (target sync every 500 steps)", "no_target": "no target network (sync every step)"}
COLORS = {"baseline": "tab:blue", "no_target": "tab:red"}
WINDOW = 100
SOLVED = 475
TASK4_EPISODES = 500
SMOOTH = 200  # steps, for loss / max-Q curves


@dataclass(frozen=True)
class Config:
    seeds: tuple
    out_dir: Path
    overrides: dict = field(default_factory=dict)

    @property
    def results_dir(self):
        return self.out_dir / "results"

    @property
    def figures_dir(self):
        return self.out_dir / "figures"


FULL = Config(tuple(range(5)), HERE, {"n_episodes": 700})
QUICK = Config((0, 1), HERE / "quick_run", {"n_episodes": 40, "warmup_steps": 100, "eps_decay_steps": 500,
                                            "buffer_size": 5_000, "eval_every": 10, "eval_episodes": 2})


@dataclass(frozen=True)
class RunSpec:
    variant: str
    seed: int

    @property
    def name(self):
        return f"{self.variant}_s{self.seed}"


def params_for(spec, cfg):
    return {**dqn.DEFAULTS, **cfg.overrides, **VARIANTS[spec.variant]}


def all_specs(cfg):
    return [RunSpec(v, s) for v in VARIANTS for s in cfg.seeds]


def _train_and_save(spec, params, path):
    torch.set_num_threads(1)
    r = dqn.train_dqn(dqn.make_env(), seed=spec.seed, **params)
    np.savez_compressed(path, returns=r.returns, losses=r.losses, mean_max_q=r.mean_max_q,
                        epsilons=r.epsilons, eval_episodes=r.eval_episodes, eval_returns=r.eval_returns)
    return spec.name


def load_result(path):
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def run_all(cfg, workers, force):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: cfg.results_dir / f"{s.name}.npz" for s in all_specs(cfg)}
    todo = [s for s, p in paths.items() if force or not p.exists()]
    if todo:
        print(f"training {len(todo)} runs on {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = pool.map(_train_and_save, todo, [params_for(s, cfg) for s in todo], [paths[s] for s in todo])
            for name in jobs:
                print("  done", name)
    return {s: load_result(p) for s, p in paths.items()}


def stack_truncated(arrays):
    n = min(len(a) for a in arrays)
    return np.stack([a[:n] for a in arrays])


def ma_episodes(returns, ma):
    """Episode index (1-based) of each moving-average point."""
    return np.arange(len(returns) - len(ma) + 1, len(returns) + 1)


def first_reaching(curve, level, n_episodes):
    hits = np.flatnonzero(curve >= level)
    return int(n_episodes - len(curve) + hits[0] + 1) if hits.size else None


def early_view(r, warmup_steps):
    """Metrics of one run restricted to its first TASK4_EPISODES episodes."""
    returns = r["returns"][:TASK4_EPISODES]
    ma = dqn.moving_average(returns, WINDOW)
    train_steps = max(int(returns.sum()) - warmup_steps, 1)  # +1 reward per step, so returns sum to steps
    evals = r["eval_returns"][r["eval_episodes"] <= TASK4_EPISODES]
    return {"best_ma": float(ma.max()), "final_ma": float(ma[-1]),
            "max_mean_q": float(r["mean_max_q"][:train_steps].max()),
            "eval": float(evals[-1]) if evals.size else float("nan")}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return [float(values.mean()), float(values.std())]


def over_seeds(results, cfg, variant, metric):
    return [metric(results[RunSpec(variant, seed)]) for seed in cfg.seeds]


def compute_stats(results, cfg):
    ma = lambda r: dqn.moving_average(r["returns"], WINDOW)  # noqa: E731
    stats = {"seeds": list(cfg.seeds), "window": WINDOW, "solved": SOLVED,
             "params": params_for(RunSpec("baseline", 0), cfg), "variants": {}}
    for v in VARIANTS:
        curves = over_seeds(results, cfg, v, ma)
        n_episodes = len(results[RunSpec(v, cfg.seeds[0])]["returns"])
        mean_curve = stack_truncated(curves).mean(axis=0)
        best = np.array([c.max() for c in curves])
        final = np.array([c[-1] for c in curves])
        final_eval = np.array(over_seeds(results, cfg, v, lambda r: r["eval_returns"][-1]))
        warmup = params_for(RunSpec(v, cfg.seeds[0]), cfg)["warmup_steps"]
        early = over_seeds(results, cfg, v, lambda r: early_view(r, warmup))
        stats["variants"][v] = {
            "best_ma": [float(best.mean()), float(best.std())],
            "final_ma": [float(final.mean()), float(final.std())],
            "best_ma_early": mean_std([e["best_ma"] for e in early]),
            "best_ma_early_per_seed": [e["best_ma"] for e in early],
            "final_ma_early": mean_std([e["final_ma"] for e in early]),
            "max_mean_q_early": float(np.mean([e["max_mean_q"] for e in early])),
            "eval_early": mean_std([e["eval"] for e in early]),
            "first_solved_per_seed": [first_reaching(c, SOLVED, n_episodes) for c in curves],
            "first_solved_mean_curve": first_reaching(mean_curve, SOLVED, n_episodes),
            "max_mean_q": float(np.mean(over_seeds(results, cfg, v, lambda r: r["mean_max_q"].max()))),
            "final_mean_q": float(np.mean(over_seeds(results, cfg, v, lambda r: r["mean_max_q"][-500:].mean()))),
            "final_loss": float(np.mean(over_seeds(results, cfg, v, lambda r: r["losses"][-500:].mean()))),
            "final_eval": [float(final_eval.mean()), float(final_eval.std())],
            "steps": float(np.mean(over_seeds(results, cfg, v, lambda r: len(r["epsilons"])))),
        }
    # Task 4 criterion: seed 0 within the first 500 episodes (same trajectory as a 500-episode run).
    seed0 = results[RunSpec("baseline", cfg.seeds[0])]["returns"][:TASK4_EPISODES]
    stats["seed0_best_ma"] = float(dqn.moving_average(seed0, WINDOW).max())
    stats["task4_episodes"] = TASK4_EPISODES
    return stats


def smooth(x, k=SMOOTH):
    return dqn.moving_average(x, min(k, len(x))) if len(x) else x


def _band(ax, x, runs, label, color):
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    ax.plot(x, mean, color=color, lw=1.5, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, lw=0)


def episode_views(runs):
    """Episode limits of the two views: the Task 4 budget and the full run."""
    n_episodes = min(len(r["returns"]) for r in runs)
    return min(TASK4_EPISODES, n_episodes), n_episodes


def _return_panel(ax, runs, limit, label, color, raw=False):
    ma = stack_truncated([dqn.moving_average(r["returns"], WINDOW) for r in runs])
    x = ma_episodes(runs[0]["returns"], ma[0])
    keep = x <= limit
    if raw:
        for r in runs:
            ax.plot(np.arange(1, limit + 1), r["returns"][:limit], color=color, alpha=0.12, lw=0.6)
    _band(ax, x[keep], ma[:, keep], label, color)
    ax.set(xlabel="episode", xlim=(0, limit), ylim=(0, 520), title=f"Episodes 1 to {limit}")


def fig_baseline(results, cfg):
    runs = [results[RunSpec("baseline", s)] for s in cfg.seeds]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, limit in zip(axes[:2], episode_views(runs)):
        _return_panel(ax, runs, limit, f"{WINDOW}-episode average, mean ± std", "tab:blue", raw=True)
        ax.axhline(SOLVED, color="grey", ls="--", lw=0.8, label=f"solved ({SOLVED})")
    axes[0].set_ylabel("training return")
    ev = stack_truncated([r["eval_returns"] for r in runs])
    _band(axes[2], runs[0]["eval_episodes"][: ev.shape[1]], ev, "greedy evaluation, mean ± std", "tab:green")
    axes[2].axhline(SOLVED, color="grey", ls="--", lw=0.8)
    axes[2].set(xlabel="episode", ylabel="evaluation return", ylim=(0, 520), title="Deterministic evaluation")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[2].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_diagnostics(results, cfg):
    r = results[RunSpec("baseline", cfg.seeds[0])]
    p = params_for(RunSpec("baseline", 0), cfg)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    train_x = np.arange(p["warmup_steps"] + 1, p["warmup_steps"] + 1 + len(r["losses"]))
    axes[0].plot(train_x, r["losses"], color="tab:blue", alpha=0.25, lw=0.5)
    axes[0].plot(train_x[len(train_x) - len(smooth(r["losses"])):], smooth(r["losses"]), color="tab:blue", lw=1.2)
    axes[0].set(yscale="log", xlabel="global step", ylabel="TD loss (MSE)", title="Loss (seed 0)")
    axes[1].plot(train_x, r["mean_max_q"], color="tab:purple", lw=0.8)
    axes[1].set(xlabel="global step", ylabel="mean max_a Q(s,a) over batch", title="Q-value scale (seed 0)")
    axes[2].plot(r["epsilons"], color="tab:orange", lw=1.2)
    axes[2].set(xlabel="global step", ylabel="ε", ylim=(0, 1.05), title="Exploration schedule")
    for ax in axes:
        ax.axvline(p["warmup_steps"], color="grey", ls=":", lw=0.8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_ablation(results, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for v in VARIANTS:
        runs = [results[RunSpec(v, s)] for s in cfg.seeds]
        for ax, limit in zip(axes[:2], episode_views(runs)):
            _return_panel(ax, runs, limit, LABELS[v], COLORS[v])
        q = stack_truncated([smooth(r["mean_max_q"]) for r in runs])
        axes[2].plot(q.mean(axis=0), color=COLORS[v], lw=1.2, label=LABELS[v])
    for ax in axes[:2]:
        ax.axhline(SOLVED, color="grey", ls="--", lw=0.8)
    axes[0].set_ylabel(f"{WINDOW}-episode average return, mean ± std")
    axes[2].set_yscale("symlog", linthresh=10)
    axes[2].xaxis.set_major_locator(plt.MaxNLocator(5))
    axes[2].set(xlabel="training step", ylabel="mean max_a Q(s,a), mean over seeds",
                title="Q-value scale (log axis)")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def make_figures(results, cfg):
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    for name, build in {"baseline": fig_baseline, "diagnostics": fig_diagnostics, "ablation": fig_ablation}.items():
        fig = build(results, cfg)
        fig.savefig(cfg.figures_dir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    parser.add_argument("--force", action="store_true", help="retrain even if cached")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cfg = QUICK if args.quick else FULL
    results = run_all(cfg, args.workers, args.force)
    stats = compute_stats(results, cfg)
    make_figures(results, cfg)
    (cfg.results_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"seed 0 best {WINDOW}-episode moving average: {stats['seed0_best_ma']:.1f}")
    print(f"figures in {cfg.figures_dir}, stats in {cfg.results_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
