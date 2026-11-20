"""Run the Lab 1 experiments: trains all configs, saves results/, figures/ and results/stats.json."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qlearning as ql

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ALGOS = ("q_learning", "sarsa")
LABELS = {"q_learning": "Q-learning", "sarsa": "SARSA"}
COLORS = {"q_learning": "tab:blue", "sarsa": "tab:orange"}
ALPHAS = (0.01, 0.05, 0.1, 0.3, 0.7)


@dataclass(frozen=True)
class Config:
    seeds: tuple
    n_episodes: int
    snapshot_every: int
    window: int
    last_n: int
    out_dir: Path

    @property
    def results_dir(self):
        return self.out_dir / "results"

    @property
    def figures_dir(self):
        return self.out_dir / "figures"


FULL = Config(tuple(range(5)), ql.DEFAULTS["n_episodes"], ql.DEFAULTS["snapshot_every"], 500, 1000, HERE)
QUICK = Config((0, 1), 500, 100, 50, 100, HERE / "quick_run")


@dataclass(frozen=True)
class RunSpec:
    algo: str
    alpha: float
    seed: int
    n_episodes: int
    snapshot_every: int

    @property
    def name(self):
        return f"{self.algo}_a{self.alpha:g}_s{self.seed}_n{self.n_episodes}"

    def train(self):
        params = {**ql.DEFAULTS, "alpha": self.alpha, "n_episodes": self.n_episodes,
                  "snapshot_every": self.snapshot_every}
        return ql.train(ql.make_env(), self.algo, seed=self.seed, **params)


def spec(cfg, algo, alpha, seed):
    return RunSpec(algo, alpha, seed, cfg.n_episodes, cfg.snapshot_every)


def all_specs(cfg):
    return [spec(cfg, algo, alpha, seed) for algo in ALGOS for alpha in ALPHAS for seed in cfg.seeds]


def save_result(result, path):
    episodes = np.array(sorted(result.snapshots))
    np.savez_compressed(
        path, Q=result.Q, returns=result.returns, lengths=result.lengths,
        snapshot_episodes=episodes, snapshot_Q=np.stack([result.snapshots[e] for e in episodes]),
    )


def load_result(path):
    with np.load(path) as d:
        snapshots = {int(e): q for e, q in zip(d["snapshot_episodes"], d["snapshot_Q"])}
        return ql.TrainResult(d["Q"], d["returns"], d["lengths"], snapshots)


def _train_and_save(spec, path):
    save_result(spec.train(), path)
    return spec.name


def run_all(cfg, workers, force):
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: cfg.results_dir / f"{s.name}.npz" for s in all_specs(cfg)}
    todo = [s for s, p in paths.items() if force or not p.exists()]
    if todo:
        print(f"training {len(todo)} runs on {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name in pool.map(_train_and_save, todo, [paths[s] for s in todo]):
                print("  done", name)
    return {s: load_result(p) for s, p in paths.items()}


def moving_avg(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def over_seeds(results, cfg, algo, alpha, metric):
    return np.stack([metric(results[spec(cfg, algo, alpha, seed)]) for seed in cfg.seeds])


def optimal_success(env, horizon=200):
    """Best possible P(reach goal within `horizon` steps), by backward induction on env.unwrapped.P."""
    P = env.unwrapped.P
    V = np.zeros(env.observation_space.n)
    for _ in range(horizon):
        V = np.array([max(sum(p * (r + (0 if done else V[s2])) for p, s2, r, done in P[s][a])
                          for a in P[s]) for s in P])
    return float(V[0])


def compute_stats(results, cfg):
    alpha0 = ql.DEFAULTS["alpha"]
    env = ql.make_env()
    greedy = {a: over_seeds(results, cfg, a, alpha0, lambda r: ql.evaluate(env, r.Q, 1000)) for a in ALGOS}
    final = lambda r: ql.success_rate(r.returns, cfg.last_n)
    sweep = {algo: {alpha: over_seeds(results, cfg, algo, alpha, final) for alpha in ALPHAS}
             for algo in ALGOS}

    def first_half(algo):
        curve = over_seeds(results, cfg, algo, alpha0, lambda r: moving_avg(r.returns > 0, cfg.window))
        hits = np.flatnonzero(curve.mean(axis=0) >= 0.5)
        return int(hits[0] + cfg.window) if hits.size else None

    return {
        "n_episodes": cfg.n_episodes,
        "seeds": list(cfg.seeds),
        "window": cfg.window,
        "last_n": cfg.last_n,
        "defaults": ql.DEFAULTS,
        "seed0_q_learning": final(results[spec(cfg, "q_learning", alpha0, cfg.seeds[0])]),
        "optimal": optimal_success(env),
        "greedy": {a: [float(greedy[a].mean()), float(greedy[a].std())] for a in ALGOS},
        "final": {a: [float(sweep[a][alpha0].mean()), float(sweep[a][alpha0].std())] for a in ALGOS},
        "sweep": {a: {str(al): [float(v.mean()), float(v.std())] for al, v in sweep[a].items()}
                  for a in ALGOS},
        "best_alpha": {a: max(ALPHAS, key=lambda al: sweep[a][al].mean()) for a in ALGOS},
        "half_episode": {a: first_half(a) for a in ALGOS},
        "mean_length": {a: float(over_seeds(results, cfg, a, alpha0,
                                            lambda r: r.lengths[-cfg.last_n:].mean()).mean())
                        for a in ALGOS},
    }


def _band(ax, x, runs, label, color):
    mean, std = runs.mean(axis=0), runs.std(axis=0)
    ax.plot(x, mean, label=label, color=color, lw=1.5)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, lw=0)


def fig_learning_curves(results, cfg):
    alpha = ql.DEFAULTS["alpha"]
    x = np.arange(cfg.window, cfg.n_episodes + 1)
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(10, 3.8))
    for algo in ALGOS:
        succ = over_seeds(results, cfg, algo, alpha, lambda r: moving_avg(r.returns > 0, cfg.window))
        lens = over_seeds(results, cfg, algo, alpha, lambda r: moving_avg(r.lengths, cfg.window))
        _band(ax_s, x, succ, LABELS[algo], COLORS[algo])
        _band(ax_l, x, lens, LABELS[algo], COLORS[algo])
    ax_s.set(xlabel="episode", ylabel=f"success rate (window {cfg.window})", ylim=(0, 1),
             title=f"Success rate, mean ± std over {len(cfg.seeds)} seeds")
    ax_l.set(xlabel="episode", ylabel="episode length", title="Episode length")
    for ax in (ax_s, ax_l):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def fig_alpha_sweep(results, cfg):
    x = np.arange(cfg.window, cfg.n_episodes + 1)
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.3))
    for ax, algo in zip(axes[:2], ALGOS):
        for i, alpha in enumerate(ALPHAS):
            succ = over_seeds(results, cfg, algo, alpha, lambda r: moving_avg(r.returns > 0, cfg.window))
            ax.plot(x, succ.mean(axis=0), color=cmap(i / (len(ALPHAS) - 1)), lw=1.4, label=f"α={alpha:g}")
        ax.set(xlabel="episode", ylabel=f"success rate (window {cfg.window})", ylim=(0, 1),
               title=f"{LABELS[algo]}: mean over {len(cfg.seeds)} seeds")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    for algo in ALGOS:
        finals = np.stack([over_seeds(results, cfg, algo, a, lambda r: ql.success_rate(r.returns, cfg.last_n))
                           for a in ALPHAS])
        ax.errorbar(ALPHAS, finals.mean(axis=1), yerr=finals.std(axis=1), marker="o", capsize=3,
                    color=COLORS[algo], label=LABELS[algo])
    ax.set(xscale="log", xlabel="learning rate α", ylabel=f"success rate, last {cfg.last_n} episodes",
           ylim=(0, 1), title="Final performance vs α")
    ax.set_xticks(ALPHAS, [f"{a:g}" for a in ALPHAS])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _draw_tiles(ax, desc):
    for (row, col), tile in np.ndenumerate(desc):
        if tile == "H":
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, color="black", alpha=0.75))
        if tile in "SG":
            ax.text(col, row, tile, ha="center", va="center", color="red", fontsize=11, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_policy(ax, Q, desc, title):
    n = desc.shape[0]
    ax.imshow(Q.max(axis=1).reshape(n, n), cmap="Blues", vmin=0, vmax=1)
    rows, cols, d_row, d_col = [], [], [], []
    for (row, col), tile in np.ndenumerate(desc):
        if tile in "HG":
            continue
        dr, dc = ql.ACTION_DELTAS[int(np.argmax(Q[row * n + col]))]
        rows.append(row), cols.append(col), d_row.append(dr), d_col.append(dc)
    ax.quiver(cols, rows, d_col, d_row, angles="xy", scale_units="xy", scale=2.2,
              pivot="middle", width=0.012, color="darkred")
    _draw_tiles(ax, desc)
    ax.set_title(title, fontsize=10)


def fig_value_policy(results, cfg):
    alpha, seed = ql.DEFAULTS["alpha"], cfg.seeds[0]
    desc = ql.make_env().unwrapped.desc.astype(str)
    n = desc.shape[0]
    q_run = results[spec(cfg, "q_learning", alpha, seed)]
    s_run = results[spec(cfg, "sarsa", alpha, seed)]
    episodes = sorted(q_run.snapshots)
    picks = [min(episodes, key=lambda ep: abs(ep - f * cfg.n_episodes)) for f in (0.03, 0.1, 0.33, 1.0)]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    for ax, ep in zip(axes[0], picks):
        im = ax.imshow(q_run.snapshots[ep].max(axis=1).reshape(n, n), cmap="Blues", vmin=0, vmax=1)
        _draw_tiles(ax, desc)
        ax.set_title(f"Q-learning V(s), episode {ep}", fontsize=8.5)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.02, pad=0.02)
    _draw_policy(axes[1][0], q_run.Q, desc, f"Q-learning policy, seed {seed}")
    _draw_policy(axes[1][1], s_run.Q, desc, f"SARSA policy, seed {seed}")
    for ax in axes[1][2:]:
        ax.axis("off")
    axes[1][2].text(0, 0.9, "black = hole, S = start, G = goal\narrows = argmax_a Q(s,a)\ncolour = V(s)",
                    fontsize=9, va="top", transform=axes[1][2].transAxes)
    return fig


def label_panels(fig):
    """Prefix each titled panel with (a), (b), ... so captions can refer to them."""
    titled = [ax for ax in fig.axes if ax.get_title()]
    for i, ax in enumerate(titled):
        ax.set_title(f"({chr(97 + i)}) {ax.get_title()}", fontsize=ax.title.get_fontsize())


def make_figures(results, cfg):
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    for name, build in {"learning_curves": fig_learning_curves, "alpha_sweep": fig_alpha_sweep,
                        "value_policy": fig_value_policy}.items():
        fig = build(results, cfg)
        label_panels(fig)
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
    print(f"seed 0 success rate: {stats['seed0_q_learning']:.3f}")
    print(f"figures in {cfg.figures_dir}, stats in {cfg.results_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
