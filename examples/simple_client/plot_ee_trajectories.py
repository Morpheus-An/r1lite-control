"""Compare right-gripper EE trajectories across policies from closed-loop eval logs.

Reads the jsonl files written by `main_r1lite_closedloop.py` (one file per
policy, episodes separated by `episode_start` / `episode_end` records), and
renders every episode of every policy into one 6-panel figure: 3D view, three
orthographic projections, mean x/y/z profiles, and endpoint spread.

Usage:
    python3 plot_ee_trajectories.py A=./eval_policyA.jsonl B=./eval_policyB.jsonl
    python3 plot_ee_trajectories.py ./eval_policyA.jsonl ./eval_policyB.jsonl -o cmp.png
    python3 plot_ee_trajectories.py A=a.jsonl B=b.jsonl --label 杯把 --min-points 10

Legacy logs written before episode tracking existed are still readable — those
get split on step-counter resets and named `<file stem>`.

Colors are Okabe-Ito (colour-vision-deficiency safe); policy identity is encoded
by hue, x/y/z by both hue and line style.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
import numpy as np


# Okabe-Ito: distinguishable under all common forms of colour blindness.
POLICY_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
AXIS_STYLE = {"x": ("#0072B2", "-"), "y": ("#E69F00", "--"), "z": ("#009E73", "-.")}
CJK_FONTS = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Source Han Sans CN",
             "WenQuanYi Micro Hei", "Droid Sans Fallback"]


@dataclasses.dataclass
class Episode:
    policy: str
    index: int
    label: str | None
    points: np.ndarray      # (N, 3) right-EE xyz in base_link
    times: np.ndarray       # (N,) seconds since the episode's first sample
    dry_run: bool
    timing: dict            # the episode_end footer, {} for logs without one
    grip: np.ndarray        # (N,) measured right-gripper opening, state[13]; empty if absent
    grip_cmd: np.ndarray    # (N,) commanded right-gripper opening, action[13]; empty if absent

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    @property
    def path_length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def endpoint(self, n: int = 5) -> np.ndarray:
        """Mean of the last n samples — steadier than the single final point."""
        return self.points[-min(n, len(self.points)):].mean(axis=0)


# ─── loading ────────────────────────────────────────────────────────────────


def _parse_ts(rec: dict) -> float | None:
    try:
        return datetime.datetime.fromisoformat(rec["timestamp"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return None


def load_episodes(path: pathlib.Path, policy_override: str | None,
                  include_dry_run: bool, min_points: int) -> list[Episode]:
    """Parse one eval log into episodes. Junk/plain-text lines are skipped."""
    if not path.exists():
        sys.exit(f"error: no such log file: {path}")

    labels: dict[tuple[str, int], str | None] = {}
    timings: dict[tuple[str, int], dict] = {}
    groups: dict[tuple[str, int], list[dict]] = {}
    order: list[tuple[str, int]] = []
    legacy: list[dict] = []
    n_bad = 0

    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line.startswith("{"):
                n_bad += line != ""
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue

            policy = policy_override or rec.get("policy") or path.stem
            idx = rec.get("episode_index")
            if rec.get("type") == "episode_start" and isinstance(idx, int):
                labels[(policy, idx)] = rec.get("episode_label")
                continue
            if rec.get("type") == "episode_end":
                if isinstance(idx, int):
                    timings[(policy, idx)] = rec
                continue
            if "r_ee_xyz" not in rec:                       # not a step record
                continue

            if isinstance(idx, int):
                key = (policy, idx)
                if key not in groups:
                    groups[key] = []
                    order.append(key)
                groups[key].append(rec)
            else:
                legacy.append(rec)                          # pre-episode-tracking log

    # Legacy fallback: a step counter that goes backwards means a new run started.
    if legacy:
        policy = policy_override or path.stem
        base = max((i for p, i in order if p == policy), default=-1) + 1
        prev, key = None, None
        for rec in legacy:
            step = rec.get("step")
            if prev is None or (isinstance(step, int) and step < prev):
                key = (policy, base)
                groups[key] = []
                order.append(key)
                base += 1
            prev = step if isinstance(step, int) else prev
            groups[key].append(rec)

    episodes = []
    for key in order:
        recs = groups[key]
        if not include_dry_run and any(r.get("dry_run") for r in recs):
            continue
        usable = [r for r in recs if r.get("r_ee_xyz") is not None]
        pts = [r["r_ee_xyz"] for r in usable]
        ts = [_parse_ts(r) for r in usable]
        if len(pts) < min_points:
            continue

        def grip_of(field: str) -> np.ndarray:
            """Right-gripper channel is the last of the 14 dims (see expand_qpos_to_14_dims)."""
            vals = [r[field][13] for r in usable
                    if isinstance(r.get(field), list) and len(r[field]) == 14]
            return np.asarray(vals, dtype=float) if len(vals) == len(pts) else np.empty(0)
        t0 = next((t for t in ts if t is not None), 0.0)
        times = np.array([(t - t0) if t is not None else np.nan for t in ts], dtype=float)
        if np.isnan(times).any():                           # fall back to sample index
            times = np.arange(len(pts), dtype=float)
        episodes.append(Episode(
            policy=key[0], index=key[1], label=labels.get(key),
            points=np.asarray(pts, dtype=float), times=times,
            dry_run=bool(recs[0].get("dry_run")), timing=timings.get(key, {}),
            grip=grip_of("state"), grip_cmd=grip_of("action"),
        ))

    if n_bad:
        print(f"  ({path.name}: skipped {n_bad} non-JSON line(s))")
    return episodes


def timing_report(eps_by_policy: dict[str, list[Episode]]) -> None:
    """Per-policy timing, averaged over the episode_end footers in the same log."""
    rows = []
    for policy, eps in eps_by_policy.items():
        t = [ep.timing for ep in eps if ep.timing]
        if not t:
            continue

        def avg(field: str) -> float | None:
            vals = [r[field] for r in t if isinstance(r.get(field), (int, float))]
            return float(np.mean(vals)) if vals else None

        rows.append((policy, len(t), avg("wall_s_task"), avg("infer_s_total"),
                     avg("exec_s_total"), avg("infer_ms_mean"), avg("infer_ms_p95"),
                     avg("exec_ms_mean"), avg("infer_pct_of_task"), avg("n_overruns")))
    if not rows:
        print("\n  (no episode_end records — timing table needs logs written by the "
              "current main_r1lite_closedloop.py)")
        return

    def f(v, spec=".2f"):
        return "—".rjust(7) if v is None else format(v, spec).rjust(7)

    print("\n── Timing per policy (mean over episodes) ──────")
    print("  policy        n   task_s  infer_s   exec_s  infer_ms  infer_p95  exec_ms  infer%  overrun")
    for p, n, task, inf_s, ex_s, inf_ms, inf_p95, ex_ms, pct, over in rows:
        print(f"  {p:<10} {n:>4}  {f(task)}  {f(inf_s)}  {f(ex_s)}  "
              f"{f(inf_ms, '.1f')}  {f(inf_p95, '.1f')}    {f(ex_ms, '.1f')}  "
              f"{f(pct, '.1f')}  {f(over, '.1f')}")
    print("  task_s = control-loop wall time (excl. init/warmup) · infer_s/exec_s = Σ per episode")


def mean_trajectory(episodes: list[Episode], n: int = 60) -> np.ndarray:
    """Average of all episodes resampled onto normalised time — the policy's typical path."""
    grid = np.linspace(0.0, 1.0, n)
    stack = []
    for ep in episodes:
        span = ep.times[-1] - ep.times[0]
        u = (ep.times - ep.times[0]) / span if span > 0 else np.linspace(0, 1, len(ep.times))
        stack.append(np.column_stack([np.interp(grid, u, ep.points[:, k]) for k in range(3)]))
    return np.mean(stack, axis=0)


# ─── plotting ───────────────────────────────────────────────────────────────


def pick_cjk_font(texts: list[str]) -> None:
    """Only touch rcParams if some label actually needs CJK glyphs."""
    if not any(any(ord(ch) > 0x2E80 for ch in t) for t in texts if t):
        return
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return
    print("  (warning: no CJK font found — Chinese labels may render as boxes)")


def episode_alpha(n: int) -> float:
    """Keep the mean line readable whether there are 2 episodes or 70."""
    return float(np.clip(0.85 / np.sqrt(max(n, 1)), 0.07, 0.35))


def _proj(ax, eps_by_policy, colors, ia, ib, xlabel, ylabel, title, solo=None):
    if solo is not None:
        # One episode: hue is free, so spend it on time — light → dark = start → end.
        ax.plot(solo.points[:, ia], solo.points[:, ib], color="0.75", lw=1.0, zorder=2)
        ax.scatter(solo.points[:, ia], solo.points[:, ib], c=solo.times, cmap="viridis",
                   s=14, zorder=3)
        ax.plot(solo.points[0, ia], solo.points[0, ib], "o", mfc="none", mec="0.2", mew=2,
                ms=10, zorder=4)
        ax.plot(solo.points[-1, ia], solo.points[-1, ib], "*", color="0.2", ms=17, zorder=4)
    for policy, eps in (eps_by_policy.items() if solo is None else []):
        c = colors[policy]
        a = episode_alpha(len(eps))
        for ep in eps:
            ax.plot(ep.points[:, ia], ep.points[:, ib], color=c, lw=1.0, alpha=a, zorder=2)
        m = mean_trajectory(eps)
        ax.plot(m[:, ia], m[:, ib], color=c, lw=2.6, alpha=0.95, zorder=3)
        ax.plot(m[0, ia], m[0, ib], "o", mfc="none", mec=c, mew=2, ms=9, zorder=4)
        ax.plot(m[-1, ia], m[-1, ib], "*", color=c, ms=16, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")


def series_colors(names: list[str]) -> dict[str, str]:
    """Okabe-Ito while it lasts; beyond that fall back to an ordered colormap."""
    if len(names) <= len(POLICY_COLORS):
        return {n: POLICY_COLORS[i] for i, n in enumerate(names)}
    cmap = plt.get_cmap("viridis")
    return {n: cmap(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}


def make_figure(eps_by_policy: dict[str, list[Episode]], out_path: pathlib.Path,
                title: str) -> None:
    colors = series_colors(list(eps_by_policy))
    pick_cjk_font([title] + [ep.label for eps in eps_by_policy.values() for ep in eps]
                  + list(eps_by_policy))

    all_eps = [ep for eps in eps_by_policy.values() for ep in eps]
    solo = all_eps[0] if len(all_eps) == 1 else None    # single episode → time-coloured mode
    # One episode per series (colour encodes episode): per-series means/spreads are
    # degenerate, so those panels switch to whole-set statistics instead.
    per_ep = len(all_eps) > 1 and len(all_eps) == len(eps_by_policy)
    n_series = len(eps_by_policy)

    def series_alpha(i: int) -> float:
        return 1.0 - 0.55 * i / max(n_series - 1, 1)

    fig = plt.figure(figsize=(17, 9.5))
    fig.suptitle(title, fontsize=15, y=0.975, x=0.012, ha="left", fontweight="bold")

    # 1. 3D view
    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    if solo is not None:
        ax3d.plot(solo.points[:, 0], solo.points[:, 1], solo.points[:, 2], color="0.75", lw=1.0)
        sc = ax3d.scatter(solo.points[:, 0], solo.points[:, 1], solo.points[:, 2],
                          c=solo.times, cmap="viridis", s=12, depthshade=False)
        ax3d.scatter(*solo.points[0], marker="o", s=60, facecolors="none",
                     edgecolors="0.2", linewidths=2, depthshade=False)
        ax3d.scatter(*solo.points[-1], marker="*", s=220, color="0.2", depthshade=False)
        cb = fig.colorbar(sc, ax=ax3d, pad=0.12, shrink=0.7)
        cb.set_label("time (s)", fontsize=9)
    for policy, eps in (eps_by_policy.items() if solo is None else []):
        c = colors[policy]
        a = episode_alpha(len(eps))
        for ep in eps:
            ax3d.plot(ep.points[:, 0], ep.points[:, 1], ep.points[:, 2],
                      color=c, lw=0.9, alpha=a)
        m = mean_trajectory(eps)
        label = (f"{policy}  (n={len(eps)})" if len(eps) > 1
                 else f"{policy}  ({eps[0].duration:.0f} s)")
        ax3d.plot(m[:, 0], m[:, 1], m[:, 2], color=c, lw=2.6, label=label)
        ax3d.scatter(*m[0], color=c, marker="o", s=55, depthshade=False,
                     facecolors="none", edgecolors=c, linewidths=2)
        ax3d.scatter(*m[-1], color=c, marker="*", s=210, depthshade=False)
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.set_title("3D trajectory — ○ start, ★ end", fontsize=11, loc="left")
    crowded = n_series > 8
    if solo is None:
        ax3d.legend(loc="upper left", fontsize=7 if crowded else 9,
                    ncol=2 if crowded else 1, framealpha=0.85,
                    borderpad=0.3, labelspacing=0.25, columnspacing=0.9)

    # 2-4. Orthographic projections
    _proj(fig.add_subplot(2, 3, 2), eps_by_policy, colors, 0, 1,
          "x (m)  forward →", "y (m)  ← right", "Top view (X–Y)", solo)
    _proj(fig.add_subplot(2, 3, 3), eps_by_policy, colors, 0, 2,
          "x (m)  forward →", "z (m)  up", "Side view (X–Z)", solo)
    _proj(fig.add_subplot(2, 3, 4), eps_by_policy, colors, 1, 2,
          "y (m)  ← right", "z (m)  up", "Front view (Y–Z)", solo)

    # 5. x/y/z vs time — hue AND dash encode the axis.
    ax = fig.add_subplot(2, 3, 5)
    if solo is not None:
        for k, name in enumerate("xyz"):
            c, ls = AXIS_STYLE[name]
            ax.plot(solo.times, solo.points[:, k], color=c, ls=ls, lw=2.0)
        ax.set_xlabel("time (s)")
        ax.set_title("x / y / z vs time", fontsize=11, loc="left")
    else:
        grid = np.linspace(0, 100, 60)
        for i, (policy, eps) in enumerate(eps_by_policy.items()):
            m = mean_trajectory(eps)
            for k, name in enumerate("xyz"):
                c, ls = AXIS_STYLE[name]
                ax.plot(grid, m[:, k], color=c, ls=ls, lw=2.0, alpha=series_alpha(i))
        ax.set_xlabel("episode progress (%)")
        ax.set_title("x / y / z profile" if per_ep else "Mean x / y / z profile",
                     fontsize=11, loc="left")
    ax.set_ylabel("position (m)")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(handles=[Line2D([], [], color=AXIS_STYLE[n][0], ls=AXIS_STYLE[n][1],
                              lw=2, label=n) for n in "xyz"]
                     + ([] if solo is not None or crowded else
                        [Line2D([], [], color="0.35", lw=2, alpha=series_alpha(i), label=p)
                         for i, p in enumerate(eps_by_policy)]),
              fontsize=9, ncol=2, loc="best", framealpha=0.9)

    # 6. Single episode: EE speed + gripper. Several: endpoint spread.
    if solo is not None:
        ax = fig.add_subplot(2, 3, 6)
        dt = np.diff(solo.times)
        speed = np.linalg.norm(np.diff(solo.points, axis=0), axis=1) * 100 / np.where(dt > 0, dt, np.nan)
        ax.plot(solo.times[1:], speed, color="#0072B2", lw=1.5, label="EE speed")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("EE speed (cm/s)", color="#0072B2")
        ax.tick_params(axis="y", labelcolor="#0072B2")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_title("EE speed & right gripper", fontsize=11, loc="left")
        handles = [Line2D([], [], color="#0072B2", lw=2, label="EE speed")]
        if solo.grip.size:
            axg = ax.twinx()
            axg.plot(solo.times, solo.grip, color="#E69F00", lw=2.0)
            if solo.grip_cmd.size:
                axg.plot(solo.times, solo.grip_cmd, color="#E69F00", lw=1.2, ls=":")
                handles.append(Line2D([], [], color="#E69F00", lw=1.2, ls=":", label="gripper cmd"))
            axg.set_ylabel("gripper opening (state[13])", color="#E69F00")
            axg.tick_params(axis="y", labelcolor="#E69F00")
            handles.insert(1, Line2D([], [], color="#E69F00", lw=2, label="gripper state"))
        ax.legend(handles=handles, fontsize=9, loc="best", framealpha=0.9)

        e = solo.endpoint()
        footer = (f"{solo.policy} · episode {solo.index}"
                  + (f" · {solo.label}" if solo.label else "")
                  + f" · {len(solo.points)} pts / {solo.duration:.1f} s"
                  f" · path {solo.path_length * 100:.1f} cm"
                  f" · start=({solo.points[0][0]:.3f}, {solo.points[0][1]:.3f}, {solo.points[0][2]:.3f})"
                  f" → end=({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}) m")
        fig.text(0.012, 0.012, footer, fontsize=10, color="0.25", va="bottom")
        fig.tight_layout(rect=[0, 0.05, 1, 0.955])
        fig.savefig(out_path, dpi=150, facecolor="white")
        print(f"\nwrote {out_path}")
        return

    ax = fig.add_subplot(2, 3, 6)
    lines = []
    for policy, eps in eps_by_policy.items():
        c = colors[policy]
        ends = np.array([ep.endpoint() for ep in eps])
        ax.scatter(ends[:, 0], ends[:, 1], color=c, s=70 if per_ep else 42, alpha=0.85,
                   edgecolors="white", linewidths=0.6, label=policy, zorder=3)
        cen = ends.mean(axis=0)
        if not per_ep:
            ax.plot(*cen[:2], "+", color=c, ms=18, mew=2.5, zorder=4)
        spread = float(np.linalg.norm(ends - cen, axis=1).mean())
        lines.append((policy, cen, spread, ends))
    ax.set_xlabel("x (m)  forward →")
    ax.set_ylabel("y (m)  ← right")
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9, framealpha=0.9)

    if per_ep:
        # Every series holds one episode: report spread over the whole set instead.
        ends = np.array([ep.endpoint() for ep in all_eps])
        cen = ends.mean(axis=0)
        ax.plot(*cen[:2], "+", color="0.2", ms=18, mew=2.5, zorder=4)
        d = np.linalg.norm(ends - cen, axis=1)
        pair = max(float(np.linalg.norm(a - b)) for a in ends for b in ends)
        ax.set_title("Endpoint spread — + is the mean of all", fontsize=11, loc="left")
        footer = (f"{len(all_eps)} episodes · mean endpoint = "
                  f"({cen[0]:.3f}, {cen[1]:.3f}, {cen[2]:.3f}) m · "
                  f"spread = {d.mean() * 100:.1f} cm (max {d.max() * 100:.1f} cm, "
                  f"widest pair {pair * 100:.1f} cm)")
        if not crowded:      # three per line, else the text runs off the canvas
            parts = [f"{p}: ({e[0][0]:.3f}, {e[0][1]:.3f}, {e[0][2]:.3f}) m, "
                     f"{ep.duration:.0f} s, path {ep.path_length * 100:.0f} cm"
                     for (p, _, _, e), ep in zip(lines, all_eps)]
            footer += "".join("\n" + "   |   ".join(parts[i:i + 3])
                              for i in range(0, len(parts), 3))
        else:
            footer += "   ·   per-episode numbers: see the console table"
        fig.text(0.012, 0.012, footer, fontsize=9.5, color="0.25", va="bottom")
        fig.tight_layout(rect=[0, 0.035 + 0.022 * footer.count("\n"), 1, 0.955])
        fig.savefig(out_path, dpi=150, facecolor="white")
        print(f"\nwrote {out_path}")
        return

    ax.set_title("Endpoint spread — + is the mean", fontsize=11, loc="left")
    footer = "   |   ".join(
        f"{p}: end=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) m, spread={s * 100:.1f} cm"
        for p, c, s, _ in lines)
    if len(lines) == 2:
        d = lines[0][1] - lines[1][1]
        footer += (f"\nΔ({lines[0][0]} → {lines[1][0]}) = "
                   f"({-d[0] * 100:+.1f}, {-d[1] * 100:+.1f}, {-d[2] * 100:+.1f}) cm   |   "
                   f"distance = {np.linalg.norm(d) * 100:.1f} cm")
    fig.text(0.012, 0.012, footer, fontsize=10, color="0.25", va="bottom")

    fig.tight_layout(rect=[0, 0.05, 1, 0.955])
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"\nwrote {out_path}")


# ─── cli ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot right-EE trajectories from closed-loop eval logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example: %(prog)s A=eval_policyA.jsonl B=eval_policyB.jsonl -o cmp.png")
    ap.add_argument("logs", nargs="+", metavar="[NAME=]LOG.jsonl",
                    help="one eval log per policy; NAME= overrides the label shown")
    ap.add_argument("-o", "--out", type=pathlib.Path,
                    default=pathlib.Path("./ee_trajectories.png"))
    ap.add_argument("--title", default="Right-gripper EE trajectories — policy comparison")
    ap.add_argument("--min-points", type=int, default=5,
                    help="drop episodes with fewer samples (aborted runs); default 5")
    ap.add_argument("--include-dry-run", action="store_true",
                    help="keep --dry-run episodes (excluded by default)")
    ap.add_argument("--label", default=None,
                    help="only keep episodes whose --episode-label contains this text")
    ap.add_argument("--episodes", default=None,
                    help="comma-separated episode indices to keep, e.g. 0,3,4")
    ap.add_argument("--color-by", choices=["auto", "policy", "episode"], default="auto",
                    help="what hue encodes; auto = per episode for a single policy, "
                         "per policy otherwise")
    args = ap.parse_args()

    keep = ({int(s) for s in args.episodes.split(",") if s.strip() != ""}
            if args.episodes else None)

    eps_by_policy: dict[str, list[Episode]] = {}
    for spec in args.logs:
        name, _, raw = spec.rpartition("=")
        path = pathlib.Path(raw)
        for ep in load_episodes(path, name or None, args.include_dry_run, args.min_points):
            if keep is not None and ep.index not in keep:
                continue
            if args.label and args.label not in (ep.label or ""):
                continue
            eps_by_policy.setdefault(ep.policy, []).append(ep)

    if not eps_by_policy:
        sys.exit("error: no episodes matched — check --min-points / --label / --include-dry-run")

    print("\n── Episodes loaded ─────────────────────────────")
    for policy, eps in eps_by_policy.items():
        ends = np.array([ep.endpoint() for ep in eps])
        cen = ends.mean(axis=0)
        print(f"  {policy}: {len(eps)} episode(s)  "
              f"mean {np.mean([e.duration for e in eps]):.1f}s / "
              f"{np.mean([len(e.points) for e in eps]):.0f} pts / "
              f"path {np.mean([e.path_length for e in eps]) * 100:.1f} cm")
        print(f"      endpoint mean=({cen[0]:.3f}, {cen[1]:.3f}, {cen[2]:.3f}) m  "
              f"spread={np.linalg.norm(ends - cen, axis=1).mean() * 100:.1f} cm")

    timing_report(eps_by_policy)

    per_episode = (args.color_by == "episode"
                   or (args.color_by == "auto" and len(eps_by_policy) == 1))
    if per_episode and len(all_eps := [e for v in eps_by_policy.values() for e in v]) > 1:
        print("\n── Per episode ─────────────────────────────────")
        print("   ep   pts   dur_s  rate_Hz  path_cm  grip_min          endpoint (m)")
        for ep in all_eps:
            rate = len(ep.points) / ep.duration if ep.duration > 0 else float("nan")
            e = ep.endpoint()
            grip = f"{ep.grip.min():8.1f}" if ep.grip.size else "       —"
            print(f"  {ep.index:>3} {len(ep.points):>5} {ep.duration:>7.1f} {rate:>8.2f} "
                  f"{ep.path_length * 100:>8.1f} {grip}   "
                  f"({e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f})")
        print("  grip_min = smallest measured right-gripper opening; ~99 means it never closed")

    # With a single policy the hue channel is free — spend it on episode identity.
    if per_episode:
        multi = len(eps_by_policy) > 1
        eps_by_policy = {
            (f"{ep.policy} ep{ep.index}" if multi else f"ep {ep.index}"): [ep]
            for eps in eps_by_policy.values() for ep in eps
        }

    make_figure(eps_by_policy, args.out, args.title)


if __name__ == "__main__":
    main()
