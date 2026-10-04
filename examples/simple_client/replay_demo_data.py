"""Replay an expert LeRobot v2.1 trajectory directly on the R1 Lite robot.

Diagnostic tool. If direct replay is also jittery, the dataset itself is
the source of the jitter and the trained policy is faithfully reproducing
it. If replay is smooth, the issue lives in the training/inference/deploy
pipeline (preprocessing, normalization, control frequency, etc.).

The script reads `action.left_arm`, `action.left_gripper`, `action.right_arm`,
`action.right_gripper` from one episode's parquet, assembles them into the
14-D layout `env.control` expects (`[L1..L6, L_grip, R1..R6, R_grip]`),
optionally warms up by interpolating from the robot's current pose to the
first action, then sends one command per frame at the requested rate.

Usage:
    # Dry run first to inspect what would be sent (no robot motion).
    python replay_lerobot.py \\
        --src /home/ntu/Desktop/ant/lpf/r1lite2lerobot/data/pick_cup_data_cleaned/pick_cup/pick_cup \\
        --episode-index 0 --dry-run

    # Real replay of episode 0 at the dataset's recording rate (15 Hz).
    python replay_lerobot.py --episode-index 0

    # Same trajectory at the deploy-side rate, to test the frequency hypothesis.
    python replay_lerobot.py --episode-index 0 --fps 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
    _BACKEND = "pyarrow"
except ImportError:  # fall back to polars (already used by main_r1lite.py)
    import polars as pl
    _BACKEND = "polars"


def _read_parquet_columns(parquet_path: Path, columns: list[str]) -> dict[str, list]:
    """Read selected columns from a parquet file. Returns {col: python list}."""
    if _BACKEND == "pyarrow":
        table = pq.read_table(str(parquet_path), columns=columns)
        return {c: table.column(c).to_pylist() for c in columns}
    df = pl.read_parquet(str(parquet_path), columns=columns)
    return {c: df[c].to_list() for c in columns}


# ─── data loading ───────────────────────────────────────────────────────────


def load_episode_actions(src: Path, ep_idx: int) -> tuple[np.ndarray, int]:
    """Read one episode's parquet and assemble a (T, 14) action array.

    Layout matches `RobotEnv.control(action)`:
        action[0:6]  = left arm joints
        action[6]    = left gripper
        action[7:13] = right arm joints
        action[13]   = right gripper

    Returns (actions, fps).
    """
    info_path = src / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text())
    chunks_size = int(info["chunks_size"])
    fps = int(info["fps"])
    data_tmpl = info["data_path"]

    chunk = ep_idx // chunks_size
    parquet = src / data_tmpl.format(episode_chunk=chunk, episode_index=ep_idx)
    if not parquet.is_file():
        raise FileNotFoundError(parquet)

    cols = ("action.left_arm", "action.left_gripper", "action.right_arm", "action.right_gripper")
    data = _read_parquet_columns(parquet, list(cols))

    l_arm = np.asarray(data["action.left_arm"], dtype=np.float32)        # (T, 6)
    r_arm = np.asarray(data["action.right_arm"], dtype=np.float32)       # (T, 6)
    l_grip = np.asarray(data["action.left_gripper"], dtype=np.float32)   # (T,) or (T,1)
    r_grip = np.asarray(data["action.right_gripper"], dtype=np.float32)

    # Gripper columns may be stored as scalar lists [x] or scalars; normalize.
    if l_grip.ndim == 2:
        l_grip = l_grip[:, 0]
    if r_grip.ndim == 2:
        r_grip = r_grip[:, 0]

    T = l_arm.shape[0]
    if not (r_arm.shape[0] == l_grip.shape[0] == r_grip.shape[0] == T):
        raise ValueError(
            f"length mismatch in episode {ep_idx}: l_arm={l_arm.shape}, r_arm={r_arm.shape}, "
            f"l_grip={l_grip.shape}, r_grip={r_grip.shape}"
        )

    actions = np.concatenate(
        [l_arm, l_grip[:, None], r_arm, r_grip[:, None]], axis=1
    ).astype(np.float32)
    return actions, fps


# ─── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/home/ntu/Desktop/ant/lpf/r1lite2lerobot/data/"
            "pick_cup_data_cleaned/pick_cup/pick_cup"
        ),
        help="LeRobot v2.1 dataset root",
    )
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="replay rate in Hz; defaults to info.json fps (15)",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="cap number of frames to replay (None = full episode)"
    )
    p.add_argument(
        "--warmup-seconds",
        type=float,
        default=2.0,
        help="linearly interpolate from current robot state to action[0] over this many seconds",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the action sequence and exit; never instantiates RobotEnv",
    )
    p.add_argument(
        "--countdown",
        type=float,
        default=3.0,
        help="wait this many seconds before sending the first command",
    )
    args = p.parse_args()

    actions, dataset_fps = load_episode_actions(args.src, args.episode_index)
    fps = float(args.fps) if args.fps is not None else float(dataset_fps)
    if args.limit is not None:
        actions = actions[: args.limit]

    print(f"episode {args.episode_index}: {len(actions)} frames, dataset fps={dataset_fps}, replay fps={fps}")
    print(f"first action: L_arm={actions[0, 0:6].round(3)} L_grip={actions[0, 6]:.1f} "
          f"R_arm={actions[0, 7:13].round(3)} R_grip={actions[0, 13]:.1f}")
    print(f"last  action: L_arm={actions[-1, 0:6].round(3)} L_grip={actions[-1, 6]:.1f} "
          f"R_arm={actions[-1, 7:13].round(3)} R_grip={actions[-1, 13]:.1f}")

    # Numerical sanity: any frame-to-frame action delta unusually large?
    diffs = np.abs(np.diff(actions, axis=0))
    max_arm_jump = diffs[:, list(range(6)) + list(range(7, 13))].max()
    max_grip_jump = diffs[:, [6, 13]].max()
    print(f"max single-step joint delta: {max_arm_jump:.4f} rad")
    print(f"max single-step gripper delta: {max_grip_jump:.2f}")

    if args.dry_run:
        print("\n--- DRY RUN: not connecting to robot, dumping first 5 + last 5 frames ---")
        for i in list(range(min(5, len(actions)))) + list(range(max(0, len(actions) - 5), len(actions))):
            print(f"  [{i:4d}] {actions[i].tolist()}")
        return

    # Defer ROS imports so --dry-run can run on machines without ROS2.
    from examples.simple_client.robot_env import RobotEnv

    print("\nconnecting to robot...")
    env = RobotEnv()
    time.sleep(1)

    frames, state = env.update_obs_window()
    if state is None or "qpos" not in state:
        raise RuntimeError("could not read initial robot state")
    current = np.asarray(state["qpos"], dtype=np.float32)
    if current.shape != (14,):
        raise RuntimeError(
            f"expected 14-D qpos from RobotEnv (post-fix), got shape {current.shape}. "
            "Make sure robot_env.py:get_joint_states has been patched."
        )

    print(f"current robot qpos (14D): {current.tolist()}")
    delta_arm = np.abs(actions[0, [0,1,2,3,4,5,7,8,9,10,11,12]] - current[[0,1,2,3,4,5,7,8,9,10,11,12]]).max()
    delta_grip = np.abs(actions[0, [6, 13]] - current[[6, 13]]).max()
    print(f"distance from current to first action: arm_max_delta={delta_arm:.3f} rad, "
          f"grip_max_delta={delta_grip:.2f}")

    if args.warmup_seconds > 0:
        n_warmup = max(1, int(args.warmup_seconds * fps))
        print(f"warming up: interpolating to action[0] over {args.warmup_seconds}s ({n_warmup} steps)...")
        for k in range(n_warmup):
            alpha = (k + 1) / n_warmup
            interp = (1.0 - alpha) * current + alpha * actions[0]
            env.control(interp.tolist())
            time.sleep(1.0 / fps)

    if args.countdown > 0:
        print(f"\n⚠️  starting replay in {args.countdown:.1f}s — press Ctrl+C to abort\n")
        time.sleep(args.countdown)

    print(f"replaying {len(actions)} frames at {fps:.1f} Hz...")
    t0 = time.time()
    dt = 1.0 / fps
    log_every = max(1, int(fps))  # 1 status print per second

    try:
        for i, a in enumerate(actions):
            target_t = t0 + (i + 1) * dt
            env.control(a.tolist())
            if i % log_every == 0 or i == len(actions) - 1:
                # Compare command vs current state for the right arm + gripper.
                _, st = env.update_obs_window()
                cur = np.asarray(st["qpos"], dtype=np.float32) if st and "qpos" in st else None
                if cur is not None and cur.shape == (14,):
                    track_err = np.abs(a[7:14] - cur[7:14]).max()
                    print(f"  [{i:4d}/{len(actions)}] R_arm={a[7:13].round(3)} R_grip={a[13]:.1f} "
                          f"track_err_R={track_err:.3f}")
                else:
                    print(f"  [{i:4d}/{len(actions)}] R_arm={a[7:13].round(3)} R_grip={a[13]:.1f}")
            sleep_for = max(0.0, target_t - time.time())
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\naborted by user")
    finally:
        elapsed = time.time() - t0
        ideal = len(actions) * dt
        print(f"replay finished: elapsed={elapsed:.2f}s ideal={ideal:.2f}s ({len(actions)/elapsed:.2f} Hz effective)")


if __name__ == "__main__":
    main()
