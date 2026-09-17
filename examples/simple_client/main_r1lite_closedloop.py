"""Closed-loop 15 Hz client for R1 Lite — receding-horizon variant of main_r1lite.py.

Why this exists
---------------
The original `main_r1lite.py` sends ~22 inference requests per outer iteration
(2 warmup + 20 timing measurements) with the SAME observation, then executes
the LAST returned 8-action chunk open-loop with 50 ms sleeps between actions.

On the server side, the policy's `ObservationAdapter` pushes every incoming
observation into a `deque(maxlen=n_obs_steps)`. With 22 identical pushes the
deque ends up filled with 8 copies of the same frame, so the policy sees a
"history" of 8 identical frames — completely out of distribution from training
where each history step is a real frame at 15 Hz spanning 0.53 s of motion.

This client fixes that by querying once per control tick at 15 Hz (matching
the LeRobot dataset's recording fps) and executing only `actions[0]` from
each returned chunk (receding-horizon). Each request carries a fresh real
observation, so the server's rolling history reflects real temporal evolution.

Wire protocol and helpers are identical to `main_r1lite.py`; the only changes
are in the control loop. The unmodified server (`serve_policy.py` or
`replay_policy.py`) works as-is.

Usage:
    python main_r1lite_closedloop.py --env r1lite --host <upstream-ip> \\
        --hz 15 --max-steps 600 --dry-run        # 40 s dry run

    python main_r1lite_closedloop.py --env r1lite --host <upstream-ip>

Evaluation logging
------------------
One script run == one episode. Give each policy its own `--log-file` and tag it
with `--policy-name`; every run appends an `episode_start` header, its `step`
records, and an `episode_end` footer to that file, so repeated runs stay
separable with no manual bookkeeping:

    python main_r1lite_closedloop.py --host <ip> \\
        --policy-name A --log-file ./eval_policyA.jsonl --log-every 1

Plot the result with `plot_ee_trajectories.py A=eval_policyA.jsonl B=eval_policyB.jsonl`.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime
import enum
import io
import json
import logging
import os
import pathlib
import time

import cv2
import numpy as np
import requests
import tyro
from openpi_client import websocket_client_policy as _websocket_client_policy

from robot_env import RobotEnv


logger = logging.getLogger(__name__)


# ─── shared constants & helpers (kept identical to main_r1lite.py) ─────────

PROMPTS = "Collaborate with the human. Pick up the target object and handover to the human hand."

# 'joint' for joint angles; 'eepose' for end-effector pose.
CONTROL_MODE = "joint"

LOG_IMAGE_DIR = "./log_images_r1lite_closedloop"

# NOTE 2026-09-16: was "image_right" (physical RIGHT head lens). Training used
# observation.images.head_rgb == /hdas/camera_head/left_raw, i.e. the LEFT lens,
# so every deployment through 2026-09-16 10:38 fed the policy an unseen viewpoint
# (~105 px of stereo parallax at 1280x720, ~18 px after the 224 resize).
CAMERA_MAPPING = {
    "cam_head": "image_left",
    "cam_left_wrist": "image_left_wrist",
    "cam_right_wrist": "image_right_wrist",
}


def prepare_image(img: np.ndarray, size: int = 224) -> str:
    """Aspect-preserving resize + black-bar pad to size×size; return base64 PNG."""
    if img is None:
        raise ValueError("prepare_image received img=None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"prepare_image expected numpy.ndarray, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"prepare_image expected [H, W, C], got {img.shape}")

    target_h = target_w = size
    cur_h, cur_w = img.shape[:2]

    if (cur_h, cur_w) != (target_h, target_w):
        ratio = max(cur_w / target_w, cur_h / target_h)
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)
        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        pad_h = max(0, int((target_h - resized_h) / 2))
        pad_w = max(0, int((target_w - resized_w) / 2))
        padded[pad_h:pad_h + resized_h, pad_w:pad_w + resized_w] = resized
    else:
        padded = img

    _, buf = cv2.imencode('.png', padded)
    return base64.b64encode(io.BytesIO(buf).getvalue()).decode('utf-8')


def expand_qpos_to_14_dims(qpos: np.ndarray) -> np.ndarray:
    """Same as main_r1lite.py — 12D legacy → 14D server format. 14D passes through."""
    if len(qpos) == 14:
        return np.array(qpos, dtype=np.float32)
    if len(qpos) != 12:
        raise ValueError(f"Expected 12D or 14D state, got {len(qpos)}D")

    r_arm_5d = qpos[0:5]
    r_grip = qpos[5]
    l_arm_5d = qpos[6:11]
    l_grip = qpos[11]
    l_arm_6d = np.concatenate([l_arm_5d, [0.0]])
    r_arm_6d = np.concatenate([r_arm_5d, [0.0]])
    return np.array(
        np.concatenate([l_arm_6d, [l_grip], r_arm_6d, [r_grip]]),
        dtype=np.float32,
    )


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as h:
        h.write(json.dumps(record, ensure_ascii=False) + "\n")


def next_episode_index(path: pathlib.Path | None) -> int:
    """Highest `episode_index` already in the log + 1, so appending never collides.

    Tolerates junk lines (older logs have hand-inserted plain-text markers) and
    legacy records that predate episode tracking — those simply don't count.
    """
    if path is None or not path.exists():
        return 0
    best = -1
    with path.open("r", encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                idx = json.loads(line).get("episode_index")
            except json.JSONDecodeError:
                continue
            if isinstance(idx, int):
                best = max(best, idx)
    return best + 1


def setup_logging_directory() -> None:
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")


# ─── CLI ────────────────────────────────────────────────────────────────────


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    R1LITE = "r1lite"


@dataclasses.dataclass
class Args:
    host: str = "10.100.40.254"
    port: int | None = 8000
    api_key: str | None = None
    env: EnvMode = EnvMode.R1LITE
    hz: float = 15.0
    """Control loop rate. 15 matches the LeRobot dataset recording fps; this is what training assumed."""
    max_steps: int | None = None
    """Stop after this many control ticks (None = run until Ctrl+C)."""
    dry_run: bool = False
    """If true, don't actually send commands to the robot — useful for protocol smoke checks."""
    log_file: pathlib.Path | None = pathlib.Path("./log_actions_r1lite_closedloop.jsonl")
    """Trajectory log. Use ONE file per policy under evaluation; episodes are separated inside it."""
    log_every: int = 5
    """Append a jsonl record every N ticks (1 = every step)."""
    policy_name: str = "unnamed"
    """Name of the policy being evaluated (e.g. 'A', 'pi0_base'). Stamped on every record."""
    episode_label: str | None = None
    """Optional free-text tag for this episode (e.g. the grasp target). Stored in the episode header."""
    save_images_every: int | None = None
    """Save head+wrist PNG snapshots every N ticks. None disables."""
    summary_file: pathlib.Path | None = pathlib.Path("./task_times_r1lite_closedloop.jsonl")
    """Append one per-task summary (total inference time + total wall time) when the run ends."""


# ─── main ───────────────────────────────────────────────────────────────────


def main(args: Args) -> None:
    t_proc_start = time.perf_counter()          # monotonic clock for total-task-time
    ts_start = datetime.datetime.now()          # human-readable run-start stamp
    setup_logging_directory()
    print(f"cam_head <- {CAMERA_MAPPING['cam_head']}; "
          f"training used observation.images.head_rgb == the LEFT head lens "
          f"(/hdas/camera_head/left_raw/image_raw_color/compressed)")
    env: RobotEnv | None = None

    try:
        # 1. Robot env
        print("Initializing robot environment (ROS2)...")
        env = RobotEnv()
        time.sleep(2)
        print("Initialization complete.")

        if args.dry_run:
            print("\n" + "🟡 " * 25)
            print("DRY RUN: actions will NOT be sent to the robot.")
            print("🟡 " * 25 + "\n")
        else:
            print("\n" + "🟢 " * 25)
            print("LIVE MODE: actions WILL be sent to the robot.")
            print("🟢 " * 25 + "\n")

        # 2. Connect to inference server (ONCE, reused for all control ticks).
        print(f"Connecting to inference server at {args.host}:{args.port}...")
        try:
            policy = _websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
            )
            logger.info(f"Server metadata: {policy.get_server_metadata()}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to server: {e}")
            return

        # 3. Warm up once with the current observation (load model, jit etc.).
        print("Warmup (2 queries)...")
        warm_obs, _ = _build_obs(env, save_image_dir=None)
        if warm_obs is None:
            print("Could not read initial observation; aborting.")
            return
        for _ in range(2):
            policy.infer(warm_obs)

        # 4. Episode bookkeeping: one script run == one episode. The index is
        #    derived from what's already in the log, so repeated runs appending
        #    to the same file stay separable without any manual bookkeeping.
        episode_index = next_episode_index(args.log_file)
        episode_id = f"{args.policy_name}-{ts_start.strftime('%Y%m%d_%H%M%S')}"
        if args.log_file is not None:
            append_jsonl(args.log_file, {
                "type": "episode_start",
                "timestamp": ts_start.isoformat(timespec="milliseconds"),
                "policy": args.policy_name,
                "episode_index": episode_index,
                "episode_id": episode_id,
                "episode_label": args.episode_label,
                "dry_run": args.dry_run,
                "hz": args.hz,
                "log_every": args.log_every,
                "host": f"{args.host}:{args.port}",
                "prompt": PROMPTS,
                "control_mode": CONTROL_MODE,
            })

        # 5. Closed-loop 15 Hz control: take chunk[0] each tick.
        target_period = 1.0 / float(args.hz)
        print(f"\nEpisode {episode_index}  (policy={args.policy_name}, id={episode_id}"
              + (f", label={args.episode_label}" if args.episode_label else "") + ")")
        print(f"  → logging trajectory to {args.log_file}")
        print(f"Entering closed-loop control at {args.hz:.1f} Hz "
              f"(period = {target_period * 1000:.1f} ms). Press Ctrl+C to stop.\n")

        next_tick = time.time()
        loop_start = time.perf_counter()        # task clock: excludes init + connect + warmup above
        step = 0
        infer_times_ms = []     # policy.infer() round-trip, client-side (incl. network)
        exec_times_ms = []      # env.control() — pushing the action onto the ROS2 topics
        n_overruns = 0          # ticks that missed the 1/hz deadline
        stop_reason = "unknown"
        try:
            while True:
                # Wait for the next tick (skip if we already overran).
                now = time.time()
                slack = next_tick - now
                if slack > 0:
                    time.sleep(slack)
                else:
                    n_overruns += 1
                    if step > 0 and step % 30 == 0:
                        print(f"  [warn] tick overrun by {-slack * 1000:.0f} ms — control loop is slow")
                next_tick += target_period

                # Fresh observation per tick (the whole point of this client).
                save_dir = (
                    LOG_IMAGE_DIR
                    if args.save_images_every and step % args.save_images_every == 0
                    else None
                )
                obs, r_ee_xyz = _build_obs(env, save_image_dir=save_dir)
                if obs is None:
                    if step % 15 == 0:
                        print("  [skip] no observation this tick")
                    continue

                # Single inference, single action consumed.
                t0 = time.perf_counter()
                result = policy.infer(obs)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                infer_times_ms.append(infer_ms)

                actions = np.asarray(result.get("actions", []))
                if actions.size == 0:
                    if step % 15 == 0:
                        print("  [skip] empty action chunk")
                    continue
                action_first = actions[0].astype(np.float32)

                # Execute.
                exec_ms = 0.0
                if not args.dry_run:
                    t1 = time.perf_counter()
                    if CONTROL_MODE == "joint":
                        env.control(action_first.tolist())
                    elif CONTROL_MODE == "eepose":
                        env.control_eef(action_first.tolist())
                    exec_ms = (time.perf_counter() - t1) * 1000.0
                    exec_times_ms.append(exec_ms)

                # Periodic status print.
                if step % int(max(1, args.hz)) == 0:  # ≈ 1 line/sec
                    r_arm = action_first[7:13]
                    r_grip = action_first[13]
                    mean_infer = float(np.mean(infer_times_ms[-int(args.hz):]))
                    print(f"  [t={step / args.hz:6.2f}s  step={step:4d}]  "
                          f"R_arm={np.round(r_arm, 3).tolist()}  R_grip={r_grip:5.1f}  "
                          f"infer={infer_ms:5.1f}ms (avg {mean_infer:5.1f}ms)")

                # Periodic jsonl logging.
                if args.log_file is not None and step % args.log_every == 0:
                    append_jsonl(args.log_file, {
                        "type": "step",
                        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                        "policy": args.policy_name,
                        "episode_index": episode_index,
                        "episode_id": episode_id,
                        "step": step,
                        "dry_run": args.dry_run,
                        "action": action_first.tolist(),
                        "state": obs["state"],
                        "r_ee_xyz": r_ee_xyz,   # right-gripper base_link xyz; may be None
                        "infer_ms": infer_ms,   # policy round-trip (incl. network)
                        "exec_ms": exec_ms,     # env.control() publish; 0.0 under --dry-run
                    })

                step += 1
                if args.max_steps is not None and step >= args.max_steps:
                    stop_reason = "max_steps"
                    print(f"\nReached --max-steps={args.max_steps}; stopping.")
                    break

        except KeyboardInterrupt:
            stop_reason = "ctrl_c"
            print("\nInterrupted by user.")

        # Final summary + per-task timing record.
        t_end = time.perf_counter()
        infer_s_total = float(np.sum(infer_times_ms)) / 1000.0   # Σ of all policy.infer() calls
        exec_s_total = float(np.sum(exec_times_ms)) / 1000.0     # Σ of all env.control() calls
        wall_s_total = t_end - t_proc_start                      # script start → stop (incl. init+warmup)
        wall_s_task = t_end - loop_start                         # control loop only (excl. init+warmup)
        infer = np.asarray(infer_times_ms) if infer_times_ms else None
        ex = np.asarray(exec_times_ms) if exec_times_ms else None
        if infer is not None:
            print(f"\nRan {step} ticks at target {args.hz:.1f} Hz "
                  f"(target period {target_period * 1000:.1f} ms, {n_overruns} overruns).")
            print(f"Inference latency: mean={infer.mean():.1f}ms  p50={np.median(infer):.1f}ms  "
                  f"p95={np.quantile(infer, 0.95):.1f}ms  p99={np.quantile(infer, 0.99):.1f}ms")
        if ex is not None:
            print(f"Execution latency: mean={ex.mean():.1f}ms  p95={np.quantile(ex, 0.95):.1f}ms")

        print("\n── Task timing ──────────────────────────────")
        print(f"  Total policy inference : {infer_s_total:8.2f} s   (Σ of {len(infer_times_ms)} infer calls)")
        print(f"  Total action execution : {exec_s_total:8.2f} s   (Σ of {len(exec_times_ms)} control calls)")
        print(f"  Total task wall time   : {wall_s_total:8.2f} s   (script start → stop, incl. init+warmup)")
        print(f"  Control-loop time      : {wall_s_task:8.2f} s   (task only, excl. init+warmup)")

        def _q(a, q):
            return round(float(np.quantile(a, q)), 2) if a is not None else None

        episode_footer = {
            "timestamp_start": ts_start.isoformat(timespec="milliseconds"),
            "timestamp_end": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "policy": args.policy_name,
            "episode_index": episode_index,
            "episode_id": episode_id,
            "episode_label": args.episode_label,
            "stop_reason": stop_reason,
            "dry_run": args.dry_run,
            "hz": args.hz,
            "n_steps": step,
            "n_overruns": n_overruns,
            "infer_s_total": round(infer_s_total, 3),
            "exec_s_total": round(exec_s_total, 3),
            "wall_s_total": round(wall_s_total, 3),
            "wall_s_task": round(wall_s_task, 3),
            "infer_ms_mean": round(float(infer.mean()), 2) if infer is not None else None,
            "infer_ms_p50": _q(infer, 0.50),
            "infer_ms_p95": _q(infer, 0.95),
            "infer_ms_p99": _q(infer, 0.99),
            "exec_ms_mean": round(float(ex.mean()), 2) if ex is not None else None,
            "exec_ms_p95": _q(ex, 0.95),
            "infer_pct_of_task": round(100.0 * infer_s_total / wall_s_task, 1) if wall_s_task > 0 else None,
        }
        # Close the episode in the trajectory log itself, so one file is self-contained.
        if args.log_file is not None:
            append_jsonl(args.log_file, {"type": "episode_end", **episode_footer})
            print(f"  → closed episode {episode_index} in {args.log_file}")
        if args.summary_file is not None:
            append_jsonl(args.summary_file, episode_footer)
            print(f"  → appended task summary to {args.summary_file}")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise
    finally:
        print("Program finished.")


def _build_obs(env: RobotEnv, save_image_dir: str | None) -> tuple[dict | None, list | None]:
    """Read one fresh frame+state; return (request dict, right-EE xyz) or (None, None)."""
    frames, state = env.update_obs_window()
    if state is None or not frames:
        return None, None
    qpos = state.get("qpos")
    if qpos is None:
        return None, None
    qpos_14 = expand_qpos_to_14_dims(np.array(qpos, dtype=np.float32))

    # Right-gripper EE position (x, y, z) in base_link frame. get_eef_poses() in
    # robot_env.py is right-first: [R_xyz(0:3), R_quat(3:7), R_grip(7),
    # L_xyz(8:11), L_quat(11:15), L_grip(15)].
    eef_pose = state.get("eef_pose")
    r_ee_xyz = [float(v) for v in eef_pose[0:3]] if eef_pose is not None else None

    encoded = {}
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    for server_name, env_name in CAMERA_MAPPING.items():
        img = frames.get(env_name)
        if img is None or img.size == 0:
            continue
        encoded[server_name] = prepare_image(img)
        if save_image_dir is not None:
            cv2.imwrite(os.path.join(save_image_dir, f"{server_name}_{ts}.png"), img)

    return {
        "state": qpos_14.tolist(),
        "images": encoded,
        "prompt": PROMPTS,
    }, r_ee_xyz


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
