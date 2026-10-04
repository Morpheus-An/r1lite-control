import dataclasses
import enum
import hashlib
import logging
import pathlib
import time
import datetime
import os
import cv2
import requests
import base64
import io
import json

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

from robot_env import RobotEnv

logger = logging.getLogger(__name__)

# Specifying the task for robot control
# PROMPTS = "Collaborate with the human. The human is pointing at target mango. Pick up the target and wait for the next instruction."

# PROMPTS = "Collaborate with the human. Pick up the target object and wait for human hand changes to next gesture."
# PROMPTS = "Collaborate with the human. The human hand is pointing at target carrot. Pick up the target object and wait for human hand changes to next gesture."
PROMPTS = "Collaborate with the human. Pick up the target object and handover to the human hand."

# PROMPTS = "Collaborate with the human. The human hand is keeping open for receiving the target peach. Handover the target object to the human hand when human hand changes gesture."

# Robot control mode: 'eepose' for end-effector pose, 'joint' for joint angles
CONTROL_MODE = "joint"

# Directory to save periodic image logs
LOG_IMAGE_DIR = "./log_images_r1lite_ros2"

# Mapping from server-expected camera names to environment camera names
CAMERA_MAPPING = {
    "cam_head": "image_right",
    "cam_left_wrist": "image_left_wrist",
    "cam_right_wrist": "image_right_wrist",
}

# --- HELPER FUNCTIONS ---


# def prepare_image(img: np.ndarray, size: int = 224) -> str:
#     """Resize with pad (server-compatible) and return base64-encoded PNG string."""
#     if img is None:
#         raise ValueError("prepare_image received img=None")
#     if not isinstance(img, np.ndarray):
#         raise TypeError(f"prepare_image expected numpy.ndarray, got {type(img)}")
#     if img.ndim != 3:
#         raise ValueError(f"prepare_image expected image shape [H, W, C], got {img.shape}")

#     target_h = size
#     target_w = size
#     cur_h, cur_w = img.shape[:2]

#     if (cur_h, cur_w) != (target_h, target_w):
#         ratio = max(cur_w / target_w, cur_h / target_h)
#         resized_h = int(cur_h / ratio)
#         resized_w = int(cur_w / ratio)

#         # PIL bilinear in the server corresponds to OpenCV linear interpolation.
#         resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

#         padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
#         pad_h = max(0, int((target_h - resized_h) / 2))
#         pad_w = max(0, int((target_w - resized_w) / 2))
#         padded[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
#     else:
#         padded = img

#     _, img_buffer = cv2.imencode('.png', padded)
#     img_bytes = io.BytesIO(img_buffer).getvalue()
#     b64_string = base64.b64encode(img_bytes).decode('utf-8')
#     return b64_string
def prepare_image(img: np.ndarray, size: int = 224) -> str:
    """Pad-to-square then resize and return base64-encoded PNG string.

    Matches the training preprocessing pipeline exactly:
    1. Pad shorter edge with zeros to make image square
    2. Resize square image to (size, size) using INTER_AREA
    3. Encode as PNG and base64
    """
    if img is None:
        raise ValueError("prepare_image received img=None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"prepare_image expected numpy.ndarray, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"prepare_image expected image shape [H, W, C], got {img.shape}")

    h, w = img.shape[:2]

    # Step 1: Pad to square (shorter edge padded with zeros, centered)
    if h != w:
        max_side = max(h, w)
        padded = np.zeros((max_side, max_side, img.shape[2]), dtype=img.dtype)
        pad_h = (max_side - h) // 2
        pad_w = (max_side - w) // 2
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = img
    else:
        padded = img

    # Step 2: Resize to target size using INTER_AREA (matches training)
    if padded.shape[0] != size or padded.shape[1] != size:
        padded = cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)

    # Step 3: Encode as base64 PNG
    _, img_buffer = cv2.imencode('.png', padded)
    img_bytes = io.BytesIO(img_buffer).getvalue()
    b64_string = base64.b64encode(img_bytes).decode('utf-8')
    return b64_string


def expand_qpos_to_14_dims(qpos_12d: np.ndarray) -> np.ndarray:
    """Convert robot 12D output to server 14D format.
    Robot output: [R1-R5(5), R_grip(1), L1-L5(5), L_grip(1)] = 12D
    Server expects: [L1-L6(6), L_grip(1), R1-R6(6), R_grip(1)] = 14D
    """
    if len(qpos_12d) == 14:
        return np.array(qpos_12d, dtype=np.float32)
    if len(qpos_12d) != 12:
        raise ValueError(f"Expected 12D or 14D state, got {len(qpos_12d)}D")
    
    # Extract components from robot output
    r_arm_5d = qpos_12d[0:5]      # R1-R5
    r_grip = qpos_12d[5]           # R_grip
    l_arm_5d = qpos_12d[6:11]      # L1-L5
    l_grip = qpos_12d[11]          # L_grip
    
    # Expand to 14D by inserting 0 for 6th joint
    l_arm_6d = np.concatenate([l_arm_5d, [0.0]])  # L1-L5 + L6(set to 0)
    r_arm_6d = np.concatenate([r_arm_5d, [0.0]])  # R1-R5 + R6(set to 0)
    
    # Reorder to server format: [L1-L6, L_grip, R1-R6, R_grip]
    qpos_14d = np.concatenate([l_arm_6d, [l_grip], r_arm_6d, [r_grip]])
    return np.array(qpos_14d, dtype=np.float32)


def setup_logging_directory():
    """Creates the logging directory if it doesn't exist."""
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    """Append a record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


# ------------------------


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    R1LITE = "r1lite"


class ControlMode(enum.Enum):
    """How the main loop drives `policy.infer()`.

    - LEGACY: capture observation ONCE per outer iteration, run a timing
      benchmark of `num_steps` inferences with that same observation, then
      execute the last action chunk on the robot. Reproduces the stuck-observation
      symptom in `episode_latents.npz` (15-20 bit-identical z_real entries) —
      keep this only for backward-compatibility benchmarks.
    - CLOSED_LOOP: capture a fresh observation before every `policy.infer()`
      call, execute the action chunk between calls. This is the correct
      deployment behavior.
    """

    LEGACY = "legacy"
    CLOSED_LOOP = "closed_loop"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    # host: str = "192.168.31.160"
    host: str = "10.100.40.254"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of policy inference cycles to run. Set to <= 0 to loop indefinitely.
    num_steps: int = 20
    # How many actions from each chunk to execute before re-inferring with a fresh observation.
    # Defaults to None == execute the entire chunk (legacy behavior).
    actions_per_inference: int | None = None
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.ALOHA_SIM
    # Dry run mode: only show actions without executing them on the robot.
    dry_run: bool = False
    # Optional JSONL log file for actions and states.
    log_file: pathlib.Path | None = pathlib.Path("./log_actions_r1lite.jsonl")
    # Which control loop to run. See ControlMode docstring for trade-offs.
    control_mode: ControlMode = ControlMode.CLOSED_LOOP


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


def _capture_observation(env: "RobotEnv", step_idx: int) -> tuple[dict | None, np.ndarray | None]:
    """Capture a single fresh (state, images) observation from the robot.

    Returns (obs_fn, qpos_state_14d) on success, (None, None) on transient failure.
    Each call refreshes from the latest ROS messages — DO NOT reuse the result
    across multiple `policy.infer()` calls (that was the original stuck-obs bug).
    """
    frames, state = env.update_obs_window()
    if state is None or not frames:
        return None, None

    qpos_state = state.get("qpos")
    if qpos_state is None:
        return None, None
    qpos_state_14d = expand_qpos_to_14_dims(np.array(qpos_state, dtype=np.float32))

    encoded_images: dict[str, str] = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    for server_name, env_name in CAMERA_MAPPING.items():
        img = frames.get(env_name)
        if img is not None and img.size > 0:
            encoded_images[server_name] = prepare_image(img)
            log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}_step{step_idx}.png")
            cv2.imwrite(log_path, img)
        else:
            print(f"Warning: Image for '{server_name}' is not available.")

    if not encoded_images:
        return None, None

    # Diagnostic hash log per WebSocket send (per bug-report Action Item #2):
    # if these hashes are constant across consecutive steps, the robot/sensor
    # layer is delivering stale data — same root cause as the server-side
    # bit-identical-z_real symptom.
    state_hash = hashlib.md5(qpos_state_14d.tobytes()).hexdigest()[:8]
    image_hashes = {k: hashlib.md5(v.encode()).hexdigest()[:8] for k, v in encoded_images.items()}
    print(f"  [step={step_idx}] state_hash={state_hash} image_hashes={image_hashes} qpos[7:13]={np.round(qpos_state_14d[7:13], 4).tolist()}")

    obs_fn = {
        "state": qpos_state_14d.tolist(),
        "images": encoded_images,
        "prompt": PROMPTS,
    }
    return obs_fn, qpos_state_14d


def _run_closed_loop(env, policy, args: "Args", timing_recorder: "TimingRecorder") -> None:
    """Closed-loop control: capture a fresh observation BEFORE each policy.infer()
    call, then execute the action chunk so the robot moves before the next capture.

    This is the correct deployment behavior. Every WebSocket send carries a
    freshly-captured (state, images) pair, so `episode_latents.npz` will record
    genuinely varying observations.
    """
    max_steps = args.num_steps if args.num_steps and args.num_steps > 0 else None
    step_idx = 0

    while True:
        if max_steps is not None and step_idx >= max_steps:
            print(f"Reached configured num_steps={max_steps}. Exiting.")
            break

        print("\n" + "=" * 50)
        print(f"[Cycle {step_idx}] 1. Capturing fresh observation...")

        obs_fn, qpos_state_14d = _capture_observation(env, step_idx=step_idx)
        if obs_fn is None:
            print("Warning: No state or image data received. Retrying in 1s.")
            time.sleep(1)
            continue

        # --- 2. Inference ---
        print(f"[Cycle {step_idx}] 2. Running inference...")
        inference_start = time.time()
        try:
            action = policy.infer(obs_fn)
        except Exception as e:
            print(f"Error during inference: {e}. Retrying after 5s.")
            time.sleep(5)
            continue
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        for key, value in action.get("server_timing", {}).items():
            timing_recorder.record(f"server_{key}", value)
        for key, value in action.get("policy_timing", {}).items():
            timing_recorder.record(f"policy_{key}", value)

        actions = action.get("actions")
        if actions is None or len(actions) == 0:
            print("No actions received from the model. Skipping.")
            step_idx += 1
            continue

        actions_np = np.array(actions)
        n_total = len(actions_np)
        n_execute = args.actions_per_inference if args.actions_per_inference else n_total
        n_execute = max(1, min(n_execute, n_total))
        print(f"[Cycle {step_idx}] 3. Executing {n_execute}/{n_total} actions from chunk before re-inferring...")

        # --- 3. Execute action chunk ---
        for i in range(n_execute):
            action_np = np.array(actions_np[i], dtype=np.float32)
            print(
                f"  [Step {i + 1}/{n_execute}] L_arm={np.round(action_np[0:6], 3)} "
                f"L_grip={action_np[6]:.3f} | R_arm={np.round(action_np[7:13], 3)} "
                f"R_grip={action_np[13]:.3f}"
            )

            if not args.dry_run:
                if CONTROL_MODE == "eepose":
                    env.control_eef(action_np)
                elif CONTROL_MODE == "joint":
                    env.control(action_np)

                # Read real state for logging (NOT for re-inference — re-inference
                # happens at the top of the next cycle with a freshly captured obs).
                try:
                    _, state_rt = env.update_obs_window()
                    if state_rt is not None and "qpos" in state_rt:
                        qpos_state_14d_rt = expand_qpos_to_14_dims(
                            np.array(state_rt.get("qpos"), dtype=np.float32)
                        )
                    else:
                        qpos_state_14d_rt = qpos_state_14d.copy()
                except Exception as e:
                    print(f"    Warning: Failed to read real state after action: {e}")
                    qpos_state_14d_rt = qpos_state_14d.copy()
            else:
                print("    [DRY RUN] Action NOT executed on robot")
                qpos_state_14d_rt = qpos_state_14d.copy()

            if args.log_file is not None:
                append_jsonl(
                    args.log_file,
                    {
                        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                        "control_mode": args.control_mode.value,
                        "cycle": step_idx,
                        "step_index": i + 1,
                        "num_actions": n_execute,
                        "chunk_size": n_total,
                        "dry_run": args.dry_run,
                        "action_raw": action_np.tolist(),
                        "action_parts": {
                            "l_arm": action_np[0:6].tolist(),
                            "l_grip": float(action_np[6]),
                            "r_arm": action_np[7:13].tolist(),
                            "r_grip": float(action_np[13]),
                        },
                        "state_obs": qpos_state_14d_rt.tolist(),
                        "state_obs_parts": {
                            "l_arm": qpos_state_14d_rt[0:6].tolist(),
                            "l_grip": float(qpos_state_14d_rt[6]),
                            "r_arm": qpos_state_14d_rt[7:13].tolist(),
                            "r_grip": float(qpos_state_14d_rt[13]),
                        },
                    },
                )

            time.sleep(1.0 / 15)

        step_idx += 1

        if step_idx % 10 == 0:
            timing_recorder.print_all_stats()
            if args.timing_file is not None:
                timing_recorder.write_parquet(args.timing_file)


def _run_legacy(env, policy, args: "Args", timing_recorder: "TimingRecorder") -> None:
    """Legacy control loop: matches the pre-fix behavior.

    Per outer iteration:
      1. Capture observation ONCE.
      2. Run num_steps inferences with that SAME observation (timing benchmark).
      3. Execute the LAST returned action chunk on the robot.
      4. Repeat forever.

    This intentionally reproduces the stuck-observation symptom that the
    server-side diagnostic captured (15-20 bit-identical z_real entries in
    episode_latents.npz). Only useful for backward-compatibility benchmarks
    or for reproducing the original bug — for real deployment, prefer
    --control-mode closed_loop (the default).
    """
    while True:
        # --- i. Get Observations (ONCE per outer iteration) ---
        print("\n" + "=" * 50)
        print("[LEGACY] 1. Gathering robot state and images...")

        frames, state = env.update_obs_window()
        if state is None or not frames:
            print("Warning: No state or image data received. Skipping this cycle.")
            time.sleep(1)
            continue
        qpos_state = state.get("qpos")
        if qpos_state is not None:
            print(f"DEBUG: qpos_state shape={np.array(qpos_state).shape}, "
                  f"min={np.min(qpos_state):.4f}, max={np.max(qpos_state):.4f}")

        # --- ii. Prepare Data for Server ---
        print("[LEGACY] 2. Preparing data for inference server...")
        encoded_images: dict[str, str] = {}
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for server_name, env_name in CAMERA_MAPPING.items():
            img = frames.get(env_name)
            if img is not None and img.size > 0:
                encoded_images[server_name] = prepare_image(img)
                log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                cv2.imwrite(log_path, img)
            else:
                print(f"Warning: Image for '{server_name}' is not available.")

        qpos_state_14d = expand_qpos_to_14_dims(np.array(qpos_state, dtype=np.float32))
        print(f"DEBUG: State converted to 14D: {qpos_state_14d.shape}")

        obs_fn = {
            "state": qpos_state_14d.tolist(),
            "images": encoded_images,
            "prompt": PROMPTS,
        }

        # --- iii. Timing benchmark loop: num_steps infers with the SAME obs ---
        print(f"[LEGACY] 3. Running {args.num_steps} inferences with the same observation (benchmark)...")
        for _ in tqdm.trange(args.num_steps, desc="Running policy (legacy)"):
            inference_start = time.time()
            action = policy.infer(obs_fn)
            timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
            for key, value in action.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in action.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)

        timing_recorder.print_all_stats()
        if args.timing_file is not None:
            timing_recorder.write_parquet(args.timing_file)

        # --- iv. Parse and Execute Actions from the LAST inference ---
        print("[LEGACY] 4. Parsing and executing actions from the last inference...")
        actions = action.get("actions")
        if actions is None or len(actions) == 0:
            print("No actions received from the model. Skipping.")
            continue

        current_state = qpos_state_14d.copy()

        for i, act in enumerate(np.array(actions)):
            action_np = np.array(act, dtype=np.float32)
            print(
                f"[Step {i + 1}/{len(actions)}] Action (14D): "
                f"L_arm={np.round(action_np[0:6], 3)} L_grip={action_np[6]:.3f} | "
                f"R_arm={np.round(action_np[7:13], 3)} R_grip={action_np[13]:.3f}"
            )

            if not args.dry_run:
                if CONTROL_MODE == "eepose":
                    env.control_eef(action_np)
                elif CONTROL_MODE == "joint":
                    env.control(action_np)

                try:
                    _, state_rt = env.update_obs_window()
                    if state_rt is not None and "qpos" in state_rt:
                        qpos_state_14d_rt = expand_qpos_to_14_dims(
                            np.array(state_rt.get("qpos"), dtype=np.float32)
                        )
                    else:
                        qpos_state_14d_rt = qpos_state_14d.copy()
                except Exception as e:
                    print(f"  Warning: Failed to read real state after action: {e}")
                    qpos_state_14d_rt = qpos_state_14d.copy()
            else:
                print("  [DRY RUN] Action NOT executed on robot")
                qpos_state_14d_rt = qpos_state_14d.copy()

            # Simulated velocity integration (matches original legacy behavior).
            dt = 0.05
            velocity_scale = 0.5
            action_velocity = (action_np - current_state) * velocity_scale / dt
            current_state = current_state + action_velocity * dt
            current_state = np.clip(current_state, -np.pi, np.pi)

            if args.log_file is not None:
                append_jsonl(
                    args.log_file,
                    {
                        "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                        "control_mode": args.control_mode.value,
                        "step_index": i + 1,
                        "num_actions": len(actions),
                        "dry_run": args.dry_run,
                        "action_raw": action_np.tolist(),
                        "action_parts": {
                            "l_arm": current_state[0:6].tolist(),
                            "l_grip": float(current_state[6]),
                            "r_arm": current_state[7:13].tolist(),
                            "r_grip": float(current_state[13]),
                        },
                        "state_sim": current_state.tolist(),
                        "state_obs": qpos_state_14d_rt.tolist(),
                        "state_obs_parts": {
                            "l_arm": qpos_state_14d_rt[0:6].tolist(),
                            "l_grip": float(qpos_state_14d_rt[6]),
                            "r_arm": qpos_state_14d_rt[7:13].tolist(),
                            "r_grip": float(qpos_state_14d_rt[13]),
                        },
                    },
                )

            # Update obs_fn["state"] (kept for parity with original behavior;
            # not re-sent to the server in this iteration).
            obs_fn["state"] = current_state.tolist()
            time.sleep(1.0 / 15)

        print("[LEGACY] Action sequence execution complete.")


def main(args: Args) -> None:
    """Main function to connect to the robot, run the control loop, and handle shutdown."""
    setup_logging_directory()
    env = None
    timing_recorder = TimingRecorder()

    try:
        # --- A. INITIALIZATION ---
        print("Initializing robot environment (ROS2)...")
        env = RobotEnv()
        time.sleep(2)  # Wait for environment to stabilize
        print("Initialization complete!")

        if args.dry_run:
            print("\n" + "🟡 " * 25)
            print("⚠️  DRY RUN MODE ENABLED - Actions will NOT be executed on the robot")
            print("🟡 " * 25 + "\n")
        else:
            print("\n" + "🟢 " * 25)
            print("✓ LIVE MODE - Actions WILL be executed on the robot")
            print("🟢 " * 25 + "\n")

        if args.log_file is not None:
            print(f"Logging actions and states to {args.log_file}")

        # Connect to the inference server ONCE (was inside the loop, causing
        # a fresh handshake every cycle).
        print(f"Connecting to inference server {args.host}:{args.port}...")
        try:
            policy = _websocket_client_policy.WebsocketClientPolicy(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
            )
            logger.info(f"Server metadata: {policy.get_server_metadata()}")
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with server: {e}. Exiting.")
            return

        # Warmup with a single fresh observation so the model is loaded server-side.
        warmup_obs, _ = _capture_observation(env, step_idx=-1)
        if warmup_obs is not None:
            print("Warming up server (2 inferences with the current observation)...")
            for _ in range(2):
                policy.infer(warmup_obs)

        # --- B. DISPATCH TO THE SELECTED CONTROL LOOP ---
        print(f"Entering main control loop (mode={args.control_mode.value})...")
        if args.control_mode == ControlMode.LEGACY:
            _run_legacy(env, policy, args, timing_recorder)
        else:
            _run_closed_loop(env, policy, args, timing_recorder)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- C. SHUTDOWN ---
        try:
            timing_recorder.print_all_stats()
            if args.timing_file is not None:
                timing_recorder.write_parquet(args.timing_file)
        except Exception:
            pass
        # if env:
        #     env.shutdown()
        print("Program finished.")


def _random_observation_r1lite() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_head": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
