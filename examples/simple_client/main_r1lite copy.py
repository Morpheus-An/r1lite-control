import dataclasses
import enum
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

from examples.simple_client.robot_env import RobotEnv

logger = logging.getLogger(__name__)

# Specifying the task for robot control
# PROMPTS = "Collaborate with the human. The human is pointing at target mango. Pick up the target and wait for the next instruction."
PROMPTS = "Collaborate with the human. Pick up the target object and wait for human hand changes to next gesture."
# PROMPTS = "Collaborate with the human. The human hand is pointing at target carrot. Pick up the target object and wait for human hand changes to next gesture."
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


def prepare_image(img: np.ndarray, size: int = 224) -> str:
    """Resize with pad (server-compatible) and return base64-encoded PNG string."""
    if img is None:
        raise ValueError("prepare_image received img=None")
    if not isinstance(img, np.ndarray):
        raise TypeError(f"prepare_image expected numpy.ndarray, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"prepare_image expected image shape [H, W, C], got {img.shape}")

    target_h = size
    target_w = size
    cur_h, cur_w = img.shape[:2]

    if (cur_h, cur_w) != (target_h, target_w):
        ratio = max(cur_w / target_w, cur_h / target_h)
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)

        # PIL bilinear in the server corresponds to OpenCV linear interpolation.
        resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((target_h, target_w, img.shape[2]), dtype=img.dtype)
        pad_h = max(0, int((target_h - resized_h) / 2))
        pad_w = max(0, int((target_w - resized_w) / 2))
        padded[pad_h : pad_h + resized_h, pad_w : pad_w + resized_w] = resized
    else:
        padded = img

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
    # Number of steps to run the policy for.
    num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.ALOHA_SIM
    # Dry run mode: only show actions without executing them on the robot.
    dry_run: bool = False
    # Optional JSONL log file for actions and states.
    log_file: pathlib.Path | None = pathlib.Path("./log_actions_r1lite.jsonl")


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


def main(args: Args) -> None:
    """Main function to connect to the robot, run the control loop, and handle shutdown."""
    setup_logging_directory()
    env = None

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

        # --- B. MAIN CONTROL LOOP ---
        print("Entering main control loop...")
        while True:
            # --- i. Get Observations ---
            print("\n" + "=" * 50)
            print("1. Gathering robot state and images...")

            frames, state = env.update_obs_window()
            if state is None or not frames:
                print("Warning: No state or image data received. Skipping this cycle.")
                time.sleep(1)
                continue
            eef_pose_state = state.get("eef_pose")
            qpos_state = state.get("qpos")
            if qpos_state is not None:
                print(f"DEBUG: qpos_state shape={np.array(qpos_state).shape}, min={np.min(qpos_state):.4f}, max={np.max(qpos_state):.4f}")

            # --- ii. Prepare Data for Server ---
            print("2. Preparing data for inference server...")
            encoded_images: dict[str, np.ndarray] = {}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            for server_name, env_name in CAMERA_MAPPING.items():
                img = frames.get(env_name)
                if img is not None and img.size > 0:
                    encoded_images[server_name] = prepare_image(img)
                    log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}.png")
                    cv2.imwrite(log_path, img)
                else:
                    print(f"Warning: Image for '{server_name}' is not available.")

            # Convert state to 14D format if necessary
            qpos_state_14d = expand_qpos_to_14_dims(np.array(qpos_state, dtype=np.float32))
            print(f"DEBUG: State converted to 14D: {qpos_state_14d.shape}")
            
            obs_fn = {
                "state": qpos_state_14d.tolist(),
                "images": encoded_images,
                "prompt": PROMPTS,
            }

            # --- iii. Send Request to Server ---
            print(f"3. Sending request to server {args.host}...")
            try:
                policy = _websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                    api_key=args.api_key,
                )
                logger.info(f"Server metadata: {policy.get_server_metadata()}")
            except requests.exceptions.RequestException as e:
                print(f"Error communicating with server: {e}. Retrying after 5s.")
                time.sleep(5)
                continue

            # Send a few observations to make sure the model is loaded.
            for _ in range(2):
                policy.infer(obs_fn)

            timing_recorder = TimingRecorder()

            for _ in tqdm.trange(args.num_steps, desc="Running policy"):
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

            # --- iv. Parse and Execute Actions ---
            print("4. Parsing and executing actions...")
            actions = action.get("actions")
            if actions is None or len(actions) == 0:
                print("No actions received from the model. Skipping.")
                continue

            # Track state during action execution
            current_state = qpos_state_14d.copy()
            
            for i, act in enumerate(np.array(actions)):
                action_np = np.array(act, dtype=np.float32)
                print(f"[Step {i + 1}/{len(actions)}] Action (14D): L_arm={np.round(action_np[0:6], 3)} L_grip={action_np[6]:.3f} | R_arm={np.round(action_np[7:13], 3)} R_grip={action_np[13]:.3f}")
                print(f"  Raw action: {action_np.tolist()}")

                if not args.dry_run:
                    if CONTROL_MODE == "eepose":
                        env.control_eef(action_np)
                    elif CONTROL_MODE == "joint":
                        env.control(action_np)
                    
                    # [NEW] Read real robot state after executing action
                    try:
                        frames_rt, state_rt = env.update_obs_window()
                        if state_rt is not None and "qpos" in state_rt:
                            qpos_state_rt = state_rt.get("qpos")
                            qpos_state_14d_rt = expand_qpos_to_14_dims(np.array(qpos_state_rt, dtype=np.float32))
                        else:
                            qpos_state_14d_rt = qpos_state_14d.copy()
                    except Exception as e:
                        print(f"    Warning: Failed to read real state after action: {e}")
                        qpos_state_14d_rt = qpos_state_14d.copy()
                else:
                    print("  [DRY RUN] Action NOT executed on robot")
                    qpos_state_14d_rt = qpos_state_14d.copy()

                # Simulate state evolution (simple velocity integration)
                dt = 0.05
                velocity_scale = 0.5
                action_velocity = (action_np - current_state) * velocity_scale / dt
                current_state = current_state + action_velocity * dt
                current_state = np.clip(current_state, -np.pi, np.pi)

                l_arm = current_state[0:6]
                l_grip = current_state[6]
                r_arm = current_state[7:13]
                r_grip = current_state[13]
                print(f"  State (sim): L_arm={np.round(l_arm, 3)} L_grip={l_grip:.3f} | R_arm={np.round(r_arm, 3)} R_grip={r_grip:.3f}")

                if args.log_file is not None:
                    # Extract real state components
                    r_arm_real = qpos_state_14d_rt[7:13]
                    r_grip_real = qpos_state_14d_rt[13]
                    l_arm_real = qpos_state_14d_rt[0:6]
                    l_grip_real = qpos_state_14d_rt[6]
                    
                    append_jsonl(
                        args.log_file,
                        {
                            "timestamp": datetime.datetime.now().isoformat(timespec="milliseconds"),
                            "step_index": i + 1,
                            "num_actions": len(actions),
                            "dry_run": args.dry_run,
                            "action_raw": action_np.tolist(),
                            "action_parts": {
                                "l_arm": l_arm.tolist(),
                                "l_grip": float(l_grip),
                                "r_arm": r_arm.tolist(),
                                "r_grip": float(r_grip),
                            },
                            "state_sim": current_state.tolist(),
                            "state_obs": qpos_state_14d_rt.tolist(),
                            "state_obs_parts": {
                                "l_arm": l_arm_real.tolist(),
                                "l_grip": float(l_grip_real),
                                "r_arm": r_arm_real.tolist(),
                                "r_grip": float(r_grip_real),
                            },
                        },
                    )
                
                # Update observation for next inference
                obs_fn["state"] = current_state.tolist()

                time.sleep(0.05)  # Short pause between actions
                
                # Print current simulated state
                if (i + 1) % 10 == 0 or i == len(actions) - 1:
                    print("  → Simulated State snapshot saved")

            print("Action sequence execution complete.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        # --- C. SHUTDOWN ---
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
