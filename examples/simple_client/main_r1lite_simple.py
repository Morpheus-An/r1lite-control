"""
简化版本：直接参考 main_r1lite.py 的结构，添加状态模拟功能。
"""
import dataclasses
import datetime
import logging
import os
import pathlib
import time
import base64
import cv2
import io
import numpy as np
import requests
import tyro
import polars as pl
import rich
import rich.console
import rich.table

from openpi_client import websocket_client_policy as _websocket_client_policy
from examples.simple_client.robot_env import RobotEnv

logger = logging.getLogger(__name__)

PROMPTS = "Pick up the peach and place it into the tray"
LOG_IMAGE_DIR = "./log_images_r1lite_ros2"
CONTROL_MODE = "joint"

CAMERA_MAPPING = {
    "cam_head": "image_right",
    "cam_left_wrist": "image_left_wrist",
    "cam_right_wrist": "image_right_wrist",
}


def prepare_image(img: np.ndarray, size: int = 224) -> str:
    """Resize image and return as base64-encoded PNG string."""
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    # Encode as PNG
    is_success, img_buffer = cv2.imencode('.png', resized)
    img_bytes = io.BytesIO(img_buffer).getvalue()
    # Encode as base64
    b64_string = base64.b64encode(img_bytes).decode('utf-8')
    return b64_string


def expand_qpos_to_14_dims(qpos_12d: np.ndarray) -> np.ndarray:
    """扩展12维qpos到14维标准格式。"""
    if len(qpos_12d) == 14:
        return qpos_12d.copy()
    elif len(qpos_12d) == 12:
        expanded = np.zeros(14)
        expanded[0:5] = qpos_12d[6:11]
        expanded[5] = 0.0
        expanded[6] = qpos_12d[11]
        expanded[7:12] = qpos_12d[0:5]
        expanded[12] = 0.0
        expanded[13] = qpos_12d[5]
        return expanded
    else:
        raise ValueError(f"Unexpected qpos dimension: {len(qpos_12d)}, expected 12 or 14")


def setup_logging_directory() -> None:
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")


class StateSimulator:
    """Simulates state changes based on actions without executing on real robot."""

    def __init__(self, dt: float = 0.05):
        self.dt = dt
        self.velocity_scale = 0.5

    def simulate_step(self, qpos_current: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simulate one action step with proper 14D state/action handling."""
        # 扩展qpos到14维
        if len(qpos_current) == 12:
            qpos = expand_qpos_to_14_dims(qpos_current)
        elif len(qpos_current) == 14:
            qpos = qpos_current.copy()
        else:
            raise ValueError(f"Expected qpos to be 12D or 14D, got {len(qpos_current)}D")
        
        if len(action) != 14:
            raise ValueError(f"Expected action to be 14D, got {len(action)}D")
        
        qpos_next = qpos.copy()
        
        # Action: [R1-R6(0-5), R_grip(6), L1-L6(7-12), L_grip(13)]
        # State:  [L1-L6(0-5), L_grip(6), R1-R6(7-12), R_grip(13)]
        qpos_next[0:6] = qpos[0:6] + action[7:13] * self.dt * self.velocity_scale
        qpos_next[6] = action[13]
        qpos_next[7:13] = qpos[7:13] + action[0:6] * self.dt * self.velocity_scale
        qpos_next[13] = action[6]
        qpos_next = np.clip(qpos_next, -np.pi, np.pi)
        
        return qpos_next


class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p50": float(np.quantile(times, 0.50)),
            "p95": float(np.quantile(times, 0.95)),
        }

    def print_all_stats(self) -> None:
        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Mean (ms)", justify="right", style="yellow")
        table.add_column("P50 (ms)", justify="right", style="magenta")
        table.add_column("P95 (ms)", justify="right", style="magenta")

        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            table.add_row(key, f"{stats['mean']:.1f}", f"{stats['p50']:.1f}", f"{stats['p95']:.1f}")

        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


@dataclasses.dataclass
class Args:
    host: str = "192.168.31.160"
    port: int | None = 8000
    api_key: str | None = None
    num_steps: int = 20
    timing_file: pathlib.Path | None = None
    send_interval_s: float = 0.2


def main(args: Args) -> None:
    """Send observations to the server, intercept inference outputs, and execute actions."""
    setup_logging_directory()

    print("Initializing robot environment (ROS2)...")
    env = RobotEnv()
    time.sleep(2)
    print("Initialization complete!")

    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        logger.info("Server metadata: %s", policy.get_server_metadata())
    except requests.exceptions.RequestException as exc:
        raise SystemExit(f"Error communicating with server: {exc}") from exc

    print("Entering simulation loop with server inference...")
    
    # Initialize with real robot state
    initial_frames, initial_state = env.update_obs_window()
    if initial_state is None:
        raise SystemExit("Failed to get initial robot state!")
    
    qpos_12d = np.array(initial_state.get("qpos"))
    current_qpos = expand_qpos_to_14_dims(qpos_12d)
    print(f"[Initial State (12D)] {np.round(qpos_12d, 3)}")
    print(f"[Initial State (14D)] {np.round(current_qpos, 3)}")

    # Prepare initial images
    encoded_images: dict[str, np.ndarray] = {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for server_name, env_name in CAMERA_MAPPING.items():
        img = initial_frames.get(env_name)
        if img is not None and img.size > 0:
            encoded_images[server_name] = prepare_image(img)
            log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}_INITIAL.png")
            cv2.imwrite(log_path, img)
            print(f"  ✓ {server_name}: {img.shape}")
        else:
            raise RuntimeError(f"Missing image: {server_name}")

    print("\n[Closed-Loop Simulation Mode]")
    print("  Vision: Fixed to initial frames")
    print("  State: Evolving based on simulated actions")
    print()

    # Warmup: Send a few dummy observations
    print("[Warmup] Sending 2 warmup observations...")
    dummy_obs = {
        "state": current_qpos.tolist(),
        "images": encoded_images,  # Already base64 strings
        "prompt": PROMPTS,
    }
    for i in range(2):
        try:
            policy.infer(dummy_obs)
            print(f"  ✓ Warmup step {i+1}/2")
        except Exception as e:
            print(f"  Warning: {e}")
    print()

    # Main loop
    timing_recorder = TimingRecorder()
    state_simulator = StateSimulator(dt=0.05)
    step = 0

    try:
        while True:
            if args.num_steps and step >= args.num_steps:
                break

            qpos_state = current_qpos.copy()
            # Print full 14D state
            l_arm = qpos_state[0:6]
            l_grip = qpos_state[6]
            r_arm = qpos_state[7:13]
            r_grip = qpos_state[13]
            print(f"[Step {step + 1}] L_arm={np.round(l_arm, 3)} L_grip={l_grip:.3f} | R_arm={np.round(r_arm, 3)} R_grip={r_grip:.3f}")

            obs = {
                "state": qpos_state.tolist(),
                "images": encoded_images,  # Already base64 strings
                "prompt": PROMPTS,
            }

            # Send inference
            inference_start = time.time()
            action_response = policy.infer(obs)
            timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))

            # Record timings
            for key, value in action_response.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in action_response.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)

            print(f"  Response keys: {list(action_response.keys())}")

            # Extract actions
            actions = action_response.get("actions")
            if actions is None or len(actions) == 0:
                print("  No actions received. Skipping.")
                step += 1
                time.sleep(args.send_interval_s)
                continue

            # Simulate state updates (no real robot execution)
            actions_array = [np.array(act, dtype=np.float32) for act in actions]
            qpos_next = qpos_state
            for i, action in enumerate(actions_array):
                qpos_next = state_simulator.simulate_step(qpos_next, action)
            
            print(f"  Action sequence ({len(actions)} actions) simulated")
            print(f"  State delta: Δmin={np.min(qpos_next - qpos_state):.4f}, Δmax={np.max(qpos_next - qpos_state):.4f}")
            
            current_qpos = qpos_next
            step += 1
            time.sleep(args.send_interval_s)

        # Print statistics
        timing_recorder.print_all_stats()
        if args.timing_file is not None:
            timing_recorder.write_parquet(args.timing_file)

        print("\n✓ Simulation loop finished successfully!")

    except KeyboardInterrupt:
        print("\n⚠ Program interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
