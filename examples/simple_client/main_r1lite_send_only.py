import dataclasses
import datetime
import logging
import os
import pathlib
import time
import base64
import io
import cv2
import numpy as np
import requests
import tyro
import tqdm
import polars as pl
import rich
import rich.console
import rich.table

from openpi_client import websocket_client_policy as _websocket_client_policy
from examples.simple_client.robot_env import RobotEnv

logger = logging.getLogger(__name__)

PROMPTS = "Pick up the peach and place it into the tray"
LOG_IMAGE_DIR = "./log_images_r1lite_ros2"

# Robot control mode: 'eepose' for end-effector pose, 'joint' for joint angles
CONTROL_MODE = "joint"

CAMERA_MAPPING = {
    "cam_head": "image_right",      # Server expects "cam_head", maps to robot's "image_right"
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
    """
    扩展12维qpos到14维标准格式。
    
    robot_env.py返回的12维格式:
    [R1-R5, R_grip, L1-L5, L_grip]
    
    期望的14维标准格式 (r1lite_policy.py):
    [L1-L6, L_grip, R1-R6, R_grip]
    
    注意: 这个映射假设机器人只有5个活跃的关节+1个夹爪
    第6个关节位置补0或补原值（需要根据实际情况调整）
    """
    if len(qpos_12d) == 14:
        # 已经是14维，直接返回
        return qpos_12d.copy()
    elif len(qpos_12d) == 12:
        # 从12维扩展到14维
        # robot_env格式: [R1-R5(5), R_grip(1), L1-L5(5), L_grip(1)]
        #                  0-4          5         6-10       11
        # 目标格式: [L1-L6(6), L_grip(1), R1-R6(6), R_grip(1)]
        #             0-5        6        7-12      13
        
        expanded = np.zeros(14)
        
        # 左臂: L1-L5 从qpos的6-10位置拷贝到0-4
        expanded[0:5] = qpos_12d[6:11]
        # 左臂第6个关节（缺失），使用0或上一个值
        expanded[5] = 0.0  # 或者使用 qpos_12d[10]
        # 左夹爪: 从qpos的11位置拷贝到6
        expanded[6] = qpos_12d[11]
        
        # 右臂: R1-R5 从qpos的0-4位置拷贝到7-11
        expanded[7:12] = qpos_12d[0:5]
        # 右臂第6个关节（缺失），使用0或上一个值
        expanded[12] = 0.0  # 或者使用 qpos_12d[4]
        # 右夹爪: 从qpos的5位置拷贝到13
        expanded[13] = qpos_12d[5]
        
        return expanded
    else:
        raise ValueError(f"Unexpected qpos dimension: {len(qpos_12d)}, expected 12 or 14")


def setup_logging_directory() -> None:
    if not os.path.exists(LOG_IMAGE_DIR):
        os.makedirs(LOG_IMAGE_DIR)
        print(f"Created log directory: {LOG_IMAGE_DIR}")


class DatasetFrameLoader:
    """Load image frames from a dataset video file or image sequence folder."""
    
    def __init__(self, dataset_path: pathlib.Path | None = None):
        """
        Initialize dataset frame loader.
        
        Args:
            dataset_path: Path to video file (mp4/avi) or folder with image files
        """
        self.video_capture = None
        self.image_files = []
        self.current_frame_idx = 0
        self.loaded = False
        
        if dataset_path is not None:
            self.load_dataset(dataset_path)
    
    def load_dataset(self, dataset_path: pathlib.Path) -> None:
        """Load dataset from video or image folder."""
        dataset_path = pathlib.Path(dataset_path)
        
        # Try as video file first
        if dataset_path.is_file() and dataset_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"Loading video dataset: {dataset_path}")
            self.video_capture = cv2.VideoCapture(str(dataset_path))
            if not self.video_capture.isOpened():
                raise RuntimeError(f"Failed to open video: {dataset_path}")
            self.loaded = True
            self.current_frame_idx = 0
            print(f"  Video loaded. Total frames: {int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))}")
        
        # Try as image folder
        elif dataset_path.is_dir():
            print(f"Loading image sequence from: {dataset_path}")
            # Find all image files
            image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            image_files = []
            for pattern in image_patterns:
                image_files.extend(sorted(dataset_path.glob(pattern)))
            
            if not image_files:
                raise RuntimeError(f"No image files found in {dataset_path}")
            
            self.image_files = sorted(image_files)  # Sort by name
            self.loaded = True
            self.current_frame_idx = 0
            print(f"  Found {len(self.image_files)} images")
        else:
            raise RuntimeError(f"Dataset path not found or unsupported format: {dataset_path}")
    
    def get_next_frame(self) -> dict[str, np.ndarray] | None:
        """
        Get next frame(s) from dataset.
        
        Returns:
            Dictionary with camera names as keys and images as values (BGR)
            Returns None if dataset exhausted
        """
        if not self.loaded:
            return None
        
        # Load from video
        if self.video_capture is not None:
            ret, frame = self.video_capture.read()
            if not ret:
                return None  # End of video
            
            # Return frame mapped to camera names
            # For simplicity, use same frame for all cameras (can be extended for multi-view)
            return {
                "image_right": frame,
                "image_left_wrist": frame,
                "image_right_wrist": frame,
            }
        
        # Load from image folder
        elif self.image_files:
            if self.current_frame_idx >= len(self.image_files):
                return None  # All images loaded
            
            img_path = self.image_files[self.current_frame_idx]
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                self.current_frame_idx += 1
                return None
            
            self.current_frame_idx += 1
            return {
                "image_right": img,
                "image_left_wrist": img,
                "image_right_wrist": img,
            }
        
        return None
    
    def reset(self) -> None:
        """Reset to beginning of dataset."""
        if self.video_capture is not None:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
    
    def __del__(self):
        """Cleanup."""
        if self.video_capture is not None:
            self.video_capture.release()


class StateSimulator:
    """Simulates state changes based on actions without executing on real robot."""

    def __init__(self, dt: float = 0.05):
        """
        Args:
            dt: Time step for state integration (seconds)
        """
        self.dt = dt
        # Joint velocity scaling factor (tune based on your robot's specs)
        self.velocity_scale = 0.5

    def simulate_step(self, qpos_current: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Simulate one action step with proper 14D state/action handling.
        
        维度定义 (14D标准格式):
        State:  [L1-L6(0-5), L_grip(6), R1-R6(7-12), R_grip(13)]
        Action: [R1-R6(0-5), R_grip(6), L1-L6(7-12), L_grip(13)]
        
        Args:
            qpos_current: Current joint positions [12 or 14,]
            action: Action command [14,] from policy
            
        Returns:
            Simulated next joint positions [14,]
        """
        # 扩展qpos到14维（如果是12维）
        if len(qpos_current) == 12:
            qpos = expand_qpos_to_14_dims(qpos_current)
        elif len(qpos_current) == 14:
            qpos = qpos_current.copy()
        else:
            raise ValueError(f"Expected qpos to be 12D or 14D, got {len(qpos_current)}D")
        
        # 验证action是14维
        if len(action) != 14:
            raise ValueError(f"Expected action to be 14D, got {len(action)}D")
        
        qpos_next = qpos.copy()
        
        # 关键：action的维度顺序与state不同！
        # Action: [R1-R6(0-5), R_grip(6), L1-L6(7-12), L_grip(13)]
        # State:  [L1-L6(0-5), L_grip(6), R1-R6(7-12), R_grip(13)]
        
        # 左臂关节 (State 0-5) <- Action 7-12
        qpos_next[0:6] = qpos[0:6] + action[7:13] * self.dt * self.velocity_scale
        
        # 左夹爪 (State 6) <- Action 13
        qpos_next[6] = action[13]
        
        # 右臂关节 (State 7-12) <- Action 0-5
        qpos_next[7:13] = qpos[7:13] + action[0:6] * self.dt * self.velocity_scale
        
        # 右夹爪 (State 13) <- Action 6
        qpos_next[13] = action[6]
        
        # 关节夹爪值的范围: [-π, π]
        qpos_next = np.clip(qpos_next, -np.pi, np.pi)
        
        return qpos_next

    def simulate_action_sequence(
        self, 
        qpos_start: np.ndarray, 
        actions: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Simulate a sequence of actions.
        
        Args:
            qpos_start: Initial joint positions
            actions: List of action commands
            
        Returns:
            Tuple of (final_qpos, trajectory of qpos at each step)
        """
        trajectory = [qpos_start.copy()]
        qpos = qpos_start.copy()
        
        for action in actions:
            qpos = self.simulate_step(qpos, action)
            trajectory.append(qpos.copy())
        
        return qpos, trajectory


class ImageTransformSimulator:
    """Simulates image changes based on state changes without real robot motion."""

    def __init__(self, transform_scale: float = 0.3):
        """
        Args:
            transform_scale: Scale factor for image transformations [0.0-1.0]
                            Higher values = more aggressive image changes
        """
        self.transform_scale = transform_scale

    def compute_state_change_magnitude(
        self, 
        qpos_before: np.ndarray, 
        qpos_after: np.ndarray
    ) -> float:
        """
        Compute magnitude of state change normalized to [0, 1].
        
        Args:
            qpos_before: Initial joint positions [14,]
            qpos_after: Final joint positions [14,]
            
        Returns:
            Normalized magnitude (0.0 to 1.0)
        """
        change = np.abs(qpos_after - qpos_before)
        max_change = np.max(change)
        # Normalize to [0, 1] assuming max joint change is ~π per step
        magnitude = np.clip(max_change / np.pi, 0, 1)
        return float(magnitude)

    def transform_image_with_motion(
        self,
        img: np.ndarray,
        state_change_magnitude: float,
        change_direction: np.ndarray
    ) -> np.ndarray:
        """
        Apply perspective/motion blur to simulate image change from robot motion.
        
        Args:
            img: Input image (H, W, 3) BGR uint8
            state_change_magnitude: Magnitude of state change [0, 1]
            change_direction: Normalized direction of state change [14,]
            
        Returns:
            Transformed image (H, W, 3) BGR uint8
        """
        if state_change_magnitude < 0.01:
            return img.copy()

        img_float = img.astype(np.float32) / 255.0
        h, w = img.shape[:2]

        # Determine primary direction of motion change
        # Assuming first 7 dims are right arm, last 7 are left arm
        right_arm_change = np.mean(np.abs(change_direction[:7]))
        left_arm_change = np.mean(np.abs(change_direction[7:]))

        # Apply different transformations based on which arm moved more
        transform_intensity = state_change_magnitude * self.transform_scale

        if right_arm_change > left_arm_change:
            # Right arm moved: apply motion blur to left/center region
            img_float = self._apply_motion_blur(img_float, intensity=transform_intensity, direction="left")
        else:
            # Left arm moved: apply motion blur to right/center region
            img_float = self._apply_motion_blur(img_float, intensity=transform_intensity, direction="right")

        # Add subtle brightness/contrast changes
        img_float = np.clip(img_float * (1.0 - transform_intensity * 0.1), 0, 1)

        # Convert back to uint8
        return (img_float * 255).astype(np.uint8)

    def _apply_motion_blur(
        self,
        img: np.ndarray,
        intensity: float,
        direction: str = "left"
    ) -> np.ndarray:
        """
        Apply subtle motion blur effect to simulate camera perspective change.
        
        Args:
            img: Image as float [0, 1]
            intensity: Blur intensity [0, 1]
            direction: Direction of blur ("left", "right", "up", "down")
            
        Returns:
            Blurred image
        """
        h, w = img.shape[:2]
        kernel_size = max(1, int(3 * intensity)) | 1  # Must be odd

        # Create a simple box blur effect in specified direction
        if kernel_size <= 1:
            return img

        # Apply Gaussian-like blur by averaging neighboring pixels
        blurred = img.copy()

        if direction == "left":
            for i in range(1, kernel_size):
                blurred[:, :-i] = (blurred[:, :-i] + img[:, i:]) / 2
        elif direction == "right":
            for i in range(1, kernel_size):
                blurred[:, i:] = (blurred[:, i:] + img[:, :-i]) / 2
        elif direction == "up":
            for i in range(1, kernel_size):
                blurred[:-i, :] = (blurred[:-i, :] + img[i:, :]) / 2
        elif direction == "down":
            for i in range(1, kernel_size):
                blurred[i:, :] = (blurred[i:, :] + img[:-i, :]) / 2

        return blurred

    def augment_images_from_state_change(
        self,
        images: dict[str, np.ndarray],
        qpos_before: np.ndarray,
        qpos_after: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Apply transformations to all images based on state change.
        
        Args:
            images: Dict of images {'cam_name': img}
            qpos_before: Initial state
            qpos_after: Final state after actions
            
        Returns:
            Dict of transformed images
        """
        magnitude = self.compute_state_change_magnitude(qpos_before, qpos_after)
        direction = qpos_after - qpos_before
        direction = direction / (np.linalg.norm(direction) + 1e-6)  # Normalize

        augmented_images = {}
        for cam_name, img in images.items():
            if img is not None and img.size > 0:
                augmented_images[cam_name] = self.transform_image_with_motion(
                    img, magnitude, direction
                )
            else:
                augmented_images[cam_name] = img

        return augmented_images



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


@dataclasses.dataclass
class Args:
    host: str = "192.168.31.160"
    port: int | None = 8000
    api_key: str | None = None
    num_steps: int = 20
    timing_file: pathlib.Path | None = None
    send_interval_s: float = 0.2
    # New parameter for image handling
    use_image_augmentation: bool = False
    """If True: simulate image changes based on state delta (动态图像变换)
       If False: use fixed initial images (当前默认，固定初始图像)"""
    
    dataset_video_path: pathlib.Path | None = None
    """Path to a dataset video file or image sequence folder
       If provided, subsequent frames will be extracted from this dataset
       Format: video file (mp4/avi) or folder with numbered images (e.g., frame_0001.png)"""


def main(args: Args) -> None:
    """Send observations to the server, intercept inference outputs, and execute actions."""
    setup_logging_directory()

    print("Initializing robot environment (ROS2)...")
    env = RobotEnv()
    time.sleep(2)
    print("Initialization complete!")

    # Initialize policy connection
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
    step = 0
    state_simulator = StateSimulator(dt=0.05)
    image_simulator = ImageTransformSimulator(transform_scale=0.3)  # 0.3 = subtle changes
    
    # Initialize dataset frame loader if provided
    dataset_loader = None
    if args.dataset_video_path is not None:
        print(f"Loading dataset from: {args.dataset_video_path}")
        dataset_loader = DatasetFrameLoader(args.dataset_video_path)
        print("Dataset loaded successfully!")
    
    # Initialize with real robot state, then use simulated states
    initial_frames, initial_state = env.update_obs_window()
    if initial_state is None:
        raise SystemExit("Failed to get initial robot state!")
    
    # Expand initial state to 14D if needed
    qpos_12d = np.array(initial_state.get("qpos"))
    current_qpos = expand_qpos_to_14_dims(qpos_12d)
    print(f"[Initial State (12D)] {np.round(qpos_12d, 3)}")
    print(f"[Initial State (14D)] {np.round(current_qpos, 3)}")

    # Variables to track state history for image augmentation
    qpos_before_prev = current_qpos.copy()
    qpos_simulated_prev = current_qpos.copy()

    # **IMPORTANT**: Prepare initial images ONCE and reuse them for the entire loop
    # This implements a closed-loop simulation: fixed vision + evolving state
    initial_encoded_images: dict[str, str] = {}  # Base64 strings now
    initial_raw_images: dict[str, np.ndarray] = {}  # Keep raw images for augmentation
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for server_name, env_name in CAMERA_MAPPING.items():
        img = initial_frames.get(env_name)
        if img is not None and img.size > 0:
            initial_raw_images[server_name] = img.copy()  # Store raw BGR image
            initial_encoded_images[server_name] = prepare_image(img)
            log_path = os.path.join(LOG_IMAGE_DIR, f"{server_name}_{timestamp}_INITIAL.png")
            cv2.imwrite(log_path, img)
            print(f"  ✓ Caching initial image: {server_name}")
    
    # Verify we have at least one image
    if not initial_encoded_images:
        raise SystemExit("No valid images received from robot environment!")

    print("[Closed-Loop Simulation Mode]")
    if dataset_loader is not None:
        print("  - Vision: DATASET (from pre-recorded video/images)")
        print("  - State: Evolving based on simulated actions")
        print("  - This uses real visual data from a dataset with simulated state evolution")
    elif args.use_image_augmentation:
        print("  - Vision: AUGMENTED (motion blur based on state delta)")
        print("  - State: Evolving based on simulated actions")
        print("  - This simulates realistic camera perspective changes from arm motion")
    else:
        print("  - Vision: Fixed to initial frames (constant)")
        print("  - State: Evolving based on simulated actions")
        print("  - This tests model robustness to state changes with fixed visual feedback")
        print("  - Tip: Use --dataset-video-path to load dataset video/images")
    print()
    
    # Warmup: Send a few dummy observations to ensure model is loaded
    print("[Warmup] Sending 2 warmup observations to load model...")
    timing_recorder = TimingRecorder()
    dummy_obs = {
        "state": current_qpos.tolist(),
        "images": initial_encoded_images,
        "prompt": PROMPTS,
    }
    try:
        for i in range(2):
            policy.infer(dummy_obs)
            print(f"  Warmup {i+1}/2 complete")
    except Exception as e:
        print(f"  Warning during warmup: {e}")
    print("Warmup complete. Starting main loop.\n")

    try:
        while True:
            if args.num_steps and step >= args.num_steps:
                break

            # Use current simulated state (after first iteration, this is simulated)
            qpos_state = current_qpos.copy()
            qpos_before = qpos_state.copy()
            
            print(f"\n[Step {step + 1}] Using state: {np.round(qpos_state[:3], 3)}... (simulated)")

            # Prepare images for this iteration
            if dataset_loader is not None and step > 0:
                # Use dataset frames for subsequent steps
                print("  [DATASET FRAMES] Loading from dataset...")
                dataset_frames = dataset_loader.get_next_frame()
                if dataset_frames is not None:
                    encoded_images = {}
                    for server_name, env_name in CAMERA_MAPPING.items():
                        dataset_img = dataset_frames.get(env_name)
                        if dataset_img is not None:
                            encoded_images[server_name] = prepare_image(dataset_img)
                            print(f"    ✓ Loaded {server_name} from dataset (step {step + 1})")
                        else:
                            # Fallback to initial image if not available in dataset
                            encoded_images[server_name] = initial_encoded_images[server_name]
                else:
                    # Dataset exhausted, switch back to initial images
                    print("  [DATASET EXHAUSTED] Switching back to initial images")
                    dataset_loader = None
                    encoded_images = initial_encoded_images.copy()
            elif args.use_image_augmentation and step > 0:
                # Compute state change from previous iteration
                # (from qpos_before_prev to qpos_simulated_prev)
                prev_state_delta = qpos_simulated_prev - qpos_before_prev
                
                # Apply image augmentation based on previous state evolution
                print("  [IMAGE AUGMENTATION] Applying motion blur transform...")
                encoded_images = {}
                for server_name, raw_img in initial_raw_images.items():
                    transformed_img = image_simulator.transform_image_with_motion(
                        raw_img,
                        state_change_magnitude=image_simulator.compute_state_change_magnitude(
                            qpos_before_prev, qpos_simulated_prev
                        ),
                        change_direction=prev_state_delta
                    )
                    encoded_images[server_name] = prepare_image(transformed_img)
                    print(f"    ✓ Augmented {server_name}")
            else:
                # Use fixed initial images
                encoded_images = initial_encoded_images.copy()

            # Prepare observation - images are already base64 strings
            obs = {
                "state": qpos_state.tolist(),  # Use simulated state (convert to list for JSON)
                "images": encoded_images,       # Base64 encoded strings
                "prompt": PROMPTS,
            }

            # Send inference request to server
            inference_start = time.time()
            action_response = policy.infer(obs)
            timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))

            # Record server timing if available
            for key, value in action_response.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in action_response.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)

            print(f"  Server response: {action_response}")

            # Extract actions
            actions = action_response.get("actions")
            if actions is None or len(actions) == 0:
                print("  No actions received from the model. Skipping.")
                step += 1
                time.sleep(args.send_interval_s)
                continue

            # Convert actions to numpy arrays
            actions_array = [np.array(act, dtype=np.float32) for act in actions]

            # Simulate state changes WITHOUT executing on real robot
            print("  [CLIENT SIMULATION] Computing state trajectory...")
            qpos_simulated, trajectory = state_simulator.simulate_action_sequence(
                qpos_before, actions_array
            )

            # Log simulated trajectory
            print("    State trajectory (14D: L_arm[0:6], L_grip[6], R_arm[7:12], R_grip[13]):")
            for i, traj_state in enumerate(trajectory):
                arm_left = traj_state[0:6]
                grip_left = traj_state[6]
                arm_right = traj_state[7:13]
                grip_right = traj_state[13]
                print(f"      Step[{i}] L_arm:{np.round(arm_left[:2], 3)} L_grip:{grip_left:.3f} R_arm:{np.round(arm_right[:2], 3)} R_grip:{grip_right:.3f}")

            # Calculate state change
            simulated_change = qpos_simulated - qpos_before
            print(f"  [Simulated Delta] Δmin={np.min(simulated_change):.4f}, Δmax={np.max(simulated_change):.4f}, Δmean={np.mean(np.abs(simulated_change)):.4f}")

            print("  Action simulation complete (vision fixed, state evolved).")
            print("=" * 60)

            # Update history for next iteration's image augmentation
            qpos_before_prev = qpos_before.copy()
            qpos_simulated_prev = qpos_simulated.copy()
            
            # Update current state to simulated state for next iteration
            current_qpos = qpos_simulated.copy()

            step += 1
            time.sleep(args.send_interval_s)

        # Print final statistics
        timing_recorder.print_all_stats()

        if args.timing_file is not None:
            timing_recorder.write_parquet(args.timing_file)

        print("Send-only loop finished.")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
