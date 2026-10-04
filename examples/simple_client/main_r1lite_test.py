import dataclasses
import enum
import logging
import pathlib
import time
import sys

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

# 【新增】导入真实机器人环境
# 假设 robot_env.py 在同一目录下
try:
    from examples.simple_client.robot_env import RobotEnv
except ImportError:
    print("Error: Could not import 'RobotEnv'. Please ensure 'robot_env.py' is in the same directory.")
    sys.exit(1)

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    R1LITE = 'r1lite'


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "192.168.31.160"
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

        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

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

        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        logger.info(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)


# 【新增】真实数据适配函数
def _get_real_observation_r1lite(env: RobotEnv) -> dict:
    """
    从 RobotEnv 获取真实数据，并转换为服务器需要的格式。
    """
    # 1. 获取原始数据 (RobotEnv 返回的是 HWC 格式的 OpenCV 图像)
    frames, state_data = env.update_obs_window()
    
    # 如果还没收到数据，返回空或等待（这里简单处理为返回 zeros，防止 crash）
    if state_data is None:
        logger.warning("No state received from robot yet, returning zeros.")
        return {
            "state": np.zeros((14,), dtype=np.float32),
            "images": {
                "cam_head": np.zeros((3, 224, 224), dtype=np.uint8),
                "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
                "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            },
            "prompt": "do something",
        }

    # 2. 图像预处理辅助函数
    def process_image(img_key):
        img = frames.get(img_key)
        if img is None:
            # 如果某个相机没数据，返回全黑图
            return np.zeros((3, 224, 224), dtype=np.uint8)
        
        # 调整大小 (假设服务器需要 224x224，如果原始就是这个尺寸可跳过)
        # img = cv2.resize(img, (224, 224)) 
        
        # 格式转换: (Height, Width, Channel) -> (Channel, Height, Width)
        # OpenCV 读出来是 HWC，PyTorch 模型通常需要 CHW
        img_chw = np.transpose(img, (2, 0, 1))
        return img_chw.astype(np.uint8)

    # 3. 组装数据字典
    # robot_env 中的 key 映射到 server 需要的 key
    # 'image_left' -> 'cam_head' (假设用左眼作为头部主视角)
    return {
        "state": np.array(state_data["qpos"], dtype=np.float32), # 14维关节数据
        "images": {
            "cam_head": process_image("image_left"), 
            "cam_left_wrist": process_image("image_left_wrist"),
            "cam_right_wrist": process_image("image_right_wrist"),
        },
        "prompt": "do something",
    }


def main(args: Args) -> None:
    # 1. 变量初始化
    robot_env_instance = None
    obs_fn = None

    # 2. 根据环境类型配置 obs_fn
    if args.env == EnvMode.R1LITE:
        logger.info("Initializing Real Robot Environment (R1LITE)...")
        # 初始化真实机器人环境（会自动启动 ROS 节点）
        robot_env_instance = RobotEnv()
        # 使用 lambda 创建闭包，将 env 实例绑定到函数调用中
        obs_fn = lambda: _get_real_observation_r1lite(robot_env_instance)
    else:
        # 其他仿真模式使用原来的字典映射逻辑
        obs_fn = {
            EnvMode.ALOHA: _random_observation_aloha,
            EnvMode.ALOHA_SIM: _random_observation_aloha,
            EnvMode.DROID: _random_observation_droid,
            EnvMode.LIBERO: _random_observation_libero,
        }[args.env]

    try:
        policy = _websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
        )
        logger.info(f"Server metadata: {policy.get_server_metadata()}")

        # Send a few observations to make sure the model is loaded.
        logger.info("Sending warmup observations...")
        for _ in range(2):
            policy.infer(obs_fn())

        timing_recorder = TimingRecorder()

        # 3. 主循环执行
        for _ in tqdm.trange(args.num_steps, desc="Running policy"):
            inference_start = time.time()
            
            # 这里调用 obs_fn() 时，如果是 R1LITE，会从真实机器人拉取数据
            obs = obs_fn()
            
            action = policy.infer(obs)
            
            # 【可选】如果你想让机器人动起来，可以在这里调用 control
            # if robot_env_instance:
            #     robot_env_instance.control(action['action'])

            timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
            for key, value in action.get("server_timing", {}).items():
                timing_recorder.record(f"server_{key}", value)
            for key, value in action.get("policy_timing", {}).items():
                timing_recorder.record(f"policy_{key}", value)

        timing_recorder.print_all_stats()

        if args.timing_file is not None:
            timing_recorder.write_parquet(args.timing_file)

    except KeyboardInterrupt:
        logger.info("Test interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # 4. 确保资源释放
        if robot_env_instance is not None:
            logger.info("Shutting down robot environment...")
            robot_env_instance.shutdown()


# 保留原有的仿真数据函数，防止其他模式报错
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