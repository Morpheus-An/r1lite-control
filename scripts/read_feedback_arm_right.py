#!/usr/bin/env python3
"""Read ROS2 topic /hdas/feedback_arm_right.

Usage:
  1) If you know the Python message class (e.g. sensor_msgs.msg.JointState):
     ./scripts/read_feedback_arm_right.py --py-msg sensor_msgs.msg.JointState

  2) If you don't know the message type, use the ROS2 CLI fallback:
     ./scripts/read_feedback_arm_right.py --cli

The script supports printing messages to stdout and optionally writing to a file.
"""

import argparse
import importlib
import subprocess
import sys
import time
from typing import Optional


def run_ros2_cli(topic: str, output: Optional[str] = None) -> None:
    """Fallback that shells out to `ros2 topic echo` and streams output."""
    cmd = ["ros2", "topic", "echo", topic]
    print(f"Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        with (open(output, "a") if output else None) as f:
            for line in proc.stdout:  # type: ignore[attr-defined]
                print(line, end="")
                if f:
                    f.write(line)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


def subscribe_with_rclpy(topic: str, py_msg: str, output: Optional[str] = None, timeout: Optional[int] = None) -> None:
    """Subscribe to topic using rclpy and a provided Python message class path.

    py_msg should be like 'sensor_msgs.msg.JointState' (module path then class).
    """
    try:
        import rclpy
        from rclpy.node import Node
    except ImportError as e:
        print("rclpy is not installed or not sourced. Make sure you source your ROS2 workspace.")
        raise

    module_name, class_name = py_msg.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    msg_cls = getattr(mod, class_name)

    class MinimalNode(Node):
        def __init__(self):
            super().__init__("read_feedback_arm_right")
            # Use default QoS; adjust if you know a specific policy is needed.
            self.sub = self.create_subscription(msg_cls, topic, self.cb, 10)
            self._output = open(output, "a") if output else None
            self._start = time.time()

        def cb(self, msg):
            # Print representation and optionally write to file.
            line = repr(msg)
            print(line)
            if self._output:
                self._output.write(line + "\n")
            # honor timeout if requested
            if timeout is not None and (time.time() - self._start) > timeout:
                print("Timeout reached, shutting down subscriber.")
                rclpy.shutdown()

        def destroy(self):
            if self._output:
                self._output.close()

    rclpy.init()
    node = MinimalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Read /hdas/feedback_arm_right")
    parser.add_argument("--topic", default="/hdas/feedback_arm_right", help="Topic to subscribe to")
    parser.add_argument("--py-msg", default=None, help="Python message class path, e.g. sensor_msgs.msg.JointState")
    parser.add_argument("--cli", action="store_true", help="Use ros2 CLI fallback (ros2 topic echo)")
    parser.add_argument("--output", default=None, help="Optional path to append printed messages")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout seconds for rclpy mode")
    args = parser.parse_args()

    if args.cli or args.py_msg is None:
        print("Using ROS2 CLI fallback (ros2 topic echo). If you want rclpy subscription, pass --py-msg module.Class")
        try:
            run_ros2_cli(args.topic, args.output)
        except FileNotFoundError:
            print("ros2 CLI not found in PATH. Make sure ROS2 is installed and your environment is sourced.")
            sys.exit(2)
        return

    # Use rclpy subscription when py-msg is provided
    subscribe_with_rclpy(args.topic, args.py_msg, args.output, args.timeout)


if __name__ == "__main__":
    main()
