"""Offline-only regression tests; no ROS initialization or network connection.

Run from the checkout with:
    python3 -m unittest discover -s examples/simple_client -p test_fd_client_capture.py -v

The baseline is the locally committed, pre-change deployment entry. Override
FD_BASELINE_MAIN with an equivalent source path when testing an exported bundle.
All observations, policies, actions and clocks used by main() are fake.
"""

from __future__ import annotations

import builtins
import ast
import contextlib
import copy
import datetime
import hashlib
import io
import json
import os
import pathlib
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

import cv2
import numpy as np
import requests


HERE = pathlib.Path(__file__).resolve().parent
ENTRY = HERE / "main_r1lite_closedloop.py"
BASELINE_SHA256 = "24decd17ef9bb38bfcabe2ac910b2ace20574b400408b65989bbb0099a16d716"
SENTINEL_SECRET = "offline-only-secret-do-not-log"


def forbidden(*args, **kwargs):
    raise AssertionError("Real hardware/network access is forbidden in offline tests")


@contextlib.contextmanager
def isolated_imports():
    robot = types.ModuleType("robot_env")
    robot.RobotEnv = forbidden
    robot.__file__ = str(HERE / "robot_env.py")
    ws = types.ModuleType("openpi_client.websocket_client_policy")
    ws.WebsocketClientPolicy = forbidden
    ws.__file__ = str(HERE.parents[1] / "packages/openpi-client/src/openpi_client/websocket_client_policy.py")
    client = types.ModuleType("openpi_client")
    client.websocket_client_policy = ws
    with mock.patch.dict(sys.modules, {"robot_env": robot, "openpi_client": client,
                                      "openpi_client.websocket_client_policy": ws}), \
            mock.patch.object(socket.socket, "connect", forbidden), \
            mock.patch.object(socket.socket, "connect_ex", forbidden), \
            mock.patch.object(socket, "create_connection", forbidden):
        yield


def load_source(name, source, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    with isolated_imports():
        exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def baseline_source():
    override = os.environ.get("FD_BASELINE_MAIN")
    if override:
        source = pathlib.Path(override).read_bytes()
    else:
        source = subprocess.run(
            ["git", "show", "07d2149:examples/simple_client/main_r1lite_closedloop.py"],
            cwd=HERE, check=True, capture_output=True,
        ).stdout
    if hashlib.sha256(source).hexdigest() != BASELINE_SHA256:
        raise AssertionError("Baseline source does not match the audited original SHA256")
    return source.decode("utf-8")


def values(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: values(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [values(val) for val in value]
    return value


def observation(index):
    # These are opaque, fake wire strings, never real images or sensor data.
    return {"state": [float(index + n) for n in range(14)],
            "images": {"cam_head": f"fake-head-{index}", "cam_right_wrist": f"fake-wrist-{index}"},
            "prompt": "offline fake observation"}


def chunk(index=0):
    return {"actions": (np.arange(42, dtype=np.float64).reshape(3, 14) / 17 + index)}


class FakeClock:
    def __init__(self):
        self.current = 1000.0
        self.sleeps = []

    def time(self):
        self.current += 0.001
        return self.current

    perf_counter = time

    def monotonic_ns(self):
        return int(self.current * 1_000_000_000)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.current += seconds


class FixedDatetime(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        result = cls(2026, 9, 17, 12, 0, 0, 123000, tzinfo=datetime.timezone.utc)
        return result.astimezone(tz) if tz else result.replace(tzinfo=None)


class MainHarness:
    def __init__(self, module, *, observations=None, results=None, control_error=None,
                 connect_error=None, metadata_error=None, fd_log_dir=None,
                 dry_run=False, max_steps=2, capture_factory=None, build_error=None,
                 env_error=None, sleep_error=None, log_error=None):
        self.module = module
        self.trace = []
        self.requests = []
        self.request_refs = []
        self.controls = []
        self.logs = []
        self.clock = FakeClock()
        self.observations = copy.deepcopy(observations if observations is not None
                                          else [observation(n) for n in range(3)])
        self.original_observations = copy.deepcopy(self.observations)
        self.results = results if results is not None else [chunk(0), chunk(1), chunk(2), chunk(3)]
        self.control_error = control_error
        self.connect_error = connect_error
        self.metadata_error = metadata_error
        self.build_error = build_error
        self.env_error = env_error
        self.sleep_error = sleep_error
        self.log_error = log_error
        self.fd_log_dir = fd_log_dir
        self.capture_factory = capture_factory
        self.error = None
        self.captures = []
        self.args = module.Args(api_key=SENTINEL_SECRET, host="offline.invalid", max_steps=max_steps,
                                dry_run=dry_run, log_file=pathlib.Path("/offline-fake/legacy.jsonl"),
                                summary_file=pathlib.Path("/offline-fake/summary.jsonl"), log_every=1)
        if hasattr(self.args, "fd_log_dir"):
            self.args.fd_log_dir = fd_log_dir
        self._obs_index = 0
        self._infer_index = 0

    def build_obs(self, env, save_image_dir):
        self.trace.append(("build_obs", save_image_dir))
        if self.build_error is not None and self._obs_index == 1:
            raise self.build_error
        if self._obs_index >= len(self.observations):
            raise AssertionError("Test exhausted observations: control loop failed to terminate")
        obs = self.observations[self._obs_index]
        self._obs_index += 1
        if isinstance(obs, BaseException):
            raise obs
        return obs, [0.1, 0.2, 0.3] if obs is not None else None

    def create_env(self):
        self.trace.append(("env_create",))
        if self.env_error is not None:
            raise self.env_error
        return types.SimpleNamespace(control=self.control, control_eef=self.control_eef)

    def create_policy(self, **kwargs):
        self.trace.append(("policy_create", kwargs))
        if self.connect_error is not None:
            raise self.connect_error
        return types.SimpleNamespace(infer=self.infer, get_server_metadata=self.metadata)

    def metadata(self):
        self.trace.append(("metadata",))
        if self.metadata_error is not None:
            raise self.metadata_error
        return {"offline_mock": True}

    def infer(self, obs):
        self.request_refs.append(obs)
        snapshot = copy.deepcopy(values(obs))
        self.requests.append(snapshot)
        self.trace.append(("infer", snapshot))
        if self._infer_index >= len(self.results):
            raise AssertionError("Unexpected extra inference call")
        result = self.results[self._infer_index]
        self._infer_index += 1
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return result(obs)
        return result

    def control(self, action):
        self.trace.append(("control", copy.deepcopy(action)))
        self.controls.append(copy.deepcopy(action))
        if self.control_error is not None:
            raise self.control_error

    def control_eef(self, action):
        self.trace.append(("control_eef", copy.deepcopy(action)))
        self.controls.append(copy.deepcopy(action))
        if self.control_error is not None:
            raise self.control_error

    def append(self, path, record):
        if self.log_error is not None and record.get("type", "summary") == self.log_error[0]:
            raise self.log_error[1]
        self.logs.append((str(path), copy.deepcopy(record)))

    def run(self):
        module = self.module
        with contextlib.ExitStack() as stack:
            stack.enter_context(isolated_imports())
            stack.enter_context(mock.patch.object(module, "RobotEnv", self.create_env))
            stack.enter_context(mock.patch.object(module._websocket_client_policy, "WebsocketClientPolicy", self.create_policy))
            stack.enter_context(mock.patch.object(module, "_build_obs", self.build_obs))
            stack.enter_context(mock.patch.object(module, "time", self.clock))
            if self.sleep_error is not None:
                stack.enter_context(mock.patch.object(self.clock, "sleep", side_effect=self.sleep_error))
            stack.enter_context(mock.patch.object(module, "datetime", types.SimpleNamespace(
                datetime=FixedDatetime, timezone=datetime.timezone)))
            stack.enter_context(mock.patch.object(module, "setup_logging_directory",
                                                 lambda: self.trace.append(("legacy_image_directory",))))
            stack.enter_context(mock.patch.object(module, "next_episode_index", lambda path: 7))
            stack.enter_context(mock.patch.object(module, "append_jsonl", self.append))
            stack.enter_context(mock.patch.object(module, "prepare_image", forbidden))
            if self.fd_log_dir is not None:
                import fd_client_capture
                stack.enter_context(mock.patch.object(fd_client_capture, "_timestamps", lambda: {
                    "monotonic_ns": self.clock.monotonic_ns(),
                    "wall_time_utc": FixedDatetime.now(datetime.timezone.utc).isoformat(),
                }))
            if self.capture_factory is not None:
                import fd_client_capture
                def make_capture(*args, **kwargs):
                    instance = self.capture_factory(*args, **kwargs)
                    self.captures.append(instance)
                    return instance
                stack.enter_context(mock.patch.object(fd_client_capture, "Capture", make_capture))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            try:
                module.main(self.args)
            except BaseException as error:
                self.error = error
        return self


class OfflineMainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(HERE))
        cls.baseline = load_source("fd_original_entry", baseline_source(), ENTRY)
        cls.current = load_source("fd_modified_entry", ENTRY.read_text(), ENTRY)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fd-offline-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def run_current(self, **kwargs):
        return MainHarness(self.current, **kwargs).run()

    def capture_outputs(self, run):
        candidates = list(run.fd_log_dir.glob("*/manifest.json"))
        self.assertEqual(len(candidates), 1)
        folder = candidates[0].parent
        manifest = json.loads((folder / "manifest.json").read_text())
        events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
        summary = json.loads((folder / "summary.json").read_text())
        return folder, manifest, events, summary

    def assert_behavior_equal(self, baseline, current, *, strip_context=False, compare_logs=True):
        def normalized_trace(run):
            trace = copy.deepcopy(run.trace)
            if strip_context:
                for event in trace:
                    if event[0] == "infer":
                        event[1].pop("fd_context", None)
            return trace
        self.assertEqual(normalized_trace(baseline), normalized_trace(current))
        self.assertEqual(baseline.controls, current.controls)
        self.assertEqual(type(baseline.error), type(current.error))
        if baseline.error is not None:
            self.assertEqual(str(baseline.error), str(current.error))
        if compare_logs:
            self.assertEqual(baseline.logs, current.logs)
        self.assertEqual(baseline.clock.sleeps, current.clock.sleeps)

    def test_disabled_matches_original_values_order_logs_and_no_helper(self):
        baseline = MainHarness(self.baseline).run()
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == "fd_client_capture":
                raise AssertionError("Capture helper imported while disabled")
            return original_import(name, *args, **kwargs)
        with mock.patch.object(builtins, "__import__", guarded_import), \
                mock.patch("uuid.uuid4", forbidden), mock.patch("threading.Thread.start", forbidden), \
                mock.patch.object(pathlib.Path, "mkdir", forbidden):
            current = self.run_current()
        self.assertIsNone(current.error)
        self.assert_behavior_equal(baseline, current)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(current.requests[0], current.requests[1])

    def test_enabled_two_warmups_shallow_copy_first_action_only_and_no_secrets(self):
        baseline = MainHarness(self.baseline).run()
        current = self.run_current(fd_log_dir=self.root / "capture")
        self.assertIsNone(current.error)
        self.assert_behavior_equal(baseline, current, strip_context=True)
        self.assertEqual(len(current.requests), 4)
        self.assertEqual(len(current.controls), 2)
        self.assertEqual(current.observations, current.original_observations)
        contexts = [request["fd_context"] for request in current.requests]
        self.assertEqual([c["request_id"] for c in contexts], [0, 1, 2, 3])
        self.assertEqual([c["phase"] for c in contexts], ["warmup", "warmup", "control", "control"])
        self.assertEqual([c["warmup_index"] for c in contexts], [0, 1, None, None])
        self.assertEqual([c["control_tick"] for c in contexts], [None, None, 0, 1])
        self.assertEqual([c["legacy_step"] for c in contexts], [None, None, 0, 1])
        self.assertEqual([c["source_obs_id"] for c in contexts], [0, 0, 1, 2])
        for request, source in zip(current.request_refs, [current.observations[0]] * 2 + current.observations[1:]):
            self.assertIsNot(request, source)
            for key in ("state", "images", "prompt"):
                self.assertIs(request[key], source[key])
        self.assertEqual(current.controls[0], chunk(2)["actions"][0].astype(np.float32).tolist())
        folder, manifest, events, summary = self.capture_outputs(current)
        text = "\n".join(p.read_text() for p in folder.iterdir() if p.is_file())
        self.assertNotIn(SENTINEL_SECRET, text)
        self.assertNotIn("fake-head", text)
        self.assertNotIn("fake-wrist", text)
        self.assertEqual(summary["task_outcome"], "unknown")
        self.assertEqual(summary["termination_reason"], "max_steps")
        self.assertEqual([e["event"] for e in events].count("infer_call_start"), 4)
        self.assertEqual([e["event"] for e in events].count("control_call_start"), 2)
        for event in events:
            self.assertIsInstance(event["monotonic_ns"], int)
            self.assertIsNotNone(datetime.datetime.fromisoformat(event["wall_time_utc"]).tzinfo)
        for event in events:
            if event["event"] == "control_call_start":
                self.assertEqual(event["row_index"], 0)
                self.assertIn(event["action"], current.controls)
            if event["event"] == "infer_call_return":
                self.assertEqual(event["actions_shape"], [3, 14])
                self.assertEqual(event["fd_context_echo"], "not_returned")
            if event["event"] == "control_call_return":
                self.assertEqual(event["meaning"], "function_return_only")
                self.assertEqual(event["execution_done"], "unknown")
                self.assertEqual(event["hardware_arrival_ack"], "not_recorded")
        first_control = next(i for i, e in enumerate(events) if e["event"] == "control_call_start")
        self.assertEqual(sum(e["event"] == "infer_call_return" for e in events[:first_control]), 3)
        self.assertEqual(manifest["entry"]["path"], str(ENTRY))
        self.assertEqual(manifest["entry"]["sha256"], hashlib.sha256(ENTRY.read_bytes()).hexdigest())

    def test_invalid_observation_empty_action_and_dry_run_preserve_all_counter_axes(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                observations = [observation(0), None, observation(1), observation(2), observation(3)]
                results = [chunk(), chunk(), {"actions": []}, chunk(2), chunk(3)]
                kwargs = dict(observations=observations, results=results, dry_run=dry_run)
                baseline = MainHarness(self.baseline, **kwargs).run()
                current = self.run_current(fd_log_dir=self.root / str(dry_run), **kwargs)
                self.assertIsNone(current.error)
                self.assert_behavior_equal(baseline, current, strip_context=True)
                contexts = [request["fd_context"] for request in current.requests]
                self.assertEqual([c["request_id"] for c in contexts], list(range(5)))
                self.assertEqual([c["source_obs_id"] for c in contexts], [0, 0, 1, 2, 3])
                self.assertEqual([c["control_tick"] for c in contexts], [None, None, 1, 2, 3])
                self.assertEqual([c["legacy_step"] for c in contexts], [None, None, 0, 0, 1])
                _, _, events, summary = self.capture_outputs(current)
                skipped = [e for e in events if e["event"] == "observation_skipped"]
                self.assertEqual(len(skipped), 1)
                self.assertEqual(skipped[0]["control_tick"], 0)
                self.assertEqual(skipped[0]["legacy_step"], 0)
                self.assertIsNone(skipped[0]["request_id"])
                self.assertIsNone(skipped[0]["source_obs_id"])
                action_skipped = [e for e in events if e["event"] == "action_skipped"]
                self.assertEqual([e["reason"] for e in action_skipped],
                                 ["empty_actions", "dry_run", "dry_run"] if dry_run else ["empty_actions"])
                self.assertEqual(len(current.controls), 0 if dry_run else 2)
                self.assertEqual(current.logs[-1][1]["n_steps"], 2)
                self.assertEqual(summary["termination_reason"], "max_steps")
                self.assertEqual(summary["last_ids"], {"request_id": 4, "control_tick": 3, "source_obs_id": 3})

    def test_existing_eepose_dispatch_is_preserved(self):
        with mock.patch.object(self.baseline, "CONTROL_MODE", "eepose"), \
                mock.patch.object(self.current, "CONTROL_MODE", "eepose"):
            baseline = MainHarness(self.baseline).run()
            current = self.run_current(fd_log_dir=self.root / "eepose")
        self.assertIsNone(current.error)
        self.assert_behavior_equal(baseline, current, strip_context=True)
        self.assertEqual(sum(event[0] == "control_eef" for event in current.trace), 2)

    def test_old_server_missing_echo_and_matching_or_mismatching_echo_never_change_actions(self):
        for mode in ("missing", "matching", "mismatch"):
            with self.subTest(mode=mode):
                def response(obs):
                    result = chunk()
                    if mode != "missing":
                        result["fd_context"] = dict(obs["fd_context"])
                        if mode == "mismatch":
                            result["fd_context"]["request_id"] += 100
                    return result
                current = self.run_current(fd_log_dir=self.root / mode, results=[response] * 4)
                self.assertIsNone(current.error)
                self.assertEqual(len(current.controls), 2)
                self.assertEqual(current.controls, [chunk()["actions"][0].astype(np.float32).tolist()] * 2)
                _, _, _, summary = self.capture_outputs(current)
                self.assertEqual(summary["complete"], mode != "mismatch")

    def test_exceptions_and_ctrl_c_keep_original_exit_and_legacy_log_behavior(self):
        scenarios = {
            "robot_init_error": dict(env_error=RuntimeError("fake init failed")),
            "robot_init_ctrl_c": dict(env_error=KeyboardInterrupt()),
            "init_sleep_ctrl_c": dict(sleep_error=KeyboardInterrupt()),
            "connect_request_error": dict(connect_error=requests.exceptions.RequestException("fake connect failure")),
            "connect_error": dict(connect_error=RuntimeError("fake connect failure")),
            "connect_ctrl_c": dict(connect_error=KeyboardInterrupt()),
            "metadata_ctrl_c": dict(metadata_error=KeyboardInterrupt()),
            "warmup_infer_error": dict(results=[RuntimeError("infer warmup fake")]),
            "warmup_ctrl_c": dict(results=[KeyboardInterrupt()]),
            "control_infer_error": dict(results=[chunk(), chunk(), RuntimeError("infer control fake")]),
            "control_infer_ctrl_c": dict(results=[chunk(), chunk(), KeyboardInterrupt()]),
            "control_error": dict(control_error=RuntimeError("control fake partial publish")),
            "control_ctrl_c": dict(control_error=KeyboardInterrupt()),
            "observation_error": dict(build_error=RuntimeError("observation fake")),
            "observation_ctrl_c": dict(build_error=KeyboardInterrupt()),
            "initial_observation_absent": dict(observations=[None]),
            "episode_start_ctrl_c": dict(log_error=("episode_start", KeyboardInterrupt())),
            "legacy_step_error": dict(log_error=("step", OSError("fake legacy log failure"))),
            "legacy_step_ctrl_c": dict(log_error=("step", KeyboardInterrupt())),
            "episode_end_ctrl_c": dict(log_error=("episode_end", KeyboardInterrupt())),
            "summary_ctrl_c": dict(log_error=("summary", KeyboardInterrupt())),
        }
        for name, kwargs in scenarios.items():
            for enabled in (False, True):
                with self.subTest(scenario=name, enabled=enabled):
                    baseline = MainHarness(self.baseline, **kwargs).run()
                    current = self.run_current(fd_log_dir=self.root / name if enabled else None, **kwargs)
                    self.assert_behavior_equal(baseline, current, strip_context=enabled)
                    if enabled:
                        _, _, events, summary = self.capture_outputs(current)
                        self.assertEqual(summary["task_outcome"], "unknown")
                        if "ctrl_c" in name:
                            self.assertEqual(summary["termination_reason"], "ctrl_c")
                        errors = [e for e in events if e["event"] in ("infer_call_error", "control_call_error")]
                        if name in ("warmup_infer_error", "warmup_ctrl_c", "control_infer_error", "control_infer_ctrl_c",
                                    "control_error", "control_ctrl_c"):
                            self.assertEqual(len(errors), 1)
                            self.assertEqual(summary["exception_stage"], errors[0]["exception_stage"])
                            expected_id = 0 if name.startswith("warmup") else 2
                            self.assertEqual(errors[0]["request_id"], expected_id)
                            if name in ("control_error", "control_ctrl_c"):
                                self.assertTrue(errors[0]["partial_publish_possible"])
                        if name in ("legacy_step_error", "legacy_step_ctrl_c"):
                            self.assertEqual(len(current.controls), 1)
                            self.assertEqual(summary["final_legacy_step"], 0)
                            self.assertEqual(summary["exception_stage"], "legacy_logging")

    def test_repeat_start_never_overwrites_artifacts(self):
        fd_dir = self.root / "repeated"
        first = self.run_current(fd_log_dir=fd_dir)
        self.assertIsNone(first.error)
        before = {str(p.relative_to(fd_dir)): p.read_bytes() for p in fd_dir.rglob("*") if p.is_file()}
        second = self.run_current(fd_log_dir=fd_dir)
        self.assertIsNone(second.error)
        self.assertEqual(len(list(fd_dir.glob("*/manifest.json"))), 2)
        for relative, content in before.items():
            self.assertEqual((fd_dir / relative).read_bytes(), content)
        self.assertEqual(first.controls, second.controls)

    def test_event_write_failure_preserves_control_and_original_infer_exception(self):
        import fd_client_capture
        for infer_error in (False, True):
            with self.subTest(infer_error=infer_error):
                kwargs = {"results": [chunk(), chunk(), RuntimeError("original infer failure")]} if infer_error else {}
                baseline = MainHarness(self.baseline, **kwargs).run()
                with mock.patch.object(fd_client_capture.Capture, "_write_line", side_effect=OSError("fake disk full")):
                    current = self.run_current(fd_log_dir=self.root / str(infer_error), **kwargs)
                self.assert_behavior_equal(baseline, current, strip_context=True)
                _, _, _, summary = self.capture_outputs(current)
                self.assertFalse(summary["complete"])
                self.assertGreater(summary["write_errors"], 0)
                self.assertGreater(summary["dropped_events"], 0)

    def test_output_directory_failure_does_not_change_control(self):
        bad_root = self.root / "not-a-directory"
        bad_root.write_text("existing artifact must stay unchanged")
        baseline = MainHarness(self.baseline).run()
        current = self.run_current(fd_log_dir=bad_root)
        self.assertIsNone(current.error)
        self.assert_behavior_equal(baseline, current, strip_context=True)
        self.assertEqual(bad_root.read_text(), "existing artifact must stay unchanged")
        self.assertEqual(list(self.root.glob("**/summary.json")), [])

    def test_queue_overflow_and_bounded_flush_do_not_change_control(self):
        import fd_client_capture
        capture_cls = fd_client_capture.Capture
        gate = threading.Event()
        entered = threading.Event()
        original_write = capture_cls._write_line
        def blocked_write(instance, stream, line):
            entered.set()
            if not gate.wait(3):
                raise OSError("offline writer gate timeout")
            return original_write(instance, stream, line)
        captures = []
        def factory(*args, **kwargs):
            instance = capture_cls(*args, **kwargs, queue_capacity=1, flush_timeout=0.02)
            captures.append(instance)
            self.assertTrue(entered.wait(2), "Fake writer did not reach deterministic blocking point")
            return instance
        baseline = MainHarness(self.baseline).run()
        try:
            with mock.patch.object(capture_cls, "_write_line", blocked_write):
                started = time.monotonic()
                current = self.run_current(fd_log_dir=self.root / "overflow", capture_factory=factory)
                elapsed = time.monotonic() - started
                self.assertIsNone(current.error)
                self.assert_behavior_equal(baseline, current, strip_context=True)
                self.assertLess(elapsed, 2.0, "Capture flush blocked the mock main unexpectedly")
                self.assertFalse((captures[0].session_dir / "summary.json").exists())
                gate.set()
                captures[0]._thread.join(2)
        finally:
            gate.set()
            for capture in captures:
                capture._thread.join(2)
        _, _, _, summary = self.capture_outputs(current)
        self.assertFalse(summary["complete"])
        self.assertTrue(summary["flush_timed_out"])
        self.assertGreater(summary["dropped_events"], 0)

    def test_uuid_collision_keeps_existing_capture_untouched(self):
        import fd_client_capture
        with mock.patch.object(fd_client_capture.uuid, "uuid4", return_value="offline-collision"):
            first = self.run_current(fd_log_dir=self.root / "collision")
            folder, _, _, _ = self.capture_outputs(first)
            before = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
            second = self.run_current(fd_log_dir=self.root / "collision")
        self.assertIsNone(first.error)
        self.assertIsNone(second.error)
        self.assertEqual(first.controls, second.controls)
        self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()})

    def test_preprocessing_source_is_byte_for_byte_unchanged(self):
        baseline = baseline_source()
        current = ENTRY.read_text()
        for name in ("prepare_image", "_build_obs"):
            def source_function(source):
                node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
                return ast.get_source_segment(source, node)
            self.assertEqual(source_function(baseline), source_function(current))

    def test_real_build_obs_with_fake_frames_matches_baseline(self):
        frames = {"image_left": np.zeros((9, 13, 3), dtype=np.uint8),
                  "image_left_wrist": np.full((7, 11, 3), 19, dtype=np.uint8),
                  "image_right_wrist": np.full((5, 17, 3), 127, dtype=np.uint8)}
        state = {"qpos": np.arange(14, dtype=np.float32), "eef_pose": np.arange(16, dtype=np.float32)}
        env = types.SimpleNamespace(update_obs_window=lambda: (frames, state))
        with isolated_imports():
            old = self.baseline._build_obs(env, None)
            current = self.current._build_obs(env, None)
        self.assertEqual(old, current)
        self.assertEqual(current[0]["state"][12], 12.0)
        self.assertEqual(set(current[0]["images"]), {"cam_head", "cam_left_wrist", "cam_right_wrist"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
