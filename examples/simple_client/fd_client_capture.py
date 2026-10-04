"""Optional, bounded metadata capture; never owns policy or robot lifecycle.

Only the writer thread accesses files. A missing summary, ``complete=false``,
or ``flush_timed_out=true`` means the capture must not be treated as complete.
No images, states, prompts, full action chunks, or exception messages are kept.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import queue
import sys
import threading
import time
import uuid


def _timestamps() -> dict:
    """Separate hook for fake clocks; writer deadlines use time.monotonic()."""
    return {
        "monotonic_ns": time.monotonic_ns(),
        "wall_time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _exception_type(exc: BaseException) -> str:
    cls = type(exc)
    return f"{cls.__module__}.{cls.__name__}"


def _shape(value) -> list[int] | None:
    """Read array metadata, or the first list branch, without copying a chunk."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(n) for n in shape[:8]]
    if isinstance(value, (list, tuple)):
        result = []
        for _ in range(8):
            if not isinstance(value, (list, tuple)):
                break
            result.append(len(value))
            if not value:
                break
            value = value[0]
        return result
    return None


def _json_default(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    raise TypeError("unsupported metadata value")


class Capture:
    """Construct only when explicitly enabled; methods never retry user calls."""

    MAX_EVENT_BYTES = 32768
    MAX_ACTION_VALUES = 64

    def __init__(self, log_dir, *, entry_path, parameters, dependency_paths=None,
                 queue_capacity=256, flush_timeout=0.5):
        self.episode_uuid = str(uuid.uuid4())
        self.connection_uuid = str(uuid.uuid4())
        self.session_dir = pathlib.Path(log_dir) / self.episode_uuid
        self.stage = "session_start"
        self.control_tick = None
        self.legacy_step = None
        self.source_obs_id = None
        self.request_id = None
        self.phase = None
        self.warmup_index = None
        self._source_counter = self._request_counter = self._tick_counter = -1
        self._entry_path = str(entry_path)
        self._parameters = dict(parameters)
        self._dependency_paths = dict(dependency_paths or {})
        if dependency_paths is None:
            for name in ("robot_env", "openpi_client.websocket_client_policy",
                         "openpi_client.base_policy", "openpi_client.msgpack_numpy",
                         "numpy", "cv2", "tyro"):
                path = getattr(sys.modules.get(name), "__file__", None)
                if path:
                    self._dependency_paths[name] = str(path)
        self._dependency_paths.setdefault("fd_client_capture", __file__)
        self._queue = queue.Queue(maxsize=max(1, int(queue_capacity)))
        self._flush_timeout = max(0.0, float(flush_timeout))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closing = False
        self._terminal = None
        self._owns_dir = False
        self._deadline = None
        self._flush_timed_out = False
        self._invalid_reasons = set()
        self._counts = dict(events_attempted=0, events_enqueued=0,
                            events_written=0, dropped_events=0,
                            write_errors=0, metadata_errors=0, echo_mismatches=0)
        self.event("session_start", parameters=self._parameters,
                   target_hz=self._parameters.get("hz"),
                   dry_run=self._parameters.get("dry_run"))
        self._thread = threading.Thread(target=self._writer, name="fd-client-capture", daemon=True)
        try:
            self._thread.start()
        except Exception:
            self._thread = None
            self._invalid("writer_start_failed", write_errors=1)

    def _invalid(self, reason, **counts):
        with self._lock:
            self._invalid_reasons.add(reason)
            for key, value in counts.items():
                self._counts[key] += value

    def _context(self) -> dict:
        return dict(schema_version=1, episode_uuid=self.episode_uuid,
                    connection_uuid=self.connection_uuid, request_id=self.request_id,
                    phase=self.phase, warmup_index=self.warmup_index,
                    control_tick=self.control_tick, legacy_step=self.legacy_step,
                    source_obs_id=self.source_obs_id)

    def _record(self, name, fields):
        return dict(self._context(), event=name, **_timestamps(), **fields)

    def _line(self, record):
        line = json.dumps(record, ensure_ascii=False, allow_nan=False,
                          default=_json_default) + "\n"
        if len(line.encode("utf-8")) > self.MAX_EVENT_BYTES:
            raise ValueError("event exceeds byte bound")
        return line

    def event(self, name, **fields):
        try:
            with self._lock:
                self._counts["events_attempted"] += 1
            if self._closing:
                self._invalid("event_after_finish", dropped_events=1)
                return
            line = self._line(self._record(name, fields))
            self._queue.put_nowait((name, line))
            with self._lock:
                self._counts["events_enqueued"] += 1
            self._wake.set()
        except queue.Full:
            self._invalid("queue_full", dropped_events=1)
        except Exception:
            self._invalid("event_encoding_failed", dropped_events=1)

    def begin_control_tick(self, legacy_step):
        self._tick_counter += 1
        self.control_tick = self._tick_counter
        self.legacy_step = legacy_step
        self.phase, self.warmup_index = "control", None
        self.request_id = self.source_obs_id = None
        return self.control_tick

    def source_observation(self):
        self._source_counter += 1
        self.source_obs_id = self._source_counter
        return self.source_obs_id

    def infer(self, policy, obs, *, phase, warmup_index=None):
        self._request_counter += 1
        self.request_id = self._request_counter
        self.phase, self.warmup_index = phase, warmup_index
        if phase == "warmup":
            self.control_tick = self.legacy_step = None
        self.stage = f"{phase}_infer"
        context = self._context()
        request = dict(obs)
        request["fd_context"] = context
        self.event("infer_call_start")
        try:
            response = policy.infer(request)
        except BaseException as exc:
            self.event("infer_call_error", exception_stage=self.stage,
                       exception_type=_exception_type(exc))
            raise
        try:
            echo_status, mismatched = "not_returned", []
            if isinstance(response, dict) and "fd_context" in response:
                echoed = response["fd_context"]
                mismatched = [key for key, value in context.items()
                              if not isinstance(echoed, dict) or key not in echoed or echoed[key] != value]
                echo_status = "mismatch" if mismatched else "matched"
                if mismatched:
                    self._invalid("echo_mismatch", echo_mismatches=1)
            actions = response.get("actions") if isinstance(response, dict) else None
            shape_source = ("array_metadata" if getattr(actions, "shape", None) is not None else
                            "first_branch_inferred" if isinstance(actions, (list, tuple)) else "unknown")
            self.event("infer_call_return", actions_shape=_shape(actions), actions_shape_source=shape_source,
                       fd_context_echo=echo_status, mismatched_fields=mismatched)
        except Exception:
            self._invalid("response_metadata_failed", metadata_errors=1)
            self.event("infer_call_return", actions_shape=None, fd_context_echo="unreadable")
        return response

    def control(self, fn, action):
        self.stage = "control_call"
        fields = dict(row_index=0, action=None, execution_done="unknown",
                      ros_source_timestamps="not_recorded", topic_publish_results="not_recorded",
                      hardware_arrival_ack="not_recorded")
        try:
            if len(action) > self.MAX_ACTION_VALUES:
                self._invalid("action_row_exceeds_bound", metadata_errors=1)
                fields["action_recorded"] = False
            else:
                fields["action"] = list(action)
                fields["action_recorded"] = True
        except Exception:
            self._invalid("action_metadata_failed", metadata_errors=1)
        self.event("control_call_start", **fields)
        try:
            result = fn(action)
        except BaseException as exc:
            self.event("control_call_error", **fields, exception_stage=self.stage,
                       exception_type=_exception_type(exc), partial_publish_possible=True)
            raise
        self.event("control_call_return", **fields, meaning="function_return_only")
        return result

    def _status(self):
        with self._lock:
            timed_out = self._flush_timed_out or (self._deadline is not None and time.monotonic() >= self._deadline)
            reasons = sorted(self._invalid_reasons | ({"flush_timeout"} if timed_out else set()))
            return dict(self._counts, complete=not reasons, invalid_reasons=reasons,
                        flush_timed_out=timed_out)

    def finish(self, reason, exception=None, legacy_step=None, exception_stage=None):
        """Bounded drain; original exceptions belong to the caller, never here."""
        try:
            if self._closing:
                return
            self._deadline = time.monotonic() + self._flush_timeout
            self._terminal = self._record("session_end", dict(
                termination_reason=reason, task_outcome="unknown",
                exception_stage=(exception_stage or self.stage) if exception is not None else None,
                exception_type=_exception_type(exception) if exception is not None else None,
                final_legacy_step=legacy_step,
                last_ids=dict(request_id=self._request_counter if self._request_counter >= 0 else None,
                              control_tick=self._tick_counter if self._tick_counter >= 0 else None,
                              source_obs_id=self._source_counter if self._source_counter >= 0 else None),
                ros_source_timestamps="not_recorded", topic_publish_results="not_recorded",
                hardware_arrival_ack="not_recorded", execution_done="unknown"))
            with self._lock:
                self._counts["events_attempted"] += 1
            self._closing = True
            self._wake.set()
            if self._thread is not None:
                self._thread.join(max(0.0, self._deadline - time.monotonic()))
                if self._thread.is_alive():
                    self._flush_timed_out = True
                    self._invalid("flush_timeout")
        except Exception:
            self._invalid("finish_failed", write_errors=1)

    def _source(self, path):
        result = {"path": str(path), "sha256": "unknown"}
        try:
            source = pathlib.Path(path).resolve()
            result["path"] = str(source)
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(65536), b""):
                    digest.update(block)
            result["sha256"] = digest.hexdigest()
        except Exception as exc:
            result["error_type"] = _exception_type(exc)
            self._invalid("source_hash_failed", metadata_errors=1)
        return result

    def _write_json(self, path, payload):
        """Exclusive file creation; test injection point, called by writer only."""
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False,
                      default=_json_default)
            stream.write("\n")

    def _write_line(self, stream, line):
        """Test injection point for blocked/failing IO; never on control thread."""
        stream.write(line)
        stream.flush()

    def _write_event(self, stream, line):
        try:
            self._write_line(stream, line)
            with self._lock:
                self._counts["events_written"] += 1
        except Exception:
            self._invalid("event_write_failed", write_errors=1, dropped_events=1)

    def _write_summary(self):
        summary = dict(self._terminal, **self._status())
        summary["event"] = "session_summary"
        pending = self.session_dir / ".summary.pending.json"
        self._write_json(pending, summary)
        # Account for a timeout observed during the summary write. The caller's
        # wait is bounded; daemon IO may finish later. ``complete`` describes
        # recorded events, and is not proof of a hard filesystem deadline.
        if self._status()["flush_timed_out"] and summary["complete"]:
            pending = self.session_dir / ".summary.incomplete.pending.json"
            summary.update(self._status())
            self._write_json(pending, summary)
        # link() publishes without replacing an existing artifact. Keep its
        # small private source as crash diagnostics (same inode, no data copy).
        os.link(pending, self.session_dir / "summary.json")

    def _writer(self):
        stream = None
        sources = None
        try:
            self.session_dir.parent.mkdir(parents=True, exist_ok=True)
            self.session_dir.mkdir(exist_ok=False)
            self._owns_dir = True
            sources = dict(entry=self._source(self._entry_path),
                           dependencies={name: self._source(path) for name, path in self._dependency_paths.items()})
            manifest = dict(schema_version=1, episode_uuid=self.episode_uuid,
                            connection_uuid=self.connection_uuid, **_timestamps(), **sources,
                            parameters=self._parameters, task_outcome="unknown",
                            completeness_rule="summary.json must parse and complete must be true; not a hard IO deadline guarantee",
                            action_semantics="first action row passed to original function; return is not execution_done",
                            storage="metadata and first action row only; no images/state/prompt/action chunks",
                            ros_source_timestamps="not_recorded", topic_publish_results="not_recorded",
                            hardware_arrival_ack="not_recorded")
            self._write_json(self.session_dir / "manifest.json", manifest)
            stream = (self.session_dir / "events.jsonl").open("x", encoding="utf-8")
        except Exception:
            self._invalid("output_initialization_failed", write_errors=1)
        try:
            while True:
                try:
                    name, line = self._queue.get_nowait()
                except queue.Empty:
                    if self._closing:
                        break
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue
                if stream is None:
                    self._invalid("event_output_unavailable", dropped_events=1)
                else:
                    if name == "session_start" and sources is not None:
                        payload = json.loads(line)
                        payload.update(sources)
                        line = self._line(payload)
                    self._write_event(stream, line)
            if self._terminal is not None:
                if stream is not None:
                    self._write_event(stream, self._line(dict(self._terminal, **self._status())))
                    stream.close()
                    stream = None
                else:
                    self._invalid("terminal_output_unavailable", dropped_events=1)
                if self._owns_dir:
                    self._write_summary()
        except Exception:
            self._invalid("writer_failed", write_errors=1)
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    self._invalid("event_close_failed", write_errors=1)
