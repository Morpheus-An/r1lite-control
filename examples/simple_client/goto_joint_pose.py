"""Send the right arm to an explicit joint pose — no policy, no inference.

Why this exists
---------------
Diagnostic for "did the arm's calibration shift?". The policy is closed-loop on
both vision and proprioception, so a joint-encoder zero-offset is invisible in
the logs: command-vs-feedback tracking stays perfect (R≈1.000) because both
sides live in the same shifted frame, while the arm physically lands somewhere
else. The only way to separate that from a perception/policy problem is to drive
the arm open-loop to a joint vector that is *known* to have worked, and look at
where the gripper physically ends up.

The `grasp` preset is the mean of the commanded right-arm vectors at the seven
confirmed grasps in `eval_policy_mars_star.jsonl` (episodes 0, 1, 2, 11, 12, 15,
23 — identified as the steps where measured gripper stalled >15 units above the
command, i.e. the fingers were blocked by the peach). Those runs put the EE at
[0.4008, -0.4035, 0.2249] ± [0.013, 0.011, 0.008] m in base_link.

Interpretation
--------------
Put the peach back where it was during those runs, then:

  * gripper lands on the peach  -> encoder zeros are fine; the reach shortfall
    is upstream (perception / policy alignment).
  * gripper lands 3-4 cm off    -> calibration shifted; re-home and re-calibrate
    before trusting any further policy eval.

The script also prints the EE the robot *reports* at the target. Note that this
readback comes from the same encoders, so it cannot detect the offset by itself
— it is there to confirm the arm converged, not to validate calibration. The
physical observation is the measurement.

Safety
------
Motion is a rate-limited linear ramp from the current measured pose (default
0.2 rad/s, ~1/1.5 of the policy's own p95 step rate), not a step command. The
plan is printed and must be confirmed before anything is published. Ctrl-C stops
publishing immediately; the arm holds its last commanded target.

Usage:
    python goto_joint_pose.py --dry-run              # print the plan, send nothing
    python goto_joint_pose.py                        # ramp to the grasp pose
    python goto_joint_pose.py --right " -0.16,2.13,-0.97,-0.21,-0.013,-0.004"
    python goto_joint_pose.py --speed 0.1 --settle 5.0

Run with the ROS-enabled system interpreter (same one that runs
main_r1lite_closedloop.py).
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
import tyro

from robot_env import RobotEnv


# Mean commanded right-arm joints at the 7 confirmed grasps in
# eval_policy_mars_star.jsonl (std [0.036, 0.027, 0.034, 0.022, 0.009, 0.001] rad).
GRASP_POSE = [-0.15820, 2.12788, -0.97182, -0.20970, -0.01270, -0.00410]

# EE the robot reported at those grasps, base_link (mean, std) in metres.
GRASP_EE_MEAN = np.array([0.4008, -0.4035, 0.2249])
GRASP_EE_STD = np.array([0.0130, 0.0112, 0.0078])

# Fully-open gripper command, matching what the policy issues at episode start.
GRIPPER_OPEN = 100.0


@dataclasses.dataclass
class Args:
    # Target right-arm joints "j0,j1,j2,j3,j4,j5" in rad. Defaults to GRASP_POSE.
    right: str | None = None
    # Max joint speed for the ramp, rad/s. The policy's own p95 is ~0.76 rad/s.
    speed: float = 0.2
    # Publish rate for the ramp, Hz.
    hz: float = 20.0
    # Seconds to hold the target (republishing) before reading back.
    settle: float = 3.0
    # Right gripper command held throughout. Keep open so the fingers stay visible.
    gripper: float = GRIPPER_OPEN
    # Print the plan and exit without publishing anything.
    dry_run: bool = False
    # Skip the interactive confirmation.
    yes: bool = False


def fmt(a, nd: int = 5) -> list:
    """Round for display — float32 .tolist() otherwise prints full binary precision."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def read_state(env: RobotEnv) -> tuple[np.ndarray, np.ndarray]:
    """Return (qpos_14, right_ee_xyz). Raises if the robot has not published yet."""
    _, state = env.update_obs_window()
    if state is None or state.get("qpos") is None:
        raise RuntimeError("No robot state — is the arm bringup running?")
    qpos = np.asarray(state["qpos"], dtype=np.float32)
    if len(qpos) == 12:
        # 12D is right-first: [R0..R4, R_grip, L0..L4, L_grip]; pad each arm to 6.
        r = np.concatenate([qpos[0:5], [0.0]])
        l = np.concatenate([qpos[6:11], [0.0]])
        qpos = np.concatenate([l, [qpos[11]], r, [qpos[5]]]).astype(np.float32)
    eef = state.get("eef_pose")
    ee = np.asarray(eef[0:3], dtype=np.float32) if eef is not None else np.full(3, np.nan)
    return qpos, ee


def main(args: Args) -> None:
    target_r = np.array(
        [float(v) for v in args.right.split(",")] if args.right else GRASP_POSE,
        dtype=np.float32,
    )
    if target_r.shape != (6,):
        raise ValueError(f"--right needs exactly 6 values, got {target_r.shape[0]}")

    env = RobotEnv()
    qpos, ee0 = read_state(env)

    # state[12] is always 0: get_joint_states() overwrites the 6th right-arm
    # reading with the gripper value (robot_env.py:216), so j5 is commandable but
    # never observed. Treat the current j5 as the target's j5 for the ramp so we
    # never ramp a joint we cannot see.
    start_r = qpos[7:13].copy()
    start_r[5] = target_r[5]

    # Hold the left arm exactly where it is, and its j5 at 0.0 — which is what the
    # policy has always commanded (action[5] is identically 0.0 in every episode).
    left_hold = qpos[0:6].copy()
    left_hold[5] = 0.0
    left_grip = float(qpos[6])

    delta = target_r - start_r
    max_delta = float(np.abs(delta).max())
    step_limit = args.speed / args.hz
    n_steps = max(1, int(np.ceil(max_delta / step_limit)))
    duration = n_steps / args.hz

    print("\n=== goto_joint_pose ===")
    print(f"  current R_arm  = {fmt(qpos[7:13])}   (j5 not observable)")
    print(f"  target  R_arm  = {fmt(target_r)}")
    print(f"  delta          = {fmt(delta)}   max={max_delta:.4f} rad ({np.degrees(max_delta):.2f} deg)")
    print(f"  ramp           = {n_steps} steps @ {args.hz:g} Hz = {duration:.1f} s at <= {args.speed:g} rad/s")
    print(f"  right gripper  = {args.gripper:.1f} (held)")
    print(f"  left arm       = {fmt(left_hold)} grip={left_grip:.1f} (held)")
    print(f"  current EE     = {fmt(ee0, 4)} m")
    print(f"  expected EE    = {GRASP_EE_MEAN.tolist()} +- {(GRASP_EE_STD * 100).round(1).tolist()} cm")

    if args.dry_run:
        print("\n  --dry-run: nothing published.\n")
        return

    if max_delta > 1.0:
        print(f"\n  REFUSING: {max_delta:.2f} rad is a large move. Re-home the arm first,")
        print("  or pass an intermediate --right if you are sure.\n")
        return

    if not args.yes:
        if input("\n  Proceed? Arm will move. [y/N] ").strip().lower() not in ("y", "yes"):
            print("  Aborted; nothing published.\n")
            return

    period = 1.0 / args.hz
    try:
        for i in range(1, n_steps + 1):
            r = start_r + delta * (i / n_steps)
            env.control(np.concatenate([left_hold, [left_grip], r, [args.gripper]]).tolist())
            if i % int(args.hz) == 0 or i == n_steps:
                meas, ee = read_state(env)
                print(f"  [{i / args.hz:5.1f}s] cmd={fmt(r[:5], 4)} "
                      f"meas={fmt(meas[7:12], 4)} EE={fmt(ee, 4)}")
            time.sleep(period)

        print(f"\n  Holding target for {args.settle:g}s ...")
        t_end = time.time() + args.settle
        while time.time() < t_end:
            env.control(np.concatenate([left_hold, [left_grip], target_r, [args.gripper]]).tolist())
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n  Ctrl-C — stopped publishing. Arm holds its last commanded target.\n")
        return

    qpos_f, ee_f = read_state(env)
    err = qpos_f[7:12] - target_r[:5]
    print("\n=== result ===")
    print(f"  target  R_arm[0:5] = {fmt(target_r[:5])}")
    print(f"  reached R_arm[0:5] = {fmt(qpos_f[7:12])}")
    print(f"  joint error        = {fmt(err)}  max={np.abs(err).max():.5f} rad "
          f"({np.degrees(np.abs(err).max()):.3f} deg)")
    print(f"  reported EE        = {fmt(ee_f, 4)} m")
    print(f"  vs grasp-run EE    = {fmt((ee_f - GRASP_EE_MEAN) * 100, 2)} cm "
          f"(|d|={np.linalg.norm(ee_f - GRASP_EE_MEAN) * 100:.2f} cm)")
    print("\n  Now look at the robot, not at this number:")
    print("    fingers on the peach   -> encoder zeros OK, problem is upstream")
    print("    fingers 3-4 cm off     -> calibration shifted; re-home and re-calibrate\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
