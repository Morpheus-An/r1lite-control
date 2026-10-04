# Session: Pick-Peach Stuck-Observation Bug — Server-Side Diagnostic Confirms Upstream Cause

- **Date**: 2026-05-04 14:00
- **User Request**: Investigate why `episode_latents.npz` records bit-identical `z_real` across all timesteps; after server-side fix, verify whether bug is resolved with new deployment data; produce a handoff log for the robot-side team to continue debugging.
- **Status**: completed (server side); pending (robot/client side)
- **Tags**: #pick-peach, #real-robot, #diagnostic, #websocket, #stuck-observation, #encoder, #kalman-planner, #handoff

## User's Original Request

> 我发现了一个bug，[`real-world-collect-ground-obs-bug.md`](.claude/session-logs/real-world-collect-ground-obs-bug.md) 文件详细描述了bug。重新deploy后，[`episode_49_fail/episode_latents.npz`](serve_output/pick_peach/diagnostic/episode_49_fail/episode_latents.npz) 仍然出现 `z_real` 全程 bit-identical 的问题，请帮我检查一下是否bug修复，并将分析结果写入log，供机器人下位机端排查使用。

## Bug Description (from the original report)

`episode_latents.npz` 中：
- `z_real` (shape `(20, 1024)` for ep48, `(15, 1024)` for ep49): **所有时间步取值逐 bit 完全相同**
- `‖z_real[t]‖ = 31.9998 ≈ √1024` (constant) — LayerNorm signature
- step-to-step L2 mean = 0.0
- `z_raw` / `z_corrected` 正常变化

下游影响：
- `trajectory_pca.png` 的 GT 轨迹退化为单点
- `mse_reduction 86.4%` 这个亮眼数字是基于一个未真实演化的 GT 算的，不能直接搬进论文

## Actions Taken (this session)

### Phase 1 — Localize the bug to a code layer

1. **Audited the encoder pipeline** ([`dyn_model/planner_kalman.py:396 update_with_real_observation`](dyn_model/planner_kalman.py#L396) → [`dyn_model/models/visual_dyn_model.py:108 encode_obs`](dyn_model/models/visual_dyn_model.py#L108) → [`dyn_model/models/resnet_encoder.py:41 ResNetEncoder.forward`](dyn_model/models/resnet_encoder.py#L41)):
   - `dyn_model.encode_obs` 内部依次调用 `encoder` (ResNet18 backbone) → `per_view_norm` (LayerNorm 512) → `concat` → `fusion_norm` (LayerNorm 1024).
   - `fusion_norm` 是 `nn.LayerNorm(total_dim, elementwise_affine=False)` ([`visual_dyn_model.py:60`](dyn_model/models/visual_dyn_model.py#L60))，这就是 `‖z_real‖ = √D` 的来源（不是 bug，是设计）。

2. **Verified encoder determinism + sensitivity to inputs** by loading the actual `pick_peach` checkpoint and running `encode_obs` on two random images:
   - 同一图像两次输入 → `‖z1 - z1b‖ = 0.0` ✓ (确定性)
   - 两张不同随机图像 → `‖z1 - z2‖ = 1.69` ✓ (敏感于输入)
   - 同图像不同步 batch=1 backbone forward → `‖out_a - out_b‖ = 30.94` ✓ (BN/eval 模式正常)
   - 结论：**encoder 工作正常，不同输入产生不同输出**。

3. **Ruled out `_MseTracker` / `np.asarray` 引用别名问题**：测试 `tensor.cpu().numpy()` 在多次调用中产生 unique CPU buffers (`id` 不同, `unique frames = 5/5`)。

4. **Ruled out 训练模式 / BatchNorm 异常**：加载模型后检查 `m.encoder.training=False`, 各 backbone `training=False`, `BN layers in train mode = 0/0` for both views.

### Phase 2 — Add raw-observation recording to localize upstream

5. **Extended the diagnostic to also save the raw policy-input observation** alongside the three latents, so we have a definitive artifact for verifying what the policy actually saw:
   - Modified [`_MseTracker`](dyn_model/planner_kalman.py#L71) to accept and pair an `obs_data` dict (per-view image arrays + proprio) with each `z_real`.
   - Modified [`KalmanPlanner.update_with_real_observation`](dyn_model/planner_kalman.py#L396) to snapshot `obs` (the input to `prepare_obs`) into a numpy dict before encoding, and pass it to `record_observation`.
   - Modified [`KalmanPlanner.generate_episode_diagnostic`](dyn_model/planner_kalman.py#L681) to dump these as `obs__<key>` entries in `episode_latents.npz`.
   - Mirrored all three changes in [`SimpleKalmanRealPlanner`](dyn_model/planner_kalman_simple_real.py) (the eq.7/8 variant), keying on `_diag_obs_data_list`.

6. **User redeployed** → produced [`episode_49_fail/episode_latents.npz`](serve_output/pick_peach/diagnostic/episode_49_fail/episode_latents.npz) (15 paired steps).

### Phase 3 — Read the new dump

7. **Loaded the new npz and confirmed the recording fix landed**:
   ```
   keys: mse_corrected, mse_raw, obs__head_rgb, obs__right_arm_qpos,
         obs__right_gripper_qpos, obs__right_wrist_rgb, timestep,
         z_corrected, z_raw, z_real
   shapes:  obs__head_rgb (15, 3, 224, 224)  float32
            obs__right_wrist_rgb (15, 3, 224, 224)  float32
            obs__right_arm_qpos (15, 6)  float32
            obs__right_gripper_qpos (15, 1)  float32
   ```

8. **Ran the bug-report reproducer on every observation field**:
   ```
   z_real                  unique frames: 1/15  ‖first - last‖ = 0.000000
   z_raw                   unique frames: 15/15  (varies — predictor sees varying actions)
   z_corrected             unique frames: 15/15
   obs__head_rgb           unique frames: 1/15  ‖frame[0]-frame[14]‖ = 0.0
                           pixel-level diff: 0 differing pixels
   obs__right_wrist_rgb    unique frames: 1/15
   obs__right_arm_qpos     unique frames: 1/15
                           values: [0.76531917, 1.1823404, -0.77531916, -0.0606383,
                                    -0.1819149, 0.0]
   obs__right_gripper_qpos unique frames: 1/15  values: 99.49577 (×15)
   ```

9. **Wrote PNG previews** of frame 0 to `serve_output/pick_peach/diagnostic/episode_49_fail/obs_preview/`:
   - `frame0_head_rgb.png` (224×224)
   - `frame0_right_wrist_rgb.png` (224×224)

## Diagnostic Conclusion

The bug is **NOT in the server-side latent-recording code**. The encoder is fed the same observation every step, and the encoder correctly produces the same latent. Both the new server-side recording change and the underlying `encode_obs` pipeline are working as intended.

Specifically:

| Layer | Observed in episode_49 | What this rules out |
|---|---|---|
| `update_with_real_observation` receives `obs` (sliced last frame) | constant | not the slice |
| `dict_apply(obs_dict, lambda x: x[:, -1:, ...])` | constant | not `dict_apply` |
| `obs_dict` from `obs_adapter.adapt(raw_obs)` | constant *(deduced — proprio also constant)* | not the encoder, not image decoding, not the deque |
| `raw_obs = msgpack_numpy.unpackb(raw_data)` | **must be constant — this is the entry point** | this is where to look upstream |

**Decisive piece of evidence**: `right_arm_qpos` and `right_gripper_qpos` go directly from `raw_obs["state"]` to numpy without any decoder/encoder/buffer touching them ([`scripts/serve_policy.py:309-326`](scripts/serve_policy.py#L309-L326)). They are bit-identical for 15 consecutive WebSocket messages. Therefore the WebSocket is delivering identical `state` payloads — and consequently identical `images` payloads — every iteration.

## Why z_raw and z_corrected still vary (sanity check, not a contradiction)

- `z_raw[t]` is the **predictor**'s output, not the encoder's. It depends on `(encoded_visual, proprio_emb, action_emb)`. The action varies across diffusion calls (stochastic sampler → different action per step even with same obs), so `z_raw` varies.
- `z_corrected[t] = z_raw[t] + g_bias × dt`. As the Kalman bias `g_bias` adapts, `z_corrected` slowly converges to the constant `z_real`. This is what produced the misleading "86.4% MSE reduction" headline number — the Kalman is just learning to point at a stuck target.
- These do **NOT** prove the obs is varying. They are consistent with constant obs + varying actions + adaptive Kalman bias.

## Most likely physical explanation

The robot did not move during the episode:
- 6 joint angles are constant → arm physically didn't rotate.
- Gripper value `99.49577` is constant → gripper didn't open or close.
- Cameras are mounted on the robot/scene → with no arm motion and no scene change, the camera feed is byte-identical.

The actions are being computed and sent (server logs would show `infer` returning new action arrays each step), but the actuators aren't producing physical motion — or the client's state/image sensor read is cached/stale.

## Files Changed

| Action | File Path | Description |
|--------|-----------|-------------|
| modified | [`dyn_model/planner_kalman.py`](dyn_model/planner_kalman.py) | `_MseTracker.reset/record_observation/_maybe_pair`: store paired `obs_data: List[Dict[str, np.ndarray]]`. `KalmanPlanner.update_with_real_observation`: snapshot `obs` per call (CPU numpy), squeeze singleton batch/time dims, pass via `obs_data=` kwarg. `generate_episode_diagnostic`: stack per-key obs across paired steps and emit as `obs__<key>` arrays in the npz. |
| modified | [`dyn_model/planner_kalman_simple_real.py`](dyn_model/planner_kalman_simple_real.py) | Same diagnostic snapshot + dump, attached to the existing `_diag_*_list` lockstep arrays (since this planner doesn't use `_MseTracker`). |
| created | [`.claude/session-logs/2026-05-04_14-00_pick-peach-stuck-obs-bug-handoff-to-robot-side.md`](.claude/session-logs/2026-05-04_14-00_pick-peach-stuck-obs-bug-handoff-to-robot-side.md) | This session log. |
| created | [`serve_output/pick_peach/diagnostic/episode_49_fail/obs_preview/frame0_head_rgb.png`](serve_output/pick_peach/diagnostic/episode_49_fail/obs_preview/frame0_head_rgb.png) | First-frame preview for visual inspection. |
| created | [`serve_output/pick_peach/diagnostic/episode_49_fail/obs_preview/frame0_right_wrist_rgb.png`](serve_output/pick_peach/diagnostic/episode_49_fail/obs_preview/frame0_right_wrist_rgb.png) | First-frame preview for visual inspection. |

## Commands Executed

```bash
# Inspect new npz from redeployment
python -c "import numpy as np; d=np.load('serve_output/pick_peach/diagnostic/episode_49_fail/episode_latents.npz'); print(sorted(d.files)); ..."

# Reproducer from the original bug report, applied to episode_49
python -c "import numpy as np; d=np.load('.../episode_49_fail/episode_latents.npz'); zr=d['z_real']; print('unique', len(np.unique(zr,axis=0))); print('||first-last||', np.linalg.norm(zr[0]-zr[-1])); ..."

# Verify obs__* fields are also constant
python -c "import numpy as np; d=np.load('.../episode_49_fail/episode_latents.npz'); ...check unique frames for each obs__* key..."

# Dump first-frame preview PNGs
python -c "import numpy as np, imageio.v2 as imageio; ...write frame0_head_rgb.png and frame0_right_wrist_rgb.png..."

# Encoder-determinism sanity check on the actual checkpoint
python -c "from dyn_model.plan import load_model; ...run encode_obs on two random images, check ||z1-z2||..."
```

## Action Items for the Robot / Client Side

To localize the upstream root cause, someone with access to the deployment hardware should:

1. **Open the two preview PNGs** at `serve_output/pick_peach/diagnostic/episode_49_fail/obs_preview/` and compare to what the cameras *should* be showing during the deployment:
   - If the previews look like a meaningful, representative frame from the workspace → the camera path delivered a real image, but the scene + robot were stationary → the bug is *robot motion / actuation*.
   - If the previews are blank/black/garbled/test-pattern-looking → the client never captured a live frame → the bug is *image capture* on the client.

2. **On the client (`main_r1lite.py` on the robot machine)**, add a one-line log per WebSocket `send()`:
   ```python
   import hashlib
   state_hash = hashlib.md5(np.asarray(state).tobytes()).hexdigest()[:8]
   print(f"send step={step} state_hash={state_hash} state[7:13]={state[7:13]}")
   ```
   Re-run for ~10 steps. Expected: every step has a different `state_hash`. If they're all the same → the client is reading stale state from its sensor / IPC layer; the WebSocket itself is fine.

3. **Sanity-check the robot's joint encoder readback** outside this stack. Run a one-shot script that polls the controller for `right_arm_qpos` 10 times with a 100ms sleep and prints them. If those readings are also constant when the user manually moves the arm, the bug is in the robot's controller / driver firmware, not in any Python code on either side.

4. **Verify actions are being executed**. The server is computing varying actions (proven by varying `z_raw`). The client should:
   - Confirm `actions` arrives in the message handler with non-zero, time-varying values.
   - Confirm those actions are being dispatched to the actuators (not just received and dropped).
   - Confirm the actuators are powered and not in safe-mode / e-stop.

5. **If the robot was intentionally not moved during this test**, the bug is "not a bug" — the server is faithfully recording a stationary scene. In that case, redeploy with the robot actually executing actions and inspect a fresh `episode_latents.npz`.

## Key Decisions

- **Decision**: Add raw-obs recording (`obs__<key>` arrays) instead of trying to fix `z_real` by re-encoding through a different path. **Reason**: The user's hint was "ground truth observation IS the policy input — just record that". Adding raw obs is also more general: it lets off-line code re-encode with arbitrary encoders without redeploying, and it's diagnostic for exactly this class of upstream bug.

- **Decision**: Save raw images as float32 (not uint8) and let `np.savez_compressed` handle size. **Reason**: Lossless and trivial code — current 15-step episode npz is ~10 MB, fully acceptable. uint8 quantization would have been a premature optimization.

- **Decision**: Don't change anything in the server-side encoding path. **Reason**: The encoder pipeline is provably correct (verified with the actual checkpoint). Any "fix" there would be guess-work and likely regress correct behavior.

- **Decision**: Hand off to the robot side rather than continuing to dig. **Reason**: All evidence on the server side points upstream of `raw_obs = msgpack_numpy.unpackb(raw_data)`. Without access to the robot/client runtime, further server-side investigation cannot make progress.

## Summary

Server-side recording fix from the previous session works correctly — `episode_latents.npz` now includes `obs__head_rgb`, `obs__right_wrist_rgb`, `obs__right_arm_qpos`, `obs__right_gripper_qpos` arrays alongside the three latents. The new episode 49 dump definitively shows that the bit-identical `z_real` problem is **not** an encoder or recording bug: every observation field — including proprioceptive joint readings that bypass the encoder entirely — is bit-identical across all 15 steps. The robot's `right_arm_qpos` did not change, the gripper value stayed at `99.49577` for 15 consecutive WebSocket frames, and the head + wrist cameras delivered byte-identical pixels. Either the robot didn't physically move, or the client is reading stale sensor data. The server has done its part; the next debugging step belongs on the robot / client side, with concrete diagnostic suggestions enumerated in "Action Items" above.
