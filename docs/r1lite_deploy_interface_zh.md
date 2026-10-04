# R1 Lite 机器人部署接口说明（交接文档）

> **适用对象**：不参与本机推理代码开发、但需要与本机器人对接的另一台上位机的工程师。
> 本文只描述**对外可观测的接口与数据格式**（ROS2 话题 + WebSocket 推理协议 + 落盘日志），
> 不依赖阅读推理服务端源码。按本文即可独立实现兼容的推理服务端或机器人客户端。
>
> - 机器人本体：Galaxea R1 Lite（双 6 自由度臂 + 双夹爪 + 3 自由度躯干 + 轮式底盘）
> - 机器人侧上位机：ROS 2 Humble + Python 3.10（`/usr/bin/python3`，须先 `source /opt/ros/humble/setup.bash`）
> - 文档基准日期：2026-10-04；对应代码分支 `main`（提交 `07d2149`）

---

## 1. 系统拓扑与角色划分

```
┌──────────────────────────── 机器人侧上位机（本机，ROS2 域内）────────────────────────────┐
│                                                                                        │
│   R1 Lite 硬件 ──(ROS2 DDS)──> robot_env.py : RobotEnv                                 │
│     /hdas/feedback_*          （订阅：关节/夹爪/EE 位姿/4 路压缩图像）                  │
│     /hdas/camera_*                     │                                               │
│                                        ▼                                               │
│                              main_r1lite_closedloop.py                                 │
│                              （15 Hz 闭环控制循环）                                     │
│                                        │                                               │
│   R1 Lite 硬件 <─(ROS2 DDS)── /motion_target/*（发布：双臂关节目标 + 双夹爪开度目标）    │
└────────────────────────────────────────┼───────────────────────────────────────────────┘
                                         │  WebSocket（msgpack 二进制帧，端口 8000）
                                         ▼
┌──────────────────────────── 推理侧上位机（对方机器）──────────────────────────────────┐
│   WebsocketPolicyServer  ←  本文第 5 节即此侧必须实现的契约                             │
│   输入：state(14) + 3 路 base64-PNG 图像 + prompt                                      │
│   输出：actions (8, 14) 绝对关节目标                                                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

**关键事实：两条链路互不相同。**
- 链路 A（第 4 节）：ROS2，机器人 ↔ 本机客户端。只在机器人侧 ROS2 域内存在，对方机器看不到也不需要接入。
- 链路 B（第 5 节）：WebSocket + msgpack，本机客户端 ↔ 推理服务。**这是两台机器之间唯一的接口。**

---

## 2. 本机实际使用的部署代码清单

路径均相对仓库根 `/home/r1lite/ant/openpi-r1lite`，`~` 指 `/home/r1lite`。

| 文件 | 角色 | 日常使用频次 |
|---|---|---|
| `examples/simple_client/main_r1lite_closedloop.py` | **主力部署客户端**：15 Hz 闭环，每拍取一次新观测、只执行返回动作块的第 0 帧（receding-horizon） | 最常用（历史命令 189 次） |
| `examples/simple_client/robot_env.py` | ROS2 接入层：`RobotEnv` / `DualArmStateReader`，所有话题收发、QoS、图像解码都在这里 | 被所有客户端 import |
| `~/arm_reset.py` | 复位：双臂回零 + 双夹爪张开（100.0） | 常用（130 次） |
| `~/restore_robot_state.py` | 按名字一比一复位到已保存快照（含躯干），支持 `--dry-run` / `--list` | 常用（127 次） |
| `~/record_robot_state.py` | 只读记录命名快照到 `~/robot_snapshots/<name>.json` | 偶用 |
| `~/robot_state_io.py` | 上两者共用的纯逻辑模块（话题常量、快照格式），无 rclpy 依赖 | 库 |
| `examples/simple_client/main_r1lite.py` | 旧版开环客户端：一次推理执行整块动作（`--actions-per-inference`），另含 `_run_legacy` 计时基准模式 | 已基本退役（62 次） |
| `examples/simple_client/goto_joint_pose.py` | 诊断：开环匀速 ramp 到已知可抓取的关节位姿，用于区分"标定漂移"与"感知/策略问题" | 诊断用 |
| `examples/simple_client/replay_demo_data.py` | 诊断：直接回放 LeRobot v2.1 专家轨迹，用于判断抖动来自数据集还是部署链路 | 诊断用 |
| `examples/simple_client/fd_client_capture.py` | 可选的有界故障事件采集（`--fd-log-dir`），不保存图像与动作块 | 可选 |
| `examples/simple_client/plot_ee_trajectories.py` | 评测后处理：按策略名对比 EE 轨迹 | 评测用 |

**日常标准流程**（真机两条命令，顺序固定）：

```bash
source /opt/ros/humble/setup.bash
python3 ~/restore_robot_state.py --name 915          # 1) 复位到标定好的起始位姿
cd ~/ant/openpi-r1lite/examples/simple_client
python3 main_r1lite_closedloop.py --port 8000 --host <推理机 IP> --max-steps 1000   # 2) 跑策略
```

历史上用过的推理机地址：`192.168.31.160:8000`（近期）、代码默认值 `10.100.40.254:8000`。

> **注意**：本仓库的 `src/openpi/training/config.py` 当前工作区副本已被一份客户端脚本覆盖（332 行，`import` 的是 `examples.r1lite.robot_env`，该路径在本机不存在）。
> 真正的训练配置须从 `git show HEAD:src/openpi/training/config.py` 取。本机**不跑训练也不跑推理服务**，故未修复。

---

## 3. 14 维向量定义（最重要的一张表）

`state`（观测）与 `action`（动作）**布局完全同构**，都是 14 维 float32，左臂在前：

| 下标 | 含义 | 单位 | 典型范围 |
|---|---|---|---|
| 0–5 | 左臂关节 1–6 | 弧度 rad | 约 ±3.0 |
| 6 | **左夹爪开度** | **0–100 线性刻度** | 0 = 完全闭合，≈100 = 完全张开（实测可达 100.8） |
| 7–12 | 右臂关节 1–6 | 弧度 rad | 约 ±3.0 |
| 13 | **右夹爪开度** | **0–100 线性刻度** | 同上 |

**务必注意的三点：**
1. **夹爪不是归一化到 [0,1] 的**，也不是手指关节弧度。它是硬件反馈 `/hdas/feedback_gripper_*` 的 `position[0]`，量纲 0–100。
   （`/joint_states` 里的 `*_gripper_finger_joint*` 是另一套弧度量，**不要**混用。）
2. 动作是**绝对关节目标**，不是增量。训练端内部对手臂维度做过 delta 化
   （`use_delta_joint_actions=True`，掩码 `make_bool_mask(6, -1, 6, -1)` → 手臂 delta、夹爪绝对），
   但服务端输出前已用 `AbsoluteActions` 还原，所以**线上收发的都是绝对值**。
3. **躯干（torso）不在这 14 维里，也不被策略控制。** 客户端 `RobotEnv.control()` 里虽然有
   `torso_joint = [-0.8279, 1.7698, 0.3725, 0.0]` 这个常量，但对应的 `publish` 调用是注释掉的：
   躯干姿态由运行前的 `restore_robot_state.py` 一次性摆好，整个 episode 内保持不动。

另有一个 16 维 EE 位姿向量（仅用于日志，当前策略不使用）：
`[R_xyz(0:3), R_quat xyzw(3:7), R_grip(7), L_xyz(8:11), L_quat xyzw(11:15), L_grip(15)]` —— 注意它是**右臂在前**，
与 14 维 qpos 的左臂在前**相反**（已知技术债，见第 8 节）。坐标系为 `base_link`，单位米。

---

## 4. 链路 A：ROS2 话题（机器人侧，仅供对方机器参考 / 自建客户端时使用）

全部实现在 [robot_env.py](../examples/simple_client/robot_env.py)。

### 4.1 订阅（读状态与图像）

| 话题 | 消息类型 | 取用字段 | 说明 |
|---|---|---|---|
| `/hdas/feedback_arm_left` | `sensor_msgs/JointState` | `position[0:6]` | 左臂 6 关节，rad。长度必须为 6，否则客户端直接抛 `RuntimeError` |
| `/hdas/feedback_arm_right` | `sensor_msgs/JointState` | `position[0:6]` | 右臂 6 关节，rad |
| `/hdas/feedback_gripper_left` | `sensor_msgs/JointState` | `position[0]` | 左夹爪开度，0–100 |
| `/hdas/feedback_gripper_right` | `sensor_msgs/JointState` | `position[0]` | 右夹爪开度，0–100 |
| `/hdas/feedback_torso` | `sensor_msgs/JointState` | `position[0:4]` | 躯干 `[j1, j2, j3, yaw]`，仅前 3 个是可驱动目标（快照脚本用，部署客户端不读） |
| `/motion_control/pose_ee_arm_left` | `geometry_msgs/PoseStamped` | `pose.position` + `pose.orientation` | 左 EE，`base_link` 系，米 + 四元数 |
| `/motion_control/pose_ee_arm_right` | `geometry_msgs/PoseStamped` | 同上 | 右 EE |
| `/hdas/camera_head/left_raw/image_raw_color/compressed` | `sensor_msgs/CompressedImage` | 整帧 | **头部左目**，解码后 1280×720×3 BGR。→ 送给策略的 `cam_head` |
| `/hdas/camera_head/right_raw/image_raw_color/compressed` | `sensor_msgs/CompressedImage` | 整帧 | 头部右目，1280×720。**当前不使用**（见第 8 节坑 1） |
| `/hdas/camera_wrist_left/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 整帧 | 左腕，解码后 640×360×3 BGR → `cam_left_wrist` |
| `/hdas/camera_wrist_right/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | 整帧 | 右腕，640×360 → `cam_right_wrist` |

图像用 `cv_bridge.compressed_imgmsg_to_cv2()` 解码，得到 **BGR** `numpy.uint8` 数组。

**订阅端 QoS**（四类消息统一）：`BEST_EFFORT` + `KEEP_LAST(depth=10)` + `VOLATILE`。

### 4.2 发布（下发动作）

| 话题 | 消息类型 | 写入字段 | 来自 14 维动作的哪几位 |
|---|---|---|---|
| `/motion_target/target_joint_state_arm_left` | `sensor_msgs/JointState` | `position`（6 个 float） | `action[0:6]` |
| `/motion_target/target_joint_state_arm_right` | `sensor_msgs/JointState` | `position`（6 个 float） | `action[7:13]` |
| `/motion_target/target_position_gripper_left` | `sensor_msgs/JointState` | `position`（1 个 float，0–100） | `action[6]` |
| `/motion_target/target_position_gripper_right` | `sensor_msgs/JointState` | `position`（1 个 float，0–100） | `action[13]` |
| `/motion_target/target_joint_state_torso` | `sensor_msgs/JointState` | `position`（3–4 个 float） | **部署时不发**（已注释） |
| `/motion_target/target_pose_arm_{left,right}` | `geometry_msgs/PoseStamped` | 位置 + 四元数 | 仅 `CONTROL_MODE="eepose"` 时使用；当前为 `"joint"` |
| `/motion_target/brake_mode` | `std_msgs/Bool` | — | 已建发布者，部署中未使用 |

每条消息都会填 `header.stamp = now()`；EE 模式下还填 `header.frame_id = "base_link"`。

**发布端 QoS（有坑，务必照抄）**：部署客户端对**全部**发布者统一使用
`RELIABLE` + `KEEP_LAST(depth=10)` + `TRANSIENT_LOCAL`。

但这与订阅侧控制器的声明并不完全一致：
- 手臂目标的订阅方（JointTracker）要求 `RELIABLE` + `TRANSIENT_LOCAL`；
- 夹爪控制器 `r1_gripper_controller` 声明的是 `BEST_EFFORT` + `VOLATILE`。

因此如果自建发布者用了**默认 QoS**，消息会被静默丢弃（只在日志里留一条 DURABILITY 不兼容告警）。
另外，`BEST_EFFORT` 下**单次发布经常丢**：复位类脚本（`arm_reset.py`）的做法是
**同一条指令以 50 ms 间隔连发 20 次（共 1 s）**才能确保执行。
15 Hz 闭环客户端不重发——它靠每拍都在发新目标来保证连续性。

### 4.3 读状态时绝对不要用 `/joint_states`

本机的 `/joint_states` 由 `r1_lite_jointTracker_demo_node` republish，内容是**跟踪器下发的目标**
（与它发往 `/motion_control/control_torso` 的 `p_des` 逐字节相同），不是真实编码器值。
用手柄速度模式搬动躯干后，`/joint_states` 完全看不出变化。
**真实关节值只在 `/hdas/feedback_*` 上。**
（`/joint_states` 可用于拿到 25 个关节的名字清单：底盘 6 + 躯干 3 + 双臂 12 + 夹爪手指 4。）

---

## 5. 链路 B：WebSocket 推理协议（两台机器之间的唯一接口）

### 5.1 连接

- 客户端实现：[websocket_client_policy.py](../packages/openpi-client/src/openpi_client/websocket_client_policy.py)
- 服务端参考实现：[websocket_policy_server.py](../src/openpi/serving/websocket_policy_server.py)
- URI：`ws://<host>:<port>`，默认端口 `8000`（`host` 以 `ws` 开头时按原样当 URI 用）
- 连接参数：`compression=None`、`max_size=None`（不限帧大小）
- 可选鉴权：若客户端传了 `--api-key`，则带 HTTP 头 `Authorization: Api-Key <key>`
- 健康检查：HTTP `GET /healthz` → `200 OK`，响应体 `"OK\n"`
- 客户端连不上会**每 5 s 重试、无限等待**（`ConnectionRefusedError` 循环），不会退出

**握手**：连接建立后，**服务端必须先主动推送一帧** msgpack 编码的 metadata 字典
（内容任意，可为空字典 `{}`；客户端把它打到日志里）。客户端在收到这一帧之前不会发任何请求。

之后是严格的一问一答：客户端发一帧请求 → 服务端回一帧响应，长连接复用，不并发、不流水线。

### 5.2 帧编码：msgpack + numpy 扩展

所有帧都是 **msgpack 二进制**（不是 JSON）。numpy 数组经 `msgpack_numpy` 自定义钩子编码
（见 [msgpack_numpy.py](../packages/openpi-client/src/openpi_client/msgpack_numpy.py)）：

```python
# ndarray 被编码成这样一个 map（注意 key 是 bytes）：
{b"__ndarray__": True, b"data": <raw bytes, C order>, b"dtype": "<f4", b"shape": (8, 14)}
# numpy 标量：
{b"__npgeneric__": True, b"data": <python scalar>, b"dtype": "<f4"}
```

解码时以 `object_hook` 还原。不支持 `V`/`O`/`c` 三类 dtype（会抛 `ValueError`）。
**不使用 pickle**，跨语言可实现。

### 5.3 请求帧（客户端 → 推理服务）

每个控制拍发一次，顶层是 3 个键：

```python
{
  "state":  [float] * 14,                 # Python list（非 ndarray），见第 3 节布局，单位 rad / 0-100
  "images": {
      "cam_head":         "<base64 PNG string>",   # 必须存在
      "cam_left_wrist":   "<base64 PNG string>",   # 缺失时服务端应补零并置 mask=False
      "cam_right_wrist":  "<base64 PNG string>",
  },
  "prompt": "Collaborate with the human. Pick up the target object and handover to the human hand.",
}
```

- `state`：由 `np.float32` 数组 `.tolist()` 得到，故是 msgpack 的 float64 数组。
- `prompt`：整段 episode 内为常量字符串，定义在客户端顶部 `PROMPTS`。换任务时改这个常量。
- 某路相机当拍没有图像（`None` 或 `size == 0`）时，该键**直接不出现**在 `images` 里。
  客户端只在"状态读不到"或"四路图像全空"时才跳过该拍不发请求；
  **`cam_head` 单独缺失时请求仍会发出**，而策略侧 `R1liteInputs` 假定 `cam_head` 必然存在
  （注释原文 "Assume that base image always exists"），会直接 `KeyError`。
  服务端建议对缺失的 `cam_head` 做显式报错或补零，而不是让它穿透成异常。

**图像预处理（服务端若要自行复现训练时的几何，必须按此理解收到的图）**
见 `prepare_image()`：
1. 输入：`cv_bridge` 解码出的 **BGR** `uint8` 数组（头部 1280×720，腕部 640×360）；
2. **等比缩放** `ratio = max(W/224, H/224)`，`cv2.INTER_LINEAR`（等价于服务端侧的 PIL bilinear）；
3. 贴到 **224×224 全零（黑色）画布的中心**，上下或左右留黑边（16:9 的头部图上下各约 34 px 黑边）；
4. `cv2.imencode('.png', padded)` → `base64.b64encode(...).decode('utf-8')`。

所以服务端收到的是**标准 PNG（无损）、224×224×3**。
因为是 `cv2.imencode` 写出的，PNG 文件内通道顺序是正确的 RGB：
用 `cv2.imdecode` 解回来是 BGR，用 PIL 打开是 RGB——**两侧必须和训练时的解码方式一致**，否则等于把 R/B 通道对调。
策略侧的张量约定是 `[channel, height, width]`、`uint8`（`R1liteInputs` 内部再 `c h w -> h w c`），
相机名映射为 `cam_head → base_0_rgb`、`cam_left_wrist → left_wrist_0_rgb`、`cam_right_wrist → right_wrist_0_rgb`。

> **带宽参考**：3 张 224×224 无损 PNG，约 100–150 KB/帧 × 15 Hz ≈ 1.5–2.5 MB/s。

### 5.4 响应帧（推理服务 → 客户端）

```python
{
  "actions": <ndarray float32, shape (8, 14)>,   # 必须有；8 = action horizon（实测恒为 8）
  "server_timing": {                              # 由服务端框架自动附加
      "infer_ms": 47.8,
      "prev_total_ms": 52.3,                      # 第一帧没有这个键
  },
}
```

- `actions[i]` 的 14 维布局与请求的 `state` **完全相同**（第 3 节），且是**绝对关节目标**。
- 闭环客户端**只执行 `actions[0]`**，其余 7 行丢弃（下一拍用新观测重新推理）。
  旧的开环客户端 `main_r1lite.py` 会按 `--actions-per-inference` 顺序执行多行，每行间隔 `1/15 s`。
- `actions` 为空（`size == 0`）时客户端跳过该拍，不下发任何指令。
- **错误约定**：服务端出错时**发一帧文本（str）而不是二进制**，内容是 Python traceback，
  随后以 `INTERNAL_ERROR` 关闭连接。客户端收到 `str` 即抛 `RuntimeError`。

### 5.5 时序与性能预算

| 项目 | 值 |
|---|---|
| 控制频率 | **15 Hz**（周期 66.7 ms）—— 与 LeRobot 数据集录制帧率一致，训练即假定此频率 |
| 每拍请求数 | 1（启动时另有 2 次 warmup 推理，用同一帧观测） |
| 推理往返（客户端实测，含网络） | mean ≈ 52 ms，p50 ≈ 48 ms，p95 ≈ 73 ms，p99 ≈ 92 ms |
| ROS2 下发耗时 | mean ≈ 0.23 ms，p95 ≈ 0.35 ms |
| 推理占任务墙钟 | ≈ 72% |

**服务端若维护观测历史（`n_obs_steps` 滑窗），务必注意**：闭环客户端保证每个请求携带**全新的真实观测**，
滑窗里是真实的时间演化。而旧客户端 `main_r1lite.py` 会用**同一帧观测连发 ~22 次**请求，
把滑窗填成 8 份同一帧的拷贝——这与训练分布（每步相隔 1/15 s、8 步跨 0.53 s）严重不符。
请按"每个请求一个新时刻"来设计服务端状态。

---

## 6. 落盘数据格式

### 6.1 轨迹日志 `--log-file`（JSONL，默认 `./log_actions_r1lite_closedloop.jsonl`）

**一次脚本运行 = 一个 episode**；`episode_index` 由文件里已有的最大值 +1 自动推导，可重复追加同一文件。
三种记录类型，用 `type` 字段区分：

```jsonc
// 1) episode 头
{"type":"episode_start","timestamp":"2026-07-30T11:01:16.782","policy":"fmdit","episode_index":0,
 "episode_id":"fmdit-20260730_110116","episode_label":"pickpeach","dry_run":false,"hz":15.0,
 "log_every":5,"host":"192.168.31.160:8000","prompt":"Collaborate with the human. ...","control_mode":"joint"}

// 2) 步记录（每 --log-every 拍一条）
{"type":"step","timestamp":"2026-07-30T11:01:23.419","policy":"fmdit","episode_index":0,
 "episode_id":"fmdit-20260730_110116","step":0,"dry_run":false,
 "action":[14 个 float],            // 本拍实际下发的 actions[0]
 "state":[14 个 float],             // 本拍送进策略的观测
 "r_ee_xyz":[0.0588,-0.3350,0.3339],// 右夹爪 EE 在 base_link 的 xyz（米），可能为 null
 "infer_ms":53.23,"exec_ms":0.245}

// 3) episode 尾（同样的内容也会追加到 --summary-file）
{"type":"episode_end","timestamp_start":"...","timestamp_end":"...","policy":"fmdit","episode_index":0,
 "episode_id":"...","episode_label":"pickpeach","stop_reason":"ctrl_c","dry_run":false,"hz":15.0,
 "n_steps":213,"n_overruns":212,"infer_s_total":11.196,"exec_s_total":0.05,
 "wall_s_total":22.036,"wall_s_task":15.47,"infer_ms_mean":52.56,"infer_ms_p50":47.81,
 "infer_ms_p95":72.81,"infer_ms_p99":92.41,"exec_ms_mean":0.23,"exec_ms_p95":0.35,
 "infer_pct_of_task":72.4}
```

解析时请容错：老文件里混有手工插入的纯文本行，客户端自身也只解析以 `{` 开头的行。

> `stop_reason` 取值：`max_steps` / `ctrl_c` / `unknown`。
> `n_overruns` 是错过 1/hz 截止时间的拍数——上例 213 拍里 212 次 overrun，说明单拍推理 ~52 ms 加上
> 观测采集后已挤满 66.7 ms 预算，实际环路频率低于 15 Hz 标称值。**这是评估结果时要带上的前提。**

### 6.2 任务计时汇总 `--summary-file`（JSONL，默认 `./task_times_r1lite_closedloop.jsonl`）

每次运行结束追加一行，schema 与上面的 `episode_end` 相同（无 `type` 键）。

> **命名约定**：`eval_*.jsonl` / `times_*.jsonl` 的文件名后缀是 `--policy-name` 标签，
> **不是**模型 checkpoint 名。同一模型在不同文件里可能用不同标签（例如 `fm_dit` 与 `mars_star` 指同一个）。

### 6.3 图像快照（`--save-images-every N`）

存到 `./log_images_r1lite_closedloop/<cam_name>_<YYYY-mm-dd_HH-MM-SS-ffffff>.png`，
保存的是**原始分辨率**（1280×720 / 640×360）BGR 图，不是送给策略的 224×224 版本。

### 6.4 位姿快照 `~/robot_snapshots/<name>.json`

`record_robot_state.py` 产出，`restore_robot_state.py` 消费：

```jsonc
{
  "name": "915",
  "timestamp": "2026-09-16T21:24:52",
  "actuated": {                       // 复位真正用到的 5 个键
    "torso":   [-0.8283, 1.5745, 0.7379],          // rad
    "left_arm":  [6 个 rad],
    "right_arm": [6 个 rad],
    "left_gripper":  100.835,                       // 0-100
    "right_gripper": 96.822
  },
  "raw_joint_states": { "<joint_name>": <rad>, ... },   // /joint_states 全部 25 个关节，仅存档
  "raw_feedback": { "torso": [...4], "left_arm": [...6], "right_arm": [...6] }  // /hdas/feedback_* 真值
}
```

`actuated` 里的手臂/躯干值优先取自 `raw_feedback`（真实编码器），而非 `raw_joint_states`（见 4.3）。

### 6.5 训练数据侧（LeRobot v2.1）字段对应

| LeRobot 键 | 线上对应 |
|---|---|
| `observation.state` | 请求的 `state`（14 维，同一布局） |
| `action.left_arm` / `action.left_gripper` / `action.right_arm` / `action.right_gripper` | 拼成 `action[0:6]` / `[6]` / `[7:13]` / `[13]` |
| `observation.images.head_rgb` | `cam_head` ←—— **录制时用的是头部左目** |
| 录制帧率 | 15 fps（= 线上控制频率） |

---

## 7. 对方机器自检清单

实现一个兼容服务端后，建议按此顺序验证（前三步都不需要机器人上电）：

1. `curl http://<ip>:8000/healthz` → `OK`
2. 连上 ws 后**先推一帧 msgpack metadata**，确认客户端日志打出 `Server metadata: {...}`
3. 本机跑干跑：`python3 main_r1lite_closedloop.py --host <ip> --hz 15 --max-steps 600 --dry-run`
   —— 全链路跑通但**不向机器人下发任何指令**，可核对 `infer_ms` 与返回的 `actions` 形状
4. 核对返回 `actions` 的 shape 为 `(8, 14)`、dtype `float32`，且第 6、13 位在 0–100 量纲
5. 真机前再确认：手臂目标是绝对角、夹爪 100 ≈ 张开；第一拍动作与当前 `state` 的差值要足够小
   （闭环下第一拍若跳变过大会猛动）

---

## 8. 已知坑与历史问题（必读）

1. **头部相机用左目，不是右目。**
   训练用 `observation.images.head_rgb == /hdas/camera_head/left_raw`。
   2026-09-16 11:03 之前的部署把**右目**当作 `cam_head` 发了出去（1280×720 下约 105 px 视差，
   缩到 224 后约 18 px），那段时间的评测结果**全部作废**。当前 `CAMERA_MAPPING["cam_head"] = "image_left"`。
   注意 `main_r1lite.py` 里仍写着 `"image_right"`——该文件已退役，但别照抄。

2. **`get_joint_states()` 曾把每条臂的第 6 关节清零。**
   2026-09-16 之前它返回 12 维右臂在前的向量，客户端的 12→14 扩展函数再用 0.0 填补 J6，
   导致策略每一拍收到的 `R6 == 0.0`。而训练数据里 R6 是右臂第二活跃的关节（std 0.212 rad，范围 −0.363…+0.544），
   清零即分布外，表现为抓取前右臂/夹爪震荡。现已修为 14 维直通，且臂向量长度不为 6 时直接抛异常。
   `expand_qpos_to_14_dims()` 仅为兼容旧 12 维数据保留，14 维输入原样通过。

3. **`/joint_states` 是跟踪器回声，不是真实关节。** 见 4.3。

4. **两个取值器的左右顺序不一致**：`get_joint_states()` 左臂在前，`get_eef_poses()` 右臂在前。
   当前策略是关节空间的、`state` 不含 EE 位姿，所以无影响；**一旦改用 EE 空间策略，这是必须先统一的阻塞项**
   （改时须同步修改所有消费方，包括记录 `r_ee_xyz` 的上游）。

5. **QoS 与重发**：见 4.2。自建 ROS2 发布者时这是最常见的"消息发出去了但机器人不动"原因。

6. **躯干不受策略控制**，运行中保持 `restore_robot_state.py` 摆好的姿态；相关 publish 代码是注释状态。

7. **Python 环境**：机器人侧脚本必须用系统 `python3`（3.10）+ 已 source 的 ROS Humble。
   跑纯逻辑单测时要清掉 ROS 注入的 `PYTHONPATH`，否则 ROS 的 pytest 插件会冲突：
   `PYTHONPATH= python3 -m pytest test_robot_state_io.py -v`。

8. **15 Hz 是目标值而非实测值**：实测 overrun 比例很高（见 6.1）。
   若对方机器能把 p95 推理延迟压到 40 ms 以下，环路才真正跑在 15 Hz。

---

## 9. 关键文件索引

| 需要查什么 | 看哪里 |
|---|---|
| ROS2 话题 / QoS / 图像解码 / 14 维组装 | [examples/simple_client/robot_env.py](../examples/simple_client/robot_env.py) |
| 控制循环 / 请求构造 / 图像预处理 / 日志 schema | [examples/simple_client/main_r1lite_closedloop.py](../examples/simple_client/main_r1lite_closedloop.py) |
| WebSocket 客户端行为（重试、握手、错误） | [packages/openpi-client/src/openpi_client/websocket_client_policy.py](../packages/openpi-client/src/openpi_client/websocket_client_policy.py) |
| 服务端参考实现（含 `/healthz`、错误帧） | [src/openpi/serving/websocket_policy_server.py](../src/openpi/serving/websocket_policy_server.py) |
| msgpack + numpy 编码规则 | [packages/openpi-client/src/openpi_client/msgpack_numpy.py](../packages/openpi-client/src/openpi_client/msgpack_numpy.py) |
| 策略侧张量约定、相机名映射、夹爪空间变换 | [src/openpi/policies/r1lite_policy.py](../src/openpi/policies/r1lite_policy.py) |
| 训练配置 `pi05_r1lite`（delta 动作掩码等） | `git show HEAD:src/openpi/training/config.py`（工作区副本已被覆盖） |
| 快照格式与话题常量 | `~/robot_state_io.py` |
