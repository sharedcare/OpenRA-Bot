# OpenRA-Bot 实验报告

> **日期**: 2026-06-15，更新至 2026-07-01
> **状态**: Goal conditioning 5x baseline。经济 reward + 升级版 RuleBasedAgent 产生步兵产线。两处 ARM64 反射 bug 已修复。Entity scalar 已补 building/unit/queue counts（28 base / 34 with goal），消除“1 powr vs 10 powr”观测盲区。当前瓶颈转为 PPO 更新稳定性与 reward/action 上限：`obs_counts_goal_w06` 100/100 updates KL early stop。完整架构见 ARCHITECTURE.md。

## 目录

1. [环境与运行条件](#1-环境与运行条件)
2. [实验时间线](#2-实验时间线)
3. [关键发现与修复](#3-关键发现与修复)
4. [当前架构状态](#4-当前架构状态)
5. [训练结果汇总](#5-训练结果汇总)
6. [下次 Session 接续指南](#6-下次-session-接续指南)
7. [2026-06-18 实验更新：PPO 已能跑通，但缺强先验](#7-2026-06-18-实验更新ppo-已能跑通但缺强先验)

---

## 1. 环境与运行条件

| 项目 | 配置 |
|------|------|
| OS | macOS ARM64 (Apple Silicon) |
| Python | 3.12 (miniforge) |
| .NET Runtime | 10.0.6 (via pythonnet coreclr) |
| OpenRA | {DEV_VERSION}, ra mod |
| GPU | M1 Max (CPU training only) |
| 启动命令 | `python scripts/train_rl.py --num-steps 512 --total-updates 80 --max-episode-ticks 1500 --warmstart-episodes 10 --warmstart-epochs 15` |

**重要**: `OpenRA.runtimeconfig.json` 需要 `"rollForward": "LatestMajor"` 才能在 ARM64 上通过 pythonnet 加载。

---

## 2. 实验时间线

### Round 1-3: 训练基础设施搭建
- 初始 PPO 训练 LR=1e-2 → KL 爆炸到 15+ → entropy 崩溃
- 降低 LR 到 3e-4 → KL 可控但 mean_reward 平坦
- 发现 87% 步 reward=0 → 信号太稀疏

### Round 4-5: Reward 设计迭代
- 添加 auto_deploy + reward density (building_per_step) → nz_frac 达到 100%
- 但 building_per_step 常数背景淹没了差异化信号 → 策略学不动
- 移除 building_per_step，增强 produce_start 到 0.5 → nz_frac 回到 88%

### Round 6: 发现 value loss 主导梯度
- 诊断：vloss=0.5 是 ploss=0.001 的 500 倍
- 共享 trunk 梯度被价值函数主导 → 策略不动
- 降低 vf_coef 到 0.01，提高 ent_coef 到 0.1
- 添加 BC warm-start (bias +1.5 for produce/deploy)

### Round 7: BC warm-start + LSTM 问题
- BC warm-start 收集数据时发现 deploy=150 次 → 状态更新 bug
- 修复后 BC 预训练成功 (loss 1.72→0.23, acc ~80%)
- 但 PPO 中 LSTM 随机权重破坏了 BC encoder 特征
- 修复：warmstart 时用 feed-forward 模型 (no LSTM) + freeze encoder

### Round 8: 发现 MCV 类型检测 Bug ⭐
- **关键发现**: `PythonAPI.cs` 反射检测 `IMove` 在 ARM64 上失败
- MCV 被误归类为建筑 → deploy reward 始终为 0
- RuleBasedAgent 测试验证：累计 reward=0.000
- **修复**: 类型白名单 `_is_building()` 替代反射启发式
- 修复后：deploy reward +1.2，RuleBasedAgent 累计 +5.91

### Round 9: 决策步 mask + BO distance reward ⭐
- 数据分析：80% 步为 forced noop（生产等待），只有 20% 有选择权
- 参考 DI-star (AlphaStar 开源实现) 的 build_order distance reward
- **决策步 mask**: 只在 >1 合法动作的步上计算 policy gradient
- **BO distance reward**: Levenshtein 匹配到 `[powr, proc, barr]` 目标序列
- 信号密度从 0.7% → 98.7%，累计 reward 从 0 → 5.91

### Round 10: -inf mask 修复策略塌缩 ⭐
- 100 轮训练：前 37 轮 produce 率上升 (3.5%→7%)，KL 正常
- 第 38 轮开始策略塌缩：produce logit 学到 20+，mask penalty (-13.8) 挡不住
- **修复**: `log(clamp(1e-6))` → `-inf` 用于完全 mask 的动作
- 80 轮训练验证：KL max=0.199，零塌缩，策略稳定

---

## 3. 关键发现与修复

### 已修复的 Bug

| # | Bug | 影响 | 修复方式 | 文件 |
|---|-----|------|---------|------|
| 1 | MCV 类型检测 (ARM64 反射失败) | deploy reward=0 | 类型白名单 | `envs/openra_env.py` |
| 2 | 决策步噪声 (80% forced noop) | policy gradient=0 | 决策步 mask | `agent/agent.py` |
| 3 | Reward 信号稀疏 (0.7%) | credit assignment 不可能 | BO distance reward | `envs/openra_env.py` |
| 4 | Mask penalty 不足 | 策略塌缩到 produce spam | `-inf` 替代 `log(1e-6)` | `models/actor.py`, `agent/agent.py` |
| 5 | LSTM 破坏 BC encoder | warmstart 无效 | feed-forward + freeze | `scripts/train_rl.py` |
| 6 | .NET 8→10 兼容 | ARM64 无法加载 | rollForward | `bin/OpenRA.runtimeconfig.json` |

### 当前超参数

```python
# scripts/train_rl.py
learning_rate = 3e-4
gamma = 0.995
gae_lambda = 0.95
clip_coef = 0.1
ent_coef = 0.1
vf_coef = 0.01
update_epochs = 4
target_kl = 0.03

# Reward weights
building = 1.0
unit = 0.3
produce_start = 0.5
produce_cancel_penalty = 0.1
producing_per_step = 0.005
deploy_bonus = 0.5
# BO distance reward: +1.0 per matching building (implicit via building weight)
```

---

## 4. 当前架构状态

### 模型架构
```
obs (1134-dim vector) → VectorEncoder (MLP 512→256) → Linear(256) → policy_head (6 heads) + value_head
                                                                  [encoder FROZEN after BC warmstart]
```

### 训练流程
```
RuleBasedAgent 采集 10 episodes → BC 预训练 encoder + action_type head (15 epochs)
→ 加载预训练权重 → freeze encoder → PPO 训练 (feed-forward, no LSTM)
```

### Action Space
```
MultiDiscrete [action_type (6), unit_idx (100), target_x (128), target_y (128), target_idx (100), unit_type (7)]
```

### Reward 成分
```
BO distance reward (建筑物匹配目标序列)  +1.0/building
produce_start (开始生产)                +0.5
producing_per_step (持续生产中)         +0.005
deploy_bonus (首次部署)                +0.5
unit change                             +0.3/unit
produce_cancel_penalty                  -0.1 * progress
low_cash_penalty                        -0.2 * deficit
```

---

## 5. 训练结果汇总

### 最终稳定训练 (80 updates, -inf mask fix)

| 指标 | 值 |
|------|-----|
| KL divergence | mean=0.062, max=0.199 ✅ |
| Entropy | 0.22-0.65 ✅ |
| Produce 率 | 22-24/512 (~5%) |
| mean_reward | 0.035-0.037 |
| nz_frac | 0.85-0.93 |
| 策略塌缩 | 无 ✅ |
| Attack spam | 无 ✅ |

### 性能对比

| Agent | Produce 率 |
|-------|-----------|
| Random | ~3% |
| RuleBasedAgent | ~4% |
| PPO (BC warmstart) | ~5% |

### Round 11: Entity Observation 实施 (Phase 1)

- 新增 `utils/entity_obs.py` — 14 维 per-actor 特征 + 10 维 scalar
- 新增 `models/entity_encoder.py` — SimpleEntityEncoder (25K params, MLP mean-pool)
- 修改 Buffer/Agent/ActorCritic 支持 `gym.spaces.Dict` entity observation
- `--observation-type entity` CLI 参数
- **BC 准确率**: 100%（vs 87% with flat vector），loss 1.59→0.04
- **PPO 50 轮**: produce 率 3-9%，无上升趋势，每轮 early_stop
- **结论**: Entity encoder 完美学到 RuleBasedAgent 行为，但 PPO 仍无法超越基线。确认瓶颈不在 observation 表达力，而在 PPO 的 credit assignment 能力（300 步仅 15 个决策点）

### Round 12: Asset-value reward + 三项消融 (2026-06-15)

**动机**: RuleBasedAgent 只建固定建筑、零作战单位，且旧 reward（BO distance）建完 `[powr,proc,barr]` 即封顶 → RL 在该 reward 下无余量可超。重构为开放式净资产 reward。

**实现**:
- `reward_mode="asset"`（默认）+ `AssetValueTracker`（`envs/openra_env.py`）：reward = 新增 owned actor 的造价总和 / 1000，无上限，事件式按 actor id 去重，catalog 成本摄取 + seed fallback
- **关键教训**: per-step 惩罚（idle_cash / power_deficit）随 episode 长度累积，淹没一次性资产收益（破坏净资产 telescoping）→ 默认关掉
- `scripts/verify_asset_reward.py`：headroom 验证（RuleBasedAgent vs GreedyDevAgent）

**余量验证（PASS）**: Greedy（rule-based 经济 + 持续造步兵）严格 > RuleBasedAgent，且 gap 随时长增长：150步 +0.1，400步 +0.7（greedy 造 32 个 e1）。证明 army value 是 rule-based 留下的余量。

**意外发现**: RuleBasedAgent 其实没卡在 3 建筑 — 它的 cycle 持续造经济（400步: 3powr/3proc/3harv）。但**零作战单位**，这才是真正的余量来源。

**三项消融（均失败，PPO 仍平）**:
| 实验 | 配置 | 结果 |
|------|------|------|
| 解 BC 锚 | `--no-freeze-encoder --no-load-bc-action-head` | mean_reward 平 0.02–0.05，25 轮无趋势 |
| 决策步跳过 | `--decision-point-skip`（env.step 内跳过 forced-noop tick） | noop 仍 ~90%，无变化 |
| 激进探索 | lr 1e-3, ent 0.25, target_kl 0.2, clip 0.2 | **更差**：KL 爆到 0.3-0.66，每轮 early_stop，produce↓ noop↑(251)，reward 跌到 0.03 |

**精炼诊断（推翻旧"forced noop"框架）**:
- 旧 REPORT 称"80% forced noop"。实测 `_has_choice`（raw-only：有空队列/可动单位/可放建筑）**几乎总为 True** → 决策步跳过几乎不触发 → 那 90% noop 是策略**主动选** noop，**不是被迫**
- 所有 run 的 **KL ≈ 0.01** = 策略几乎不动。瓶颈不是 observation / reward 余量 / BC 锚 / forced-noop 密度（逐一排除），而是 **on-policy PPO 在 6-head 分解动作空间 + 单环境下的样本效率**：要造一个兵需联合命中 `produce + 正确队列 unit_idx + unit_type=e1`，低概率联合事件，单环境 256 步/轮几乎采不到成功样本去强化

**下一步候选（按杠杆）**:
1. **简化动作空间** — 把"produce e1"塌成单一动作（而非 3-head 联合），直接抬高有效动作被采样+强化的概率。最可能破墙
2. **并行环境（Phase 4）** — 8–64× 样本/wall-clock，但依赖 headless（每实例开 OpenGL 窗口）
3. 激进探索 hypers（Round 12 进行中测试）— 若仍平则确认非"更新太怯"

**激进探索结论**: 已测，**确认非"更新太怯"**。保守 hypers → 平（KL 0.01）；激进 hypers → KL 爆炸、early_stop、策略反而退化（produce↓ noop↑）。无论怎么调探索，PPO 都无法在单环境下采到足够的成功 produce-unit 样本去强化 → 锁定杠杆 #1/#2。

### Round 13: Macro 动作空间 + auto_place bug 修复 (2026-06-15)

**动机**: 选定杠杆 #1（简化动作空间）。把 6-head 联合动作塌成单一 `action_type` 类别。

**实现** (`action_space_mode="macro"`):
- action_type = `['noop'] + ['produce:<t>' for t in MACRO_PRODUCE_TYPES]`（18 类，覆盖经济/防御/步兵/车辆）
- 其余 5 个 head 经 agent 的 `_dummy_mask_like` 自动塌到 index 0（零熵、零梯度）→ **复用整套 model/agent/buffer，零改动**
- env 解析 `produce:<t>` → `_find_producer_for` 找队列 → StartProduction；MCV 部署 + 建筑放置全靠 auto_place
- warmstart 做了 macro-aware 标签映射（StartProduction→`produce:<target>`）

**修复 3 个执行 bug**（diag_macro.py 定位）:
1. **dedup 阻塞重复 produce**: `_action_ttl_steps=8` 窗口挡住连续同类生产 → macro 模式跳过 dedup
2. **auto_place 死守卫**: `_auto_place_done_items` 里一个 `player_actor_id` try-block 异常即 `return []`，导致 Done 建筑**永不放置** → 删除。**此 bug 影响所有模式**（非 macro 靠 agent 手动 build 动作掩盖了它）
3. **auto_place subject 错**: 用 `player_actor_id` 应为队列 actor id（对齐 RuleBasedAgent）→ 改 `q_id`

**结果**（macro + BC warmstart + asset reward, 50 轮, entity obs）:
- 修复后 produce→build→auto-place→actor→reward 闭环打通（diag 验证 actors 1→2→3 持续增长）
- mean_reward 变**连续**值 ~0.045（修复前卡在 {0.0156, 0.0312} = 只有 MCV+conyard）
- **agent 确实造兵了**（atype_dist 常见 9/10/11 = e1/e3/e2，rule-based 从不造兵）
- 但 mean_reward 仍在基线附近震荡（峰值 ~0.056 vs rule-based ~0.048），**未决定性超越**。方差大、频繁 early_stop

**最终诊断**: Macro 让框架真正**能用**（造兵、开放式资产增长、auto_place 修好），但单环境 PPO 仍无法稳定超基线 —— 高方差 + 样本不足。**剩余唯一杠杆 = 样本吞吐（并行环境 Phase 4，依赖 headless）**。

### Round 14: Headless 渲染器（Phase 4 前置） (2026-06-15)

**动机**: 杠杆 = 样本吞吐（并行环境）。前置 = headless（每实例不开 OpenGL 窗口，否则多开崩）。

**实现**:
- 新项目 **`OpenRA.Platforms.Null`** — `NullPlatform : IPlatform` + 全套 no-op stub（window/context/vbuf/ibuf/texture/framebuffer/shader/font/sound）。`Renderer` 构造能跑通但永不真正绘制
- 关键洞察: `PythonAPI.Step()` **只 tick 模拟**（TickImmediate/TryTick/world.Tick），从不渲染 → Null renderer 只需活过 `Game.Initialize` 构造
- `StartLocalGame(..., headless=true)` → 给 `Game.Initialize` 传 `Game.Platform=Null` → `CreatePlatform` 加载 `OpenRA.Platforms.Null.dll`（失败回退 Default，安全）
- env `headless` 参数 + train_rl `--headless` 开关

**验证**: diag + 训练全程 headless 跑通 —— **零 SDL/OpenGL 行**（对比非 headless 会打印 "Using SDL 2 with OpenGL (ANGLE)…"），warmstart + PPO + macro 闭环正常，exit 0。actors 1→2→3 正常增长。

**状态**: headless **完成**。下一步 = `SubprocVecEnv`（spawn 多进程，每进程独立 CLR + headless 引擎）+ PPO 多环境支持（当前 `agent.py` 写死 `num_envs=1`）。

### Round 15: 并行环境 SubprocVecEnv + PPO 多环境 (2026-06-16)

**实现**:
- `envs/vector_env.py` — `SubprocVecEnv`（multiprocessing **spawn**，每进程独立 CLR + headless 引擎）+ `EnvFactory`（可 pickle，强制 headless）。worker 内 auto-reset，obs 跨环境 stack
- `agent.py` — 解除 `num_envs=1` 限制；新增向量化 rollout 分支（批量 forward + `_stack_env_masks` 把 per-env mask 堆成 `(num_envs,...)`）。单环境路径保留
- **修复潜在 bug**: decision_mask 多了一次 `permute(0,2,1)` → `(B,L,N)` 与 `mb_attention_mask (B,N,L)` 不匹配。N=1 时被广播掩盖，N>1 崩。删除
- `train_rl --num-envs N`（warmstart 仍在单环境跑，再建 venv）

**验证**:
- 2-env smoke: 两 CLR + headless 共存，40 步 3.6s = 22 env-steps/s，obs stack `(2,128,14)`，干净退出
- 4-env 训练: 5 CLR（1 warmstart + 4 worker）共存无错，40 轮，atype 计数 ~512/轮，**~4× 吞吐**

**结果**（4 env, macro+BC, 40 轮）: first10 0.0432 → last10 0.0458，max 0.0564 —— **仍贴基线，未决定性超越**。21/40 轮 early_stop（target_kl=0.03 太紧，切断学习）。

**判断**: 并行**基建完整交付**（headless + SubprocVecEnv + PPO 多环境 + 4× 吞吐）。但 4 env/40 轮 + 当前 hypers 仍未破墙。下一步调参方向: 放宽 target_kl（0.03→0.1+，现在一半轮被 early_stop）、加 env 数（8-16）、加 num_steps/总轮数。基建已就位，剩下是 scale + 调参。

**调参冲刺（8 env, 1024 transitions/轮）结果**:
- target_kl 0.15 + ent 0.05 → **崩**：KL 爆到 2.0，entropy→2.0（near-uniform），reward 跌到地板 0.0312
- target_kl 0.05 + ent 0.01 + clip 0.1（保守）→ 前期稳（~0.045，峰 0.055），但 **120 轮后期 late-collapse 到地板**（last15 avg 0.027，末 5 轮 = 0.0312）
- **结论**: 更多更新 → 最终塌缩，非改善。调参/scale 解不了；PPO 在此任务上既不超基线也无法长期稳定

**最终定论（跨极多配置）**: 试遍 obs(flat/entity) × reward(BO/asset) × BC(冻/解/不载) × 动作(6head/macro) × 探索(保守/激进) × 并行(1/4/8) × KL(0.03/0.05/0.15)。**PPO 始终匹配但不决定性超越 rule-based/BC 基线（~0.048），且长跑会塌缩。** 根因非基建（已全交付）—— 是任务余量太薄（asset-optimal 发展策略 ≈ BC 已学的高效建经济，多造兵边际增益 < 方差）+ PPO 长程 credit assignment 极限。要破需换方向：**加对手 + 战斗/胜负 reward 把余量做厚**，或换算法（off-policy/model-based）。

### 历史未解决问题与当前状态

- **PPO credit assignment 极限**: 仍然成立，但表述需要更新。macro action 和并行环境已经缓解了采样问题，PPO 仍无法在发展-only 任务中稳定超越 BC/RuleBased 基线。下一步应补 teacher-KL、goal conditioning、combat/winloss，或评估 off-policy 方法。
- **无 headless 模式**: 已解决。`OpenRA.Platforms.Null` + `--headless` 已可支持本地训练和 `SubprocVecEnv`。

---

## 6. 下次 Session 接续指南

### 启动训练

```powershell
cd F:\Projects\OpenRA\OpenRA.Bot
.\scripts\train_best.ps1 -Updates 100 -RunName stronger_teacher_no_head_repeat
```

### 修改代码后需注意
- 改 `envs/openra_env.py` → 修改 reward 或 observation 需验证 RuleBasedAgent 累计 reward 单调递增
- 改 `agent/agent.py` → 关注 KL 是否爆炸 (>1.0) 和 atype_dist 是否出现 spam
- 改 `models/actor.py` → 注意 mask penalty 必须用 `-inf` 而非 `clamp`

### 关键文件位置
| 文件 | 用途 |
|------|------|
| `envs/openra_env.py` | 环境、reward、action mask |
| `agent/agent.py` | PPO 训练循环、决策步 mask |
| `models/actor.py` | 网络架构、mask penalty |
| `scripts/train_rl.py` | 训练入口、超参数 |
| `scripts/warmstart.py` | BC 数据采集和预训练 |

### 下一步计划
参考 `PLAN.md` 的 2026-06-18 修订版路线。当前不再把 flat vector → entity 当作下一步主线；entity obs 已经可用，下一步主线是实验记录收口、soft teacher-KL、goal conditioning、强化脚本专家，以及引入对手/战斗/胜负信号。

---

## 7. 2026-06-18 实验更新：PPO 已能跑通，但缺强先验

### 7.1 本轮已完成的工程进展

- `models/buffer.py`: 修复 `num_steps < seq_len` 时 `num_sequences=0` 导致 PPO 没有 batch 的问题。短 rollout 现在也会产生有效训练 batch。
- `agent/agent.py`: 增加 `batches`、`decision_steps`、`rollout_steps`、`grad_norm`、`reward_comp` 等诊断；修复 padded mask 全 0 行可能导致 `Categorical` NaN 的问题；KL 估计改成非负形式。
- `envs/openra_env.py`: 默认 asset reward，加入 `AssetValueTracker`；加入 production start / active / cancel credit；macro action 模式 `noop + produce:<type>`；修复 auto-place subject 和 Done 建筑放置问题。
- `scripts/train_rl.py`: 支持 entity / macro / headless / `num_envs` / warmstart 权重隔离 / `--no-load-bc-action-head`。
- `scripts/train_best.ps1`: 当前推荐启动脚本，默认 `MaxEpisodeTicks=1800`、`LearningRate=7e-5`、`ClipCoef=0.05`、`TargetKl=0.03`、`EntCoef=0.02`、`UpdateEpochs=4`、不硬加载 BC action head。
- `RuleBasedAgent`: 从早期三建筑脚本升级为基础宏观发展脚本：`powr -> proc -> barracks -> powr -> weap`，之后持续步兵/车辆/经济循环，并屏蔽墙、防御塔、超级武器等短期噪声动作。

### 7.2 关键实验结果

| 实验 | 配置 | 结果 | 结论 |
|------|------|------|------|
| Short rollout fix | `num_steps=32/128`, `seq_len=128` | 修复后 `batches=4`，`decision_steps` 和 `grad_norm` 正常出现 | PPO 此前存在“看起来训练、实际没更新”的边界 bug，已修复 |
| 稳定基线 | no BC head, asset reward 前 | `reward=0.0358`, `nz=0.039`, `KL=0.0017`, `batches=4.00` | 能更新但 reward 仍稀疏，策略几乎不动 |
| Asset production credit | `run_asset_prod_credit_100` | `reward=0.0646`, `nz=0.913`, `KL=0.0156`, `early=11/100` | production credit 明显改善信号密度，reward 约提升 2x，但平台期约 `0.06-0.07` |
| 更长 rollout | `256 x 100` | `reward=0.0571`, `nz=0.891`, `KL=0.0133` | 简单加长 rollout 没有更好，128 step 仍是当前主线 |
| 强化 RuleBased teacher + hard BC head | `stronger_teacher_with_head_100` | `reward=0.0455`, `last20=0.0392`, `KL=0.8674`, `early=100/100` | 直接加载 BC action head 会过度锚定且触发高 KL early stop，不推荐 |
| 强化 RuleBased teacher + no BC head | `stronger_teacher_no_head_100` | `reward=0.0679`, `last20=0.0706`, `KL=0.0324`, `early=40/100` | 当前最好趋势，但仍不够稳定 |
| UE4/KL0.03 repeat | `stronger_teacher_no_head_repeat_ue4_kl003` | `reward=0.0697`, `last20=0.0645`, `best=0.1062@34`, `KL=0.0400`, `early=56/100` | 全程均值略高、峰值持平略高，但尾段不如上一组；说明能冲高但不能稳定保持 |
| Best logging repeat | `repeat_best_logging` | `reward=0.0577`, `last20=0.0476`, `best=0.1052@31`, `final=0.0463`, `KL=0.0154`, `early=12/100` | 再次验证中途会出现高分策略，但低 KL 下仍会逐步漂回低收益策略；最终 checkpoint 明显不是最佳 |
| 降低 LR | `lr=5e-5`, 150 updates | `reward=0.0663`, `last20=0.0667`, `KL=0.0468`, `early=95/150` | 降 LR 没解决稳定性，略弱于当前最好 |
| 减少 update epochs | `target_kl=0.05`, `UpdateEpochs=2` | `reward=0.0563`, `last20=0.0592`, `KL=0.0118` | 更稳但学习弱 |
| `UpdateEpochs=3` 对照 | `target_kl=0.05`, `UpdateEpochs=3` | `reward=0.0415`, `last20=0.0397`, `KL=0.1010`, entropy 偏高 | 训练退化，出现接近随机/高熵的失败形态 |

当前两组最有价值的记录是：

- `stronger_teacher_no_head_100`: `mean=0.0679`, `last20=0.0706`, `best=0.1060@69`, `early=40/100`，尾段保持更好。
- `stronger_teacher_no_head_repeat_ue4_kl003`: `mean=0.0697`, `last20=0.0645`, `best=0.1062@34`, `early=56/100`，全程均值略高但后半段更抖。
- `repeat_best_logging`: `mean=0.0577`, `last20=0.0476`, `best=0.1052@31`, `final=0.0463`, `early=12/100`，前期能冲高，但随后稳定退回低收益区间。

这说明 asset reward + macro action + 更强脚本 teacher 的方向是有效的，但 PPO 并没有稳定收敛到高 reward 策略；它更像是在几个可行策略模式之间跳动。`repeat_best_logging` 还补充了一个重要信息：即使 KL 不高、early stop 不多，策略也会缓慢漂回低收益策略。当前最该补的是 **best checkpoint 保存** 和 **soft teacher-KL**，否则高分策略会被后续更新洗掉。

`repeat_best_logging` 的 10-update 分段趋势：

| updates | avg_reward | max_reward | avg_KL | early_lt4 | 诊断 |
|---------|------------|------------|--------|-----------|------|
| 1-10 | `0.0767` | `0.0947` | `0.0101` | `0` | 开局策略质量较好 |
| 11-40 | `0.0649` | `0.1052` | `0.0105` | `1` | update 31 达到最佳 |
| 41-70 | `0.0548` | `0.0956` | `0.0073` | `0` | 低 KL 下逐步退化 |
| 71-100 | `0.0471` | `0.0770` | `0.0300` | `11` | 后期进入低收益区间 |

### 7.3 对训练日志的诊断

典型日志中可以看到：

- `mean_reward` 在 `0.02-0.07` 间震荡，最高阶段能到 `~0.07`，但没有稳定上行趋势。
- `reward_comp` 主要来自 `asset_gained / asset_reward`，idle cash 和 power deficit penalty 当前为 0。
- `atype_dist` 经常在 `noop`、`produce:<building/unit>` 间摆动，说明 macro action 已经让 agent 能探索生产，但策略仍容易退回保守/noop 或高熵。
- KL 呈两难：过低表示策略几乎不动；过高会 early stop 或 collapse。简单调 `lr`、`target_kl`、`update_epochs` 只能在“不学”和“不稳”之间移动。
- `LoadBcActionHead` 的实验非常明确：硬加载 action head 不是 teacher-KL，它会把策略压回脚本动作分布，并导致 PPO 更新空间变窄。
- 最新 UE4/KL0.03 repeat 的高分 update 多数只有 `num_batches=1`，说明它经常刚有收益就被 KL early stop 截断；这不是“完全学不会”，而是“好策略无法稳定保留”。
- `repeat_best_logging` 则说明另一个失败模式：不需要 KL 爆炸也会退化。前期 `avg_reward=0.0767`，后期 `last20=0.0476`，mean KL 只有 `0.0154`，说明当前 PPO 更新会在低 KL 漂移中丢掉中途好策略。

### 7.4 与 DI-star / AlphaStar 复现分析的对照

DI-star 的“发展能力”不是纯 RL 从零发现，而是由多层结构支撑：

1. 大规模人类 replay 的监督学习预训练，使 RL 起点已经会完整打局。
2. 训练早期使用 teacher-KL，把策略拉在 SL teacher 附近，避免塌缩。
3. 使用目标条件 `z`，把人类 build order / cumulative stat 喂给网络，策略知道本局要朝什么发展。
4. 多路 reward 和多 value head，拆分 win/loss、build order、unit、upgrade、battle 等 credit assignment。
5. V-trace / UPGO 等 off-policy 机制，提高样本复用和长 horizon 学习能力。
6. 有对手和 battle / winloss 信号，发展动作最终能被战斗结果校准。

OpenRA.Bot 当前相当于：弱脚本 BC + 单标量 asset reward + vanilla PPO + 发展-only 场景。实验现象与这个差距一致：PPO 可以学会靠近脚本宏观发展，但很难稳定超越，因为 teacher 本身没有完整战略和作战能力，reward 也没有把“更强军队是否能赢”传回来。

### 7.5 当前结论

- PPO 实现已经不是主要 blocker；它能产生 batch、能更新、有诊断、有 headless / vectorized 基础设施。
- Asset reward + production credit 是有效改进，解决了旧 build-order reward 封顶和 reward 稀疏问题。
- Macro action 是必要简化，解决了 6-head 联合采样导致的 produce 成功样本稀少问题。
- 更强 `RuleBasedAgent` 是必要的，但“硬加载 BC action head”不是正确用法；下一步应做 frozen teacher 的 soft KL。
- 当前任务余量太薄：发展-only asset reward 下，脚本经济已经很接近局部最优，多造兵的边际收益小于 PPO 方差。
- 要继续提升，需要把训练目标从“造更多资产”推进到“在对手压力下发展并取胜”。

### 7.6 下一步建议

1. **先补实验记录能力**
   - CSV 固化 action distribution、reward components、last20、best reward、early_stop、decision_steps。
   - 保存 best checkpoint，避免只保留固定 update。
   - 目前 `repeat_ue4_kl003` 的最佳点是 `model_0034.pth`，`repeat_best_logging` 的最佳点是 `model_0031.pth`，最终 `model_0100.pth` 未必更好；后续训练应按验证指标自动保存 best。

2. **实现 soft teacher-KL**
   - 冻结一个由升级版 `RuleBasedAgent` BC 得到的 teacher。
   - PPO loss 加 action-type KL，先只约束 macro action head。
   - KL 系数从强到弱退火，避免 hard BC head 的过锚定问题。

3. **加入 goal conditioning**
   - 手写多个 BO / strategy target：经济、步兵、重工、均衡。
   - 每局采样目标，把目标编码进 scalar obs。
   - reward 从“隐式猜方向”变成“朝给定目标前进”。

4. **把 RuleBasedAgent 继续做成更好的专家基线**
   - 它不需要完美，但需要覆盖红警基本常识：补电、补矿、兵营/重工持续出兵、基础防守/进攻触发。
   - 更好的 teacher 会直接提升 BC 数据质量和 teacher-KL 锚点质量。

5. **加入对手、combat reward 和 terminal reward**
   - 发展-only 已经接近上限。
   - 先做固定弱对手，记录 kill value、lost value、battle score、win/loss。
   - 再考虑 self-play / league。

6. **暂缓大模型和大规模算法迁移**
   - Transformer、多 value head、V-trace/UPGO 都有价值，但应在 teacher-KL、goal conditioning、combat signal 有雏形后再上。
   - 否则复杂度会上升，但训练信号仍然不够。

### 7.7 新 Session 接续计划

下一次 session 不建议继续跑同配置 PPO。先把训练系统补成“能保留好策略、能解释为什么好”的状态。

#### P0: best checkpoint + richer CSV

目标：不再丢掉 `model_0031.pth` / `model_0034.pth` 这类中途峰值。

- 在 `PPOAgent.train()` 内维护：
  - `best_reward`
  - `best_update`
  - `last20_reward`
  - `early_stop`
- 保存：
  - `model_best.pth`
  - `best_metrics.json`
- `training.csv` 增加字段：
  - `early_stop`
  - `last20_reward`
  - `best_reward`
  - `best_update`
  - `mask_mean`
  - `atype_dist`
  - `reward_comp`
- 检查点策略：
  - 保留 `save_interval` 或只按固定间隔保存普通 checkpoint。
  - 每当 `mean_reward > best_reward` 时覆盖 `model_best.pth`。

验收：

```powershell
.\scripts\train_best.ps1 -Updates 100 -RunName best_ckpt_smoke
```

训练结束后应能看到：

- `checkpoints/best_ckpt_smoke/model_best.pth`
- `checkpoints/best_ckpt_smoke/best_metrics.json`
- `training.csv` 中有 `best_reward / last20_reward / atype_dist / reward_comp`

预期：即使最终 `model_0100.pth` 退化，也能自动保留中途最佳模型。

#### P1: soft teacher-KL

目标：替代 hard BC action head，让 teacher 只作为软锚，减少“低 KL 漂移”和“高 KL 截断”两种失败模式。

建议实现：

- warmstart 后保留一份 frozen teacher model。
- PPO rollout / update 时，对 macro `action_type` head 计算 teacher KL。
- 先只约束 action_type，不约束其它 dummy heads。
- loss 形态：

```text
loss = ppo_policy_loss + vf_coef * value_loss + ent_coef * entropy_loss + teacher_kl_coef * teacher_kl
```

建议参数：

```text
teacher_kl_coef = 0.05
teacher_kl_final = 0.005
teacher_kl_steps = 50
```

验收对照：

```powershell
.\scripts\train_best.ps1 -Updates 100 -RunName teacher_kl_005_decay
```

对比对象：

- `stronger_teacher_no_head_100`
- `stronger_teacher_no_head_repeat_ue4_kl003`
- `repeat_best_logging`

主要看：

- `best_reward` 是否仍能到 `~0.10`
- `last20_reward` 是否高于 `repeat_best_logging` 的 `0.0476`
- `final` 是否不再明显低于 `best`

#### P2: goal conditioning / BO library

等 P0/P1 完成后再做。目标是让 policy 看到“本局要朝什么发展”，不要只靠 reward 隐式猜。

初始 BO 目标库可以很小：

- `economy`: `powr, proc, powr, proc`
- `infantry`: `powr, proc, barr, e1, e1, e3`
- `vehicle`: `powr, proc, barr, powr, weap, 1tnk`
- `balanced`: `powr, proc, barr, powr, weap, e1, 1tnk`

每局采样一个目标，把目标编码到 entity obs 的 scalar 部分；reward 加目标进度项。这个阶段再考虑多 value head。

---

## 8. 2026-06-27 实验更新：Goal Conditioning 突破

### Round 16: Goal conditioning 实施 + 权重扫描

**动机**: PLAN.md P4，DI-star 的 `z`-conditioning 思想。给 agent 显式目标（economy/infantry/vehicle/balanced），reward 从"猜一个隐含目标"改为"朝显式目标前进"。

**实现** (`utils/goal_library.py`):
- 4 个手写目标：economy (powr×2,proc×2,hary×3) / infantry (e1×15,e3×3) / vehicle (1tnk×5,jeep×3) / balanced
- 每局均匀采样一个 goal，编码为 6 维向量（4 one-hot + 2 weights）拼接到 scalar observation
- Reward: `goal_aligned_weight × progress`（progress = 达到目标建筑/兵力数量的比例）

**Bug 发现** (2026-06-29): reset() 中 goal 采样在 observation 构建**之后**，导致每个 episode 的第一个 obs scalar 是 10 维而不是 16 维（goal_vec 缺失；当时 scalar 维度，Round 30 后为 28/34）。已在 `dbdeba4` 修复。所有之前的 goal conditioning 实验都受此影响。

**权重扫描结果（100 轮）**:

| goal_aligned_weight | best | final | vs baseline |
|---------------------|------|-------|-------------|
| 0.0 (baseline) | 0.0805 | 0.0386 | 1.0x |
| 0.2 | 0.2046 | 0.1507 | 2.5x |
| 0.4 | 0.3102 | 0.1881 | 3.9x |
| **0.6** | **0.4037** | **0.2823** | **5.0x** |
| 0.8 | 0.4690 | 0.1642 | 5.8x（塌缩更陡） |

**结论**: Goal conditioning 是当前架构下唯一有效的大幅改进。权重线性驱动峰值（每 +0.2 → +0.07-0.09 peak）。w=0.6 是最佳平衡点（峰值高 + 塌缩小）。vloss 随权重增长（w=0.6 → vloss=9.7），但没有破坏策略稳定性。

### Round 17: Best checkpoint + 增强 CSV + Teacher-KL

**Best checkpoint**: `model_best.pth` + `best_metrics.json` 在每次 peak 保存。解决 "best=0.1052@31 → final=0.0463" 的退化问题——即使最终 checkpoint 塌缩，最佳模型仍被保留。

**Teacher-KL**: 冻结 BC teacher，PPO loss 中加入 `KL(teacher||policy)`，仅约束 action_type head。退火 coef 0.05→0.005。结果：**final +50%（0.0578 vs baseline 0.0386），但峰值不变**。Teacher-KL 防塌缩但不增峰。

**CSV 增强**: 新增 `early_stop / last20_reward / best_reward / best_update / mask_mean / atype_dist / reward_comp` 列。

### Round 18: DI-star/AlphaStar 组件对齐尝试

**动机**: 研究 AlphaStar 为什么可行、我们差在哪。参考 [DI-star](https://github.com/opendilab/DI-star)。

**关键发现**:
1. **GAE(λ) 就是 TD(λ)**: 数学上等价，不需要改
2. **多路 Value Head**: AlphaStar 用的是单 value head + V-trace/UPGO，不是多路。我们尝试的简化版（预测 per-step component）无效——需要 per-component GAE λ-returns
3. **γ=0.999**: 需要 V-trace 做 off-policy 校正。单独用 → vloss 爆炸到 30，峰值反而更低
4. **GLU gating**: 让 goal z 直接 gate action_type logits。结果 KL→0，策略停更——sigmoid gate 阻止任何 action 获得高 logit
5. **vf_coef 提高**: 降低 vloss 但破坏策略稳定性（entropy 爆炸到 2.0，KL 失控到 4.6）

**完整对照**:

| 实验 | best | 结论 |
|------|------|------|
| Baseline | 0.0805 | — |
| **Goal w=0.6** | **0.4037** | ✅ **唯一有效** |
| + γ=0.999 | 0.3974 | ❌ V-trace 缺失 |
| + GLU gate | 0.4042 | ❌ KL→0 |
| + multi-V | 0.3687 | ❌ 需要 per-component λ-return |
| + vf_coef=0.1 | 0.3820 | ❌ 策略崩溃 |
| + Teacher-KL | 0.0824 | 防塌缩不增峰 |

**根因**: AlphaStar 每个组件都在完整栈中互相支撑（V-trace + UPGO + human replay SL + 1000 环境 + league training）。单独拆零件放进 PPO 框架中均无效。

### Round 19: 阶段式 Goal（三版，均失败）

尝试课程学习：phase 1 先奖励建筑完成，phase 2 再奖励造兵。

- v1 (thresh=1.0): phase 2 从未触发，best=0.3594
- v2 (thresh=0.7): phase 2 ~10%，best=0.3622
- v3 (continuous weights, thresh=0.5): phase 2 ~70%，best=0.2509

**结论**: 固定权重 goal conditioning 优于任何形式的课程学习。移动目标（mid-episode 权重变化）混淆 critic，降低峰值。

### Round 20: 可视化 + Bug 修复

- `scripts/view_best.py`: 加载 best checkpoint，终端显示 top-5 动作概率 + goal 进度，支持 `--delay` / `--stochastic`
- 修复: scalar dim 不一致——reset() 中 goal 采样顺序错误导致首步 observation 缺 goal_vec → 修复后当时 scalar 正确为 16 维；Round 30 后为 28/34 维
- 模型行为验证: 最优 checkpoint 产生 94.5% noop + 5% produce，与训练日志一致——这是学到的最优保守策略，不是 bug

### 当前最佳配置

```bash
python scripts/train_rl.py --num-steps 256 --total-updates 100 \
    --observation-type entity --action-space-mode macro --headless \
    --warmstart-episodes 10 --warmstart-epochs 15 \
    --goal-conditioning --goal-aligned-weight 0.6
```

### 剩余杠杆

1. **强化 RuleBasedAgent** (PLAN.md P2) — 覆盖完整开局 + 持续造兵
2. **加入对手** (PLAN.md P6) — 制造 combat headroom（macOS bot hang，需 Windows）
3. **并行环境** (PLAN.md P4) — SubprocVecEnv 代码就绪，增加样本效率
4. **Proper multi-value-head** (PLAN.md P5) — 需要 per-component GAE，Buffer 级改动
5. **IMPALA/V-trace** — 替代 PPO 的完整 off-policy 框架

---

## 9. 2026-06-29~07-01 实验更新：经济 Reward + Bug 修复 + 架构整理

### Round 21: ARM64 Bug 修复 — IsBuildingQueueOccupied

**发现**: PythonAPI.cs 中 `IsBuildingQueueOccupied` 用反射检测建筑队列占用状态。ARM64 上反射失败 → 永远返回 True → 所有建筑的 `StartProduction` 订单被静默丢弃。这就是为什么模型在远程模式中只造得出 powr（它偶然绕过了 guard）。

**修复**: 移除 `IsBuildingQueueOccupied` guard，游戏引擎自身处理队列占用逻辑。

**连带修复**: `RenderFrame()` 方法删除（`Renderer.BeginFrame` 是 internal 的，无法从 pythonnet 调用）。

### Round 22: RA 前置建筑依赖

**发现**: proc（矿场）需要 powr（电厂）作为前置。RuleBasedAgent 的 state-aware 逻辑先造 proc → 游戏拒绝 → 静默丢弃。powr 能造成因为它是首批可建造建筑之一。

**修复**: Agent 优先建造 powr（`n_powr < 1`），然后 proc（`n_powr >= 1`），然后 barr/weap，最后才允许额外 powr。

### Round 23-28: 经济 Reward 迭代 (V2-V8)

**V2 (纯经济)**: 去掉 goal_aligned reward (w=0.0)，全部靠资产价值 + 资源增长。best=0.1159。
- 13 种动作类型，e1/e2/e3 步兵产线出现
- 但过度建矿场（proc×66）和电厂（powr×45）

**V3 (per-step 冗余惩罚)**: 每步按超出阈值建筑数扣分。best=0.1020。
- powr 从 45 降到 36，proc 从 66 降到 23
- 但惩罚累积导致 total reward 变负

**V4 (一次性建造折扣)**: 建造时打折而非每步扣。best=0.109。
- 回退到 V2 水平

**V5 (BC barr优先)**: Agent 在额外 powr 之前先建 barr+weap。best=0.115。
- 模型大量造 e1 步兵 (×91)
- powr 仍然很多

**V6 (电力规划)**: Agent 在 power_margin<30 时强制 powr。best=0.111。
- powr 回到 133

**V7 (proc 门槛过激)**: proc discount 在 cash>2000 触发。best=0.045。
- 模型跳过 economy 直接 spam weap — 崩了

**V8 (proc 滑梯)**: proc discount >5000→10%, >3000→50%。best=0.121。
- 全部版本最高 reward，9 种动作类型
- 但实际建造率极低（单队列丢弃 80%+ 命令）

**关键教训**: 
- 条件折扣是精妙的但一过性——只在建造那步生效
- "86% produce 率"是假象——统计的是意图，80%+ 被单队列丢弃
- 所有版本都无法克服 frozen encoder 的 state-unawareness

### Round 29: ARCHITECTURE.md + 游戏 Bot 对照

**产出**: [ARCHITECTURE.md](ARCHITECTURE.md) — 完整的 obs/action/reward/model 规格。

**游戏内置 Bot 对照**: BaseBuilderBotModule 有 `MaximumExcessPower`（电厂上限）、`MinimumExcessPower`（电厂下限）、`InititalMinimumRefineryCount`（生产前先造矿场）——这些概念我们一个都没有。

**核心发现（Round 30 已修复第 1 项）**:
1. Scalar observation 缺失 building_counts——模型分不清 1 个和 10 个 powr
2. Reward 缺硬上限——游戏 bot 有 MaximumExcessPower，我们没有
3. BC frozen encoder 是当前最大瓶颈——state-unaware 导致重复造同一建筑
4. 单队列丢弃让训练时的 "action distribution" 完全不可信

### Round 30: Count-aware Scalar Obs + `train_best.ps1` 默认 Goal 配置

**动机**: Round 29 确认 scalar observation 缺失关键计数，policy 看不到已经造了多少电厂/矿场/兵营/重工/矿车/兵。Reward 和 RuleBasedAgent 都在用这些概念，但模型只能从 entity mean-pool 间接推断，导致重复建造和高方差。

**实现**:
- `utils/entity_obs.py`: `SCALAR_BASE_DIM` 从 10 扩到 28；goal conditioning 打开时为 34。
- 新增 scalar 特征：短尺度 cash、signed `power_margin`、`powr/proc/barr/weap/dome+fix/harv` 计数、步兵/车辆总量、`e1/e2+e3`、总建筑/总单位、enabled/empty/busy/done queue 统计。
- `scripts/train_best.ps1`: 默认开启 `--goal-conditioning --goal-aligned-weight 0.6`；新增 `-NoGoalConditioning`、`-GoalAlignedWeight`、`-TeacherKlCoef`、`-TeacherKlAnnealSteps`、`-AddOpponent` 参数；默认 run name 改为 `obs_counts_goal_w...`。

**验证**:
- Python AST parse 通过。
- PowerShell parser 通过。
- Env observation shape 验证：no-goal scalar `(28,)`，goal scalar `(34,)`。
- `git diff --check` 通过。

**实验：`obs_counts_goal_w06`**

命令：

```powershell
.\scripts\train_best.ps1 -Updates 100 -RunName obs_counts_goal_w06
```

结果：

| 指标 | 数值 |
|------|------|
| best | `0.3298 @ update 6` |
| final | `0.2240` |
| mean | `0.2286` |
| last20 | `0.2184` |
| mean KL | `0.1684` |
| early_stop | `100/100` |
| batches | `1.0` |

**判断**:
- 新 obs 没有破坏训练，且明显高于 no-goal/baseline；但它没有复现旧 `goal w=0.6` 的 `0.4037` peak。
- 主要原因不是“count obs 无效”，而是默认脚本配置过激/过短：`NumSteps=128`、warmstart 仅 `3/3`，`TargetKl=0.03`，结果 100/100 updates 都 KL early stop，每轮只吃 1 个 PPO batch。
- 行为上不再只偏向 powr/proc，后期大量探索 `dome/fix/apwr/barr/e2`，说明 count obs 给了策略更多状态区分，但当前 high-entropy + KL 截断让策略无法稳定收敛。

**下一条推荐实验**:

```powershell
.\scripts\train_best.ps1 `
  -Updates 100 `
  -RunName obs_counts_goal_w06_stable `
  -NumSteps 256 `
  -WarmstartEpisodes 10 `
  -WarmstartEpochs 15 `
  -LearningRate 3e-5 `
  -EntCoef 0.005 `
  -TargetKl 0.05 `
  -GoalAlignedWeight 0.6
```

验收重点：
- `early_stop` 不再接近 100/100。
- `num_batches` 平均显著高于 1。
- `last20` 回到 `0.25+`。
- `best` 接近或超过旧 `0.4037`。
- 若稳定后仍 spam `dome/fix/apwr`，下一步给 macro action mask 加硬上限：例如 `dome/fix <= 1`，`apwr` 仅在电力余量不足或特定阶段开放。
