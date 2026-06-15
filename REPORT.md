# OpenRA-Bot 实验报告

> **日期**: 2026-06-15
> **状态**: PPO 训练稳定，学到 RuleBasedAgent 基线水平。瓶颈在 observation 表达力。

## 目录

1. [环境与运行条件](#1-环境与运行条件)
2. [实验时间线](#2-实验时间线)
3. [关键发现与修复](#3-关键发现与修复)
4. [当前架构状态](#4-当前架构状态)
5. [训练结果汇总](#5-训练结果汇总)
6. [下次 Session 接续指南](#6-下次-session-接续指南)

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

### 未解决问题

- **PPO credit assignment 极限**: 无论 flat vector 还是 entity obs，PPO 都无法在 15 决策点/300 步的稀疏场景中超越 BC 基线。需要简化 action space 或换算法
- **无 headless 模式**: 每次训练打开 OpenGL 窗口

---

## 6. 下次 Session 接续指南

### 启动训练

```bash
cd /Users/sharedcare/Projects/OpenRA/OpenRA-Bot
rm -rf checkpoints && mkdir checkpoints
python scripts/train_rl.py \
    --num-steps 512 --total-updates 80 --max-episode-ticks 1500 \
    --warmstart-episodes 10 --warmstart-epochs 15
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

### 下一步计划 (Phase 1)
参考 `PLAN.md` Phase 1，将 flat vector observation 升级为 entity-based + spatial 结构。
