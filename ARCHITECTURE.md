# OpenRA-Bot 架构详解

> 2026-07-01

---

## 1. Observation Space

### 1.1 Entity Features (128 actors, 每 actor 14 维)

```
entities: Box(-1, 1, (128, 14), float32)
entity_mask: Box(0, 1, (128,), bool)
scalar: Box(-1, 1, (28,), float32)       # no goal conditioning
scalar: Box(-1, 1, (34,), float32)       # with 6-dim goal vector
```

每个 actor 的 14 维特征 (`EntityObservationBuilder._encode_actor`):

| 索引 | 名称 | 编码方式 | 取值范围 |
|------|------|---------|---------|
| 0 | owner_self | `1.0 if owner == my_owner else 0.0` | {0, 1} |
| 1 | owner_enemy | `1.0 if owner != my_owner else 0.0` | {0, 1} |
| 2 | is_building | `1.0 if type in _BUILDING_TYPES else 0.0` | {0, 1} |
| 3 | is_infantry | `1.0 if type starts with 'e' or dog/spy/...` | {0, 1} |
| 4 | health_ratio | `hp / max(max_hp, 1)` | [0, 1] |
| 5 | pos_x_norm | `cell_x / 128.0` | [0, ~2] |
| 6 | pos_y_norm | `cell_y / 128.0` | [0, ~2] |
| 7 | can_move | `1.0 if 'move' in available_orders` | {0, 1} |
| 8 | can_attack | `1.0 if 'attack' in available_orders` | {0, 1} |
| 9 | can_produce | `1.0 if 'startproduction' in available_orders` | {0, 1} |
| 10 | can_deploy | `1.0 if 'deploytransform' in available_orders` | {0, 1} |
| 11 | is_idle | `1.0 if 'idle' in order_str` | {0, 1} |
| 12 | type_embed_idx | 稳定的 type 哈希值 (0-997)，归一化到 [0,1] | [0, 1] |
| 13 | type_is_mcv | `1.0 if type == 'mcv'` | {0, 1} |

### 1.2 Scalar Features (28 维 base + 6 维 goal)

`EntityObservationBuilder._encode_scalar`:

| 索引 | 名称 | 编码方式 | 范围 |
|------|------|---------|------|
| 0 | cash | `min(cash / 10000.0, 1.0)` | [0, 1] |
| 1 | resource_fill | `min(res_total / res_cap, 1.0)` | [0, 1] |
| 2 | power_provided | `min(p_prov / 500.0, 1.0)` | [0, 1] |
| 3 | power_drained | `min(p_drain / 500.0, 1.0)` | [0, 1] |
| 4 | power_critical | `1.0 if 'critical' in p_state` | {0, 1} |
| 5 | power_low | `1.0 if 'low' in p_state` | {0, 1} |
| 6 | active_production | `min(active_items, 5) / 5.0` | [0, 1] |
| 7 | has_empty_queue | `1.0 if any queue has 0 items` | {0, 1} |
| 8 | has_done_item | `1.0 if any item has Done=True` | {0, 1} |
| 9 | game_time | `world_tick / 20000.0` | [0, 1] |
| 10 | cash_short | `min(cash / 3000.0, 1.0)` | [0, 1] |
| 11 | power_margin | `clip((provided - drained) / 200.0, -1, 1)` | [-1, 1] |
| 12 | n_powr | `(powr + apwr) / 8` | [0, 1] |
| 13 | n_proc | `proc / 6` | [0, 1] |
| 14 | n_barr | `(barr + tent) / 4` | [0, 1] |
| 15 | n_weap | `weap / 4` | [0, 1] |
| 16 | n_tech_support | `(dome + fix) / 4` | [0, 1] |
| 17 | n_harv | `harv / 12` | [0, 1] |
| 18 | n_infantry | infantry total / 50 | [0, 1] |
| 19 | n_e1 | `e1 / 30` | [0, 1] |
| 20 | n_e2_e3 | `(e2 + e3) / 24` | [0, 1] |
| 21 | n_vehicles | vehicle total / 30 | [0, 1] |
| 22 | total_buildings | owned buildings / 24 | [0, 1] |
| 23 | total_units | owned mobile units / 80 | [0, 1] |
| 24 | enabled_queues | enabled queues / 8 | [0, 1] |
| 25 | empty_queues | empty enabled queues / 8 | [0, 1] |
| 26 | busy_queues | busy enabled queues / 8 | [0, 1] |
| 27 | done_items | completed production items / 8 | [0, 1] |
| 28-31 | goal_onehot | goal conditioning 的 one-hot 编码 (4 goals) | {0, 1}⁴ |
| 32 | goal_building_weight | goal 的建筑权重 (0.1-1.0) | [0, 1] |
| 33 | goal_unit_weight | goal 的兵力权重 (0.1-0.7) | [0, 1] |

**2026-07-01 更新**: 原先的 building/unit/harvester count 盲区已经补上。`obs_counts_goal_w06` 证明新 obs 可以稳定跑通，但默认 PPO 配置过激，100/100 updates 触发 KL early stop；后续需要先稳定 PPO 更新，再判断计数特征对最终策略质量的贡献。

---

## 2. Model Architecture

### 2.1 SimpleEntityEncoder

```
entity_dim=14, scalar_dim=28 or 34, feature_dim=256

Entity path:
  128 × 14 → MLP(14→128→128) → (128*N, 128)
  → masked_mean_pool → (B, 128)

Scalar path:
  scalar_dim → MLP(scalar_dim→64→128) → (B, 128)

Fusion:
  concat(entity_128, scalar_128) = (B, 256)
  → MLP(256→256) → (B, 256)
```

### 2.2 ActorCritic

```
SimpleEntityEncoder (frozen after BC warmstart)
  → Linear(256, 256) core
  → MultiDiscretePolicy: 6 heads
      action_type: Linear(256, 18)
      unit_idx:    Linear(256, 100)
      target_x:    Linear(256, 128)
      target_y:    Linear(256, 128)
      target_idx:  Linear(256, 100)
      unit_type:   Linear(256, 7)
  → value_head: MLP(256→256→1)
  + GLU gate: Linear(scalar_dim, 18) — goal-conditioned action_type modulation
  + Multi-value heads: value_head_asset/goal/prod/other (all Linear(256→128→1))
```

总参数量: ~300K (大部分在 action heads)

---

## 3. Action Space

### 3.1 Macro Mode (当前使用)

```
action_space = MultiDiscrete([18, 100, 128, 128, 100, 7])
```

Macro 模式下只有 action_type[0] 有意义（其他 5 个头被 mask 强制为 0）。

```
MACRO_PRODUCE_TYPES = [
    'powr',   # 电厂      $300
    'proc',   # 矿场      $1400
    'barr',   # 兵营      $400
    'tent',   # 帐篷兵营   $400
    'weap',   # 重工      $2000
    'dome',   # 雷达      $1000
    'fix',    # 修理厂    $1200
    'apwr',   # 高级电厂   $500
    'e1',     # 步枪兵    $100
    'e3',     # 手雷兵    $300
    'e2',     # 火箭兵    $160
    '1tnk',   # 轻型坦克  $700
    '2tnk',   # 中型坦克  $850
    '3tnk',   # 重型坦克  $950
    'jeep',   # 吉普车    $600
    'arty',   # 火炮      $600
    'harv',   # 矿车      $1100
]

action_types = ['noop'] + ['produce:' + t for t in MACRO_PRODUCE_TYPES]
# 共 18 个动作: noop(0) + 17 produce types
```

### 3.2 动作执行流程

```
env.step(np.array([action_type_idx, 0, 0, 0, 0, 0]))
  → _execute_multidiscrete_action(action)
    → atype = action_types[action_type_idx]  # e.g. 'produce:powr'
    → _auto_deploy_mcv()  # 如需要则先部署MCV
    → _auto_place_done_items()  # 放置已完成的建筑
    → if atype.startswith('produce:'):
        wanted = atype.split(':')[1]  # 'powr'
        producer_id = _find_producer_for(wanted)
        if producer_id is not None:
           send StartProduction(producer_id, wanted)
  → api.Step() × ticks_per_step
```

### 3.3 Action Mask (单队列限制)

`_get_macro_action_mask`:
- noop (索引0) 永远合法
- produce:<t> 合法 ⇔ 存在启用的空队列 (len(Items)==0) 且 Producible 包含 t
- 队列有 ≥1 item → 所有 produce 被 mask → 模型只能选 noop

---

## 4. Reward Function

### 4.1 完整公式

```
total_reward = asset_reward
             + resource_reward
             + production_start_reward
             + production_active_reward
             - production_cancel_penalty
             - idle_cash_penalty
             - power_deficit_penalty (如 power_margin < 0)
             + goal_aligned_reward
```

### 4.2 各项详解

**asset_reward** (主信号):
```
gained = AssetValueTracker.update(raw)   # 新增 actors 的造价总和
asset_reward = asset_value(2.0) × gained / asset_value_scale(1000.0)

# 条件折扣 (一次性，建造时触发):
if powr 新造 and power_margin > 60:  asset_reward *= 0.1
elif powr 新造 and power_margin > 30: asset_reward *= 0.5
if proc 新造 and cash > 5000:         asset_reward *= 0.1
elif proc 新造 and cash > 3000:       asset_reward *= 0.5
if dome 新造 and n_dome > 1:          asset_reward *= 0.0
```

**resource_reward**:
```
delta = resources_now - resources_last_step  # 采矿收入增长
resource_reward = resource_growth_weight(1.0) × max(0, delta) / 1000.0
```

**production_start_reward**:
```
新 item 进入生产队列时:
reward = produce_start(0.5) × damp_factor
damp_factor: queue_len < 6 → 1.0, else → 0.5
```

**production_active_reward**:
```
每个活跃生产项每步:
reward = producing_per_step(0.005) × active_item_count
```

**production_cancel_penalty**:
```
cancel_count = 被取消的 in-progress item 数
penalty = produce_cancel_penalty(0.1) × cancel_count
```

**power_deficit_penalty**:
```
if power_margin < 0:
  penalty = 0.01 × abs(power_margin) / 20.0
```

**goal_aligned_reward** (goal_conditioning 启用时):
```
goal_building_progress = Σ min(owned_bld[type] / target, 1.0) / N_buildings
goal_unit_progress     = Σ min(owned_unit[type] / target, 1.0) / N_units
total_goal_progress = bld_weight × bld_progress + unit_weight × unit_progress
reward = goal_aligned_weight(0.0-0.6) × total_goal_progress
```

### 4.3 当前权重值

```python
reward_weights = {
    'asset_value': 2.0,
    'asset_value_scale': 1000.0,
    'resource_growth_weight': 1.0,
    'overbuild_penalty': 0.002,
    'produce_start': 0.5,
    'produce_cancel_penalty': 0.1,
    'producing_per_step': 0.005,
    'idle_cash_penalty': 0.0,      # 关闭
    'power_deficit_penalty': 0.0,   # 关闭
    'goal_aligned_weight': 0.0,     # goal 仅作 observation
}
```

---

## 5. 已知问题

### 5.1 Observation / Policy 问题

| 问题 | 状态 | 影响 |
|------|------|------|
| building_counts (powr/proc/barr/...) | 已补入 scalar 12-16,22 | 模型现在能区分 1 个和多个关键建筑 |
| unit_counts (e1/e2/e3/vehicle/harv) | 已补入 scalar 17-23 | 模型现在能看到兵力和矿车规模 |
| power_margin | 已补入 scalar 11 | 模型能直接看到电力余量 |
| cash_absolute | 部分缓解，新增 cash_short | `$500` vs `$5000` 的区分更明显，但仍是归一化连续特征 |
| PPO 更新稳定性 | 未解决 | `obs_counts_goal_w06` 全部 update early stop，新增 obs 的收益被 KL 截断限制 |

### 5.2 Reward 问题

1. **powr 缺硬上限**: margin<60 就给全额奖励，模型造到 margin 刚好 >60 为止。游戏 bot 用 `MaximumExcessPower` 杜绝
2. **proc 缺硬上限**: cash<3000 全额，但矿场回本后 cash 很快超过 3000，然后新 proc 只有 50% 奖励
3. **折扣是一过性的**: 只在建造那个 step 打折，不影响后续步骤。4 个 powr 后的第 5 个 powr 被打折，但第 6 个如果 margin 又 <30 则可以全额
4. **barr/weap 无折扣但是也不造**: 模型根本不选 barr/weap——策略本身的 powr 偏好太强

### 5.3 策略问题

1. **BC warmstart 冻结 encoder**: encoder 无法学习 state-dependent features
2. **BC 数据偏差**: RuleBasedAgent 96% 时间 noop，produce 中 90%+ 是 powr
3. **单队列丢弃**: 80%+ produce 命令被静默丢弃，训练时模型不知道"我的命令没生效"
4. **多步延时**: 从 produce:powr 到建筑完成约 20 步，credit assignment 困难
