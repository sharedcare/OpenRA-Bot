# OpenRA-Bot: AlphaStar-Style RTS RL Agent — 完整实施方案

> **目标**: 从当前研究基线演进为具备 AlphaStar 级别架构能力的 RTS RL agent 框架。
> **原则**: 渐进式改造，每阶段产出可运行的中间版本，不追求一步到位。

---

## 目录

1. [架构总览](#1-架构总览)
2. [Phase 0: 紧急修复与基线加固](#2-phase-0-紧急修复与基线加固)
3. [Phase 1: 观察空间升级 — Entity + Spatial](#3-phase-1-观察空间升级--entity--spatial)
4. [Phase 2: 动作空间升级 — Hierarchical Autoregressive + Spatial Head](#4-phase-2-动作空间升级--hierarchical-autoregressive--spatial-head)
5. [Phase 3: 网络架构升级 — Transformer + Pointer Network](#5-phase-3-网络架构升级--transformer--pointer-network)
6. [Phase 4: 训练基础设施 — 并行环境 + Self-Play](#6-phase-4-训练基础设施--并行环境--self-play)
7. [Phase 5: 训练策略 — League Training + Curriculum + 奖励重构](#7-phase-5-训练策略--league-training--curriculum--奖励重构)
8. [Phase 6: 引擎桥接优化](#8-phase-6-引擎桥接优化)
9. [风险与依赖](#9-风险与依赖)
10. [里程碑和检查点](#10-里程碑和检查点)

---

## 1. 架构总览

### 1.1 目标架构 (Phase 5 完成时)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python Training Loop                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Env 0   │  │  Env 1   │  │  Env 2   │  │  Env N   │  ...   │
│  │ (local)  │  │ (local)  │  │ (local)  │  │ (remote) │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │             │             │                │
│       └─────────────┴──────┬──────┴─────────────┘                │
│                            │                                     │
│              ┌─────────────▼──────────────┐                      │
│              │    ImpalaStore / Queue      │                      │
│              │   (async rollout + train)   │                      │
│              └─────────────┬──────────────┘                      │
│                            │                                     │
│              ┌─────────────▼──────────────┐                      │
│              │      PPO / IMPALA / PBT    │                      │
│              │    (distributed training)   │                      │
│              └─────────────┬──────────────┘                      │
│                            │                                     │
│              ┌─────────────▼──────────────┐                      │
│              │      League Manager         │                      │
│              │   (self-play + Elo + MMR)   │                      │
│              └────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Neural Network                                │
│                                                                  │
│   ┌───────────┐    ┌───────────┐    ┌───────────┐               │
│   │  Entity   │    │  Spatial  │    │  Scalar   │               │
│   │ Encoder   │    │  Encoder  │    │  Encoder  │               │
│   │(Transfrmr)│    │  (ResNet) │    │  (MLP)    │               │
│   └─────┬─────┘    └─────┬─────┘    └─────┬─────┘               │
│         │               │               │                        │
│         └───────────────┼───────────────┘                        │
│                         │                                        │
│              ┌──────────▼──────────┐                             │
│              │   Deep LSTM Core    │                             │
│              └──────────┬──────────┘                             │
│                         │                                        │
│     ┌───────────────────┼───────────────────┐                    │
│     │                   │                   │                    │
│  ┌──▼──┐          ┌─────▼─────┐       ┌─────▼─────┐              │
│  │Value│          │Action Type│       │Arguments  │              │
│  │Head │          │   Head    │       │  Heads    │              │
│  └─────┘          └─────┬─────┘       └─────┬─────┘              │
│                         │                   │                    │
│              ┌──────────▼───────┐   ┌───────▼──────────┐        │
│              │  delayed_action  │   │  unit_selection  │        │
│              │  (e.g., build)   │   │  (Pointer Net)   │        │
│              └──────────────────┘   └───────┬──────────┘        │
│                                             │                    │
│                               ┌─────────────▼─────────────┐     │
│                               │        target_head        │     │
│                               │  ┌─────────┐ ┌─────────┐  │     │
│                               │  │ Spatial │ │ Entity  │  │     │
│                               │  │ (2D map)│ │(Pointer)│  │     │
│                               │  └─────────┘ └─────────┘  │     │
│                               └───────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 与 AlphaStar 的对照

| 组件 | AlphaStar | 我们的目标 |
|------|-----------|-----------|
| Entity Encoder | Transformer over units, 无固定容量 | Transformer over visible actors |
| Spatial Encoder | ResNet over minimap + camera | ResNet over semantic grid |
| Scalar Encoder | MLP over economy/tech stats | MLP over cash/power/supply |
| Core | 3-layer LSTM 384-hidden | 2-3 layer LSTM 256-512 hidden |
| Action type | Categorical over ~30 function types | Categorical over 8-15 action types |
| Unit selection | Pointer network (attention) | Pointer network (attention) |
| Spatial target | 2D logits map over minimap | 2D logits map 128×128 |
| Training | IMPALA+V-trace (off-policy) | PPO first, then upgrade to IMPALA |
| Parallelism | 1000s TPU actors | 8-64 CPU actors (budget-constrained) |
| Opponents | League with past/historical/exploiter | League with past checkpoints + built-in bots |

---

## 2. Phase 0: 紧急修复与基线加固

**工期**: 1-2 周
**目标**: 修复已知 bug，确保 RL 在纯发展场景下能学到东西，建立可度量基线。

> **背景（2024-06 调试发现）**: 经过 7 轮 PPO 训练迭代发现以下关键问题：
> 1. **MCV 类型检测 Bug**: `PythonAPI.cs` 通过反射检测 `IMove` 接口在 ARM64 上失败，MCV 被误归类为建筑，deploy reward 从未触发
> 2. **80% 步为被迫 Noop**: 生产等待期间 agent 无选择权，policy gradient 在这些步上贡献零梯度+噪声，淹没了 20% 决策步的信号
> 3. **Reward 信号太稀疏**: RuleBasedAgent 150 步只有 1 步有非零 reward，无法提供有效的 credit assignment
> 4. **Action 极度不平衡**: 93% noop vs 7% 有效动作，策略塌缩到全 noop

### 2.0a MCV / 建筑类型检测修复 ✅

**文件**: `envs/openra_env.py` — `_compute_development_metrics()`

**问题**: `PythonAPI.cs` 用反射检测 `IMove` 接口判断单位是否可移动，在 macOS ARM64 上失败。MCV 的 `available_orders` 缺少 `Move`，被归类为建筑。

**方案（已实施）**: 用类型白名单替代反射启发式。

```python
# 类型白名单 — 比反射更可靠，不依赖 C# bridge 的内部实现
_BUILDING_TYPES = {'fact', 'powr', 'proc', 'barr', 'tent', 'weap',
                   'afld', 'spen', 'syrd', 'dome', ...}
_MOBILE_TYPES = {'mcv', 'e1', 'e2', 'jeep', '1tnk', ...}

def _is_building(self, actor):
    atype = actor['type'].lower()
    if atype in self._BUILDING_TYPES: return True
    if atype in self._MOBILE_TYPES:   return False
    # fallback: order heuristic
    return 'move' not in actor.get('available_orders', [])
```

**验证**: RuleBasedAgent 的 deploy 奖励从 0 变为 +1.2，确认修复有效。

### 2.0b 决策步 Policy Gradient Masking 🔴 P0

**文件**: `agent/agent.py` — PPO loss 计算

**问题**: 80% 的步只有 noop 一个合法动作（生产等待期间）。PPO 在这些步上计算 policy_loss，贡献零梯度+噪声，淹没了 20% 决策步的信号。这直接导致 produce 率无法提升。

**方案**: 只在有多于 1 个合法动作的步上计算 policy gradient。

```python
# 在 PPO loss 计算中（agent/agent.py 约 line 852）
decision_mask = (action_type_mask.sum(dim=-1) > 1).float()
valid_decision_steps = decision_mask.sum().clamp_min(1)
policy_loss = -(torch.min(unclipped, clipped) * mb_attention_mask * decision_mask).sum() / valid_decision_steps
```

**改动量**: ~5 行。预期效果：消除 80% 噪声梯度，让 produce 信号突出。

### 2.0c Build Order Distance Reward（参考 DI-star）🔴 P0

**文件**: `envs/openra_env.py` — 新增 `BuildOrderTracker`

**问题**: 当前 reward 只在 deploy 和 building_placed 时非零。RuleBasedAgent 150 步只有 1 次 reward 事件。DI-star 用 Levenshtein 距离到目标建造序列来衡量进度，reward = 距离改善量，每个建造动作立即得到反馈。

**方案**:

```python
TARGET_BUILD_ORDER = ['powr', 'proc', 'barr']  # RA 标准开局

class BuildOrderTracker:
    def __init__(self, target):
        self.target = [t.lower() for t in target]
        self.built = []
        self._last_distance = len(target)
    
    def update(self, actor_type):
        """记录新建筑，返回距离改善量作为 reward"""
        atype = actor_type.lower()
        if atype not in self.target:
            return 0.0
        self.built.append(atype)
        new_dist = self._levenshtein_match(self.built, self.target)
        reward = self._last_distance - new_dist
        self._last_distance = new_dist
        return float(reward)
    
    def _levenshtein_match(self, built, target):
        """计算已建序列与目标前缀的最长匹配子序列"""
        match = 0
        for b in built:
            if b in target[match:]:
                match = target.index(b, match) + 1
        return len(target) - match
```

**优势**: 每个正确建造动作立即得到 +1 级别的大 reward（不用等 20 步），信号密度从 1/150 提升到 ~10/150。

### 2.1 修复 target_x / target_y 独立分解问题

**文件**: `envs/openra_env.py` — `_get_action_mask()` 和 `_sample_action()`
**文件**: `models/actor.py` — `MultiDiscretePolicy`
**文件**: `agent/agent.py` — `_build_effective_masks()`

**问题**: 当前 `target_x_mask` 和 `target_y_mask` 是独立的一维 mask，导致合法 x 坐标可以和非法 y 坐标组合。

**方案**: 将动作空间改为：

```python
# 当前（错误）:
self.action_space = spaces.MultiDiscrete([
    action_types, 100, 128, 128, 100, unit_types
])
#                                         ^^^^  ^^^^
#                                    target_x  target_y  (独立分解)

# 改进（Phase 0 快速修复）:
# 将 target_x 和 target_y 合并为一个 head
self.action_space = spaces.MultiDiscrete([
    action_types,        # 操作类型
    100,                 # unit_idx
    128 * 128,           # target_cell (联合 2D 坐标，展平)
    100,                 # target_idx
    unit_types           # unit_type
])
```

同时在 mask 构造中：

```python
# 构造联合 2D mask
target_cell_mask = np.zeros((100, 128 * 128), dtype=np.uint8)
for i in range(n_units):
    for cell in valid_cells_for_unit_i:
        flat_idx = cell[1] * 128 + cell[0]
        target_cell_mask[i, flat_idx] = 1
```

**注意**: 这只是临时修复。Phase 2 会用真正的 spatial action head 替代。

### 2.2 完整加载 unit_types 映射

**文件**: `envs/openra_env.py` — `_init_unit_types_from_csv()`

**问题**: `actors.csv` 加载被注释掉了 (`# self._init_unit_types_from_csv()`), fallback 只有 7 种类型。

**方案**:
1. 取消注释 `_init_unit_types_from_csv()` 调用
2. 修复 `actors.csv` 加载路径
3. 添加运行时验证：如果 CSV 不可用，从 `producible_catalog` 动态构建类型映射

```python
def _init_unit_types_from_csv(self) -> None:
    # ... 现有代码 ...
    
def _ensure_unit_types(self, raw: Dict[str, Any]) -> None:
    """如果 CSV 加载失败，从游戏数据动态构建类型映射。"""
    if self.unit_types and len(self.unit_types) > 10:
        return  # 已从 CSV 成功加载
    
    catalog = raw.get('producible_catalog') or []
    for b in catalog:
        name = str(b.get('Name', '')).lower()
        if name and name not in self.unit_types:
            self.unit_types[name] = len(self.unit_types)
```

### 2.3 添加 Combat Reward 和 Terminal Reward

**文件**: `envs/openra_env.py` — `_compute_reward()`

**当前问题**: 只有开发向 reward，没有战斗和胜负信号。

**方案**:

```python
def _compute_reward(self, raw, action=None):
    # ... 现有的开发 reward ...
    
    # === 新增: Combat reward ===
    combat_r = self._compute_combat_reward(raw)
    
    # === 新增: Terminal reward ===
    terminal_r = self._compute_terminal_reward(raw)
    
    return dev_r + combat_r + terminal_r

def _compute_combat_reward(self, raw):
    """基于敌方单位/建筑损失的战斗奖励"""
    current_enemy = self._count_enemy_actors(raw)
    prev_enemy = self._prev_metrics.get('enemy_total')
    
    reward = 0.0
    if prev_enemy is not None:
        # 敌方损失 = 正奖励
        enemy_loss = prev_enemy - current_enemy
        if enemy_loss > 0:
            reward += enemy_loss * self.reward_weights.get('enemy_kill', 0.5)
    
    self._prev_metrics['enemy_total'] = current_enemy
    return reward

def _compute_terminal_reward(self, raw):
    """检测游戏结束条件"""
    my_units = self._count_my_military(raw)
    enemy_units = self._count_enemy_military(raw)
    
    if my_units == 0:
        return self.reward_weights.get('loss_penalty', -5.0)
    if enemy_units == 0:
        return self.reward_weights.get('win_bonus', 5.0)
    return 0.0
```

### 2.4 建立 Baselines

**文件**: 新增 `scripts/baselines/`

```python
# scripts/baselines/run_baselines.py
"""
运行和记录基线性能：

1. Random agent: 100 episodes
2. Rule-based agent: 100 episodes  
3. Current PPO best checkpoint: 100 episodes
4. Built-in bot (HAL 9001 or similar): 100 episodes

输出: baselines.csv (reward, units, buildings, survival_time per episode)
"""
```

### 2.5 修复 Docker/dev 环境问题

- 修复 `scripts/deploy.py` (当前是死文件，只有 1 行错误代码)
- 添加 `scripts/setup.py` 自动检测和配置 bin_dir
- 创建 `tests/` 目录，添加基础 smoke tests:
  - `test_env_creation.py`: 验证环境可以创建
  - `test_engine_load.py`: 验证 engine 可以加载
  - `test_obs_shape.py`: 验证 observation 形状正确
  - `test_action_mask.py`: 验证 action masks 非空且合理

### 2.6 Phase 0 产出物

- [x] MCV 类型检测修复（`_is_building` 白名单）
- [x] 决策步 mask（policy gradient 只在有选择权的步上计算）
- [x] Build Order distance reward（参考 DI-star Levenshtein 匹配）
- [x] Mask penalty `-inf` 修复（防止策略塌缩到 produce spam）
- [x] Reward 信号密度从 0.7% 提升到 98%+
- [x] PPO 训练稳定（KL max=0.199，零塌缩，零 attack spam）
- [x] 更新的 README (包含当前基线指标)
- [x] 实验报告 `REPORT.md`
- [ ] target_cell 联合 2D mask (临时修复)
- [ ] 完整 unit_types 映射
- [ ] Combat + terminal reward（推迟到加 bot 对手后）
- [ ] Baseline 评估脚本和结果
- [ ] 基础测试套件

### 2.7 Phase 0 优先级顺序

```
🔴 P0 (阻塞训练 — 必须先修):
  2.0a MCV 类型检测 ✅ (已完成)
  2.0b 决策步 mask
  2.0c BO distance reward

🟡 P1 (改善信号):
  2.1  target_cell 联合 mask
  2.2  完整 unit_types
  2.3  combat reward（可推迟到加 bot 对手后）

🟢 P2 (基础设施):
  2.4  baselines
  2.5  Docker/dev env + tests
```

---

## 3. Phase 1: 观察空间升级 — Entity + Spatial

**工期**: 3-4 周
**目标**: 将观察从 fixed-size flat vector 升级为 entity-based + spatial 结构。

### 3.1 设计新的观察空间

#### 3.1.1 Entity Observation

每个 actor 编码为固定维度的特征向量：

```python
ENTITY_FEATURES = [
    # 类型特征 (learnable embedding indices)
    'actor_type_id',          # str -> int mapping, for embedding
    'actor_type_category',    # enum: {infantry, vehicle, building, defense, resource}
    
    # 所有权
    'owner_relation',         # enum: {self, ally, enemy, neutral}
    
    # 位置 (normalized)
    'pos_x',                  # float [0, 1]
    'pos_y',                  # float [0, 1]
    'pos_z',                  # float (for flying units)
    
    # 状态
    'health_ratio',           # float [0, 1]
    'shield_ratio',           # float [0, 1] (if applicable)
    
    # 动作/能力
    'is_idle',                # bool
    'can_move',               # bool
    'can_attack',             # bool
    'can_produce',            # bool
    'current_order_id',       # int (which order is being executed)
    
    # 战斗相关
    'weapon_range',           # float normalized
    'weapon_damage',          # float normalized
    'attack_cooldown_remain', # float [0, 1]
    
    # 经济相关 (buildings)
    'is_producing',           # bool
    'production_progress',    # float [0, 1]
    'has_done_item',          # bool
    
    # 时间相关
    'time_since_last_order',  # float normalized
]
# 总计: ~20 维原始特征
```

**Python 侧实现** (`utils/entity_obs.py` 新增):

```python
class EntityObservationBuilder:
    """将 raw RLState 转换为 entity-based 观察。"""
    
    def __init__(self, unit_types: Dict[str, int], max_entities: int = 200):
        self.unit_types = unit_types
        self.max_entities = max_entities
        # Learnable 的类型 embedding 索引
        self._type_to_idx = self._build_type_index()
    
    def build(self, raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Returns:
            entities: (max_entities, entity_feature_dim) float32
            entity_mask: (max_entities,) bool
        """
        actors = raw.get('actors') or []
        my_owner = raw.get('my_owner', -1)
        production = raw.get('production') or {}
        
        n = min(len(actors), self.max_entities)
        entities = np.zeros((self.max_entities, ENTITY_FEATURE_DIM), dtype=np.float32)
        mask = np.zeros(self.max_entities, dtype=bool)
        
        for i, actor in enumerate(actors[:n]):
            entities[i] = self._encode_actor(actor, my_owner, production)
            mask[i] = True
        
        return {'entities': entities, 'entity_mask': mask}
    
    def _encode_actor(self, actor, my_owner, production):
        # ... 编码逻辑 ...
```

#### 3.1.2 Spatial Observation (Minimap)

扩展当前 128×128×10 的 image obs 到更丰富的通道：

```python
# 新通道布局 (C=18)
SPATIAL_CHANNELS = {
    # 地形 (来自 Map)
    0:  'terrain_height',         # 高度图
    1:  'terrain_impassable',     # 不可通过区域
    
    # 资源
    2:  'resource_ore',           # 矿物密度
    3:  'resource_gems',          # 宝石密度
    
    # 己方单位
    4:  'self_infantry',          # 己方步兵密度
    5:  'self_vehicle',           # 己方车辆密度
    6:  'self_building',          # 己方建筑
    7:  'self_hp_low',            # 己方低血量单位
    
    # 敌方可见单位
    8:  'enemy_infantry',
    9:  'enemy_vehicle',
    10: 'enemy_building',
    11: 'enemy_hp_low',
    
    # 战争迷雾
    12: 'fog_of_war',             # 未探索
    13: 'shroud',                 # 已探索但不可见
    
    # 可建造性
    14: 'buildable',              # 当前可建造区域
    
    # 战斗
    15: 'threat_enemy',           # 敌方威胁范围
    16: 'weapon_range_self',      # 己方武器覆盖
    
    # 全局状态 proj
    17: 'camera_focus',           # 当前注意力焦点 (可学习)
}
```

**引擎侧改造** (`PythonAPI.cs`):

需要在 `GetState()` 中添加地形提取：

```csharp
// 新增: 提取地形数据
public static int[,] GetTerrainGrid()
{
    var world = Game.OrderManager?.World;
    if (world == null) return new int[0, 0];
    
    var map = world.Map;
    var grid = new int[map.MapSize.X, map.MapSize.Y];
    foreach (var cell in map.AllCells)
    {
        var terrain = map.Tiles[cell];
        grid[cell.X, cell.Y] = terrain.Type; // 地形类型索引
    }
    return grid;
}
```

#### 3.1.3 Scalar Observation

```python
SCALAR_FEATURES = [
    'cash_normalized',           # 现金 / 10000
    'resource_fill_ratio',       # 资源存储 / 容量
    'power_balance',             # (提供 - 消耗) / max(提供, 1)
    'power_state_normal',        # bool
    'power_state_low',           # bool
    'power_state_critical',      # bool
    'supply_used',               # 已用人口 (OpenRA 可能没有此概念)
    'supply_cap',                # 人口上限
    'num_my_units',              # normalized
    'num_my_buildings',          # normalized
    'num_enemy_visible',         # normalized
    'game_time_ratio',           # 游戏时间 / max_episode_ticks
]
# 总计: ~12 维
```

#### 3.1.4 完整的 Gym Observation Space

```python
class ObservationSpace(gym.spaces.Dict):
    def __init__(self, max_entities=200, entity_dim=20, 
                 spatial_h=128, spatial_w=128, spatial_c=18,
                 scalar_dim=12):
        super().__init__({
            'entities': spaces.Box(-1, 1, (max_entities, entity_dim)),
            'entity_mask': spaces.Box(0, 1, (max_entities,), dtype=bool),
            'spatial': spaces.Box(0, 255, (spatial_h, spatial_w, spatial_c), dtype=np.uint8),
            'scalar': spaces.Box(-1, 1, (scalar_dim,)),
        })
```

### 3.1.5 实施建议：从最小可行集开始

当前发展阶段（纯建造，无对手），entity 特征中以下维度是零或无用：
- `weapon_range`, `weapon_damage`, `attack_cooldown_remain` — 无战斗
- `shield_ratio`, `pos_z` — OpenRA RA mod 无护盾/飞行单位
- `owner_relation`（enemy/ally）— 无其他玩家

**建议**: Phase 1 先用 ~12 维最小可行 entity 特征，等 Phase 3 加 Transformer 时扩展。

同样，spatial channels 中的 `threat_enemy`(15)、`weapon_range_self`(16) 在没有对手时全为零，建议分阶段启用。

### 3.2 实现文件结构

```
OpenRA-Bot/
├── utils/
│   ├── entity_obs.py       # 新增: Entity observation builder
│   ├── spatial_obs.py      # 新增: Spatial observation builder
│   ├── scalar_obs.py       # 新增: Scalar observation builder
│   └── obs.py              # 修改: 整合三种 builder
├── envs/
│   └── openra_env.py       # 修改: 支持新的 observation_space
```

### 3.3 Phase 1 产出物

- [ ] `utils/entity_obs.py` — Entity observation builder
- [ ] `utils/spatial_obs.py` — 扩展的空间观察 (含地形)
- [ ] `utils/scalar_obs.py` — Scalar 特征提取
- [ ] `openra_env.py` — 支持 `observation_type="entity"` 和 dict space
- [ ] Engine 侧: `GetTerrainGrid()` 方法
- [ ] 测试: 验证所有 observation 形状和数值范围
- [ ] 向后兼容: `observation_type="vector"` 仍然工作

---

## 4. Phase 2: 动作空间升级 — Hierarchical Autoregressive + Spatial Head

**工期**: 3-4 周
**目标**: 用 hierarchical autoregressive 动作空间替代当前的 factorized MultiDiscrete。

### 4.1 新的动作结构

```python
action = {
    # Level 0: 选择操作类型
    'action_type': Categorical([
        'noop',
        'move',              # 移动单位到空间目标
        'attack',            # 攻击敌方单位
        'harvest',           # 采集资源
        'build',             # 放置建筑
        'produce',           # 开始生产
        'cancel_production', # 取消生产
        'deploy',            # 部署/变形
        'repair',            # 修理
        'sell',              # 卖掉建筑
        'guard',             # 守护单位/区域
        'stop',              # 停止当前命令
    ]),
    
    # Level 1: 选择单位 (条件化于 action_type != noop)
    'unit_selection': Categorical(n_entities),  # pointer network 输出
    
    # Level 2: 目标 (条件化于 action_type)
    'target_spatial': Categorical(128 * 128),  # 用于 move, build, guard
    'target_entity': Categorical(n_entities),  # 用于 attack, repair, guard
    'unit_type': Categorical(n_unit_types),    # 用于 produce, build
    
    # Level 2: 队列管理
    'queued': Categorical(2),  # 是否加入队列
}
```

### 4.2 网络中的 Autoregressive 实现

关键是**按依赖顺序采样**，每步将之前的采样结果 feedback 给后续 head：

```python
# models/action_head.py (新增)

class HierarchicalActionHead(nn.Module):
    def __init__(self, d_model, action_type_dim, max_entities, 
                 spatial_h, spatial_w, unit_type_dim):
        super().__init__()
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        
        # Action type head
        self.action_type_head = nn.Linear(d_model, action_type_dim)
        
        # Unit selection (Pointer Network style)
        self.unit_query = nn.Linear(d_model, d_model)
        self.unit_keys = nn.Linear(d_model, d_model)
        
        # Spatial target head (deconvolution network)
        self.spatial_head = SpatialActionHead(d_model, spatial_h, spatial_w)
        
        # Entity target head (conditioned on action type + selected unit)
        self.target_entity_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # d_model (state) + d_model (selected_unit)
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Unit type selection
        self.unit_type_head = nn.Linear(d_model, unit_type_dim)
        
        # Embeddings for action type (for conditioning)
        self.action_type_embed = nn.Embedding(action_type_dim, d_model)
    
    def forward(self, features, entity_embeddings, entity_mask, 
                spatial_features, action_type=None):
        """
        Args:
            features: (B, d_model) — LSTM 输出
            entity_embeddings: (B, N, d_model) — encoder 输出的实体嵌入
            entity_mask: (B, N) — 哪些实体有效
            spatial_features: (B, C, H, W) — spatial encoder 输出
            action_type: 训练时提供，推理时为 None
        """
        # ... autoregressive 推理/采样逻辑
```

### 4.3 Autoregressive 采样流程

```python
def sample_action(self, features, entity_embeddings, entity_mask, 
                  spatial_features, action_masks):
    """Autoregressive action sampling."""
    B = features.shape[0]
    device = features.device
    
    # Step 1: Action type
    action_type_logits = self.action_type_head(features)
    action_type_logits = apply_mask(action_type_logits, action_masks['action_type'])
    action_type = Categorical(logits=action_type_logits).sample()
    
    # Step 2: Condition features on action type
    action_embed = self.action_type_embed(action_type)  # (B, d_model)
    cond_features = features + action_embed
    
    # Step 3: Unit selection (Pointer Network)
    queries = self.unit_query(cond_features)  # (B, d_model)
    keys = self.unit_keys(entity_embeddings)  # (B, N, d_model)
    unit_logits = torch.einsum('bd,bnd->bn', queries, keys) / math.sqrt(d_model)
    unit_logits = apply_mask(unit_logits, action_masks['unit_selection'])
    unit_idx = Categorical(logits=unit_logits).sample()
    
    # Step 4: Gather selected unit embedding
    selected_unit_embed = entity_embeddings[
        torch.arange(B), unit_idx
    ]  # (B, d_model)
    unit_cond_features = cond_features + selected_unit_embed
    
    # Step 5: Target (depends on action type)
    if is_spatial_action(action_type):
        target_spatial_logits = self.spatial_head(
            spatial_features, unit_cond_features
        )
        target_spatial_logits = apply_mask(
            target_spatial_logits, action_masks['target_spatial']
        )
        target = Categorical(logits=target_spatial_logits.flatten(1)).sample()
    elif is_entity_target_action(action_type):
        target_entity_logits = self._entity_target(
            unit_cond_features, entity_embeddings
        )
        target_entity_logits = apply_mask(
            target_entity_logits, action_masks['target_entity']
        )
        target = Categorical(logits=target_entity_logits).sample()
    
    # Step 6: Unit type (for produce/build)
    if is_production_action(action_type):
        unit_type_logits = self.unit_type_head(unit_cond_features)
        unit_type_logits = apply_mask(unit_type_logits, action_masks['unit_type'])
        unit_type = Categorical(logits=unit_type_logits).sample()
    
    return Action(action_type, unit_idx, target, unit_type, ...)
```

### 4.4 Spatial Action Head

```python
class SpatialActionHead(nn.Module):
    """
    输出一个 2D spatial logits map (H, W).
    用于移动目标、建筑放置等。
    
    输入:
    - spatial_features: (B, C, H, W) — CNN encoder 输出
    - conditioning: (B, d_model) — 条件化向量 (操作类型 + 选中单位)
    """
    
    def __init__(self, d_model, h, w, spatial_channels=64):
        super().__init__()
        self.h, self.w = h, w
        
        # 处理条件化向量到空间维度
        self.cond_proj = nn.Linear(d_model, spatial_channels)
        
        # 反卷积层: 将特征放大到目标分辨率
        # 输入: (spatial_c + cond_c) channels at (H/16, W/16)
        # 输出: 1 channel at (H, W)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(spatial_channels * 2, 32, 4, 2, 1),  # -> H/8
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),                   # -> H/4
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1),                    # -> H/2
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, 2, 1),                     # -> H
        )
    
    def forward(self, spatial_features, conditioning):
        B, C, H_s, W_s = spatial_features.shape
        
        # 将条件化传播到空间维度
        cond = self.cond_proj(conditioning)  # (B, cond_c)
        cond_spatial = cond[:, :, None, None].expand(B, -1, H_s, W_s)
        
        # 合并并解码
        combined = torch.cat([spatial_features, cond_spatial], dim=1)
        logits_map = self.deconv(combined)
        
        # 裁剪/插值到目标尺寸
        if logits_map.shape[2:] != (self.h, self.w):
            logits_map = F.interpolate(
                logits_map, (self.h, self.w), mode='bilinear'
            )
        
        return logits_map.squeeze(1)  # (B, H, W)
```

### 4.5 环境侧 Action Mask 改造

**文件**: `envs/openra_env.py` — `_get_action_mask()`

需要输出符合新动作结构的 mask：

```python
def _get_action_mask(self, raw, my_units, enemy_units):
    return {
        'action_type': np.array([...]),  # (n_actions,)
        'unit_selection': np.array([...]),  # (n_entities,) — 哪些实体可被选中
        'target_spatial': np.array([...]),  # (128, 128) — 2D 联合 mask
        'target_entity': np.array([...]),  # (n_entities,) — 哪些实体可作为目标
        'unit_type': np.array([...]),  # (n_unit_types,)
    }
```

### 4.6 Phase 2 产出物

- [ ] `models/action_head.py` — HierarchicalActionHead + SpatialActionHead
- [ ] `envs/openra_env.py` — 新动作空间 + action mask 重构
- [ ] `agent/agent.py` — PPOAgent 适配 autoregressive 动作
- [ ] `models/actor.py` — ActorCritic 适配新动作头
- [ ] 测试: 验证 autoregressive 采样正确性
- [ ] 测试: 验证联合 2D spatial mask 正确性

---

## 5. Phase 3: 网络架构升级 — Transformer + Pointer Network

**工期**: 3-4 周
**目标**: 用 Transformer entity encoder + ResNet spatial encoder 替换当前 MLP/CNN。

### 5.1 Entity Transformer Encoder

```python
# models/entity_encoder.py (新增)

class EntityTransformerEncoder(nn.Module):
    """
    对变长实体列表进行编码。
    
    - 支持 self-attention 和 cross-attention (后续可用于 multi-agent)
    - 输出 per-entity embeddings 和 pooled global embedding
    """
    
    def __init__(
        self,
        entity_feature_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_entities: int = 200,
    ):
        super().__init__()
        
        # 类别特征的 Embedding 层
        self.type_embed = nn.Embedding(100, 32)  # unit type
        self.owner_embed = nn.Embedding(4, 8)     # self/ally/enemy/neutral
        
        # 合并后投影
        self.input_proj = nn.Linear(
            entity_feature_dim - 1 + 32 + 8,  # -1 for raw type index
            d_model
        )
        
        # Positional encoding (sinusoidal)
        self.register_buffer('pos_encoding', self._build_positional_encoding(max_entities, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Pooling
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, entities, entity_mask):
        """
        Args:
            entities: (B, N, F) — raw entity features
            entity_mask: (B, N) — True for valid entities
        Returns:
            entity_embeddings: (B, N, d_model) — per-entity
            global_embedding: (B, d_model) — pooled
        """
        B, N, _ = entities.shape
        
        # Split and embed categorical features
        type_idx = entities[:, :, 0].long()  # actor type index
        owner_idx = entities[:, :, 2].long()  # owner relation
        continuous = entities[:, :, 3:]  # rest of features
        
        type_e = self.type_embed(type_idx)
        owner_e = self.owner_embed(owner_idx)
        
        # Concatenate and project
        combined = torch.cat([type_e, owner_e, continuous], dim=-1)
        x = self.input_proj(combined)
        
        # Add positional encoding
        x = x + self.pos_encoding[:N, :].unsqueeze(0)
        
        # Self-attention (mask padding)
        src_key_padding_mask = ~entity_mask  # True = ignore
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling over valid entities
        pooled = (x * entity_mask.unsqueeze(-1)).sum(dim=1) / entity_mask.sum(dim=1, keepdim=True).clamp_min(1)
        global_embedding = self.pool_proj(pooled)
        
        return x, global_embedding
    
    @staticmethod
    def _build_positional_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
```

### 5.2 Spatial ResNet Encoder

```python
# models/spatial_encoder.py (新增)

class SpatialResNetEncoder(nn.Module):
    """ResNet-style encoder for minimap/spatial observations."""
    
    def __init__(self, in_channels=18, d_model=256):
        super().__init__()
        
        # ResNet blocks
        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )  # 128 -> 32
        
        self.layer1 = self._make_layer(64, 64, 3, stride=1)   # 32
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # 16
        self.layer3 = self._make_layer(128, 256, 6, stride=2) # 8
        
        # Project to d_model
        self.proj = nn.Conv2d(256, d_model, 1)
    
    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_c, out_c, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)
    
    def forward(self, spatial):
        x = self.entry(spatial)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        return x  # (B, d_model, 8, 8)
```

### 5.3 整合: AlphaStarActorCritic

```python
# models/alpha_star_actor_critic.py (新增)

class AlphaStarActorCritic(nn.Module):
    def __init__(self, ...):
        super().__init__()
        
        # Encoders
        self.entity_encoder = EntityTransformerEncoder(...)
        self.spatial_encoder = SpatialResNetEncoder(...)
        self.scalar_encoder = ScalarMLPEncoder(...)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(entity_dim + spatial_dim + scalar_dim, d_model),
            nn.ReLU(),
        )
        
        # Core
        self.lstm = nn.LSTM(d_model, hidden_size, num_layers=3, batch_first=True)
        
        # Heads (from Phase 2)
        self.action_head = HierarchicalActionHead(
            d_model=hidden_size, ...
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def forward(self, obs, hidden):
        # Encode
        entity_emb, entity_global = self.entity_encoder(
            obs['entities'], obs['entity_mask']
        )
        spatial_emb = self.spatial_encoder(obs['spatial'])
        scalar_emb = self.scalar_encoder(obs['scalar'])
        
        # Pool spatial
        spatial_pooled = spatial_emb.mean(dim=[2, 3])
        
        # Fuse
        fused = torch.cat([entity_global, spatial_pooled, scalar_emb], dim=-1)
        lstm_input = self.fusion(fused)
        
        # LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Action + value
        logits = self.action_head(
            output, entity_emb, obs['entity_mask'], spatial_emb
        )
        value = self.value_head(output)
        
        return logits, value, hidden
```

### 5.4 Phase 3 产出物

- [ ] `models/entity_encoder.py` — EntityTransformerEncoder
- [ ] `models/spatial_encoder.py` — SpatialResNetEncoder
- [ ] `models/scalar_encoder.py` — ScalarMLPEncoder
- [ ] `models/alpha_star_actor_critic.py` — 整合网络
- [ ] 保持向后兼容: 旧 `ActorCritic` 仍然可用
- [ ] 测试: 验证 forward pass 各维度正确

---

## 6. Phase 4: 训练基础设施 — 并行环境 + Self-Play

**工期**: 4-6 周
**目标**: 支持多环境并行训练和自我对弈。

### 6.1 多环境并行方案

#### 方案 A: Python Multiprocessing (推荐先实现)

利用 Python 的 `multiprocessing` 创建多个独立进程，每个进程运行一个 `OpenRAEnv` 实例。

```python
# envs/vector_env.py (新增)

import multiprocessing as mp
from multiprocessing import Process, Queue, Pipe

class SubprocVecEnv:
    """
    基于 Python multiprocessing 的向量化环境。
    每个子进程运行一个独立的 OpenRAEnv 实例。
    """
    
    def __init__(self, env_fns):
        """
        env_fns: List[Callable] — 每个返回一个 OpenRAEnv 实例
        """
        self.n_envs = len(env_fns)
        self.processes = []
        self.parent_conns = []
        
        for i, env_fn in enumerate(env_fns):
            parent_conn, child_conn = Pipe()
            process = Process(
                target=self._worker,
                args=(i, env_fn, child_conn),
                daemon=True,
            )
            self.parent_conns.append(parent_conn)
            self.processes.append(process)
            process.start()
    
    @staticmethod
    def _worker(idx, env_fn, conn):
        """子进程主循环。"""
        env = env_fn()
        while True:
            cmd, data = conn.recv()
            if cmd == 'step':
                obs, reward, done, truncated, info = env.step(data)
                conn.send((obs, reward, done, truncated, info))
            elif cmd == 'reset':
                obs, info = env.reset()
                conn.send((obs, info))
            elif cmd == 'close':
                env.close()
                conn.close()
                break
    
    def reset(self):
        for conn in self.parent_conns:
            conn.send(('reset', None))
        results = [conn.recv() for conn in self.parent_conns]
        obs = [r[0] for r in results]
        infos = [r[1] for r in results]
        return self._stack_obs(obs), infos
    
    def step(self, actions):
        for conn, action in zip(self.parent_conns, actions):
            conn.send(('step', action))
        results = [conn.recv() for conn in self.parent_conns]
        obs, rewards, dones, truncateds, infos = zip(*results)
        return (
            self._stack_obs(obs),
            np.array(rewards),
            np.array(dones),
            np.array(truncateds),
            infos,
        )
    
    def _stack_obs(self, obs_list):
        """Stack dict observations across environments."""
        if isinstance(obs_list[0], dict):
            return {
                k: np.stack([o[k] for o in obs_list])
                for k in obs_list[0].keys()
            }
        return np.stack(obs_list)
```

**关键挑战和解决方案**:

- **CLR 隔离**: pythonnet 的 CLR 不能在 fork 后的进程中安全使用。解决方法：
  - 使用 `spawn` 而非 `fork` 作为 mp start method
  - 每个子进程独立调用 `ensure_engine()` 加载自己的 CLR

- **⚠️ Headless 模式是前提**: `spawn` 后每个子进程都会打开一个 OpenGL 窗口。必须在 Phase 0/1 先解决 headless 游戏启动（`StartLocalGame` 的 `headless=True` 参数或类似机制），否则多环境并行不可行（macOS 上开 8 个 OpenRA 窗口会崩溃）。

- **端口冲突**: 如果使用远程 lobby 模式，每个子进程需要不同端口。解决方法：
  - 本地模式不使用网络，没有端口冲突
  - 远程模式使用端口范围分配

### 6.2 Self-Play 基础设施

```python
# envs/self_play_env.py (新增)

class SelfPlayOpenRAEnv(OpenRAEnv):
    """
    支持两个 Python 控制玩家的 OpenRA 环境。
    用于 self-play 训练。
    """
    
    def __init__(self, ...):
        super().__init__(...)
        self._opponent_model = None  # 对手模型
    
    def set_opponent(self, model):
        """设置对手模型 (可以是检查点或不同策略)。"""
        self._opponent_model = model
    
    def step(self, action_main):
        """
        主 agent 执行 action。
        对手通过 _opponent_model 选择 action。
        """
        # 主 agent action
        self._execute_action(action_main, player=0)
        
        # 对手 action
        opponent_action = self._opponent_model.act(obs_player1, info_player1)
        self._execute_action(opponent_action, player=1)
        
        # 推进引擎
        api = self._openra['PythonAPI']
        for _ in range(self.ticks_per_step):
            api.Step()
        
        # 返回主 agent 视角的观察和奖励
        return obs_player0, self._compute_reward(player=0), ...
```

**引擎侧改造**:

`StartLocalGame` 需要支持多个由 Python 控制的槽位：

```csharp
// PythonAPI.cs 新增
public static void StartLocalGameSelfPlay(string modId, string mapUid, string binDir)
{
    // ... 初始化代码 ...
    JoinLocal(force: true);
    
    // 添加两个 Python 控制的客户端
    // Client 0: 主 agent (槽位 Multi0)
    // Client 1: 对手 agent (槽位 Multi1)
    
    EnsurePlayableAgent(mapUid, "Multi0");
    AddPythonClient("Multi1");  // 新增方法
}
```

### 6.3 评估框架

```python
# scripts/evaluate.py (新增)

class EloEvaluator:
    """
    Elo rating system for evaluating agents.
    
    使用方式:
    - 定期从 league training 中取出 agents 对战
    - 更新 Elo rating
    - 记录对战结果到 CSV
    """
    
    def __init__(self, k_factor=32, initial_rating=1500):
        self.ratings = {}  # model_id -> rating
        self.k_factor = k_factor
    
    def play_match(self, agent_a, agent_b, n_games=10):
        """两个 agent 进行 n 场对战，返回胜率。"""
        wins_a, wins_b, draws = 0, 0, 0
        for i in range(n_games):
            # 交替先后手
            if i % 2 == 0:
                result = self._play_game(agent_a, agent_b)
            else:
                result = self._play_game(agent_b, agent_a)
            # ...
        return wins_a / n_games, wins_b / n_games
    
    def update_ratings(self, model_a_id, model_b_id, score_a):
        """Update Elo ratings based on match result."""
        ra = self.ratings[model_a_id]
        rb = self.ratings[model_b_id]
        ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400))
        self.ratings[model_a_id] = ra + self.k_factor * (score_a - ea)
        self.ratings[model_b_id] = rb + self.k_factor * ((1 - score_a) - (1 - ea))
```

### 6.4 Phase 4 产出物

- [ ] `envs/vector_env.py` — SubprocVecEnv
- [ ] `envs/self_play_env.py` — 双玩家环境
- [ ] Engine 侧双 Python 玩家支持
- [ ] `scripts/evaluate.py` — Elo 评估框架
- [ ] 适配 PPOAgent 到并行环境
- [ ] 负载测试: 8 环境并行 > 5x 吞吐量提升

---

## 7. Phase 5: 训练策略 — League Training + Curriculum + 奖励重构

**工期**: 4-6 周
**目标**: 实现 self-play league training、课程学习、和最终奖励设计。

### 7.1 League Training

AlphaStar 的核心训练机制。实现简化版：

```python
# training/league.py (新增)

class LeagueManager:
    """
    管理 self-play league:
    - Main Agents: 当前训练中的 agents
    - League Exploiters: 专门击败特定 main agent 的 agents
    - Main Exploiters: 专门击败整个 league 的 agents
    - Historical agents: 过去的检查点
    - Built-in bots: OpenRA 内置 AI
    """
    
    def __init__(self, eval_dir, agent_factory):
        self.main_agents = []        # List[MainAgent]
        self.league_exploiters = []  # List[LeagueExploiter]
        self.main_exploiters = []    # List[MainExploiter]
        self.historical = []         # List[HistoricalAgent]
        self.builtin_bots = []       # List[BuiltinBot]
        
        self.match_history = []
        self.elo_evaluator = EloEvaluator()
    
    def sample_opponent(self, current_agent_id, strategy='pfsp'):
        """
        PFSP (Prioritized Fictitious Self-Play) 采样:
        - 优先选择最近的或强力的对手
        - 也偶尔选择过去的对手（防止遗忘）
        """
        # PFSP 采样权重
        weights = self._compute_pfsp_weights(current_agent_id)
        opponent = np.random.choice(self._all_opponents(), p=weights)
        return opponent
    
    def add_checkpoint(self, agent_id, model_path, step):
        """将当前模型添加为历史对手。"""
        self.historical.append(HistoricalAgent(agent_id, model_path, step))
        # 限制历史对手数量
        if len(self.historical) > 50:
            # 保留最早的 + 最近的，均匀下采样
            self._downsample_historical()
    
    def get_training_opponent_distribution(self):
        """返回训练对手的分布统计。"""
        return {
            'main_agents': len(self.main_agents),
            'exploiters': len(self.league_exploiters),
            'historical': len(self.historical),
            'builtin': len(self.builtin_bots),
        }
```

### 7.2 奖励重构

**文件**: `envs/rewards.py` (从 `openra_env.py` 中拆分)

```python
# envs/rewards.py (新增)

class RewardCalculator:
    """模块化奖励计算，支持多目标。"""
    
    def __init__(self):
        self.reward_components = OrderedDict({
            'win_loss': WinLossReward(weight=10.0),
            'combat': CombatReward(weight=0.1),
            'economy': EconomyReward(weight=0.05),
            'exploration': ExplorationReward(weight=0.01),
            'production': ProductionReward(weight=0.02),
        })
        
        # 状态追踪
        self.prev_state = None
    
    def compute(self, raw_state, action):
        rewards = {}
        for name, component in self.reward_components.items():
            rewards[name] = component(raw_state, self.prev_state, action)
        total = sum(rewards.values())
        self.prev_state = raw_state
        return total, rewards  # 返回总奖励和各组分


class WinLossReward:
    """
    最重要的奖励信号: 胜负。
    
    在回合结束时:
    - 胜利: +1
    - 失败: -1
    - 平局: 0
    
    在回合中:
    - 预估胜率变化 (可选，基于价值函数)
    """
    
    def __init__(self, weight=1.0, win_bonus=10.0, loss_penalty=-10.0):
        self.weight = weight
        self.win_bonus = win_bonus
        self.loss_penalty = loss_penalty
    
    def __call__(self, state, prev_state, action):
        if self._all_enemy_eliminated(state):
            return self.win_bonus * self.weight
        if self._all_self_eliminated(state):
            return self.loss_penalty * self.weight
        return 0.0


class CombatReward:
    """
    基于战斗的奖励:
    - 造成伤害: proportional to damage dealt
    - 击杀单位: bonus per kill
    - 击杀建筑: bonus per building destroyed
    """
    
    def __init__(self, weight=0.1, damage_scale=0.001, kill_bonus=0.5):
        self.weight = weight
    
    def __call__(self, state, prev_state, action):
        if prev_state is None:
            return 0.0
        
        reward = 0.0
        # 敌方损失
        enemy_loss = self._count_enemy(prev_state) - self._count_enemy(state)
        if enemy_loss > 0:
            reward += enemy_loss * 0.5
        
        return reward * self.weight


class EconomyReward:
    """
    基于经济的奖励:
    - 资源采集效率
    - 资源利用率 (不囤积)
    - 电力平衡
    """
    # ...
```

### 7.3 课程学习

```python
# training/curriculum.py (新增)

class CurriculumManager:
    """
    渐进难度课程:
    
    Stage 1: Economy only (无对手, 学习建造和采集)
    Stage 2: 静态对手 (固定位置的简单 AI)
    Stage 3: 全对战 (自由对战, 使用内置 bot)
    Stage 4: 多地图泛化
    """
    
    STAGES = [
        {
            'name': 'economy',
            'opponent_type': 'none',
            'maps': ['single_map'],
            'max_episode_ticks': 5000,
            'success_criteria': 'avg_reward > 100',
            'min_updates': 50,
        },
        {
            'name': 'static_opponent',
            'opponent_type': 'easy_bot',
            'maps': ['single_map'],
            'max_episode_ticks': 10000,
            'success_criteria': 'win_rate > 0.6',
            'min_updates': 100,
        },
        {
            'name': 'full_combat',
            'opponent_type': 'medium_bot',
            'maps': ['single_map'],
            'max_episode_ticks': 20000,
            'success_criteria': 'win_rate > 0.5',
        },
        {
            'name': 'generalization',
            'opponent_type': 'self_play',
            'maps': ['map_1', 'map_2', 'map_3', 'map_4'],
            'max_episode_ticks': 20000,
        },
    ]
    
    def __init__(self):
        self.current_stage = 0
        self.stage_update_count = 0
        self.stage_rewards = []
    
    def should_advance(self, eval_results):
        """检查是否应该进入下一阶段。"""
        stage = self.STAGES[self.current_stage]
        if self.stage_update_count < stage['min_updates']:
            return False
        criteria = stage['success_criteria']
        # ... 检查是否满足进阶条件
        return False
    
    def advance(self):
        self.current_stage = min(self.current_stage + 1, len(self.STAGES) - 1)
        self.stage_update_count = 0
        self.stage_rewards.clear()
    
    def get_config(self):
        return self.STAGES[self.current_stage]
```

### 7.4 监督预训练 (Behavior Cloning)

```python
# training/behavior_cloning.py (新增)

class BehaviorCloning:
    """
    使用 expert demonstration (内置 bot 或人类 replay) 进行监督预训练。
    
    AlphaStar 中这是关键步骤:
    1. 从内置 bot / 人类 replay 中提取 (obs, action) pairs
    2. 用 cross-entropy loss 训练 policy, 用 MSE loss 训练 value
    3. 预训练模型作为 PPO 的初始权重
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def collect_demonstrations(self, env, expert_agent, n_episodes=100):
        """收集专家演示数据。"""
        dataset = []
        for ep in range(n_episodes):
            obs, info = env.reset()
            episode = []
            while True:
                action = expert_agent.act(obs, info)
                episode.append({
                    'obs': obs,
                    'action': action,
                    'info': info,
                })
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            dataset.append(episode)
        return dataset
    
    def train(self, dataset, n_epochs=10, batch_size=256):
        """Behavior cloning training loop."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        for epoch in range(n_epochs):
            for batch in self._yield_batches(dataset, batch_size):
                obs = self._to_tensor(batch['obs'])
                target_action = self._to_tensor(batch['action'])
                
                logits, value, _ = self.model(obs)
                
                # Action type loss (cross-entropy)
                action_loss = F.cross_entropy(
                    logits['action_type'], target_action['action_type']
                )
                
                # Unit selection loss
                unit_loss = F.cross_entropy(
                    logits['unit_selection'], target_action['unit_selection']
                )
                
                # Target losses
                target_loss = F.cross_entropy(
                    logits['target'], target_action['target']
                )
                
                total_loss = action_loss + unit_loss + target_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
```

### 7.5 Phase 5 产出物

- [ ] `training/league.py` — League Manager
- [ ] `envs/rewards.py` — 模块化奖励系统
- [ ] `training/curriculum.py` — 课程管理器
- [ ] `training/behavior_cloning.py` — 监督预训练
- [ ] `scripts/distributed_train.py` — 分布式训练入口
- [ ] 集成测试: 完整 self-play + league 训练循环

---

## 8. Phase 6: 引擎桥接优化

**工期**: 2-3 周
**目标**: 优化 PythonAPI.cs 性能，减少状态提取延迟。

### 8.1 缓存反射调用

**文件**: `OpenRA.Game/PythonAPI.cs`

```csharp
// 在 PythonAPI 类中添加静态缓存字段
private static class CachedReflection
{
    // 缓存 Assembly.GetType() 结果
    public static readonly Type PlayerResourcesType;
    public static readonly Type PowerManagerType;
    public static readonly Type ProductionQueueType;
    public static readonly Type BuildingInfoType;
    public static readonly Type BuildingUtilsType;
    
    // 缓存 MethodInfo
    public static readonly MethodInfo TraitsImplementing;
    public static readonly MethodInfo TraitOrDefault;
    public static readonly MethodInfo CanPlaceBuilding;
    public static readonly MethodInfo IsCloseEnoughToBase;
    public static readonly MethodInfo AllQueued;
    public static readonly MethodInfo GetProductionCost;
    
    // 缓存 FieldInfo (经济/电力)
    public static readonly FieldInfo PlayerResources_Cash;
    public static readonly FieldInfo PlayerResources_Resources;
    public static readonly FieldInfo PlayerResources_ResourceCapacity;
    public static readonly FieldInfo PowerManager_PowerProvided;
    public static readonly FieldInfo PowerManager_PowerDrained;
    public static readonly FieldInfo PowerManager_PowerState;
    
    static CachedReflection()
    {
        var assemblies = AppDomain.CurrentDomain.GetAssemblies();
        
        PlayerResourcesType = assemblies
            .Select(a => a.GetType("OpenRA.Mods.Common.Traits.PlayerResources", false))
            .FirstOrDefault(t => t != null);
        
        // ... 其余类型和成员的缓存 ...
    }
}
```

### 8.2 增量状态更新

```csharp
// 新增: 仅返回自上次调用以来变化的实体
public static RLStateDelta GetStateDelta(int lastWorldTick)
{
    var currentState = GetState();
    
    var changedActors = currentState.Actors
        .Where(a => /* 检测 a 在上次 tick 之后发生了变化 */)
        .ToArray();
    
    var removedActorIds = /* 检测已消失的 actor IDs */;
    
    return new RLStateDelta
    {
        WorldTick = currentState.WorldTick,
        ChangedActors = changedActors,
        RemovedActorIds = removedActorIds,
        FullEconomyInfo = currentState.PlayerCash, // 经济信息总是全量发送，因为很小
        // ... 
    };
}
```

### 8.3 批量 Order Feasibility Check

**当前问题**: `CheckOrderFeasibility` 在构造 action mask 时为
- 每个单位的 4-6 个邻接 cell (move feasibility)
- 每个己方单位 × 每个敌方单位 (attack feasibility)

独立调用，产生大量 C#↔Python 穿越。

**方案**: 提供批量版本：

```csharp
// 批量移动可行性检查
public static bool[] CheckMoveFeasibilityBulk(
    uint[] subjectActorIds, int[] cellXs, int[] cellYs)
{
    var results = new bool[subjectActorIds.Length];
    for (int i = 0; i < subjectActorIds.Length; i++)
    {
        results[i] = CheckOrderFeasibility(
            subjectActorIds[i], "Move", "Cell",
            new CPos(cellXs[i], cellYs[i]).Bits, 0, 0
        );
    }
    return results;
}
```

### 8.4 可建造区域计算优化

**当前**: 遍历 map.AllCells 的**每个** cell 做 `CanPlaceBuilding` + `IsCloseEnoughToBase`。

**优化**: 
- 使用 `FindTilesInAnnulus` 或 `FindTilesInCircle` 只检查基地周围的 cell
- 缓存最近的检查结果，仅在建筑被放置/摧毁时重新计算
- 为每种建筑类型缓存 `AdjacentCells` 要求

### 8.5 Phase 6 产出物

- [ ] `CachedReflection` static class
- [ ] `GetStateDelta()` 增量状态方法
- [ ] `CheckMoveFeasibilityBulk()` 和 `CheckAttackFeasibilityBulk()`
- [ ] PlaceableAreas 增量更新
- [ ] 性能基准: Phase 6 前后对比
- [ ] 目标: GetState() 延迟 < 10ms (当前可能 50-200ms)

---

## 9. 风险与依赖

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **Action 不平衡 (93% noop)** | PPO 策略塌缩到全 noop，produce 率无法提升 | 决策步 mask（2.0b）；简化 action space（Phase 2） |
| **Reward 信号太稀疏** | 150 步仅 1 次 reward 事件，credit assignment 不可能 | BO distance reward（2.0c）；RuleBasedAgent 验证 reward 设计 |
| **C# 反射跨平台不稳定** | MCV 类型检测在 ARM64 上失败，deploy reward=0 | 类型白名单替代反射（2.0a）；Phase 6 编译时依赖替代全反射 |
| pythonnet CLR 不能在 fork 进程中使用 | 阻塞多环境并行 (Phase 4) | 使用 `spawn` 而非 `fork`；每个子进程独立加载 CLR |
| **OpenRA 多实例开窗口** | 每个子进程打开一个 OpenGL 窗口，多环境不可行 | Phase 0/1 解决 headless 模式（P0 依赖） |
| OpenRA 不支持多实例 | 阻塞并行环境 (Phase 4) | 每个进程独立加载 Game assemblies |
| 引擎反射调用不稳定 (私有 API 变更) | 阻塞状态提取 (全部 Phase) | Phase 6 编译时可选依赖 Mods.Common 而非全反射 |
| LSTM + Transformer 训练不稳定 | 阻塞训练 (Phase 3+) | 从小模型开始递增；使用 warmup + gradient clipping |
| Self-play 计算成本高 | 阻塞 Phase 5 | 先实现简化版 (仅2个对手轮流)；逐步扩展 |
| OpenRA 版本升级破坏兼容性 | 阻塞所有 Phase | 锁定 OpenRA 版本；在 CI 中自动检测兼容性 |

---

## 10. 里程碑和检查点

### M0: 基线加固 ✅ (Phase 0 核心完成)

**核心成功标准**（基于实战调试发现）:
- [x] MCV 类型检测修复 — deploy reward 正确触发
- [x] RuleBasedAgent 150 步累计 reward 单调递增（BO distance reward 验证通过）
- [x] PPO + BC warmstart produce 率 ~5%（达到 RuleBasedAgent 水平）
- [x] PPO 训练稳定 80+ 轮（KL max=0.199，零塌缩，零 attack spam）
- [x] Reward 信号密度 98%+（从 0.7% 提升）
- [x] Entity Observation 实施（Phase 1 核心）— BC 准确率 100%，训练稳定
- [ ] ⚠️ PPO 无法超越 BC 基线（跨架构一致结论：瓶颈在 credit assignment，不在 observation）
- [ ] target_cell 联合 mask 修复
- [ ] 完整 unit_types 映射
- [ ] 基线指标记录

### M1: 观察重构 (5 周) — Phase 0+1 完成
- [ ] Entity observation builder 可用且 tested
- [ ] Spatial observation 含地形和迷雾通道
- [ ] 旧 `observation_type="vector"` 仍然工作
- [ ] PPO 在新 observation space 上达到旧基线水平

### M2: 动作重构 (8 周) — Phase 2 完成
- [ ] Hierarchical action head 替代旧的 MultiDiscretePolicy
- [ ] Spatial action head (2D logits map) 工作正常
- [ ] Autoregressive sampling 正确且高效
- [ ] Action masks 适配新动作空间
- [ ] PPO 训练在新动作空间上收敛

### M3: 网络升级 (11 周) — Phase 3 完成
- [ ] Entity Transformer encoder 替代 fixed-size MLP
- [ ] Spatial ResNet encoder 可用
- [ ] AlphaStarActorCritic 全整合网络 forward pass 正确
- [ ] 训练在新网络上收敛 (可能比旧网络更慢但效果更好)

### M4: 并行训练 (13 周) — Phase 4 完成
- [ ] SubprocVecEnv 支持 4-8 并行环境
- [ ] Self-play 环境可用 (双 Python 玩家)
- [ ] Elo evaluator + 定期评估
- [ ] 吞吐量 > 5x 提升 vs 单环境

### M5: League Training (18 周) — Phase 5 完成
- [ ] League manager + PFSP 采样
- [ ] Curriculum 四阶段
- [ ] Behavior cloning 预训练
- [ ] 在 3+ 地图上的 self-play 训练
- [ ] Agent 在 full game 设置上击败最强内置 bot > 50% 胜率

### M6: 生产就绪 (20 周) — Phase 6 + Polish
- [ ] Engine bridge 性能: GetState < 10ms
- [ ] 完整测试套件
- [ ] 文档和教程
- [ ] Docker 部署
- [ ] 预训练模型发布

---

## 附录 A: 文件结构变更总览

```
OpenRA-Bot/
├── README.md                          # 更新
├── PLAN.md                            # 本文件
├── requirements.txt                   # 添加依赖 (einops, wandb, etc.)
│
├── envs/
│   ├── openra_env.py                  # 重构: 新 obs + action spaces
│   ├── vector_env.py                  # 新增: 多环境并行
│   ├── self_play_env.py               # 新增: self-play
│   ├── rewards.py                     # 新增: 奖励模块
│   └── wrappers.py                    # 保留, 可能需要适配
│
├── models/
│   ├── entity_encoder.py              # 新增: Transformer entity encoder
│   ├── spatial_encoder.py             # 新增: ResNet spatial encoder
│   ├── scalar_encoder.py              # 新增: MLP scalar encoder
│   ├── action_head.py                 # 新增: hierarchical action head
│   ├── alpha_star_actor_critic.py     # 新增: 整合网络
│   ├── actor.py                       # 保留 (向后兼容)
│   └── buffer.py                      # 修改: 支持 dict obs
│
├── training/
│   ├── __init__.py                    # 新增
│   ├── league.py                      # 新增: League manager
│   ├── curriculum.py                  # 新增: 课程学习
│   ├── behavior_cloning.py            # 新增: 监督预训练
│   └── distributed.py                 # 新增: 分布式训练工具
│
├── agent/
│   ├── agent.py                       # 重构: 适配新动作空间
│   └── evaluator.py                   # 新增: Elo evaluation
│
├── utils/
│   ├── entity_obs.py                  # 新增: Entity obs builder
│   ├── spatial_obs.py                 # 新增: Spatial obs builder
│   ├── scalar_obs.py                  # 新增: Scalar obs builder
│   ├── obs.py                         # 修改: 整合三种 builder
│   ├── engine.py                      # 修改: 支持多进程
│   └── ...
│
├── scripts/
│   ├── train_rl.py                    # 重构: 支持新网络
│   ├── distributed_train.py           # 新增: 分布式训练
│   ├── evaluate.py                    # 新增: 评估脚本
│   ├── collect_demos.py               # 新增: 收集专家演示
│   └── baselines/
│       └── run_baselines.py           # 新增: 基线评估
│
└── tests/
    ├── test_env_creation.py
    ├── test_obs_shapes.py
    ├── test_action_masks.py
    ├── test_entity_encoder.py
    ├── test_action_head.py
    └── test_league.py
```

## 附录 B: 新增依赖

```
# requirements.txt 新增
pythonnet>=3.0.0         # .NET CLR 桥接（核心依赖，之前漏了）
einops>=0.6.0           # 张量操作简化
wandb>=0.15.0           # 实验追踪
hydra-core>=1.3.0       # 配置管理
pytest>=7.0.0           # 测试 (从可选升级为必需)
tensorboard>=2.10.0     # 训练监控
moviepy>=1.0.3          # replay 视频生成 (可选)
```

---

*本计划为路线图而非严格时间表。每个 Phase 的进度取决于实际开发中的发现和验证结果。关键原则是每个 Phase 都产出可运行的中间版本。*
