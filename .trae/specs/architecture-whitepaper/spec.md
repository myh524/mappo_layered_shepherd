# 项目架构白皮书 Spec

## Why
用户需要全面理解本项目的底层逻辑和架构设计，以便进行二次开发。通过输出详细的架构分析文档，帮助开发者快速掌握项目的技术栈、目录结构、核心数据流和模块依赖关系。

## What Changes
- 创建《项目架构白皮书》文档，包含技术栈识别、目录结构语义映射、核心数据流、模块依赖关系四个核心章节
- 分析项目的架构模式（分层架构 + CTDE模式）
- 梳理各模块之间的调用关系和依赖层次

## Impact
- Affected specs: 无（纯文档输出）
- Affected code: 无代码修改，仅创建文档

---

# 项目架构白皮书

## 一、技术栈识别

### 1.1 编程语言与版本

| 语言 | 版本 | 用途 |
|------|------|------|
| Python | 3.8+ | 主要开发语言 |

### 1.2 核心框架与库

| 组件 | 版本 | 用途 |
|------|------|------|
| PyTorch | 1.8.1 | 深度学习框架，构建神经网络 |
| NumPy | 1.19.4 | 数值计算，数组操作 |
| Gym | 0.10.5 | 强化学习环境接口标准 |
| TensorBoardX | 2.1 | 训练可视化 |
| Wandb | 0.10.31 | 实验跟踪与可视化 |
| PyTorch Geometric | 2.0.4 | 图神经网络支持（预留） |
| torch-scatter | 2.0.7 | 稀疏张量操作 |
| torch-cluster | 1.5.9 | 图聚类操作 |

### 1.3 架构模式判断

本项目采用 **分层架构 + CTDE模式**：

#### 分层架构
```
┌─────────────────────────────────────────┐
│           应用层 (Application)          │  ← train.py
├─────────────────────────────────────────┤
│           算法层 (Algorithm)            │  ← MAPPO训练器
├─────────────────────────────────────────┤
│           策略层 (Policy)               │  ← Actor-Critic网络
├─────────────────────────────────────────┤
│           数据层 (Buffer)               │  ← 经验回放缓冲区
├─────────────────────────────────────────┤
│           环境层 (Environment)          │  ← Gym环境接口
└─────────────────────────────────────────┘
```

#### CTDE模式
- **CTDE (Centralized Training Decentralized Execution)**：中心化训练，去中心化执行
- **训练阶段**：Critic网络使用全局共享观测（share_observation），Actor网络使用局部观测
- **执行阶段**：每个智能体仅使用局部观测进行决策

---

## 二、目录结构语义映射

### 2.1 根目录结构

```
high_layer/
├── envs/                    # 环境模块：定义强化学习环境
│   ├── __init__.py         # 模块导出
│   ├── sheep_entity.py     # 羊实体类：Boids模型实现
│   ├── sheep_scenario.py   # 场景管理类：状态管理
│   └── sheep_flock.py      # 环境主类：Gym接口实现
│
├── onpolicy/               # 算法核心模块：MAPPO实现
│   ├── algorithms/         # 算法实现
│   │   ├── utils/         # 网络工具模块
│   │   │   ├── mlp.py          # MLP基础网络
│   │   │   ├── rnn.py          # RNN层实现
│   │   │   ├── bounded_act.py  # 有界动作层
│   │   │   ├── distributions.py # 分布定义
│   │   │   └── popart.py       # PopArt归一化
│   │   ├── MAPPOPolicy.py      # MAPPO策略封装
│   │   ├── mappo.py            # MAPPO训练器
│   │   ├── actor_critic.py     # 通用Actor-Critic
│   │   └── sheep_actor_critic.py # 羊群专用Actor-Critic
│   │
│   ├── runner/            # 训练执行器
│   │   └── shared/
│   │       ├── base_runner.py  # 基础Runner类
│   │       └── mpe_runner.py   # MPE环境Runner
│   │
│   ├── scripts/           # 训练脚本
│   │   └── train_mpe.sh        # MPE训练启动脚本
│   │
│   └── utils/             # 工具模块
│       ├── shared_buffer.py    # 共享经验缓冲区
│       ├── separated_buffer.py # 分离经验缓冲区
│       ├── valuenorm.py        # 值归一化
│       └── util.py             # 通用工具函数
│
├── utils/                  # 全局工具模块
│   ├── logger.py          # 日志工具
│   └── utils.py           # 通用工具函数
│
├── tests/                  # 单元测试
│   ├── __init__.py
│   ├── test_env.py        # 环境测试
│   └── test_network.py    # 网络测试
│
├── scripts/                # Shell脚本集合
│   ├── run_mappo.sh       # MAPPO训练脚本
│   ├── train_sheep.sh     # 羊群训练脚本
│   └── ...                # 其他对比实验脚本
│
├── docs/                   # 文档目录
│   └── PROJECT_PLAN.md    # 项目计划文档
│
├── train.py               # 【入口文件】主训练脚本
├── simple_test.py         # 简单测试脚本
├── requirement.txt        # 依赖清单
└── README.md              # 项目说明
```

### 2.2 入口文件

| 入口文件 | 用途 | 启动命令 |
|---------|------|---------|
| `train.py` | 主训练入口 | `python train.py --env_name sheep_herding` |
| `simple_test.py` | 快速测试入口 | `python simple_test.py` |
| `tests/test_env.py` | 环境单元测试 | `python -m pytest tests/test_env.py` |
| `tests/test_network.py` | 网络单元测试 | `python -m pytest tests/test_network.py` |

---

## 三、核心数据流

### 3.1 数据流概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           训练数据流                                      │
└──────────────────────────────────────────────────────────────────────────┘

     ┌─────────┐
     │  reset  │ ─────────────────────────────────────────┐
     └────┬────┘                                          │
          │                                               │
          ▼                                               │
     ┌─────────┐     ┌─────────────┐     ┌───────────┐   │
     │ 观测obs │ ──▶ │ Actor网络   │ ──▶ │ 动作action│   │
     └────┬────┘     │ (策略网络)  │     └─────┬─────┘   │
          │          └─────────────┘           │         │
          │                                    │         │
          │          ┌─────────────┐           │         │
          └─────────▶│ Critic网络  │           │         │
                     │ (价值网络)  │           │         │
                     └──────┬──────┘           │         │
                            │                  │         │
                            ▼                  ▼         │
                     ┌───────────┐      ┌───────────┐   │
                     │  value    │      │  env.step │   │
                     └─────┬─────┘      └─────┬─────┘   │
                           │                  │         │
                           │                  ▼         │
                           │           ┌───────────┐    │
                           │           │ obs,reward│    │
                           │           │ done,info │    │
                           │           └─────┬─────┘    │
                           │                 │          │
                           ▼                 ▼          │
                    ┌─────────────────────────────┐     │
                    │    SharedReplayBuffer       │     │
                    │  (经验回放缓冲区)            │     │
                    └──────────────┬──────────────┘     │
                                   │                    │
                                   ▼                    │
                    ┌─────────────────────────────┐     │
                    │    compute_returns (GAE)    │     │
                    └──────────────┬──────────────┘     │
                                   │                    │
                                   ▼                    │
                    ┌─────────────────────────────┐     │
                    │    PPO Update (训练更新)     │     │
                    └──────────────┬──────────────┘     │
                                   │                    │
                                   └────────────────────┘
```

### 3.2 详细数据流步骤

#### 阶段1: 环境初始化
```python
# train.py -> SheepTrainer.__init__()
env = SheepFlockEnv(
    world_size=(50.0, 50.0),
    num_sheep=10,
    num_herders=3,
    episode_length=100
)
```

#### 阶段2: 观测收集
```python
# SheepFlockEnv._get_obs() -> SheepScenario.get_observation()
obs = [
    flock_center[0:2],      # 羊群质心
    flock_direction[0:2],   # 羊群方向
    flock_spread,           # 扩散度
    num_sheep,              # 羊数量
    target_position[0:2],   # 目标位置
    herder_position[0:2]    # 机械狗位置
]
```

#### 阶段3: 动作生成
```python
# SheepActorCritic.get_actions()
actions = Actor(obs)  # 输出站位参数
# actions = [radius_mean, radius_std, angle_mean, concentration]
```

#### 阶段4: 环境交互
```python
# SheepFlockEnv.step(actions)
herder_positions = _sample_herder_positions(actions)  # 采样站位
scenario.update_herders(herder_positions)             # 更新机械狗位置
scenario.update_sheep(dt)                             # 更新羊群状态（Boids模型）
reward = _compute_reward()                            # 计算奖励
```

#### 阶段5: 经验存储
```python
# SharedReplayBuffer.insert()
buffer.insert(share_obs, obs, rnn_states, actions, 
              action_log_probs, values, rewards, masks)
```

#### 阶段6: GAE计算
```python
# SharedReplayBuffer.compute_returns()
gae = 0
for step in reversed(range(episode_length)):
    delta = reward + gamma * next_value * mask - value
    gae = delta + gamma * gae_lambda * mask * gae
    returns[step] = gae + value
```

#### 阶段7: PPO更新
```python
# R_MAPPO.train()
for epoch in range(ppo_epoch):
    for mini_batch in data_generator:
        # 计算重要性权重
        ratio = exp(new_log_prob - old_log_prob)
        # PPO裁剪目标
        surr1 = ratio * advantage
        surr2 = clip(ratio, 1-epsilon, 1+epsilon) * advantage
        policy_loss = -min(surr1, surr2)
        # 价值损失
        value_loss = (return - value)^2
        # 总损失
        loss = value_loss * coef + policy_loss - entropy * entropy_coef
```

### 3.3 核心配置文件

| 配置位置 | 配置内容 | 文件 |
|---------|---------|------|
| 命令行参数 | 训练超参数、环境参数 | `train.py::parse_args()` |
| 环境配置 | 奖励权重、Boids参数 | `sheep_flock.py::reward_config` |
| 网络配置 | 隐藏层大小、激活函数 | `train.py::parse_args()` |
| 依赖配置 | Python包版本 | `requirement.txt` |

---

## 四、模块依赖关系

### 4.1 模块层次结构

```
                    ┌─────────────────┐
                    │    train.py     │  应用层
                    │   (入口脚本)     │
                    └────────┬────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ SheepTrainer  │ │ SheepFlockEnv │ │ SheepActorCrit│
    │  (训练控制)    │ │   (环境)      │ │   (网络)      │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            │                 │                 │
            ▼                 ▼                 ▼
    ┌───────────────────────────────────────────────────┐
    │                   onpolicy/                       │
    │  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
    │  │  algorithms │  │   runner    │  │   utils   │ │
    │  │  (算法层)   │  │  (执行层)   │  │  (数据层) │ │
    │  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │
    │         │                │               │       │
    │         └────────────────┼───────────────┘       │
    │                          │                       │
    └──────────────────────────┼───────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   envs/ (环境层)     │
                    │  SheepScenario      │
                    │  SheepEntity        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  utils/ (工具层)     │
                    │  numpy, torch等     │
                    └─────────────────────┘
```

### 4.2 模块分类

#### 基础底层模块（无外部依赖）

| 模块 | 文件 | 职责 |
|------|------|------|
| SheepEntity | `envs/sheep_entity.py` | 单个羊实体，实现Boids行为规则 |
| 工具函数 | `utils/utils.py` | 通用工具函数 |
| 日志工具 | `utils/logger.py` | 日志记录 |

#### 核心业务模块

| 模块 | 文件 | 职责 | 依赖 |
|------|------|------|------|
| SheepScenario | `envs/sheep_scenario.py` | 场景状态管理 | SheepEntity |
| SheepFlockEnv | `envs/sheep_flock.py` | Gym环境接口 | SheepScenario |
| SheepActor | `algorithms/sheep_actor_critic.py` | 策略网络 | MLPBase, BoundedACTLayer |
| SheepCritic | `algorithms/sheep_actor_critic.py` | 价值网络 | MLPBase |
| SharedReplayBuffer | `utils/shared_buffer.py` | 经验存储 | - |
| R_MAPPO | `algorithms/mappo.py` | MAPPO训练器 | Policy, Buffer |

#### 应用层模块

| 模块 | 文件 | 职责 | 依赖 |
|------|------|------|------|
| SheepTrainer | `train.py` | 训练流程控制 | Env, Policy, Buffer |
| Runner | `runner/shared/base_runner.py` | 训练执行器基类 | Policy, Buffer |

### 4.3 调用关系图

```
train.py
    │
    ├──▶ envs/
    │       ├── SheepFlockEnv
    │       │       ├── SheepScenario
    │       │       │       └── SheepEntity (多个)
    │       │       └── Gym Spaces
    │       └── SheepFlockEnvWrapper (多进程包装)
    │
    ├──▶ onpolicy/algorithms/
    │       ├── sheep_actor_critic.py
    │       │       ├── SheepActor
    │       │       │       ├── MLPBase
    │       │       │       └── BoundedACTLayer
    │       │       └── SheepCritic
    │       │               └── MLPBase
    │       └── mappo.py
    │               └── R_MAPPO (训练器)
    │
    └──▶ onpolicy/utils/
            ├── shared_buffer.py
            │       └── SharedReplayBuffer
            └── valuenorm.py
                    └── ValueNorm
```

### 4.4 关键接口定义

#### 环境接口
```python
class SheepFlockEnv:
    observation_space: gym.spaces.Box    # 观测空间 (10,)
    action_space: gym.spaces.Box         # 动作空间 (4,)
    share_observation_space: gym.spaces.Box  # 共享观测空间 (36,)
    
    def reset() -> np.ndarray            # 返回初始观测
    def step(actions) -> Tuple[obs, reward, done, info]
    def get_shared_obs() -> np.ndarray   # 返回共享观测
```

#### 策略接口
```python
class SheepActorCritic:
    def get_actions(cent_obs, obs, rnn_states, masks) 
        -> Tuple[values, actions, log_probs, rnn_states]
    
    def get_values(cent_obs, rnn_states, masks) -> values
    
    def evaluate_actions(cent_obs, obs, actions, ...)
        -> Tuple[values, log_probs, entropy]
```

#### 缓冲区接口
```python
class SharedReplayBuffer:
    def insert(share_obs, obs, actions, rewards, ...)
    def compute_returns(next_value)
    def feed_forward_generator(advantages, num_mini_batch) -> Generator
```

---

## 五、二次开发指南

### 5.1 扩展环境

1. 修改 `envs/sheep_entity.py` 添加新的羊群行为规则
2. 修改 `envs/sheep_scenario.py` 添加新的场景元素
3. 修改 `envs/sheep_flock.py` 调整观测/动作空间

### 5.2 修改网络结构

1. 修改 `onpolicy/algorithms/utils/mlp.py` 调整MLP结构
2. 修改 `onpolicy/algorithms/sheep_actor_critic.py` 调整Actor-Critic结构
3. 修改 `onpolicy/algorithms/utils/bounded_act.py` 调整动作输出层

### 5.3 调整训练参数

1. 修改 `train.py::parse_args()` 中的默认参数
2. 通过命令行参数覆盖默认值

### 5.4 添加新算法

1. 在 `onpolicy/algorithms/` 下创建新的算法文件
2. 继承或参考 `mappo.py` 的结构
3. 在 `train.py` 中集成新算法
