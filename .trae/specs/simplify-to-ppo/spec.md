# 简化架构：从MAPPO改为标准PPO Spec

## Why
当前项目使用MAPPO框架，但实际上是**单一中央控制器**架构，并非真正的多智能体系统。使用标准PPO可以：
1. 减少代码复杂度
2. 提高训练效率
3. 更容易调试和维护

## What Changes
- 移除多智能体相关的维度扩展（num_agents维度）
- 简化Buffer结构，移除SharedReplayBuffer中的多智能体维度
- 简化Actor-Critic网络接口
- 保留核心功能：站位参数输出、Boids环境交互

## Impact
- Affected specs: 训练流程
- Affected code: 
  - `train.py` - 移除num_agents维度扩展
  - `onpolicy/utils/shared_buffer.py` - 简化为单智能体Buffer
  - `onpolicy/algorithms/sheep_actor_critic.py` - 简化接口

---

## 架构分析

### 当前问题

| 特征 | 当前实现 | 实际需求 |
|------|---------|---------|
| 决策主体 | 伪装成多智能体 | 单一中央控制器 |
| 动作空间 | 复制给每个"智能体" | 单一动作输出 |
| 奖励信号 | 复制给每个"智能体" | 单一团队奖励 |
| Buffer维度 | `(episode, threads, agents, ...)` | `(episode, threads, ...)` |

### 当前代码问题示例

```python
# train.py 第253行 - 不必要的维度扩展
rewards_batch = np.array([[rewards] * num_agents] * n_rollout_threads)

# 实际上只需要
rewards_batch = np.array([rewards] * n_rollout_threads)
```

### 目标架构
┌─────────────────────────────────────────────────┐
│          单一中央控制器（标准PPO）               │
│                                                 │
│   输入: 羊群状态观测 (obs_dim=10)               │
│   输出: 统一站位参数 (action_dim=4)             │
│   执行: 所有机械狗按参数采样站位                │
│   奖励: 单一团队奖励                            │
└─────────────────────────────────────────────────┘

## 简化方案

### 1. Buffer简化
- 移除 `num_agents` 维度
- 数据形状从 `(episode_length, n_rollout_threads, num_agents, ...)` 
- 简化为 `(episode_length, n_rollout_threads, ...)`

### 2. 网络接口简化
- 移除 `share_obs` 和 `obs` 的区分（单智能体不需要）
- 简化 `get_actions()` 接口

### 3. 训练循环简化
- 移除 `num_agents` 相关的循环和扩展
- 直接使用单一观测和动作维度
- 奖励信号直接应用于所有机械狗

## 实现步骤

### 1. 缓冲区修改
- 调整 `SharedReplayBuffer` 为单智能体版本
- 移除 `num_agents` 相关的索引操作

### 2. 网络修改
- 调整 `SheepActorCritic` 网络结构
- 移除 `share_obs` 相关的代码

### 3. 训练脚本修改
- 调整 `train.py` 中的循环结构
- 移除 `num_agents` 相关的奖励计算
- 直接应用奖励信号到所有机械狗