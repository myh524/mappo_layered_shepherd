# Bug修复与训练推进规范

## Why
项目当前处于阶段3（训练调试），但代码存在多个关键Bug导致无法正常运行测试和训练。需要先修复这些Bug，然后继续推进训练功能开发。

## What Changes
- 修复train.py中的维度不匹配问题
- 修复BoundedACTLayer中log_probs计算错误
- 修复SheepActorCritic.evaluate_actions参数不匹配问题
- 修复SharedReplayBuffer数据维度处理问题
- 添加正确的环境依赖检查

## Impact
- Affected specs: 训练系统、网络结构、数据缓冲
- Affected code: 
  - train.py
  - onpolicy/algorithms/sheep_actor_critic.py
  - onpolicy/algorithms/utils/bounded_act.py
  - onpolicy/utils/shared_buffer.py

## ADDED Requirements

### Requirement: Bug修复 - train.py维度处理
系统SHALL正确处理环境观测和动作的维度转换。

#### Scenario: 正确的观测维度处理
- **WHEN** 环境返回观测时
- **THEN** 观测应正确扩展为(n_rollout_threads, num_agents, obs_dim)维度

#### Scenario: 正确的动作采样
- **WHEN** 策略输出动作时
- **THEN** 动作应正确reshape为(num_agents, action_dim)供环境使用

### Requirement: Bug修复 - BoundedACTLayer log_probs
系统SHALL正确计算有界动作的log概率。

#### Scenario: log_probs计算
- **WHEN** 计算动作的log概率时
- **THEN** 应先将动作从action_space转换回tanh空间再计算

### Requirement: Bug修复 - evaluate_actions接口
系统SHALL提供正确的evaluate_actions接口。

#### Scenario: 参数顺序
- **WHEN** 调用evaluate_actions时
- **THEN** 参数顺序应为(cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, ...)

### Requirement: 训练流程完整性
系统SHALL能够完整运行训练流程。

#### Scenario: 训练启动
- **WHEN** 运行train.py时
- **THEN** 应能成功初始化环境、网络、缓冲区并开始训练

#### Scenario: 训练循环
- **WHEN** 训练进行中时
- **THEN** 应能正确收集经验、计算回报、更新网络

## MODIFIED Requirements

### Requirement: 共享观测空间维度
共享观测空间维度应为(obs_dim * num_herders,)而非(obs_dim * num_herders + 6,)，以保持一致性。

## REMOVED Requirements
无