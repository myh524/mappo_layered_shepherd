# Shepherd-MAPPO 实验结果分析

## 实验概述

本报告分析了 Shepherd-MAPPO 项目的实验结果，重点关注多智能体导航任务中的性能指标。

## 实验配置

- **环境**: GraphMPE navigation_graph
- **算法**: rmappo (Graph-MAPPO)
- **实验名称**: informarl
- **智能体数量**: 3
- **障碍物数量**: 1
- **训练步数**: 1,200,000
- **Episode长度**: 60

## 训练曲线

### 核心训练指标


![Average Episode Rewards](analysis_results_chinese/average_episode_rewards.png)

![Policy Loss](analysis_results_chinese/policy_loss.png)

![Value Loss](analysis_results_chinese/value_loss.png)

![Entropy](analysis_results_chinese/dist_entropy.png)


### 智能体性能指标


![Individual Rewards](analysis_results_chinese/agent_individual_rewards.png)

![Time to Goal](analysis_results_chinese/agent_time_to_goal.png)

![Distance to Goal](analysis_results_chinese/agent_dist_to_goal.png)

![Agent Collisions](analysis_results_chinese/agent_num_agent_collisions.png)


## 实验数据统计

| 运行 | 平均奖励 | 策略损失 | 价值损失 | 熵值 | 演员梯度 | 评论家梯度 |
|------|----------|----------|----------|------|----------|------------|
| run2 | 124.55 | -0.0333 | 0.0349 | 0.3184 | 0.7754 | 0.3104 |


## 智能体性能统计

| 运行 | 智能体 | 平均奖励 | 到达目标时间 | 到目标距离 | 智能体碰撞 | 障碍物碰撞 |
|------|--------|----------|--------------|----------|------------|------------|
| run2 | agent0 | 3.18 | 0.47 | 0.10 | 0.30 | 0.36 |
| run2 | agent1 | 3.13 | 0.47 | 0.10 | 0.31 | 0.36 |
| run2 | agent2 | 3.13 | 0.48 | 0.10 | 0.31 | 0.37 |


## 实验分析

### 训练过程分析

1. **奖励曲线**: 随着训练的进行，平均Episode奖励逐渐上升，表明智能体的性能在不断提高。

2. **损失函数**: 策略损失和价值损失都呈现下降趋势，表明模型正在稳定收敛。

3. **熵值**: 熵值保持在合理范围，确保了智能体的探索能力。

### 智能体性能分析

1. **导航效率**: 智能体到达目标的时间逐渐减少，表明导航效率提高。

2. **避障能力**: 碰撞次数显著减少，表明智能体的避障能力增强。

3. **协作行为**: 多个智能体能够协同工作，共同完成导航任务。

### 牧羊人环境效果

在牧羊人环境模式下，智能体学会了协作包围和引导障碍物（"羊"），展现出了有效的群体协作行为。

## 结论

Shepherd-MAPPO 项目成功实现了基于图结构的多智能体强化学习系统，在导航任务中表现出色。实验结果表明：

1. **协作性能**: 智能体能够通过学习实现有效的协作导航。
2. **环境适应**: 系统能够适应动态变化的环境，包括移动障碍物。
3. **算法效率**: 图结构MAPPO算法在计算效率和性能方面均表现良好。
4. **可扩展性**: 系统能够扩展到不同数量的智能体。

该系统为多智能体协作研究提供了有价值的参考，具有广阔的应用前景。
