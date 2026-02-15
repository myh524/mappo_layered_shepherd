# Tasks

- [ ] Task 1: 创建简化的PPO Buffer
  - [ ] SubTask 1.1: 创建 `onpolicy/utils/ppo_buffer.py`
  - [ ] SubTask 1.2: 移除num_agents维度
  - [ ] SubTask 1.3: 实现简化的insert和compute_returns

- [ ] Task 2: 简化Actor-Critic网络
  - [ ] SubTask 2.1: 创建 `onpolicy/algorithms/ppo_actor_critic.py`
  - [ ] SubTask 2.2: 简化get_actions接口
  - [ ] SubTask 2.3: 移除share_obs相关代码

- [ ] Task 3: 重构训练脚本
  - [ ] SubTask 3.1: 创建新的 `train_ppo.py`
  - [ ] SubTask 3.2: 移除num_agents维度扩展
  - [ ] SubTask 3.3: 简化训练循环

- [ ] Task 4: 测试验证
  - [ ] SubTask 4.1: 创建单元测试
  - [ ] SubTask 4.2: 验证训练流程
  - [ ] SubTask 4.3: 对比训练效果

# Task Dependencies
- Task 2 depends on Task 1
- Task 3 depends on Task 2
- Task 4 depends on Task 3