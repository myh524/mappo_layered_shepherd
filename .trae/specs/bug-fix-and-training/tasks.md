# Tasks

## Phase 1: Bug修复

- [x] Task 1: 修复BoundedACTLayer log_probs计算错误
  - [x] SubTask 1.1: 在TanhNormal类中添加from_action_space方法，将动作从action_space转换回tanh空间
  - [x] SubTask 1.2: 修改log_probs方法，先转换动作再计算log概率
  - [x] SubTask 1.3: 验证log_probs计算正确性

- [x] Task 2: 修复SheepActorCritic.evaluate_actions参数问题
  - [x] SubTask 2.1: 修改evaluate_actions方法签名，添加cent_obs参数
  - [x] SubTask 2.2: 确保参数顺序与train.py调用一致
  - [x] SubTask 2.3: 添加单元测试验证接口正确性

- [x] Task 3: 修复train.py维度处理问题
  - [x] SubTask 3.1: 修复warmup中的观测维度扩展逻辑
  - [x] SubTask 3.2: 修复collect中的动作reshape逻辑
  - [x] SubTask 3.3: 修复insert中的数据维度处理
  - [x] SubTask 3.4: 修复compute_returns中的维度问题

- [x] Task 4: 修复SharedReplayBuffer兼容性问题
  - [x] SubTask 4.1: 检查buffer初始化参数
  - [x] SubTask 4.2: 确保buffer与连续动作空间兼容
  - [x] SubTask 4.3: 修复feed_forward_generator的输出格式

## Phase 2: 训练验证

- [x] Task 5: 运行环境测试
  - [x] SubTask 5.1: 安装必要依赖(gym, torch, numpy)
  - [x] SubTask 5.2: 运行test_env.py验证环境正确性
  - [x] SubTask 5.3: 运行test_network.py验证网络正确性

- [x] Task 6: 运行训练测试
  - [x] SubTask 6.1: 运行简短训练测试(20 episodes)
  - [x] SubTask 6.2: 验证训练循环无错误
  - [x] SubTask 6.3: 检查奖励是否合理变化

## Phase 3: 项目推进

- [ ] Task 7: 完善训练配置
  - [ ] SubTask 7.1: 创建训练配置文件
  - [ ] SubTask 7.2: 添加课程学习配置
  - [ ] SubTask 7.3: 添加超参数配置

- [ ] Task 8: 添加训练脚本
  - [ ] SubTask 8.1: 创建train_sheep.sh脚本
  - [ ] SubTask 8.2: 添加日志记录功能
  - [ ] SubTask 8.3: 添加模型保存/加载功能

# Task Dependencies
- [Task 2] depends on [Task 1]
- [Task 3] depends on [Task 2]
- [Task 4] depends on [Task 3]
- [Task 5] depends on [Task 4]
- [Task 6] depends on [Task 5]
- [Task 7] depends on [Task 6]
- [Task 8] depends on [Task 7]