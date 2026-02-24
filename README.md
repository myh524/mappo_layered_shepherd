# InforMARL: 可扩展的多智能体强化学习框架

## 项目简介

InforMARL是一个基于图神经网络的多智能体强化学习（MARL）框架，专为解决有限局部观察条件下的多智能体导航和碰撞避免问题而设计。通过智能信息聚合，InforMARL能够在分散式决策环境中实现高效的多智能体协作。

### 核心特点

- **基于图神经网络的信息聚合**：利用GNN有效聚合局部邻居信息，提高决策质量
- **可扩展性**：在测试时能够很好地适应具有任意数量智能体和障碍物的环境
- **灵活性**：可与任何标准MARL算法（如MAPPO）结合使用
- **高效训练**：相比基线方法，具有更好的样本效率和性能
- **真实环境模拟**：提供了与图神经网络兼容的导航环境

## 目录结构

```
shepherd_mappo/
├── onpolicy/           # 算法实现，包含MAPPO等
│   ├── algorithms/     # 算法核心实现
│   ├── config.py       # 配置文件
│   ├── envs/           # 环境包装器
│   ├── runner/         # 训练和评估运行器
│   ├── scripts/        # 训练脚本
│   └── utils/          # 工具函数
├── multiagent/         # 环境实现
│   ├── custom_scenarios/ # 自定义场景
│   ├── environment.py  # 环境核心
│   └── MPE_env.py      # MPE环境包装器
├── scripts/            # 各种训练和测试脚本
├── baselines/          # 基线算法实现
├── utils/              # 通用工具
├── requirement.txt     # 依赖项
└── README.md           # 项目说明
```

## 安装指南

### 依赖项

- Python 3.8+
- PyTorch 1.11.0
- PyTorch Geometric 2.0.4
- PyTorch Scatter 2.0.8
- PyTorch Sparse 0.6.12
- Gym 0.10.5+
- NumPy
- NumPy-STL

### 安装步骤

1. 克隆仓库：
   ```bash
   git clone https://github.com/nsidn98/InforMARL.git
   cd InforMARL
   ```

2. 安装依赖：
   ```bash
   pip install -r requirement.txt
   ```

3. 安装PyTorch Geometric相关包（根据您的CUDA版本）：
   ```bash
   TORCH="1.11.0"
   CUDA="cu113"  # 根据您的CUDA版本调整
   pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
   pip install torch-geometric
   ```

## 快速开始

### 训练InforMARL

```bash
python -u onpolicy/scripts/train_mpe.py --use_valuenorm --use_popart \
--project_name "informarl" \
--env_name "GraphMPE" \
--algorithm_name "rmappo" \
--seed 0 \
--experiment_name "informarl" \
--scenario_name "navigation_graph" \
--num_agents 3 \
--collision_rew 5 \
--n_training_threads 1 --n_rollout_threads 128 \
--num_mini_batch 1 \
--episode_length 25 \
--num_env_steps 2000000 \
--ppo_epoch 10 --use_ReLU --gain 0.01 --lr 7e-4 --critic_lr 7e-4 \
--user_name "marl" \
--use_cent_obs "False" \
--graph_feat_type "relative" \
--auto_mini_batch_size --target_mini_batch_size 128
```

### 环境使用示例

```python
from multiagent.environment import MultiAgentGraphEnv
from multiagent.policy import InteractivePolicy

# 创建参数对象
class Args:
    def __init__(self):
        self.num_agents:int=3
        self.world_size=2
        self.num_scripted_agents=0
        self.num_obstacles:int=3
        self.collaborative:bool=False 
        self.max_speed:Optional[float]=2
        self.collision_rew:float=5
        self.goal_rew:float=5
        self.min_dist_thresh:float=0.1
        self.use_dones:bool=False
        self.episode_length:int=25
        self.max_edge_dist:float=1
        self.graph_feat_type:str='global'
args = Args()

# 创建场景和环境
scenario = Scenario()
world = scenario.make_world(args)
env = MultiAgentGraphEnv(world=world, reset_callback=scenario.reset_world, 
                    reward_callback=scenario.reward, 
                    observation_callback=scenario.observation, 
                    graph_observation_callback=scenario.graph_observation,
                    info_callback=scenario.info_callback, 
                    done_callback=scenario.done,
                    id_callback=scenario.get_id,
                    update_graph=scenario.update_graph,
                    shared_viewer=False)

# 重置环境
obs_n, agent_id_n, node_obs_n, adj_n = env.reset()

# 执行步骤
while True:
    # 获取每个智能体的动作
    act_n = []
    for i, policy in enumerate(policies):
        act_n.append(policy.action(obs_n[i]))
    # 环境步进
    obs_n, agent_id_n, node_obs_n, adj_n, reward_n, done_n, info_n = env.step(act_n)
    # 渲染环境
    env.render()
```

## 核心功能

### 1. 图神经网络信息聚合

InforMARL使用图神经网络聚合智能体局部邻居的信息，为每个智能体提供更全面的环境感知。这使得智能体能够做出更明智的决策，特别是在复杂的多智能体环境中。

### 2. 多智能体导航环境

提供了专门为图神经网络设计的导航环境，支持：
- 多个智能体同时导航
- 动态障碍物
- 目标点到达奖励
- 碰撞避免惩罚
- 可配置的世界大小和智能体数量

### 3. 与多种MARL算法兼容

InforMARL的信息聚合模块可以与多种MARL算法结合使用，如：
- MAPPO (Multi-Agent Proximal Policy Optimization)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- QMIX
- VDN

### 4. 可扩展性

InforMARL在训练时使用固定数量的智能体，但在测试时能够扩展到任意数量的智能体，这使得它非常适合实际应用场景。

## 算法原理

InforMARL的核心思想是通过图神经网络智能地聚合局部信息，具体步骤如下：

1. **环境建模**：将环境中的智能体、障碍物和目标点建模为图中的节点
2. **信息聚合**：使用GNN聚合每个智能体局部邻居的信息
3. **决策制定**：将聚合的信息输入到actor网络中生成动作
4. **价值评估**：使用聚合的全局信息输入到critic网络中评估状态价值

这种方法允许智能体在仅具有局部观察的情况下做出全局最优的决策。

## 实验结果

InforMARL在多智能体导航任务上表现出色，相比其他基线方法：
- 更高的成功率
- 更少的碰撞
- 更好的样本效率
- 更强的可扩展性

## 应用场景

- **自主导航**：多机器人协作导航
- **交通管理**：智能交通系统中的车辆协调
- **无人机集群**：多无人机协同任务
- **仓库自动化**：多AGV协同工作
- **任何需要多智能体协作的场景**

## 贡献

我们欢迎社区贡献，特别是：
- 添加新的场景和环境
- 改进算法性能
- 扩展到新的应用领域
- 修复bug和改进文档

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 引用

如果您在研究中使用了本项目，请引用以下论文：

```bibtex
@article{nayak22informarl,
  doi = {10.48550/ARXIV.2211.02127},
  url = {https://arxiv.org/abs/2211.02127},
  author = {Nayak, Siddharth and Choi, Kenneth and Ding, Wenqi and Dolan, Sydney and Gopalakrishnan, Karthik and Balakrishnan, Hamsa},
  keywords = {Multiagent Systems (cs.MA), Artificial Intelligence (cs.AI), Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## 联系方式

如有问题或建议，请通过以下方式联系我们：
- 提交GitHub Issue
- 发送邮件至：sidnayak@mit.edu

---

感谢您对InforMARL的关注和支持！