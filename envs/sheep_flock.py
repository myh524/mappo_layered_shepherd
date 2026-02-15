"""
SheepFlockEnv: 羊群引导强化学习环境
实现Gym风格的多智能体环境接口
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
from gym import spaces
from envs.sheep_scenario import SheepScenario


class SheepFlockEnv:
    """
    羊群引导多智能体环境
    
    高层控制器通过观测羊群状态，输出站位参数，
    指导机械狗围堵和引导羊群到达目标位置。
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (50.0, 50.0),
        num_sheep: int = 10,
        num_herders: int = 3,
        episode_length: int = 100,
        dt: float = 0.1,
        reward_config: Optional[Dict[str, float]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        初始化环境
        
        Args:
            world_size: 世界大小
            num_sheep: 羊的数量
            num_herders: 机械狗数量
            episode_length: 每个episode的最大步数
            dt: 时间步长
            reward_config: 奖励配置
            random_seed: 随机种子
        """
        self.world_size = world_size
        self.num_sheep = num_sheep
        self.num_herders = num_herders
        self.episode_length = episode_length
        self.dt = dt
        self.random_seed = random_seed
        
        self.reward_config = reward_config or {
            'distance_reward_weight': 1.0,
            'spread_penalty_weight': 0.1,
            'success_bonus': 10.0,
            'timeout_penalty': -5.0,
            'collision_penalty': -0.5,
        }
        
        self.scenario = SheepScenario(
            world_size=world_size,
            num_sheep=num_sheep,
            num_herders=num_herders,
            random_seed=random_seed,
        )
        
        self._setup_spaces()
        
        self.current_step = 0
        self.prev_distance = None
        
        self._seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _setup_spaces(self):
        """设置观测空间和动作空间"""
        self.obs_dim = 10
        self.action_dim = 4
        
        obs_low = np.array([-np.inf] * self.obs_dim, dtype=np.float32)
        obs_high = np.array([np.inf] * self.obs_dim, dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        action_low = np.array([0.0, 0.0, -np.pi, 0.0], dtype=np.float32)
        action_high = np.array([20.0, 5.0, np.pi, 1.0], dtype=np.float32)
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        
        self.share_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim * self.num_herders + 6,),
            dtype=np.float32,
        )
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始观测
        """
        self.current_step = 0
        
        if self._seed is not None:
            np.random.seed(self._seed)
            self._seed = None
        
        target_pos = np.array([
            self.world_size[0] * np.random.uniform(0.7, 0.9),
            self.world_size[1] * np.random.uniform(0.3, 0.7)
        ])
        
        self.scenario.reset(target_position=target_pos)
        self.prev_distance = self.scenario.get_distance_to_target()
        
        return self._get_obs()
    
    def get_shared_obs(self) -> np.ndarray:
        """
        获取共享观测
        
        Returns:
            共享观测，形状为 (num_herders, obs_dim)
        """
        return self.scenario.get_shared_observation()

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        执行动作
        
        Args:
            actions: 动作数组，形状为 (num_herders, action_dim)
                    每个动作包含:
                    - radius_mean: 站位半径
                    - radius_std: 半径标准差
                    - angle_mean: 聚集角度
                    - concentration: 聚集度
        
        Returns:
            obs: 观测
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        self.current_step += 1
        
        herder_positions = self._sample_herder_positions(actions)
        self.scenario.update_herders(herder_positions)
        
        self.scenario.update_sheep(self.dt)
        
        reward = self._compute_reward()
        
        done = self._check_done()
        
        info = self._get_info()
        
        obs = self._get_obs()
        
        return obs, reward, done, info
    
    def _sample_herder_positions(self, actions: np.ndarray) -> np.ndarray:
        """
        根据动作采样机械狗站位
        
        动作解释:
        - radius_mean: 围绕羊群质心的半径
        - radius_std: 半径变化范围
        - angle_mean: 围绕羊群质心的角度均值
        - concentration: 聚集度 [0, 1]，越高站位越集中
        """
        flock_center = self.scenario.get_flock_center()
        positions = np.zeros((self.num_herders, 2), dtype=np.float32)
        
        base_angle_step = 2 * np.pi / self.num_herders
        
        for i in range(self.num_herders):
            action = actions[i] if actions.ndim > 1 else actions
            
            radius_mean = float(action[0])
            radius_std = float(action[1])
            angle_mean = float(action[2])
            concentration = float(action[3])
            
            radius = np.random.normal(radius_mean, radius_std * (1 - concentration))
            radius = max(1.0, min(radius, 25.0))
            
            angle_offset = base_angle_step * i
            angle = angle_mean + angle_offset
            angle += np.random.uniform(-0.5, 0.5) * (1 - concentration)
            
            pos = flock_center + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])
            
            pos = np.clip(pos, [0, 0], self.world_size)
            positions[i] = pos
        
        return positions
    
    def _compute_reward(self) -> float:
        """计算奖励"""
        reward = 0.0
        
        current_distance = self.scenario.get_distance_to_target()
        
        if self.prev_distance is not None:
            distance_delta = self.prev_distance - current_distance
            reward += distance_delta * self.reward_config['distance_reward_weight']
        
        self.prev_distance = current_distance
        
        spread = self.scenario.get_flock_spread()
        if spread > 10.0:
            reward -= (spread - 10.0) * self.reward_config['spread_penalty_weight']
        
        if self.scenario.is_flock_at_target(threshold=5.0):
            reward += self.reward_config['success_bonus']
        
        return float(reward)
    
    def _check_done(self) -> bool:
        """检查episode是否结束"""
        if self.scenario.is_flock_at_target(threshold=5.0):
            return True
        
        if self.current_step >= self.episode_length:
            return True
        
        return False
    
    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        return self.scenario.get_observation()
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            'step': self.current_step,
            'distance_to_target': self.scenario.get_distance_to_target(),
            'flock_spread': self.scenario.get_flock_spread(),
            'is_success': self.scenario.is_flock_at_target(threshold=5.0),
            'flock_state': self.scenario.get_flock_state(),
        }
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        渲染环境
        
        Args:
            mode: 渲染模式，'human' 或 'rgb_array'
        """
        if mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            self._render_human()
    
    def _render_human(self):
        """控制台渲染"""
        print(f"\n--- Step {self.current_step} ---")
        print(f"Flock center: {self.scenario.get_flock_center()}")
        print(f"Flock spread: {self.scenario.get_flock_spread():.2f}")
        print(f"Distance to target: {self.scenario.get_distance_to_target():.2f}")
        print(f"Herder positions: {self.scenario.get_herder_positions()}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """生成RGB图像"""
        img_size = 400
        scale = img_size / max(self.world_size)
        
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
        
        target = self.scenario.get_target_position()
        tx, ty = int(target[0] * scale), int(target[1] * scale)
        img[max(0, ty-10):min(img_size, ty+10), max(0, tx-10):min(img_size, tx+10)] = [0, 200, 0]
        
        flock_state = self.scenario.get_flock_state()
        for pos in flock_state['positions']:
            px, py = int(pos[0] * scale), int(pos[1] * scale)
            if 0 <= px < img_size and 0 <= py < img_size:
                img[max(0, py-3):min(img_size, py+3), max(0, px-3):min(img_size, px+3)] = [200, 200, 200]
        
        for hpos in self.scenario.get_herder_positions():
            hx, hy = int(hpos[0] * scale), int(hpos[1] * scale)
            if 0 <= hx < img_size and 0 <= hy < img_size:
                img[max(0, hy-5):min(img_size, hy+5), max(0, hx-5):min(img_size, hx+5)] = [0, 0, 200]
        
        return img
    
    def close(self):
        """关闭环境"""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)
    
    def get_env_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        return {
            'num_agents': self.num_herders,
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'episode_length': self.episode_length,
            'world_size': self.world_size,
        }


class SheepFlockEnvWrapper:
    """
    多进程环境包装器
    
    用于创建多个并行环境，符合MAPPO训练框架的接口
    """
    
    def __init__(
        self,
        num_envs: int = 1,
        **kwargs
    ):
        """
        初始化包装器
        
        Args:
            num_envs: 并行环境数量
            **kwargs: 传递给SheepFlockEnv的参数
        """
        self.num_envs = num_envs
        self.envs = [SheepFlockEnv(**kwargs) for _ in range(num_envs)]
        
        self.num_agents = self.envs[0].num_herders
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.share_observation_space = self.envs[0].share_observation_space
    
    def reset(self) -> np.ndarray:
        """重置所有环境"""
        obs_list = [env.reset() for env in self.envs]
        return np.array(obs_list)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        在所有环境中执行动作
        
        Args:
            actions: 形状为 (num_envs, num_agents, action_dim) 的动作数组
        
        Returns:
            obs: 观测数组
            rewards: 奖励数组
            dones: 结束标志数组
            infos: 信息列表
        """
        obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        
        for i, env in enumerate(self.envs):
            env_actions = actions[i] if actions.ndim > 2 else actions
            obs, reward, done, info = env.step(env_actions)
            obs_list.append(obs)
            rewards_list.append(reward)
            dones_list.append(done)
            infos_list.append(info)
        
        return (
            np.array(obs_list),
            np.array(rewards_list),
            np.array(dones_list),
            infos_list,
        )
    
    def render(self, mode: str = 'human'):
        """渲染第一个环境"""
        return self.envs[0].render(mode)
    
    def close(self):
        """关闭所有环境"""
        for env in self.envs:
            env.close()
    
    def seed(self, seed: Optional[int] = None):
        """设置随机种子"""
        for i, env in enumerate(self.envs):
            env.seed(seed + i if seed is not None else None)