"""
SheepScenario: 羊群引导场景管理类
管理羊群、机械狗、目标位置等场景元素
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from envs.sheep_entity import SheepEntity


class SheepScenario:
    """
    场景管理类
    
    管理整个羊群引导场景，包括:
    - 羊群实体
    - 机械狗位置
    - 目标位置
    - 场景边界
    """
    
    def __init__(
        self,
        world_size: Tuple[float, float] = (50.0, 50.0),
        num_sheep: int = 10,
        num_herders: int = 3,
        target_position: Optional[np.ndarray] = None,
        sheep_config: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        """
        初始化场景
        
        Args:
            world_size: 世界大小 (width, height)
            num_sheep: 羊的数量
            num_herders: 机械狗数量
            target_position: 目标位置，如果为None则随机生成
            sheep_config: 羊的配置参数
            random_seed: 随机种子
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.world_size = world_size
        self.num_sheep = num_sheep
        self.num_herders = num_herders
        
        self.sheep_config = sheep_config or {
            'max_speed': 1.0,
            'max_force': 0.1,
            'perception_radius': 5.0,
            'separation_radius': 2.0,
        }
        
        self.sheep: List[SheepEntity] = []
        self.herder_positions: np.ndarray = np.zeros((num_herders, 2), dtype=np.float32)
        self.target_position = target_position
        
        self.boids_weights = {
            'separation': 1.5,
            'alignment': 1.0,
            'cohesion': 1.0,
            'evasion': 2.0,
            'boundary': 1.0,
        }
        
        self._init_scenario()
    
    def _init_scenario(self):
        """初始化场景元素"""
        self._init_sheep()
        self._init_herders()
        self._init_target()
    
    def _init_sheep(self):
        """初始化羊群"""
        self.sheep = []
        center = np.array([self.world_size[0] * 0.3, self.world_size[1] * 0.5])
        spread = 5.0
        
        for _ in range(self.num_sheep):
            pos = center + np.random.uniform(-spread, spread, 2)
            pos = np.clip(pos, [1, 1], [self.world_size[0]-1, self.world_size[1]-1])
            
            sheep = SheepEntity(
                position=pos,
                max_speed=self.sheep_config['max_speed'],
                max_force=self.sheep_config['max_force'],
                perception_radius=self.sheep_config['perception_radius'],
                separation_radius=self.sheep_config['separation_radius'],
            )
            self.sheep.append(sheep)
    
    def _init_herders(self):
        """初始化机械狗位置"""
        self.herder_positions = np.zeros((self.num_herders, 2), dtype=np.float32)
        start_x = self.world_size[0] * 0.1
        spacing = self.world_size[1] / (self.num_herders + 1)
        
        for i in range(self.num_herders):
            self.herder_positions[i] = [
                start_x,
                spacing * (i + 1)
            ]
    
    def _init_target(self):
        """初始化目标位置"""
        if self.target_position is None:
            self.target_position = np.array([
                self.world_size[0] * 0.85,
                self.world_size[1] * 0.5
            ], dtype=np.float32)
        else:
            self.target_position = np.array(self.target_position, dtype=np.float32)
    
    def reset(
        self,
        target_position: Optional[np.ndarray] = None,
        random_seed: Optional[int] = None,
    ):
        """
        重置场景
        
        Args:
            target_position: 新的目标位置
            random_seed: 新的随机种子
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        if target_position is not None:
            self.target_position = np.array(target_position, dtype=np.float32)
        
        self._init_sheep()
        self._init_herders()
    
    def update_herders(self, positions: np.ndarray):
        """
        更新机械狗位置
        
        Args:
            positions: 新的机械狗位置数组，形状为 (num_herders, 2)
        """
        positions = np.array(positions, dtype=np.float32)
        
        for i in range(min(len(positions), self.num_herders)):
            self.herder_positions[i] = np.clip(
                positions[i],
                [0, 0],
                [self.world_size[0], self.world_size[1]]
            )
    
    def update_sheep(self, dt: float = 0.1):
        """
        更新羊群状态
        
        应用Boids规则并更新每只羊的位置
        """
        herder_list = [self.herder_positions[i] for i in range(self.num_herders)]
        
        for sheep in self.sheep:
            sheep.apply_boids_rules(
                all_sheep=self.sheep,
                herders=herder_list,
                world_size=self.world_size,
                weights=self.boids_weights,
            )
            sheep.update(dt)
            
            sheep.position = np.clip(
                sheep.position,
                [0, 0],
                [self.world_size[0], self.world_size[1]]
            )
    
    def get_flock_center(self) -> np.ndarray:
        """获取羊群质心位置"""
        if not self.sheep:
            return np.zeros(2, dtype=np.float32)
        
        positions = np.array([s.position for s in self.sheep])
        return np.mean(positions, axis=0)
    
    def get_flock_spread(self) -> float:
        """获取羊群扩散度"""
        if len(self.sheep) < 2:
            return 0.0
        
        center = self.get_flock_center()
        positions = np.array([s.position for s in self.sheep])
        distances = np.linalg.norm(positions - center, axis=1)
        return float(np.std(distances))
    
    def get_flock_direction(self) -> np.ndarray:
        """获取羊群主方向"""
        if not self.sheep:
            return np.zeros(2, dtype=np.float32)
        
        velocities = np.array([s.velocity for s in self.sheep])
        avg_velocity = np.mean(velocities, axis=0)
        
        norm = np.linalg.norm(avg_velocity)
        if norm > 1e-6:
            return avg_velocity / norm
        return np.zeros(2, dtype=np.float32)
    
    def get_flock_state(self) -> Dict[str, Any]:
        """
        获取羊群状态信息
        
        Returns:
            包含质心、方向、扩散度等信息的字典
        """
        return {
            'center': self.get_flock_center(),
            'direction': self.get_flock_direction(),
            'spread': self.get_flock_spread(),
            'num_sheep': len(self.sheep),
            'positions': np.array([s.position for s in self.sheep]),
            'velocities': np.array([s.velocity for s in self.sheep]),
        }
    
    def get_distance_to_target(self) -> float:
        """获取羊群质心到目标的距离"""
        center = self.get_flock_center()
        return float(np.linalg.norm(center - self.target_position))
    
    def is_flock_at_target(self, threshold: float = 5.0) -> bool:
        """检查羊群是否到达目标"""
        return self.get_distance_to_target() < threshold
    
    def get_herder_positions(self) -> np.ndarray:
        """获取所有机械狗位置"""
        return self.herder_positions.copy()
    
    def get_target_position(self) -> np.ndarray:
        """获取目标位置"""
        return self.target_position.copy()
    
    def get_observation(self) -> np.ndarray:
        """
        获取观测向量
        
        观测向量结构:
        - [0:2] 羊群质心位置
        - [2:4] 羊群主方向向量
        - [4] 羊群扩散度
        - [5] 羊群数量 (归一化)
        - [6:8] 目标位置
        - [8:10] 当前机械狗位置
        """
        flock_state = self.get_flock_state()
        
        obs = np.zeros(10, dtype=np.float32)
        obs[0:2] = flock_state['center'] / self.world_size[0]
        obs[2:4] = flock_state['direction']
        obs[4] = flock_state['spread'] / 10.0
        obs[5] = flock_state['num_sheep'] / 30.0
        obs[6:8] = self.target_position / self.world_size[0]
        obs[8:10] = self.herder_positions[0] / self.world_size[0]
        
        return obs
    
    def get_shared_observation(self) -> np.ndarray:
        """
        获取共享观测向量 (用于中心化Critic)
        
        Shape: (obs_dim * num_herders + 6,) = (10 * 3 + 6,) = (36,)
        """
        flock_state = self.get_flock_state()
        
        shared_obs = []
        
        for i in range(self.num_herders):
            shared_obs.extend(flock_state['center'] / self.world_size[0])
            shared_obs.extend(flock_state['direction'])
            shared_obs.append(flock_state['spread'] / 10.0)
            shared_obs.append(flock_state['num_sheep'] / 30.0)
            shared_obs.extend(self.target_position / self.world_size[0])
            shared_obs.extend(self.herder_positions[i] / self.world_size[0])
        
        shared_obs.extend(self.target_position / self.world_size[0])
        shared_obs.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(shared_obs, dtype=np.float32)
    
    def set_boids_weights(self, weights: Dict[str, float]):
        """设置Boids规则权重"""
        self.boids_weights.update(weights)
    
    def __repr__(self):
        return (
            f"SheepScenario("
            f"world_size={self.world_size}, "
            f"num_sheep={self.num_sheep}, "
            f"num_herders={self.num_herders}, "
            f"target={self.target_position})"
        )