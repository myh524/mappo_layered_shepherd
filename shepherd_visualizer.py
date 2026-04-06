"""
Shepherd MAPPo 模型可视化运行脚本
加载训练好的模型并实时渲染环境，使用与外部项目相同的背景风格

键盘控制：
- 空格键：暂停/继续
- 'n' 键：跳过当前 episode，进入下一个 episode
"""

import argparse
import os
import sys

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import torch
import numpy as np
import matplotlib

# 处理显示后端
temp_parser = argparse.ArgumentParser(add_help=False)
temp_parser.add_argument('--no_display', action='store_true', default=False)
temp_args, _ = temp_parser.parse_known_args()

if temp_args.no_display:
    matplotlib.use('Agg')
else:
    import warnings
    
    backend_set = False
    if sys.platform == 'linux':
        if 'DISPLAY' not in os.environ:
            matplotlib.use('Agg')
            backend_set = True
        else:
            try:
                import tkinter
                matplotlib.use('TkAgg')
                backend_set = True
            except (ImportError, RuntimeError):
                pass
    
    if not backend_set:
        try:
            matplotlib.use('Qt5Agg')
        except:
            try:
                matplotlib.use('TkAgg')
            except:
                matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button

from multiagent.MPE_env import GraphMPEEnv
from onpolicy.runner.shared.graph_mpe_runner import GMPERunner
from onpolicy.config import get_config, graph_config


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


@dataclass
class VisualizationState:
    current_step: int = 0
    total_reward: float = 0.0
    episode_done: bool = False
    success: bool = False
    action_history: List[np.ndarray] = None
    
    def __post_init__(self):
        if self.action_history is None:
            self.action_history = []


def parse_args():
    parser = argparse.ArgumentParser(description='Shepherd MAPPo 模型可视化')
    
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--render_delay', type=int, default=30)
    
    parser.add_argument('--num_agents', type=int, default=5)
    parser.add_argument('--num_obstacles', type=int, default=1)
    parser.add_argument('--world_size', type=float, default=2.0)
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--save_gif', type=str, default=None,
                       help='Path to save GIF of the entire episode')
    parser.add_argument('--save_video', type=str, default=None,
                       help='Path to save video of the entire episode')
    parser.add_argument('--save_episode_gif', action='store_true', default=False,
                       help='Save each episode as a GIF')
    parser.add_argument('--save_step_png', action='store_true', default=False,
                       help='Save each step as a PNG image in a new folder')
    parser.add_argument('--no_display', action='store_true', default=False,
                       help='Force non-interactive mode (save images instead of display)')
    
    return parser.parse_args()


class ShepherdVisualizer:
    def __init__(self, env, runner, args):
        self.env = env
        self.runner = runner
        self.args = args
        
        self.state = VisualizationState()
        self.fig = None
        self.ax = None
        self.ax_reward = None
        self.paused = False
        self.skip_episode = False
        self.frames = []
        
        self.current_obs = None
        self.reward_history = []
        
        self.interactive = not args.no_display and matplotlib.get_backend() != 'agg'
        # 创建输出目录
        self.output_dir = 'visualization_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not self.interactive:
            print("\nWarning: No interactive backend available or --no_display flag set.")
            print("Running in non-interactive mode.")
            print("Images will be saved to 'visualization_output/' directory.")
        
    def setup_figure(self):
        # 始终创建一个没有奖励面板的图形
        self.fig = plt.figure(figsize=(10, 10))
        self.ax = self.fig.add_axes([0.05, 0.1, 0.9, 0.85])
        
        world_size = self.args.world_size
        self.ax.set_xlim(-world_size/2, world_size/2)
        self.ax.set_ylim(-world_size/2, world_size/2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')  # 与外部项目相同的背景色
        
        self.ax_reward = None
        
        # 连接键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        plt.show(block=False)
        
    def toggle_pause(self, event):
        self.paused = not self.paused
        
    def skip_to_next_episode(self, event):
        self.skip_episode = True
        self.paused = False
        
    def on_key_press(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')
        elif event.key == 'n':
            self.skip_to_next_episode(event)
        
    def reset_episode(self):
        self.state = VisualizationState()
        self.current_obs = self.env.reset()
        
        self.reward_history = []
        
    def get_action(self, obs):
        # 使用 runner 的策略获取动作
        with torch.no_grad():
            # 对于 GraphMPE 环境，obs 是一个元组，包含 obs, agent_id, node_obs, adj
            # 我们需要将其转换为正确的维度格式
            obs_np, agent_id, node_obs, adj = obs
            
            # 初始化 RNN 状态和掩码
            batch_size = 1
            num_agents = self.args.num_agents
            recurrent_N = self.runner.recurrent_N
            hidden_size = self.runner.hidden_size
            
            rnn_states = np.zeros(
                (batch_size, num_agents, recurrent_N, hidden_size),
                dtype=np.float32
            )
            masks = np.ones(
                (batch_size, num_agents, 1),
                dtype=np.float32
            )
            
            # 调用 policy.act 方法获取动作
            self.runner.trainer.prep_rollout()
            action, rnn_states = self.runner.trainer.policy.act(
                np.concatenate(obs_np),
                np.concatenate(node_obs),
                np.concatenate(adj),
                np.concatenate(agent_id),
                np.concatenate(rnn_states),
                np.concatenate(masks),
                deterministic=True,
            )
            actions = np.array(np.split(_t2n(action), self.runner.n_rollout_threads))
            
            # 处理动作，转换为环境期望的格式
            action_space = self.env.action_space[0]
            if action_space.__class__.__name__ == "Discrete":
                actions_env = np.squeeze(np.eye(action_space.n)[actions], 2)
            else:
                raise NotImplementedError
        # 对于包装后的环境，我们需要返回一个包含动作的列表
        return actions_env.tolist()
    
    def render_env(self):
        self.ax.clear()
        
        world_size = self.args.world_size
        self.ax.set_xlim(-world_size/2, world_size/2)
        self.ax.set_ylim(-world_size/2, world_size/2)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#f5f5f5')  # 与外部项目相同的背景色
        
        # 绘制障碍物（如果是 shepherd 环境）
        if hasattr(self.env, 'envs') and hasattr(self.env.envs[0].world, 'use_shepherd_env') and self.env.envs[0].world.use_shepherd_env:
            for obstacle in self.env.envs[0].world.obstacles:
                pos = obstacle.state.p_pos
                # 绘制障碍物
                self.ax.scatter(pos[0], pos[1], 
                               c='red', s=225, alpha=0.9, marker='o', edgecolors='darkred')
        
        # 绘制智能体
        for agent in self.env.envs[0].world.agents:
            pos = agent.state.p_pos
            self.ax.scatter(pos[0], pos[1], 
                           c='blue', s=180, alpha=0.7, marker='s', edgecolors='darkblue',
                           label='Shepherd Agent' if agent.name == 'agent 0' else None)
        
        # 绘制目标点
        # for landmark in self.env.envs[0].world.landmarks:
        #     pos = landmark.state.p_pos
        #     self.ax.scatter(pos[0], pos[1], 
        #                    c='green', s=60, alpha=0.7, marker='*', edgecolors='darkgreen',
        #                    label='Goal' if landmark.name == 'landmark 0' else None)
        
        # 添加图例
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8)
        
        # 绘制信息文本
        obs = self.env.envs[0]._get_obs(self.env.envs[0].world.agents[0]) if hasattr(self.env.envs[0], '_get_obs') else []
        obs_text = f'Step: {self.state.current_step}'
        self.ax.text(0.02, 0.98, obs_text, transform=self.ax.transAxes,
                    fontsize=8, ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        status = "PAUSED" if self.paused else "RUNNING"
        self.ax.set_title(f'Step {self.state.current_step} | {status}', 
                     fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
    

    
    def run_episode(self, episode_num: int, total_episodes: int):
        print(f"\n{'='*50}")
        print(f"Episode {episode_num}/{total_episodes}")
        print(f"{'='*50}")
        
        self.reset_episode()
        self.state.episode_done = False
        
        if self.fig is None:
            self.setup_figure()
        
        for step in range(self.args.episode_length):
            if self.interactive:
                while self.paused:
                    plt.pause(0.1)
                
                if self.skip_episode:
                    self.skip_episode = False
                    break
            
            # 获取动作
            action = self.get_action(self.current_obs)
            
            # 执行动作
            obs, ag_ids, node_obs, adj, reward, done, info = self.env.step(action)
            
            self.state.current_step += 1
            self.state.total_reward += sum(sum(r) for r in reward)
            self.state.action_history.append(action.copy())
            
            self.render_env()
            
            if self.args.save_gif or self.args.save_video or self.args.save_episode_gif:
                self.fig.canvas.draw()
                self.frames.append(np.array(self.fig.canvas.renderer.buffer_rgba()))
            
            if self.interactive:
                plt.pause(self.args.render_delay / 1000.0)
            else:
                if step % 10 == 0:
                    save_path = os.path.join(self.output_dir, f'episode_{episode_num}_step_{step:04d}.png')
                    self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
            # 保存每个 step 为 PNG 图片
            if self.args.save_step_png:
                # 创建一个新的文件夹来保存当前 episode 的 step 图片
                step_dir = os.path.join(self.output_dir, f'episode_{episode_num}_steps')
                os.makedirs(step_dir, exist_ok=True)
                save_path = os.path.join(step_dir, f'step_{step:04d}.png')
                self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
            self.current_obs = (obs, ag_ids, node_obs, adj)
            
            if np.all(done):
                break
        
        if not self.interactive:
            final_path = os.path.join(self.output_dir, f'episode_{episode_num}_final.png')
            self.fig.savefig(final_path, dpi=100, bbox_inches='tight')
            print(f"  Saved visualization to: {final_path}")
        
        # 保存 episode 为 GIF
        if self.args.save_episode_gif and len(self.frames) > 0:
            gif_path = os.path.join(self.output_dir, f'episode_{episode_num}.gif')
            try:
                import imageio
                # 确保所有帧的形状相同
                if len(self.frames) > 0:
                    # 获取第一帧的形状
                    shape = self.frames[0].shape
                    # 调整所有帧的形状
                    frames = []
                    for frame in self.frames:
                        if frame.shape == shape:
                            frames.append(frame)
                    if len(frames) > 0:
                        imageio.mimsave(gif_path, frames, fps=20)
                        print(f"  Saved episode GIF to: {gif_path}")
            except ImportError:
                print("Warning: imageio required to save GIF")
        
        # 清空帧列表，为下一个 episode 做准备
        self.frames = []
        
        print(f"\nEpisode {episode_num} finished:")
        print(f"  Steps: {self.state.current_step}, Reward: {float(self.state.total_reward):.3f}")
        print(f"  Result: {'SUCCESS!' if self.state.success else 'COMPLETED'}")
        
        return self.state.success, self.state.total_reward, self.state.current_step
    
    def run(self):
        # 当保存每个 step 为 PNG 图片时，强制使用非交互式模式
        if self.args.save_step_png:
            self.interactive = False
        
        successes = []
        total_rewards = []
        total_steps = []
        
        for ep in range(self.args.num_episodes):
            success, reward, steps = self.run_episode(ep + 1, self.args.num_episodes)
            successes.append(success)
            total_rewards.append(reward)
            total_steps.append(steps)
        
        print(f"\n{'='*50}")
        print("Visualization completed!")
        print(f"Success rate: {sum(successes)}/{len(successes)} ({sum(successes)/len(successes)*100:.1f}%)")
        print(f"Average reward: {np.mean(total_rewards):.3f}")
        
        if self.args.save_gif:
            self.save_gif(self.args.save_gif)
        if self.args.save_video:
            self.save_video(self.args.save_video)
        
        plt.close('all')
    
    def save_gif(self, path: str):
        try:
            import imageio
            imageio.mimsave(path, self.frames, fps=20)
            print(f"GIF saved to: {path}")
        except ImportError:
            print("Warning: imageio required to save GIF")
    
    def save_video(self, path: str):
        try:
            import imageio
            imageio.mimsave(path, self.frames, fps=20)
            print(f"Video saved to: {path}")
        except ImportError:
            print("Warning: imageio required to save video")


def main():
    args = parse_args()
    
    print(f"Model directory: {args.model_dir}")
    print(f"Config: {args.num_agents} agents, {args.num_obstacles} obstacles, world size {args.world_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 配置环境参数
    parser = get_config()
    parser.set_defaults(
        env_name="GraphMPE",
        algorithm_name="rmappo",
        experiment_name="informarl",
        scenario_name="navigation_graph",
        num_agents=args.num_agents,
        num_scripted_agents=0,
        num_obstacles=args.num_obstacles,
        world_size=args.world_size,
        episode_length=args.episode_length,
        seed=args.seed,
        n_training_threads=1,
        n_rollout_threads=1,
        use_render=True,
        render_episodes=args.num_episodes,
        model_dir=args.model_dir,
        user_name="marl",
        use_cent_obs=False,
        graph_feat_type="relative",
        use_shepherd_env=True,
        save_gifs=False,  # 我们使用自定义的保存逻辑
        # 以下是 navigation_graph.py 需要的参数
        collaborative=True,
        max_speed=2.0,
        collision_rew=5.0,
        goal_rew=5.0,
        min_dist_thresh=0.1,
        use_dones=False,
        max_edge_dist=1.0
    )
    
    all_args, parser = graph_config([], parser)
    
    # 创建环境
    from onpolicy.envs.env_wrappers import GraphDummyVecEnv
    
    def make_env():
        def _thunk():
            env = GraphMPEEnv(all_args)
            env.seed(args.seed)
            return env
        return _thunk
    
    envs = GraphDummyVecEnv([make_env()])
    
    # 创建 runner
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": None,
        "num_agents": args.num_agents,
        "device": device,
        "run_dir": None,
    }
    
    runner = GMPERunner(config)
    
    # 加载模型
    if args.model_dir is not None:
        print(f"Restoring from checkpoint stored in {args.model_dir}")
        runner.restore()
    
    # 创建可视化器
    visualizer = ShepherdVisualizer(envs, runner, args)
    visualizer.run()
    
    envs.close()


if __name__ == '__main__':
    main()
