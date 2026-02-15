"""
Main training script for Sheep Herding High-Level Controller
"""

import argparse
import os
import torch
import numpy as np
from datetime import datetime

from envs import SheepFlockEnv, SheepFlockEnvWrapper
from onpolicy.algorithms.sheep_actor_critic import SheepActorCritic
from onpolicy.utils.shared_buffer import SharedReplayBuffer

class SimpleLogger:
    def __init__(self, save_folder, **kwargs):
        self.save_folder = save_folder
        import os
        os.makedirs(save_folder, exist_ok=True)
        self.log_file = open(os.path.join(save_folder, 'training_log.txt'), 'w')

    def log(self, metrics):
        msg = ' | '.join([f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}'
                         for k, v in metrics.items()])
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='sheep_herding')
    parser.add_argument('--scenario_name', type=str, default='default')

    parser.add_argument('--num_sheep', type=int, default=10)
    parser.add_argument('--num_herders', type=int, default=3)
    parser.add_argument('--episode_length', type=int, default=100)
    parser.add_argument('--world_size', type=float, nargs=2, default=[50.0, 50.0])

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_env_steps', type=int, default=1000000)
    parser.add_argument('--n_rollout_threads', type=int, default=1)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--ppo_epoch', type=int, default=15)
    parser.add_argument('--num_mini_batch', type=int, default=1)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--value_loss_coef', type=float, default=1)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_grad_norm', type=float, default=10.0)

    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--layer_N', type=int, default=1)
    parser.add_argument('--use_ReLU', action='store_true', default=True)
    parser.add_argument('--use_orthogonal', action='store_true', default=True)
    parser.add_argument('--use_feature_normalization', action='store_true', default=True)
    parser.add_argument('--use_popart', action='store_true', default=False)
    parser.add_argument('--use_valuenorm', action='store_true', default=False)
    parser.add_argument('--use_gae', action='store_true', default=True)
    parser.add_argument('--use_linear_lr_decay', action='store_true', default=True)
    parser.add_argument('--use_clipped_value_loss', action='store_true', default=True)

    parser.add_argument('--recurrent_N', type=int, default=1)
    parser.add_argument('--use_naive_recurrent_policy', action='store_true', default=False)
    parser.add_argument('--use_recurrent_policy', action='store_true', default=False)

    parser.add_argument('--gain', type=float, default=0.01)
    parser.add_argument('--use_policy_active_masks', action='store_true', default=True)
    parser.add_argument('--stacked_frames', type=int, default=1)

    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default='sheep_herding')

    parser.add_argument('--algorithm_name', type=str, default='mappo')
    parser.add_argument('--experiment_name', type=str, default='')
    parser.add_argument('--use_proper_time_limits', action='store_true', default=False)

    args = parser.parse_args()
    return args


def make_train_env(args):
    """Create training environment"""
    env = SheepFlockEnv(
        world_size=tuple(args.world_size),
        num_sheep=args.num_sheep,
        num_herders=args.num_herders,
        episode_length=args.episode_length,
        random_seed=args.seed,
    )
    return env


def make_eval_env(args):
    """Create evaluation environment"""
    env = SheepFlockEnv(
        world_size=tuple(args.world_size),
        num_sheep=args.num_sheep,
        num_herders=args.num_herders,
        episode_length=args.episode_length,
        random_seed=args.seed + 1000,
    )
    return env


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SheepTrainer:
    """Trainer class for sheep herding"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        set_seed(args.seed)

        self.env = make_train_env(args)
        self.num_agents = args.num_herders

        self.policy = SheepActorCritic(
            args,
            self.env.observation_space,
            self.env.share_observation_space,
            self.env.action_space,
            self.device,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=args.lr,
        )

        self.buffer = SharedReplayBuffer(
            args,
            self.num_agents,
            self.env.observation_space,
            self.env.share_observation_space,
            self.env.action_space,
        )

        from onpolicy.utils.valuenorm import ValueNorm
        self.value_normalizer = ValueNorm(1) if args.use_valuenorm else None

        run_dir = os.path.join(
            'results',
            args.env_name,
            args.scenario_name,
            args.algorithm_name,
            f'seed{args.seed}',
        )
        if args.experiment_name:
            run_dir = os.path.join(run_dir, args.experiment_name)

        self.run_dir = run_dir
        self.save_dir = os.path.join(run_dir, 'models')
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = SimpleLogger(run_dir)

        self.total_num_steps = 0
        self.episode = 0

    def warmup(self):
        """Warmup by collecting initial observations"""
        obs = self.env.reset()
        share_obs = self.env.get_shared_obs()

        n_rollout_threads = self.args.n_rollout_threads
        num_agents = self.num_agents

        share_obs_batch = np.tile(share_obs[np.newaxis, np.newaxis, :], 
                                   (n_rollout_threads, num_agents, 1))
        obs_batch = np.tile(obs[np.newaxis, np.newaxis, :], 
                            (n_rollout_threads, num_agents, 1))

        self.buffer.share_obs[0] = share_obs_batch.copy()
        self.buffer.obs[0] = obs_batch.copy()

    def collect(self, step):
        """Collect actions from policy"""
        self.policy.eval()

        n_rollout_threads = self.args.n_rollout_threads
        num_agents = self.num_agents

        with torch.no_grad():
            value, action, action_log_prob, rnn_states, rnn_states_critic = \
                self.policy.get_actions(
                    self.buffer.share_obs[step].reshape(-1, *self.buffer.share_obs.shape[3:]),
                    self.buffer.obs[step].reshape(-1, *self.buffer.obs.shape[3:]),
                    self.buffer.rnn_states[step].reshape(-1, *self.buffer.rnn_states.shape[3:]),
                    self.buffer.rnn_states_critic[step].reshape(-1, *self.buffer.rnn_states_critic.shape[3:]),
                    self.buffer.masks[step].reshape(-1, *self.buffer.masks.shape[3:]),
                    deterministic=False,
                )

        values = value.cpu().numpy().reshape(n_rollout_threads, num_agents, 1)
        actions = action.cpu().numpy().reshape(n_rollout_threads, num_agents, -1)
        action_log_probs = action_log_prob.cpu().numpy().reshape(n_rollout_threads, num_agents, 1)
        rnn_states_out = rnn_states.cpu().numpy().reshape(n_rollout_threads, num_agents, *rnn_states.shape[1:])
        rnn_states_critic_out = rnn_states_critic.cpu().numpy().reshape(n_rollout_threads, num_agents, *rnn_states_critic.shape[1:])

        return values, actions, action_log_probs, rnn_states_out, rnn_states_critic_out

    def insert(self, data):
        """Insert data into buffer"""
        obs, rewards, dones, infos, values, actions, action_log_probs, \
            rnn_states, rnn_states_critic = data

        n_rollout_threads = self.args.n_rollout_threads
        num_agents = self.num_agents

        if isinstance(dones, bool):
            dones_array = np.array([[dones] * num_agents] * n_rollout_threads, dtype=bool)
        elif isinstance(dones, np.ndarray):
            if dones.ndim == 0:
                dones_array = np.array([[bool(dones)] * num_agents] * n_rollout_threads, dtype=bool)
            else:
                dones_array = np.tile(dones.reshape(1, -1), (n_rollout_threads, 1))
        else:
            dones_array = np.array([[bool(dones)] * num_agents] * n_rollout_threads, dtype=bool)
        
        masks = np.ones((n_rollout_threads, num_agents, 1), dtype=np.float32)
        masks[dones_array] = 0.0

        rnn_states[dones_array] = np.zeros(
            (dones_array.sum(), self.args.recurrent_N, self.args.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones_array] = np.zeros(
            (dones_array.sum(), self.args.recurrent_N, self.args.hidden_size),
            dtype=np.float32,
        )

        share_obs_raw = self.env.get_shared_obs()
        share_obs = np.tile(share_obs_raw[np.newaxis, np.newaxis, :], (n_rollout_threads, num_agents, 1))
        obs_batch = np.tile(obs[np.newaxis, np.newaxis, :], (n_rollout_threads, num_agents, 1))
        
        if isinstance(rewards, (int, float)):
            rewards_batch = np.array([[rewards] * num_agents] * n_rollout_threads).reshape(n_rollout_threads, num_agents, 1)
        else:
            rewards_batch = np.array([[rewards] * num_agents] * n_rollout_threads).reshape(n_rollout_threads, num_agents, 1)

        self.buffer.insert(
            share_obs,
            obs_batch,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards_batch,
            masks,
        )

    def compute_returns(self):
        """Compute returns for training"""
        self.policy.eval()

        with torch.no_grad():
            next_values = self.policy.get_values(
                self.buffer.share_obs[-1].reshape(-1, *self.buffer.share_obs.shape[3:]),
                self.buffer.rnn_states_critic[-1].reshape(-1, *self.buffer.rnn_states_critic.shape[3:]),
                self.buffer.masks[-1].reshape(-1, *self.buffer.masks.shape[3:]),
            )
            next_values = next_values.cpu().numpy().reshape(self.args.n_rollout_threads, self.num_agents, 1)

        self.buffer.compute_returns(next_values, self.value_normalizer)

    def train(self):
        """Train the policy"""
        self.policy.train()

        advantages = self.buffer.returns[:-1] - self.buffer.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        total_loss = 0

        for _ in range(self.args.ppo_epoch):
            data_generator = self.buffer.feed_forward_generator(
                advantages,
                self.args.num_mini_batch,
            )

            for sample in data_generator:
                share_obs_batch, obs_batch, rnn_states_batch, \
                    rnn_states_critic_batch, actions_batch, \
                    value_preds_batch, return_batch, masks_batch, \
                    active_masks_batch, old_action_log_probs_batch, \
                    adv_targ, available_actions_batch = sample

                values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                    share_obs_batch,
                    obs_batch,
                    rnn_states_batch,
                    rnn_states_critic_batch,
                    actions_batch,
                    masks_batch,
                    available_actions_batch,
                    active_masks_batch,
                )

                old_action_log_probs_batch = torch.from_numpy(old_action_log_probs_batch).to(self.device)
                adv_targ = torch.from_numpy(adv_targ).to(self.device)
                value_preds_batch = torch.from_numpy(value_preds_batch).to(self.device)
                return_batch = torch.from_numpy(return_batch).to(self.device)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1 - self.args.clip_param,
                                   1 + self.args.clip_param) * adv_targ

                action_loss = -torch.min(surr1, surr2).mean()

                if self.args.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(
                            -self.args.clip_param, self.args.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                loss = value_loss * self.args.value_loss_coef + \
                       action_loss - \
                       dist_entropy * self.args.entropy_coef

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / self.args.ppo_epoch

    def save(self):
        """Save model checkpoint"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_num_steps': self.total_num_steps,
            'episode': self.episode,
        }
        torch.save(checkpoint, os.path.join(
            self.save_dir, f'model_{self.total_num_steps}.pt'))

    def run(self):
        """Main training loop"""
        self.warmup()

        episodes = int(self.args.num_env_steps) // self.args.episode_length

        for episode in range(episodes):
            if self.args.use_linear_lr_decay:
                self.adjust_lr(episode, episodes)

            obs = self.env.reset()
            
            rnn_states = np.zeros((self.args.n_rollout_threads, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((self.args.n_rollout_threads, self.num_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
            masks = np.ones((self.args.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.args.episode_length):
                values, actions, action_log_probs, new_rnn_states, new_rnn_states_critic = \
                    self.collect(step)

                actions_env = actions[0]
                obs, rewards, dones, infos = self.env.step(actions_env)

                data = (
                    obs, rewards, dones, infos, values, actions,
                    action_log_probs, new_rnn_states, new_rnn_states_critic
                )
                self.insert(data)

                self.total_num_steps += 1

            self.compute_returns()
            train_loss = self.train()

            self.episode = episode

            if episode % self.args.log_interval == 0:
                avg_reward = np.mean(self.buffer.rewards) * self.args.episode_length
                print(f"Episode {episode}/{episodes}, "
                      f"Steps: {self.total_num_steps}, "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Loss: {train_loss:.4f}")

                self.logger.log({
                    'episode': episode,
                    'total_steps': self.total_num_steps,
                    'avg_reward': avg_reward,
                    'train_loss': train_loss,
                })

            if episode % self.args.save_interval == 0:
                self.save()

        self.save()
        print("Training completed!")

    def adjust_lr(self, episode, episodes):
        """Linear learning rate decay"""
        lr = self.args.lr * (1 - episode / episodes)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def main():
    args = parse_args()
    print(f"Training with args: {args}")

    trainer = SheepTrainer(args)
    trainer.run()


if __name__ == '__main__':
    main()