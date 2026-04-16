import wandb
import os
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def _tb_scalar(v):
    """Scalar for TensorBoard / float(); detach tensors to avoid autograd warnings."""
    if torch.is_tensor(v):
        return v.detach().float().cpu().item()
    return float(v)

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
            self.gif_dir = Path(self.save_dir) / "gifs"
            self.checkpoint_dir = str(Path(self.save_dir) / "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        elif self.use_render:
            # Rendering-only runs still need stable dirs for optional saves.
            self.run_dir = Path(config["run_dir"])
            self.save_dir = str(self.run_dir / "models")
            os.makedirs(self.save_dir, exist_ok=True)
            self.checkpoint_dir = str(self.run_dir / "models" / "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.gif_dir = self.run_dir / "gifs"
            self.gif_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = Path(config["run_dir"])
            self.log_dir = str(self.run_dir / "logs")
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / "models")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.checkpoint_dir = str(self.run_dir / "models" / "checkpoints")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            self.gif_dir = self.run_dir / "gifs"
            self.gif_dir.mkdir(parents=True, exist_ok=True)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from onpolicy.algorithms.mat.algorithm.transformer_policy import (
                TransformerPolicy as Policy,
            )
        else:
            # This repo variant keeps MAPPO/RMAPPO trainers in flat modules,
            # not under onpolicy.algorithms.r_mappo.*.
            if self.env_name == "GraphMPE":
                from onpolicy.algorithms.graph_mappo import GR_MAPPO as TrainAlgo
                from onpolicy.algorithms.graph_MAPPOPolicy import GR_MAPPOPolicy as Policy
            else:
                from onpolicy.algorithms.mappo import R_MAPPO as TrainAlgo
                from onpolicy.algorithms.MAPPOPolicy import R_MAPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(
                self.all_args,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
                self.num_agents,
                device=self.device,
            )
        else:
            if self.env_name == "GraphMPE":
                # Graph policy needs node + edge observation spaces.
                self.policy = Policy(
                    self.all_args,
                    self.envs.observation_space[0],
                    share_observation_space,
                    self.envs.node_observation_space[0],
                    self.envs.edge_observation_space[0],
                    self.envs.action_space[0],
                    device=self.device,
                )
            else:
                self.policy = Policy(
                    self.all_args,
                    self.envs.observation_space[0],
                    share_observation_space,
                    self.envs.action_space[0],
                    device=self.device,
                )

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        else:
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)
        
        # buffer
        if self.env_name == "GraphMPE":
            # This repo variant uses GraphReplayBuffer (not GraphSharedReplayBuffer).
            from onpolicy.utils.graph_buffer import GraphReplayBuffer

            self.buffer = GraphReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.node_observation_space[0],
                self.envs.agent_id_observation_space[0],
                self.envs.share_agent_id_observation_space[0],
                self.envs.adj_observation_space[0],
                self.envs.action_space[0],
            )
        else:
            self.buffer = SharedReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                share_observation_space,
                self.envs.action_space[0],
            )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(
        self,
        episode=0,
        total_num_steps: Optional[int] = None,
        save_checkpoint: bool = False,
    ):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_dir, episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

            if save_checkpoint and total_num_steps is not None:
                ckpt = {
                    "total_num_steps": int(total_num_steps),
                    "episode": int(episode),
                    "actor_state_dict": policy_actor.state_dict(),
                    "critic_state_dict": policy_critic.state_dict(),
                }
                ckpt_path = str(Path(self.checkpoint_dir) / f"model_{int(total_num_steps)}.pt")
                torch.save(ckpt, ckpt_path)
                # Keep a stable pointer for downstream tooling.
                latest_path = str(Path(self.checkpoint_dir) / "model_latest.pt")
                shutil.copyfile(ckpt_path, latest_path)

                meta_path = str(Path(self.checkpoint_dir) / "checkpoint_meta.jsonl")
                with open(meta_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "total_num_steps": int(total_num_steps),
                                "episode": int(episode),
                                "file": os.path.basename(ckpt_path),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            # Checkpoints may be saved on CUDA; eval/render on CPU must remap storages.
            _map = self.device
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + "/actor.pt", map_location=_map
            )
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(
                    str(self.model_dir) + "/critic.pt", map_location=_map
                )
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                # Use flat scalar tags. `add_scalars` creates a deep directory tree
                # (one folder per tag) which makes TensorBoard navigation painful.
                tag = "train/" + str(k).replace("/", ".")
                self.writter.add_scalar(tag, _tb_scalar(v), total_num_steps)
        if not self.use_wandb:
            # Make metrics show up promptly in TensorBoard.
            try:
                self.writter.flush()
            except Exception:
                pass

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    tag = "env/" + str(k).replace("/", ".")
                    self.writter.add_scalar(tag, float(np.mean(v)), total_num_steps)
        if not self.use_wandb:
            try:
                self.writter.flush()
            except Exception:
                pass

    def process_infos(self, infos):
        """Aggregate per-env, per-agent info dicts for logging (GraphMPE / MPE)."""
        env_infos = {}
        if infos is None:
            return env_infos
        rows = list(infos)

        def _is_number(x):
            try:
                return np.isscalar(x) and np.isfinite(x)
            except Exception:
                return False

        for aid in range(self.num_agents):
            by_key = {}
            for row in rows:
                try:
                    ag = row[aid]
                except (TypeError, IndexError, KeyError):
                    continue
                if not isinstance(ag, dict):
                    continue
                for k, v in ag.items():
                    if _is_number(v):
                        by_key.setdefault(k, []).append(float(v))
            for k, vals in by_key.items():
                if vals:
                    env_infos[f"agent{aid}/{k}"] = vals
        return env_infos

    def get_collisions(self, env_infos):
        if not env_infos:
            return 0.0
        ckeys = [k for k in env_infos if "collision" in k.lower()]
        if not ckeys:
            return 0.0
        return float(np.mean([np.mean(env_infos[k]) for k in ckeys]))

    def get_fraction_episodes(self, env_infos):
        """Return (fraction_all_agents_reached_goal, mean_fraction_reached)."""
        n = self.num_agents
        keys = ["agent%i/Time_req_to_goal" % i for i in range(n)]
        if any(k not in env_infos for k in keys):
            return 0.0, 0.0
        n_t = len(env_infos[keys[0]])
        frac = []
        for t in range(n_t):
            reached = [env_infos[k][t] >= 0.0 for k in keys]
            frac.append(float(all(reached)))
        mean_frac = float(np.mean(frac)) if frac else 0.0
        return mean_frac, mean_frac
