import time
import wandb
import os
import json
import shutil
import numpy as np
from itertools import chain
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional

from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


def _tb_scalar(v):
    if torch.is_tensor(v):
        return v.detach().float().cpu().item()
    return float(v)


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]

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

        if self.use_render:
            import imageio

            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / "gifs")
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
            self.save_dir = str(self.run_dir / "models")
            os.makedirs(self.save_dir, exist_ok=True)
            self.checkpoint_dir = str(self.run_dir / "models" / "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.checkpoint_dir = str(Path(self.save_dir) / "checkpoints")
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            else:
                self.run_dir = config["run_dir"]
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

        from onpolicy.algorithms.mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.MAPPOPolicy import R_MAPPOPolicy as Policy

        # from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        # from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            # policy network
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            # buffer
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
            )
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(
                next_value, self.trainer[agent_id].value_normalizer
            )

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(
        self,
        episode: int = 0,
        total_num_steps: Optional[int] = None,
        save_checkpoint: bool = False,
    ):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(
                policy_critic.state_dict(),
                str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt",
            )

            if save_checkpoint and total_num_steps is not None:
                ckpt = {
                    "total_num_steps": int(total_num_steps),
                    "episode": int(episode),
                    "agent_id": int(agent_id),
                    "actor_state_dict": policy_actor.state_dict(),
                    "critic_state_dict": policy_critic.state_dict(),
                }
                agent_ckpt_dir = Path(self.checkpoint_dir) / f"agent{agent_id}"
                agent_ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = str(agent_ckpt_dir / f"model_{int(total_num_steps)}.pt")
                torch.save(ckpt, ckpt_path)
                latest_path = str(agent_ckpt_dir / "model_latest.pt")
                shutil.copyfile(ckpt_path, latest_path)

                meta_path = str(agent_ckpt_dir / "checkpoint_meta.jsonl")
                with open(meta_path, "a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "total_num_steps": int(total_num_steps),
                                "episode": int(episode),
                                "agent_id": int(agent_id),
                                "file": os.path.basename(ckpt_path),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    def restore(self):
        _map = self.device
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + "/actor_agent" + str(agent_id) + ".pt",
                map_location=_map,
            )
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + "/critic_agent" + str(agent_id) + ".pt",
                map_location=_map,
            )
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    tag = ("train/agent%i/" % agent_id) + str(k).replace("/", ".")
                    self.writter.add_scalar(tag, _tb_scalar(v), total_num_steps)
        if not self.use_wandb:
            try:
                self.writter.flush()
            except Exception:
                pass

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
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
