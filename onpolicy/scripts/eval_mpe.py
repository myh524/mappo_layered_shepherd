#!/usr/bin/env python
import argparse
from distutils.util import strtobool
import sys
import os
import shutil
import time
from typing import Dict
import wandb
import numpy as np
from pathlib import Path
import torch

import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from utils.utils import print_args, print_box
from onpolicy.config import get_config, graph_config
from multiagent.MPE_env import MPEEnv, GraphMPEEnv
from onpolicy.envs.env_wrappers import (
    SubprocVecEnv,
    DummyVecEnv,
    GraphSubprocVecEnv,
    GraphDummyVecEnv,
)


def make_render_env(all_args: argparse.Namespace):
    def get_env_fn(rank: int):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            elif all_args.env_name == "GraphMPE":
                env = GraphMPEEnv(all_args)
            else:
                print(f"Can not support the {all_args.env_name} environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        if all_args.env_name == "GraphMPE":
            return GraphDummyVecEnv([get_env_fn(0)])
        return DummyVecEnv([get_env_fn(0)])
    else:
        if all_args.env_name == "GraphMPE":
            return GraphSubprocVecEnv(
                [get_env_fn(i) for i in range(all_args.n_rollout_threads)]
            )
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="simple_spread",
        help="Which scenario to run on",
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")
    parser.add_argument(
        "--num_obstacles", type=int, default=3, help="Number of obstacles"
    )
    parser.add_argument(
        "--collaborative",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Number of agents in the env",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=2,
        help="Max speed for agents. NOTE that if this is None, "
        "then max_speed is 2 with discrete action space",
    )
    parser.add_argument(
        "--collision_rew",
        type=float,
        default=5,
        help="The reward to be negated for collisions with other "
        "agents and obstacles",
    )
    parser.add_argument(
        "--goal_rew",
        type=float,
        default=5,
        help="The reward to be added if agent reaches the goal",
    )
    parser.add_argument(
        "--min_dist_thresh",
        type=float,
        default=0.1,
        help="The minimum distance threshold to classify whether "
        "agent has reached the goal or not",
    )
    parser.add_argument(
        "--use_dones",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether we want to use the 'done=True' "
        "when agent has reached the goal or just return False like "
        "the `simple.py` or `simple_spread.py`",
    )

    all_args = parser.parse_known_args(args)[0]

    return all_args


def modify_args(
    model_dir: str,
    args: argparse.Namespace,
    exclude_args: list = [
        "model_dir",
        "num_agents",
        "num_obstacles",
        "num_landmarks",
        "render_episodes",
        "world_size",
        "seed",
    ],
):
    """
    Modify the args used to train the model
    """
    import yaml

    with open(str(model_dir) + "/config.yaml") as f:
        ydict = yaml.load(f)

    print("_" * 50)
    for k, v in ydict.items():
        if k in exclude_args:
            print(f"Using {k} = {vars(args)[k]}")
            # print(f"Skipping {k} with value {args.k}")
            continue
        # all args have 'values' and 'desc' as keys
        if type(v) == dict:
            if "value" in v.keys():
                # print(f'Setting attr {k} to {ydict[k]["value"]}')
                setattr(args, k, ydict[k]["value"])
    print("_" * 50)

    # set some args manually
    args.cuda = False
    args.use_wandb = False
    args.use_render = True
    args.save_gifs = True
    args.n_rollout_threads = 1

    return args


def main(args):
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)  # ✅ 解析 --scenario_name 等自定义参数
    all_args, parser = graph_config(sys.argv[1:], parser)  # ✅ 补充 GNN 参数
    # all_args = parse_args(args, parser)
    # all_args = modify_args(all_args.model_dir, all_args)

    # -----------------------------
    # visualize.py-style rendering UX
    # - auto headless detection (no DISPLAY/WAYLAND_DISPLAY)
    # - allow ms-based delay (--render_delay_ms) instead of seconds (--ifi)
    # - allow explicit GIF output path (--gif_path)
    # - avoid requiring wandb run dir during evaluation rendering
    # -----------------------------
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    headless = bool(getattr(all_args, "force_headless", False) or not has_display)

    if getattr(all_args, "render_delay_ms", None) is not None:
        try:
            all_args.ifi = float(all_args.render_delay_ms) / 1000.0
        except Exception:
            pass

    if getattr(all_args, "gif_path", None):
        all_args.save_gifs = True

    if headless:
        # In headless mode, saving gifs is the safest default.
        all_args.save_gifs = True

    # Make eval rendering self-contained (no wandb directory assumptions)
    all_args.use_wandb = False

    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "rmappg":
        assert (
            all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy
        ), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mappg":
        assert (
            all_args.use_recurrent_policy and all_args.use_naive_recurrent_policy
        ) == False, "check recurrent policy!"
    else:
        raise NotImplementedError

    assert all_args.use_render, "Need to set use_render be True"
    assert not (
        all_args.model_dir == None or all_args.model_dir == ""
    ), "set model_dir first"
    assert all_args.n_rollout_threads == 1, "only support to use 1 env to render."

    device = torch.device("cpu")

    # run dir
    # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    # if not run_dir.exists():
    #     os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents
    # When use_wandb=False, runners expect a concrete run_dir.
    # model_dir points to .../runX/models, so run_dir is .../runX
    run_dir = Path(all_args.model_dir).parent

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        if all_args.env_name == "GraphMPE":
            from onpolicy.runner.shared.graph_mpe_runner import GMPERunner as Runner
        else:
            from onpolicy.runner.shared.mpe_runner import MPERunner as Runner
    else:
        if all_args.env_name == "GraphMPE":
            raise NotImplementedError
        from onpolicy.runner.separated.mpe_runner import MPERunner as Runner

    # print_args(config['all_args'])

    runner = Runner(config)
    # actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    # runner.policy.actor.load_state_dict(actor_state_dict)
    def _render_mpl_graph_mpe(runner, envs, all_args):
        """Matplotlib renderer that bypasses multiagent/rendering.py (pyglet/OpenGL)."""
        import matplotlib

        if headless or all_args.save_gifs:
            matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        interactive = plt.get_backend().lower() != "agg"

        # unwrap base env (GraphDummyVecEnv -> MultiAgentGraphEnv)
        base_env = None
        if hasattr(envs, "envs") and len(envs.envs) > 0:
            base_env = envs.envs[0]
        if base_env is None:
            raise RuntimeError("mpl_render requires a non-subprocess env (n_rollout_threads=1).")

        # Match repo-root `visualize.py` (setup_figure): larger interactive window.
        fig, ax = plt.subplots(figsize=(10, 10))
        frames = []

        def _draw_world():
            ax.clear()
            world = base_env.world
            half = float(getattr(world, "world_size", 2.0)) / 2.0
            pad = max(half * 0.06, 0.2)
            ax.set_xlim(-half - pad, half + pad)
            ax.set_ylim(-half - pad, half + pad)
            ax.set_aspect("equal")
            ax.set_facecolor("#f5f5f5")
            ax.add_patch(
                patches.Rectangle(
                    (-half, -half),
                    2 * half,
                    2 * half,
                    fill=False,
                    edgecolor="0.4",
                    linewidth=1.2,
                )
            )

            # Entities: data-coordinate patches; slightly smaller than physics `size`
            # so the frame reads clearly (purely visual).
            _mpl_icon_scale = 0.35
            _mpl_icon_floor = 0.0022  # min radius as fraction of half-width

            def _entity_patches(entities, default_color, label, n_sides: int = 0):
                if not entities:
                    return
                for i, e in enumerate(entities):
                    p = np.asarray(e.state.p_pos, dtype=float)
                    c = getattr(e, "color", None)
                    if c is None:
                        c = np.asarray(default_color, dtype=float)
                    else:
                        c = np.asarray(c, dtype=float)
                    r = float(getattr(e, "size", 0.05)) * _mpl_icon_scale
                    r = max(r, half * _mpl_icon_floor)
                    leg = label if i == 0 else "_nolegend_"
                    if n_sides == 4:
                        poly = patches.RegularPolygon(
                            (float(p[0]), float(p[1])),
                            numVertices=4,
                            radius=r * np.sqrt(2.0),
                            orientation=np.pi / 4.0,
                            facecolor=c,
                            edgecolor="0.15",
                            linewidth=0.7,
                            alpha=0.9,
                            label=leg,
                        )
                        ax.add_patch(poly)
                    else:
                        circ = patches.Circle(
                            (float(p[0]), float(p[1])),
                            radius=r,
                            facecolor=c,
                            edgecolor="0.15",
                            linewidth=0.7,
                            alpha=0.9,
                            label=leg,
                        )
                        ax.add_patch(circ)

            _entity_patches(
                getattr(world, "agents", []),
                default_color=np.array([0.2, 0.4, 0.9]),
                label="agents",
                n_sides=4,
            )
            _entity_patches(
                getattr(world, "obstacles", []),
                default_color=np.array([0.85, 0.2, 0.2]),
                label="obstacles",
                n_sides=0,
            )
            _entity_patches(
                getattr(world, "landmarks", []),
                default_color=np.array([0.2, 0.8, 0.2]),
                label="landmarks",
                n_sides=0,
            )
            _entity_patches(
                getattr(world, "flock_entities", []),
                default_color=np.array([0.6, 0.6, 0.6]),
                label="flock",
                n_sides=0,
            )
            _entity_patches(
                getattr(world, "scripted_agents_goals", []),
                default_color=np.array([0.1, 0.7, 0.1]),
                label="goals",
                n_sides=0,
            )

            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)

        obs, agent_id, node_obs, adj = envs.reset()
        rnn_states = np.zeros(
            (all_args.n_rollout_threads, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((all_args.n_rollout_threads, all_args.num_agents, 1), dtype=np.float32)

        render_delay_s = (float(all_args.render_delay_ms) / 1000.0) if getattr(all_args, "render_delay_ms", None) is not None else float(all_args.ifi)

        for ep in range(int(all_args.render_episodes)):
            obs, agent_id, node_obs, adj = envs.reset()
            rnn_states[...] = 0.0
            masks[...] = 1.0

            for step in range(int(all_args.episode_length)):
                runner.trainer.prep_rollout()
                action, rnn_states_out = runner.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(node_obs),
                    np.concatenate(adj),
                    np.concatenate(agent_id),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(action.detach().cpu().numpy(), all_args.n_rollout_threads))
                rnn_states = np.array(np.split(rnn_states_out.detach().cpu().numpy(), all_args.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                obs, agent_id, node_obs, adj, rewards, dones, infos = envs.step(actions_env)

                masks = np.ones((all_args.n_rollout_threads, all_args.num_agents, 1), dtype=np.float32)
                masks[dones == True] = 0.0

                _draw_world()
                ax.set_title(f"Episode {ep+1}/{all_args.render_episodes} | Step {step+1}/{all_args.episode_length}", fontsize=11)

                if all_args.save_gifs:
                    fig.canvas.draw()
                    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
                    frames.append(frame)
                elif interactive:
                    plt.show(block=False)
                    plt.pause(max(0.001, render_delay_s))
                else:
                    time.sleep(max(0.0, render_delay_s))

        if all_args.save_gifs and frames:
            import imageio

            out_dir = Path(run_dir) / "gifs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "render.gif"
            imageio.mimsave(str(out_path), frames, duration=float(all_args.ifi))

        plt.close(fig)

    if getattr(all_args, "mpl_render", False):
        _render_mpl_graph_mpe(runner, envs, all_args)
    else:
        runner.render()

    # If user requested a specific GIF path, move/copy it there.
    gif_path = getattr(all_args, "gif_path", None)
    if gif_path:
        src = Path(run_dir) / "gifs" / "render.gif"
        dst = Path(gif_path)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copyfile(src, dst)
                print(f"Saved GIF to: {dst}")
            else:
                print(f"Warning: expected GIF not found at {src}")
        except Exception as e:
            print(f"Warning: failed to write gif_path={dst}: {e}")

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
