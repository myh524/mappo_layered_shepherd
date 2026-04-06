#!/bin/sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

env="GraphMPE"
scenario="navigation_graph"
algo="rmappo"
exp="informarl"
seed=1
num_agents=7
num_obstacles=1
episode_length=50
world_size=2.0

# 是否保存每个 episode 为 GIF
SAVE_EPISODE_GIF=true

# 是否保存每个 step 为 PNG 图片
SAVE_STEP_PNG=true

CUDA_VISIBLE_DEVICES=0 python -u shepherd_visualizer.py \
    --model_dir onpolicy/results/${env}/${scenario}/${algo}/${exp}/run2/models \
    --num_episodes 1 \
    --render_delay 30 \
    --num_agents ${num_agents} \
    --num_obstacles ${num_obstacles} \
    --world_size ${world_size} \
    --episode_length ${episode_length} \
    --seed ${seed} \
    $(if [ "$SAVE_EPISODE_GIF" = true ]; then echo "--save_episode_gif"; fi) \
    $(if [ "$SAVE_STEP_PNG" = true ]; then echo "--save_step_png"; fi)
    
