#!/bin/sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# visualize.py 风格：
# - 无 DISPLAY 时 eval_mpe.py 会自动 headless 并默认保存 GIF
# - SHOW_RENDER_DELAY_MS 用毫秒控制帧间隔
# - SHOW_SAVE_GIF 指定 GIF 输出路径

env="GraphMPE"
scenario="navigation_graph"
algo="rmappo"
exp="informarl"
seed=0
num_agents=6
num_obstacles=1
episode_length=100
render_episodes=5
world_size=100

MODEL_DIR="${1:-../results/sheep_herding/${scenario}/${algo}/seed${seed}/20260415_161642/models}"

EXTRA_ARGS=""
if [ -n "${SHOW_RENDER_DELAY_MS:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --render_delay_ms ${SHOW_RENDER_DELAY_MS}"
fi
if [ -n "${SHOW_SAVE_GIF:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --gif_path ${SHOW_SAVE_GIF}"
fi

CUDA_VISIBLE_DEVICES=0 python -u onpolicy/scripts/eval_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_obstacles ${num_obstacles} \
    --world_size ${world_size} \
    --collision_rew 5 \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --episode_length ${episode_length} \
    --use_render \
    --render_episodes ${render_episodes} \
    --model_dir "${MODEL_DIR}" \
    --user_name "marl" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --mpl_render \
    --use_shepherd_env \
    ${EXTRA_ARGS}

# 可选：在上一行末尾追加参数，例如
#   --save_gifs
#   --use_dones True

