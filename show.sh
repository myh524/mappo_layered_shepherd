#!/bin/sh

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

env="GraphMPE"
scenario="navigation_graph"
algo="rmappo"
exp="informarl"
seed=0
num_agents=6
num_obstacles=1
episode_length=100

CUDA_VISIBLE_DEVICES=0 python -u onpolicy/scripts/eval_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_obstacles ${num_obstacles} \
    --collision_rew 5 \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --episode_length ${episode_length} \
    --use_render \
    --render_episodes 5 \
    --model_dir onpolicy/results/${env}/${scenario}/${algo}/${exp}/run2/models \
    --user_name "marl" \
    --use_cent_obs "False" \
    --graph_feat_type "relative" \
    --use_shepherd_env \
    --save_gifs \
    # --use_dones "True" \

