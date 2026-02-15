#!/bin/bash

# Training script for Sheep Herding High-Level Controller
# Usage: ./train_sheep.sh [options]

# Default parameters
env="sheep_herding"
scenario="default"
num_sheep=10
num_herders=3
episode_length=100
seed=1

# Training parameters
lr=0.0005
ppo_epoch=15
num_mini_batch=1
clip_param=0.2
value_loss_coef=1
entropy_coef=0.01
gamma=0.99
gae_lambda=0.95

# Network parameters
hidden_size=64
layer_N=1

# Training settings
num_env_steps=1000000
n_rollout_threads=4
save_interval=100
log_interval=10
use_wandb=False

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_sheep)
            num_sheep="$2"
            shift 2
            ;;
        --num_herders)
            num_herders="$2"
            shift 2
            ;;
        --episode_length)
            episode_length="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --num_env_steps)
            num_env_steps="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Training Sheep Herding High-Level Controller"
echo "=========================================="
echo "Environment: $env"
echo "Num sheep: $num_sheep"
echo "Num herders: $num_herders"
echo "Episode length: $episode_length"
echo "Seed: $seed"
echo "Learning rate: $lr"
echo "Total steps: $num_env_steps"
echo "=========================================="

# Activate conda environment and run training
cd /home/hmy524/github_project/high_layer

/home/hmy524/miniconda3/envs/informarl38/bin/python3 train.py \
    --env_name $env \
    --scenario_name $scenario \
    --num_sheep $num_sheep \
    --num_herders $num_herders \
    --episode_length $episode_length \
    --seed $seed \
    --lr $lr \
    --ppo_epoch $ppo_epoch \
    --num_mini_batch $num_mini_batch \
    --clip_param $clip_param \
    --value_loss_coef $value_loss_coef \
    --entropy_coef $entropy_coef \
    --gamma $gamma \
    --gae_lambda $gae_lambda \
    --hidden_size $hidden_size \
    --layer_N $layer_N \
    --num_env_steps $num_env_steps \
    --n_rollout_threads $n_rollout_threads \
    --save_interval $save_interval \
    --log_interval $log_interval \
    --use_wandb $use_wandb \
    --algorithm_name mappo \
    --use_ReLU