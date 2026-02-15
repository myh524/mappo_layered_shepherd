import sys
sys.path.insert(0, '/home/hmy524/github_project/high_layer')

import torch
import numpy as np
from envs import SheepFlockEnv
from onpolicy.algorithms.sheep_actor_critic import SheepActorCritic

class Args:
    hidden_size = 64
    gain = 0.01
    use_orthogonal = True
    use_policy_active_masks = True
    use_naive_recurrent_policy = False
    use_recurrent_policy = False
    recurrent_N = 1
    use_popart = False
    use_feature_normalization = True
    use_ReLU = True
    stacked_frames = 1
    layer_N = 1

def test_training():
    print("Testing training components...")
    
    env = SheepFlockEnv(num_sheep=5, num_herders=2, episode_length=20)
    args = Args()
    
    policy = SheepActorCritic(args, env.observation_space, env.action_space)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    
    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")
    
    for step in range(10):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        rnn_states = torch.zeros(1, 1, args.hidden_size)
        masks = torch.ones(1, 1)
        
        with torch.no_grad():
            value, action, log_prob, _, _ = policy.get_actions(
                obs_tensor, rnn_states, rnn_states, masks, deterministic=False
            )
        
        action_np = action.numpy().reshape(env.num_herders, -1)
        obs, reward, done, info = env.step(action_np)
        
        print(f"Step {step}: reward={reward:.4f}, done={done}")
        
        if done:
            break
    
    print("\nTraining components test passed!")
    return True

if __name__ == "__main__":
    test_training()