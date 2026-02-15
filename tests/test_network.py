"""
Network structure tests for sheep herding high-level controller
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from gym import spaces


class MockArgs:
    """Mock arguments for network initialization"""
    def __init__(self):
        self.hidden_size = 64
        self.gain = 0.01
        self.use_orthogonal = True
        self.use_policy_active_masks = True
        self.use_naive_recurrent_policy = False
        self.use_recurrent_policy = False
        self.recurrent_N = 1
        self.use_popart = False
        self.use_feature_normalization = True
        self.use_ReLU = True
        self.stacked_frames = 1
        self.layer_N = 1


def test_bounded_act_layer():
    """Test bounded action layer"""
    print("Testing BoundedACTLayer...")

    from onpolicy.algorithms.utils.bounded_act import BoundedACTLayer

    action_space = spaces.Box(
        low=np.array([0.0, 0.0, -np.pi, 0.0]),
        high=np.array([20.0, 5.0, np.pi, 1.0]),
        dtype=np.float32
    )

    act_layer = BoundedACTLayer(action_space, inputs_dim=64)

    x = torch.randn(4, 64)
    actions, log_probs = act_layer(x, deterministic=False)

    assert actions.shape == (4, 4), f"Expected shape (4, 4), got {actions.shape}"

    assert torch.all(actions[:, 0] >= 0.0) and torch.all(actions[:, 0] <= 20.0), \
        "radius_mean out of bounds"
    assert torch.all(actions[:, 1] >= 0.0) and torch.all(actions[:, 1] <= 5.0), \
        "radius_std out of bounds"
    assert torch.all(actions[:, 2] >= -np.pi) and torch.all(actions[:, 2] <= np.pi), \
        "angle_mean out of bounds"
    assert torch.all(actions[:, 3] >= 0.0) and torch.all(actions[:, 3] <= 1.0), \
        "concentration out of bounds"

    print("BoundedACTLayer test passed!")


def test_sheep_actor():
    """Test sheep actor network"""
    print("Testing SheepActor...")

    from onpolicy.algorithms.sheep_actor_critic import SheepActor

    args = MockArgs()
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.array([0.0, 0.0, -np.pi, 0.0]),
        high=np.array([20.0, 5.0, np.pi, 1.0]),
        dtype=np.float32
    )

    actor = SheepActor(args, obs_space, action_space)

    obs = torch.randn(8, 10)
    rnn_states = torch.zeros(8, 1, 64)
    masks = torch.ones(8, 1)

    actions, log_probs, new_rnn_states = actor(obs, rnn_states, masks, deterministic=True)

    assert actions.shape == (8, 4), f"Expected shape (8, 4), got {actions.shape}"
    assert log_probs.shape[0] == 8, f"Expected batch size 8, got {log_probs.shape[0]}"

    print("SheepActor test passed!")


def test_sheep_critic():
    """Test sheep critic network"""
    print("Testing SheepCritic...")

    from onpolicy.algorithms.sheep_actor_critic import SheepCritic

    args = MockArgs()
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    critic = SheepCritic(args, obs_space)

    obs = torch.randn(8, 10)
    rnn_states = torch.zeros(8, 1, 64)
    masks = torch.ones(8, 1)

    values, new_rnn_states = critic(obs, rnn_states, masks)

    assert values.shape == (8, 1), f"Expected shape (8, 1), got {values.shape}"

    print("SheepCritic test passed!")


def test_sheep_actor_critic():
    """Test combined actor-critic"""
    print("Testing SheepActorCritic...")

    from onpolicy.algorithms.sheep_actor_critic import SheepActorCritic

    args = MockArgs()
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    cent_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.array([0.0, 0.0, -np.pi, 0.0]),
        high=np.array([20.0, 5.0, np.pi, 1.0]),
        dtype=np.float32
    )

    model = SheepActorCritic(args, obs_space, cent_obs_space, action_space)

    obs = torch.randn(8, 10)
    cent_obs = torch.randn(8, 30)
    rnn_states_actor = torch.zeros(8, 1, 64)
    rnn_states_critic = torch.zeros(8, 1, 64)
    masks = torch.ones(8, 1)

    values, actions, log_probs, new_rnn_actor, new_rnn_critic = model.get_actions(
        cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, deterministic=True
    )

    assert values.shape == (8, 1)
    assert actions.shape == (8, 4)

    eval_values, eval_log_probs, entropy = model.evaluate_actions(
        cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks
    )

    assert eval_values.shape == (8, 1)
    assert eval_log_probs.shape[0] == 8

    print("SheepActorCritic test passed!")


def test_action_bounds_consistency():
    """Test that actions are always within bounds"""
    print("Testing action bounds consistency...")

    from onpolicy.algorithms.sheep_actor_critic import SheepActorCritic

    args = MockArgs()
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
    cent_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(30,), dtype=np.float32)
    action_space = spaces.Box(
        low=np.array([0.0, 0.0, -np.pi, 0.0]),
        high=np.array([20.0, 5.0, np.pi, 1.0]),
        dtype=np.float32
    )

    model = SheepActorCritic(args, obs_space, cent_obs_space, action_space)

    for _ in range(10):
        obs = torch.randn(16, 10)
        cent_obs = torch.randn(16, 30)
        rnn_states_actor = torch.zeros(16, 1, 64)
        rnn_states_critic = torch.zeros(16, 1, 64)
        masks = torch.ones(16, 1)

        _, actions, _, _, _ = model.get_actions(
            cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, deterministic=False
        )

        assert torch.all(actions[:, 0] >= 0.0) and torch.all(actions[:, 0] <= 20.0)
        assert torch.all(actions[:, 1] >= 0.0) and torch.all(actions[:, 1] <= 5.0)
        assert torch.all(actions[:, 2] >= -np.pi) and torch.all(actions[:, 2] <= np.pi)
        assert torch.all(actions[:, 3] >= 0.0) and torch.all(actions[:, 3] <= 1.0)

    print("Action bounds consistency test passed!")


if __name__ == "__main__":
    test_bounded_act_layer()
    test_sheep_actor()
    test_sheep_critic()
    test_sheep_actor_critic()
    test_action_bounds_consistency()

    print("\n" + "="*50)
    print("All network tests passed!")
    print("="*50)