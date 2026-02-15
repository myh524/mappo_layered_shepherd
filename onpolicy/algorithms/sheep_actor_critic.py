"""
Sheep Actor-Critic Networks for High-Level Decision Controller
Specialized networks for outputting formation parameters
"""

import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, Optional
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.bounded_act import BoundedACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space


class SheepActor(nn.Module):
    """
    Actor network for sheep herding high-level controller.
    
    Outputs formation parameters:
    - radius_mean: Standing radius around flock center [0, 20]
    - radius_std: Radius variation [0, 5]
    - angle_mean: Gathering angle [-pi, pi]
    - concentration: How concentrated the formation is [0, 1]
    """
    
    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu")
    ) -> None:
        super(SheepActor, self).__init__()
        self.hidden_size = args.hidden_size
        
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        obs_shape = get_shape_from_obs_space(obs_space)
        self.base = MLPBase(args, obs_shape)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )
        
        self.act = BoundedACTLayer(
            action_space, self.hidden_size, self._use_orthogonal, self._gain
        )
        
        self.to(device)
    
    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute actions from observations.
        
        Args:
            obs: Observation inputs
            rnn_states: Hidden states for RNN (if used)
            masks: Mask for hidden state initialization
            available_actions: Not used for continuous actions
            deterministic: Whether to use deterministic actions
        
        Returns:
            actions: Formation parameters
            action_log_probs: Log probabilities of actions
            rnn_states: Updated RNN hidden states
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        
        return actions, action_log_probs, rnn_states
    
    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None
    ) -> Tuple[Tensor, Tensor]:
        """
        Evaluate actions for training.
        
        Returns:
            action_log_probs: Log probabilities
            dist_entropy: Distribution entropy
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        
        actor_features = self.base(obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None,
        )
        
        return action_log_probs, dist_entropy


class SheepCritic(nn.Module):
    """
    Critic network for sheep herding high-level controller.
    
    Takes centralized observation (shared observation) and outputs value estimate.
    """
    
    def __init__(
        self,
        args,
        cent_obs_space,
        device=torch.device("cpu")
    ) -> None:
        super(SheepCritic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][
            self._use_orthogonal
        ]
        
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.base = MLPBase(args, cent_obs_shape)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        
        self.to(device)
    
    def forward(self, cent_obs, rnn_states, masks) -> Tuple[Tensor, Tensor]:
        """
        Compute value estimates.
        
        Args:
            cent_obs: Centralized observation
            rnn_states: Hidden states for RNN
            masks: Mask for hidden state initialization
        
        Returns:
            values: Value function predictions
            rnn_states: Updated RNN hidden states
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        critic_features = self.base(cent_obs)
        
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        values = self.v_out(critic_features)
        
        return values, rnn_states


class SheepActorCritic(nn.Module):
    """
    Combined Actor-Critic for sheep herding.
    
    This class wraps the actor and critic networks and provides
    a unified interface for action selection and value estimation.
    """
    
    def __init__(
        self,
        args,
        obs_space,
        cent_obs_space,
        action_space,
        device=torch.device("cpu")
    ):
        super(SheepActorCritic, self).__init__()
        
        self.actor = SheepActor(args, obs_space, action_space, device)
        self.critic = SheepCritic(args, cent_obs_space, device)
        
        self.to(device)
    
    def get_actions(
        self,
        cent_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        masks,
        available_actions=None,
        deterministic=False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Get actions and values for rollout.

        Returns:
            values: Value estimates
            actions: Selected actions
            action_log_probs: Log probabilities
            rnn_states_actor: Updated actor RNN states
            rnn_states_critic: Updated critic RNN states
        """
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        actions, action_log_probs, rnn_states_actor = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks) -> Tensor:
        """Get value estimates only."""
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values
    
    def evaluate_actions(
    self,
    cent_obs,
    obs,
    rnn_states_actor,
    rnn_states_critic,
    action,
    masks,
    available_actions=None,
    active_masks=None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evaluate actions for training update.

        Returns:
            values: Value estimates
            action_log_probs: Log probabilities
            dist_entropy: Distribution entropy
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        action_log_probs, dist_entropy = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )

        return values, action_log_probs, dist_entropy
        