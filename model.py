from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
import torch.optim as optim

from torch.distributions import Categorical
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from copy import deepcopy

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gym
import torch

import numpy as np

from functools import reduce
from operator import __add__

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces

from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy

from einops import rearrange

import utils


### THESE IMPORTS ARE FROM SOURCE SB3
import collections
import copy
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.preprocessing import (
    get_action_dim,
    is_image_space,
    maybe_transpose,
    preprocess_obs,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)



import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    BasePolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
import os
from gradients import GradientTracker


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")


class Conv2dSamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(
            reduce(
                __add__,
                [
                    (k // 2 + (k - 2 * (k // 2)) - 1, k // 2)
                    for k in self.kernel_size[::-1]
                ],
            )
        )

    def forward(self, input):
        return self._conv_forward(
            # self.zero_pad_2d(input.cuda()), self.weight.cuda(), self.bias.cuda()
            self.zero_pad_2d(input), self.weight, self.bias
        )


class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            Conv2dSamePadding(n_input_channels, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Conv2dSamePadding(128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Conv2dSamePadding(256, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Flatten(start_dim=1),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            test_input = th.as_tensor(observation_space.sample()[None]).float()
            test_input = rearrange(test_input, 'b h w c -> b c h w')
            n_flatten = self.cnn(
                test_input
            ).shape[-1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations = observations.to(memory_format=torch.channels_last)
        # import pdb; pdb.set_trace()

        observations = rearrange(observations, 'b h w c -> b c h w')
        return self.linear(self.cnn(observations))


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 256,
        last_layer_dim_vf: int = 256,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU(),
            nn.Linear(last_layer_dim_vf, last_layer_dim_vf), nn.ReLU()
        )

    # def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    #     """
    #     :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
    #         If all layers are shared, then ``latent_policy == latent_value``
    #     """
    #     return self.forward_actor(features), self.forward_critic(features)
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class WrappedNetwork(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Box, 
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        features_dim: int = 256,
        last_layer_dim_pi: int = 256,
        last_layer_dim_vf: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._policy = CustomActorCriticPolicy(observation_space, action_space, lr_schedule, *args, **kwargs)
        self._mlp_extractor = CustomNetwork(features_dim, last_layer_dim_pi, last_layer_dim_vf)
        self._features_extractor = CustomCNNFeatureExtractor(observation_space, features_dim)
        print(f"action_space: {action_space}")
        self._action_net = nn.Linear(features_dim, 8)
        self._value_net = nn.Linear(features_dim, 1)

    # def forward(self, observations):
    #     latent = self._features_extractor(observations)
    #     policy_latent, value_latent = self._mlp_extractor(latent)
    #     return self._action_net(policy_latent), self._value_net(value_latent)
    def forward(self, observations):
        latent = self._features_extractor(observations)
        policy_latent = self._mlp_extractor(latent)
        return self._action_net(policy_latent)




class CustomPPO(PPO):
    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        exp_path: str = None
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )
        self._gradient_tracker_scratch = GradientTracker(exp_path=exp_path)
        self.exp_path = exp_path


    def train_supervised(self, epochs=10, batch_size=64, lr=1e-5):
        """
        Train feature_extractor, mlp_extractor, action_net, and value_net via supervised learning.

        Args:
            dataset (dict): A dataset containing:
                - "observations": np.array of shape (N, obs_dim)
                - "actions": np.array of shape (N,)
                - "values": np.array of shape (N,)
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for supervised training.
            lr (float): Learning rate.
        """

        # supervised_rollout_buffer = RolloutBuffer()

        # Define optimizer
        
        # import pdb; pdb.set_trace()

        # Loss functions
        action_loss_fn = nn.CrossEntropyLoss()
        value_loss_fn = nn.MSELoss()

        file_pattern = (
            "/home/jupyter-msiper/bootstrapping-rl/data/zelda/pod_trajs_normal_*.npz"
        )
        # file_pattern = (
        #     "/home/jupyter-msiper/bootstrapping-rl/data/zelda/lg_expert_*.npz"
        # )
        data = utils.merge_npz_files(file_pattern)
        # print(data)
        # data = np.load("/home/jupyter-msiper/bootstrapping-rl/data/zeldaold/pod_trajs_normal_14_sampling.npz")

        # dataset = {
        #     "observations": torch.tensor.from_numpy(data["obs"]),
        #     "actions": data["actions"],
        #     "values": data["rewards"],
        #     "episode_starts": data["episode_starts"],
        #     "rewards": data["rewards"],
        #     "log_probs": np.array([0 for _ in range(len(data["obs"]))])

        # }

        # Ensure dataset is in PyTorch tensors
        # import pdb; pdb.set_trace()
        observations = torch.tensor(data["obs"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(data["actions"], dtype=torch.float32).to(self.device)
        # actions = torch.nn.functional.one_hot(torch.argmax(actions, dim=1) + 1, num_classes=9).float().to(self.device)
        values = torch.tensor(data["rewards"], dtype=torch.float32).to(self.device)
        print(f"Dataset size: {len(observations)}")
        # import pdb; pdb.set_trace()
        # observations = torch.tensor(data["expert_observations"][:], dtype=torch.float32).to(self.device)
        # actions = torch.tensor(data["expert_actions"], dtype=torch.long).to(self.device)
        # values = torch.tensor(data["expert_rewards"], dtype=torch.float32).to(self.device)

        best_loss = np.inf
        best_pct_train_correct = -np.inf
        loss_per_epoch = np.inf
        action_loss = 0
        value_loss = 0
        total_loss = 0
        pct_train_correct = 0
        total_action_loss = 0
        total_value_loss = 0
        # policy = deepcopy(self.policy)
        # action_net = deepcopy(self.policy.action_net)
        # value_net = deepcopy(self.policy.value_net)
        # mlp_extractor = deepcopy(self.policy.mlp_extractor)
        # features_extractor = deepcopy(self.policy.features_extractor)
        net = WrappedNetwork(self.env.observation_space, self.env.action_space, lambda x: lr, features_dim=256, last_layer_dim_pi=256, last_layer_dim_vf=256)
        # net.load_state_dict(
        #     torch.load("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/sl_policy.pth", weights_only=True, map_location=self.device)
        # )
        # print(net.summary())
        optimizer = optim.Adam(net.parameters(), lr=lr)
        net.to('cuda')
        net.train()

        # Training loop
        for epoch in range(epochs):
            
            
            perm = torch.randperm(observations.size(0))  # Shuffle data
            total_loss = 0
            steps_per_epoch = 0
            total_correct = 0
            total_action_loss = 0
            total_value_loss = 0
            action_loss = 0
            value_loss = 0
            total_loss = 0
            

            for i in range(0, observations.size(0), batch_size):
                batch_idx = perm[i : i + batch_size]
                obs_batch = observations[batch_idx]
                action_batch = actions[batch_idx]
                value_batch = values[batch_idx]
                
                
                action_preds, value_preds = net(obs_batch)
                
                action_loss = action_loss_fn(action_preds.float(), action_batch.float())
                print(f"action_preds: {action_preds.shape}")
                print(f"action_batch: {action_batch.shape}")
                value_loss = value_loss_fn(value_preds, value_batch)
                

                loss = torch.stack([action_loss, value_loss]).sum()
                net.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss   
                total_correct += ((torch.max(action_preds, dim=1).indices == torch.max(action_batch, dim=1).indices).type(torch.float).sum().item())
                steps_per_epoch += len(action_batch)
                total_action_loss += action_loss
                total_value_loss += value_loss

                

            pct_train_correct = round(total_correct/steps_per_epoch,10)
            loss_per_epoch = round(total_loss.item()/steps_per_epoch,10)
            action_loss_per_epoch = round(total_action_loss.item()/steps_per_epoch,10)
            value_loss_per_epoch = round(total_value_loss.item()/steps_per_epoch,10)
            
            print(
                f"Epoch {epoch + 1}/{epochs}, Action Loss: {action_loss_per_epoch:.4f}, Value Loss: {value_loss_per_epoch:.4f}, Epoch loss: {loss_per_epoch}, Correct Predictions: {pct_train_correct}"
            )
            
            if pct_train_correct > best_pct_train_correct:
                best_pct_train_correct = pct_train_correct
                best_loss = loss_per_epoch
                # Save trained modules
                torch.save(
                    net._features_extractor.state_dict(),
                    f"{self.exp_path}/feature_extractor.pth",
                )
                torch.save(
                    net._mlp_extractor.state_dict(),
                    f"{self.exp_path}/mlp_extractor.pth",
                )
                torch.save(
                    net._action_net.state_dict(),
                    f"{self.exp_path}/action_net.pth",
                )
                torch.save(
                    net._value_net.state_dict(),
                    f"{self.exp_path}/value_net.pth",
                )
                torch.save(
                    net.state_dict(),
                    f"{self.exp_path}/sl_policy.pth",
                )
                print(f"New best loss found: {best_loss}, saving model components")

            if best_pct_train_correct >= 0.9999:
                break

        # Save trained modules
        torch.save(
            net._features_extractor.state_dict(),
            f"{self.exp_path}/latest_feature_extractor.pth",
        )
        torch.save(
            net._mlp_extractor.state_dict(),
            f"{self.exp_path}/latest_mlp_extractor.pth",
        )
        torch.save(
            net._action_net.state_dict(),
            f"{self.exp_path}/latest_action_net.pth",
        )
        torch.save(
            net._value_net.state_dict(),
            f"{self.exp_path}/latest_value_net.pth",
        )
        torch.save(
            net.state_dict(),
            f"{self.exp_path}/latest_sl_policy.pth",
        )
        print(f"New best loss found: {best_loss}, saving model components")
        print("Supervised training complete. Saving model components...")

    def train_actor_supervised(self, epochs=10, batch_size=64, lr=1e-5):
        """
        Train feature_extractor, mlp_extractor, action_net, and value_net via supervised learning.

        Args:
            dataset (dict): A dataset containing:
                - "observations": np.array of shape (N, obs_dim)
                - "actions": np.array of shape (N,)
                - "values": np.array of shape (N,)
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for supervised training.
            lr (float): Learning rate.
        """

        # supervised_rollout_buffer = RolloutBuffer()

        # Define optimizer
        
        # import pdb; pdb.set_trace()

        # Loss functions
        action_loss_fn = nn.CrossEntropyLoss()
        value_loss_fn = nn.MSELoss()

        file_pattern = (
            "/home/jupyter-msiper/bootstrapping-rl/data/zelda/pod_trajs_normal_*.npz"
        )
        # file_pattern = (
        #     "/home/jupyter-msiper/bootstrapping-rl/data/zelda/lg_expert_*.npz"
        # )
        data = utils.merge_npz_files(file_pattern)
        # print(data)
        # data = np.load("/home/jupyter-msiper/bootstrapping-rl/data/zeldaold/pod_trajs_normal_14_sampling.npz")

        # dataset = {
        #     "observations": torch.tensor.from_numpy(data["obs"]),
        #     "actions": data["actions"],
        #     "values": data["rewards"],
        #     "episode_starts": data["episode_starts"],
        #     "rewards": data["rewards"],
        #     "log_probs": np.array([0 for _ in range(len(data["obs"]))])

        # }

        # Ensure dataset is in PyTorch tensors
        # import pdb; pdb.set_trace()
        observations = torch.tensor(data["obs"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(data["actions"], dtype=torch.float32).to(self.device)
        # actions = torch.nn.functional.one_hot(torch.argmax(actions, dim=1) + 1, num_classes=9).float().to(self.device)
        values = torch.tensor(data["rewards"], dtype=torch.float32).to(self.device)
        print(f"Dataset size: {len(observations)}")
        # import pdb; pdb.set_trace()
        # observations = torch.tensor(data["expert_observations"][:], dtype=torch.float32).to(self.device)
        # actions = torch.tensor(data["expert_actions"], dtype=torch.long).to(self.device)
        # values = torch.tensor(data["expert_rewards"], dtype=torch.float32).to(self.device)

        best_loss = np.inf
        best_pct_train_correct = -np.inf
        loss_per_epoch = np.inf
        action_loss = 0
        value_loss = 0
        total_loss = 0
        pct_train_correct = 0
        total_action_loss = 0
        total_value_loss = 0
        # policy = deepcopy(self.policy)
        # action_net = deepcopy(self.policy.action_net)
        # value_net = deepcopy(self.policy.value_net)
        # mlp_extractor = deepcopy(self.policy.mlp_extractor)
        # features_extractor = deepcopy(self.policy.features_extractor)
        net = WrappedNetwork(self.env.observation_space, self.env.action_space, lambda x: lr, features_dim=256, last_layer_dim_pi=256, last_layer_dim_vf=256)
        # net.load_state_dict(
        #     torch.load("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/supervised_training/sl_policy.pth", weights_only=True, map_location=self.device)
        # )
        # print(net.summary())
        optimizer = optim.Adam(net.parameters(), lr=lr)
        net.to('cuda')
        net.train()

        # Training loop
        for epoch in range(epochs):
            
            
            perm = torch.randperm(observations.size(0))  # Shuffle data
            total_loss = 0
            steps_per_epoch = 0
            total_correct = 0
            total_action_loss = 0
            total_value_loss = 0
            action_loss = 0
            value_loss = 0
            total_loss = 0
            

            for i in range(0, observations.size(0), batch_size):
                batch_idx = perm[i : i + batch_size]
                obs_batch = observations[batch_idx]
                action_batch = actions[batch_idx]
                value_batch = values[batch_idx]
                
                
                action_preds = net(obs_batch)
                print(f"action_preds: {action_preds.shape}")
                print(f"action_batch: {action_batch.shape}")
                
                action_loss = action_loss_fn(action_preds.float(), action_batch.float())    

                loss = action_loss
                net.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss   
                total_correct += ((torch.max(action_preds, dim=1).indices == torch.max(action_batch, dim=1).indices).type(torch.float).sum().item())
                steps_per_epoch += len(action_batch)
                total_action_loss += action_loss

                

            pct_train_correct = round(total_correct/steps_per_epoch,10)
            action_loss_per_epoch = round(total_action_loss.item()/steps_per_epoch,10)
            loss_per_epoch = round(total_action_loss.item()/steps_per_epoch,10)
            
            print(
                f"Epoch {epoch + 1}/{epochs}, Action Loss: {action_loss_per_epoch:.4f}, Epoch loss: {loss_per_epoch}, Correct Predictions: {pct_train_correct}"
            )
            
            if pct_train_correct > best_pct_train_correct:
                best_pct_train_correct = pct_train_correct
                best_loss = loss_per_epoch
                # Save trained modules
                torch.save(
                    net._features_extractor.state_dict(),
                    f"{self.exp_path}/feature_extractor.pth",
                )
                torch.save(
                    net._mlp_extractor.state_dict(),
                    f"{self.exp_path}/mlp_extractor.pth",
                )
                torch.save(
                    net._action_net.state_dict(),
                    f"{self.exp_path}/action_net.pth",
                )
                torch.save(
                    net._value_net.state_dict(),
                    f"{self.exp_path}/value_net.pth",
                )
                torch.save(
                    net.state_dict(),
                    f"{self.exp_path}/sl_policy.pth",
                )
                print(f"New best loss found: {best_loss}, saving model components")

            if best_pct_train_correct >= 0.9999:
                break

        # Save trained modules
        torch.save(
            net._features_extractor.state_dict(),
            f"{self.exp_path}/latest_feature_extractor.pth",
        )
        torch.save(
            net._mlp_extractor.state_dict(),
            f"{self.exp_path}/latest_mlp_extractor.pth",
        )
        torch.save(
            net._action_net.state_dict(),
            f"{self.exp_path}/latest_action_net.pth",
        )
        torch.save(
            net._value_net.state_dict(),
            f"{self.exp_path}/latest_value_net.pth",
        )
        torch.save(
            net.state_dict(),
            f"{self.exp_path}/latest_sl_policy.pth",
        )
        print(f"New best loss found: {best_loss}, saving model components")
        print("Supervised training complete. Saving model components...")

    def load_supervised_weights(
        self,
        feature_extractor_path,
        mlp_extractor_path,
        action_net_path,
        value_net_path,
    ):
        """
        Load saved weights into the policy components.

        Args:
            feature_extractor_path (str): Path to saved feature_extractor weights.
            mlp_extractor_path (str): Path to saved mlp_extractor weights.
            action_net_path (str): Path to saved action_net weights.
            value_net_path (str): Path to saved value_net weights.
        """
        print("Loading trained model components...")

        self.policy.features_extractor.load_state_dict(
            torch.load(feature_extractor_path, map_location=self.device)
        )
        self.policy.mlp_extractor.load_state_dict(
            torch.load(mlp_extractor_path, map_location=self.device)
        )
        self.policy.action_net = nn.Linear(256, 8)
        self.policy.action_net.load_state_dict(
            torch.load(action_net_path, map_location=self.device)
        )
        self.policy.value_net.load_state_dict(
            torch.load(value_net_path, map_location=self.device)
        )

        print("Model components loaded successfully.")
        # return self
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                self._gradient_tracker_scratch.track_gradients(self.policy)
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._gradient_tracker_scratch.plot_gradients(f"Gradient Updates Epoch {epoch}")
            self._n_updates += 1
            if not continue_training:

                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        # import pdb; pdb.set_trace()
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


        
        