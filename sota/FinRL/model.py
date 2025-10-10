"""
Custom DDPG model that LLM can evolve.
Based on Stable Baselines3's DDPG.
"""
from typing import Any, Optional, Union
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.td3.policies import TD3Policy

# --OPTION--
class CustomDDPG(DDPG):
    """
    Custom DDPG that LLM can evolve.
    Inherits from SB3's DDPG and allows custom policy_kwargs.
    """
    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # LLM can evolve the network architecture here
        if policy_kwargs is None:
            policy_kwargs = {}

# --OPTION-- 
        # Default custom architecture (LLM evolves this)
        if "net_arch" not in policy_kwargs:
            policy_kwargs["net_arch"] = dict(
                pi=[256, 256],  # Actor network layers
                qf=[256, 256]   # Critic network layers
            )
# --OPTION--
        # Call parent constructor, or try modifying it
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
        )