import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from utils import make_vec_envs as mkvenv
from model import CustomActorCriticPolicy, CustomCNNFeatureExtractor, CustomPPO

rng = np.random.default_rng(0)
# env = make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=rng,
#     n_envs=1,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
# )
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )
kwargs = {
        'cropped_size': 22,
        'render_rank': 0,
        # 'render': render,
        "change_percentage": 1.0,
        "trials": 1000,
        "verbose": True,
        # "experiment": experiment
    }

env = mkvenv("zelda-narrow-v0", "narrow", None, 1, **kwargs)
expert = CustomPPO.load("")


rollouts = rollout.rollout(
    expert,
    env,
    rollout.make_sample_until(min_timesteps=None, min_episodes=50),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)
