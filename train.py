import os
import argparse
import datetime
import pathlib

import gym
import torch

# from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from model import CustomPPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

import matplotlib 

from pathlib import Path
from utils import load_model
# from model import FullyConvPolicyBigMap, FullyConvPolicySmallMap, CustomPolicyBigMap, CustomPolicySmallMap
from model import CustomActorCriticPolicy, CustomCNNFeatureExtractor



import numpy as np
import os

import numpy as np
from utils import make_vec_envs as mkvenv, make_env as mkenv

from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback



np.seterr(all='ignore') 


PROJECT_ROOT = os.getenv("PROJECT_ROOT")
if not PROJECT_ROOT:
    raise RuntimeError("The env var `PROJECT_ROOT` is not set.")
    
logdir = "/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/ppo_100M_steps"
def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward

    # Print stats every 1000 calls
    if (n_steps + 1) % 10 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 100:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, we save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(os.path.join(log_dir, 'best_model.pkl'))
            else:
                print("Saving latest model")
                _locals['self'].save(os.path.join(log_dir, 'latest_model.pkl'))
        else:
            # print('{} monitor entries'.format(len(x)))
            pass
    n_steps += 1
    # Returning False will stop training early
    return True


def main(game, representation, experiment, steps, n_cpu, render, logging, tb_log_dir, **kwargs):
    resume = kwargs.get('resume', False)
    if game == "binary":
        kwargs['cropped_size'] = 28
    elif game == "zelda":
        kwargs['cropped_size'] = 22
    elif game == "sokoban":
        kwargs['cropped_size'] = 10

    # global log_dir

    # log_dir = kwargs["logdir"]
    # if resume:
    #     model = load_model(experiment_path)
        
    kwargs = {
        **kwargs,
        'render_rank': 0,
        'render': render,
        "change_percentage": 1.0,
        "trials": 1000,
        "verbose": True,
        "experiment": experiment
    }
    new_logger = configure(f"./experiments/{game}/{experiment}/", ["stdout", "csv", "tensorboard"])
    env = mkvenv("zelda-narrow-v0", "narrow", None, 1, **kwargs)
    policy_kwargs = dict(
        features_extractor_class=CustomCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = CustomPPO(CustomActorCriticPolicy, env=env, policy_kwargs=policy_kwargs, verbose=2, exp_path=f"./experiments/{game}/{experiment}", device = "cuda" if torch.cuda.is_available() else "cpu")
    # model = CustomPPO.load("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/ppo_baseline/rl_model_300000_steps_up_to_300k_steps.zip", env)
    model.policy.to("cuda")
    # model.set_env(env)
    model.set_logger(new_logger)
    
    config = {
        "policy_type": CustomActorCriticPolicy,
        "total_timesteps": steps,
        "env_id": "zelda-narrow-v0",
    }
    run = wandb.init(
        project="ppo_baseline",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )
    checkpoint_callback = CheckpointCallback(save_freq=steps//10, save_path=f"./experiments/{game}/{experiment}/")
    # Separate evaluation env
    
    eval_env = mkvenv("zelda-narrow-v0", "narrow", None, 1, **kwargs)
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./experiments/{game}/{experiment}/best_model",
                                log_path=f"./experiments/{game}/{experiment}/results", eval_freq=steps//10)
    wandb_callback = WandbCallback(
            model_save_path=f"models/{run.id}",
            verbose=2,
        )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
    load_path = "/home/jupyter-msiper/bootstrapping-rl/saved_models"
    
    # model = model.load_supervised_weights(
    #     f"{load_path}/feature_extractor.pth",
    #     f"{load_path}/mlp_extractor.pth",
    #     f"{load_path}/action_net.pth",
    #     f"{load_path}/value_net.pth",

    # )
    model.train_actor_supervised(epochs=250, batch_size=16, lr=0.00003)
    # model.learn(steps, callback=callbacks)
    
    run.finish()
    
    # NOTE: Bootsrapping section!!
    # dataset_path = os.path.join(PROJECT_ROOT, "data", game, "pod_trajs_normal_1_sampling.npz")
    # print(f"Loading supervised dataset from {dataset_path}")
    # data = np.load(dataset_path)
    # # import pdb; pdb.set_trace()
    
    # dataset = {
    #     "observations": data["obs"],
    #     "actions": data["actions"],
    #     "values": data["rewards"],
    #     "episode_starts": data["episode_starts"],
    #     "rewards": data["rewards"],
    #     "log_probs": np.array([0 for _ in range(len(data["obs"]))])

    # }

    # model.train_supervised(dataset, epochs=2000, batch_size=16, lr=3e-4)


    # Load the trained model components
    # model.load_supervised_weights(
    #     feature_extractor_path="feature_extractor.pth",
    #     mlp_extractor_path="mlp_extractor.pth",
    #     action_net_path="action_net.pth",
    #     value_net_path="value_net.pth"
    # )
    # ======================================================================================================


    # Now, continue PPO training using the loaded components
    # model.learn(total_timesteps=100_000)



def parse_args():
    parser = argparse.ArgumentParser(
        prog='Training script',
        description='This is the training script for a non-bootstrapped agent'
    )
    parser.add_argument('--game', '-g', choices=['zelda', 'loderunner'], default="zelda") 
    parser.add_argument('--representation', '-r', default='narrow')
    parser.add_argument('--experiment', default="ppo_baseline")
    # parser.add_argument('--n_steps', default=0, type=int)
    parser.add_argument('--steps', default=100000000, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--logging', default=True, type=bool)
    parser.add_argument('--n_cpu', default=1, type=int)
    parser.add_argument('--tb_log_dir', default="tb_log", type=str)
    parser.add_argument('--resume', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    game = args.game
    representation = args.representation
    experiment = args.experiment
    # n_steps = args.n_steps
    steps = args.steps
    render = args.render
    logging = args.logging
    n_cpu = args.n_cpu
    kwargs = {
        'resume': args.resume
    }
    best_mean_reward = -np.inf
    # experiment_path = PROJECT_ROOT + "/data/" + args.game + "/experiments/" + args.experiment
    # experiment_filepath = pathlib.Path(experiment_path)
    # if not experiment_filepath.exists():
    #     os.makedirs(str(experiment_filepath))

    # kwargs["logdir"] = experiment_path
    # log_dir = experiment_path
    # main(game, representation, experiment, steps, n_cpu, render, logging, experiment_path, args.tb_log_dir, **kwargs)
    main(game, representation, experiment, steps, n_cpu, render, logging, args.tb_log_dir, **kwargs)