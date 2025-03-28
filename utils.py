"""
Helper functions for train, infer, and eval modules.
"""
import os
import re
import glob
import numpy as np
from gym_pcgrl import wrappers

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import gymnasium as gym

import hashlib
import struct
import pandas as pd
import np_utils

def merge_npz_files(file_pattern):
    """
    Merges multiple npz files into a single npz file.

    Args:
        file_pattern (str): A glob-style pattern to match the npz files to be merged.
        output_filename (str): The name of the output npz file.
    """
    all_data = {}
    file_list = glob.glob(file_pattern)

    for filename in file_list:
        with np.load(filename) as data:
            for key in data:
                if key in all_data:
                    all_data[key] = np.concatenate((all_data[key], data[key]))
                else:
                    all_data[key] = data[key]

    return all_data

class RenderMonitor(Monitor):
    """
    Wrapper for the environment to save data in .csv files.
    """
    def __init__(self, env, rank, log_dir, **kwargs):
        self.log_dir = log_dir
        self.rank = rank
        self.render_gui = kwargs.get('render', False)
        self.render_rank = kwargs.get('render_rank', 0)
        if log_dir is not None:
            log_dir = os.path.join(log_dir, str(rank))
        Monitor.__init__(self, env, log_dir)

    def step(self, action):
        if self.render_gui and self.rank == self.render_rank:
            self.render()
        output = Monitor.step(self, action)
        return output

def get_action(obs, env, model, action_type=True):
    action = None
    if action_type == 0:
        action, _ = model.predict(obs)
    elif action_type == 1:
        action_prob = model.action_probability(obs)[0]
        action = np.random.choice(a=list(range(len(action_prob))), size=1, p=action_prob)
    else:
        action = np.array([env.action_space.sample()])
    return action

def get_exp_name(game, representation, experiment, **kwargs):
    exp_name = '{}_{}'.format(game, representation)
    if experiment is not None:
        exp_name = '{}_{}'.format(exp_name, experiment)
    return exp_name

def max_exp_idx(exp_name):
    log_dir = os.path.join("./runs", exp_name)
    log_files = glob.glob('{}*'.format(log_dir))
    if len(log_files) == 0:
        n = 0
    else:
        log_ns = [re.search('_(\d+)', f).group(1) for f in log_files]
        n = max(log_ns)
    return int(n)

def load_model(log_dir, full_path=None):
    if full_path:
        model = PPO.load(full_path)
        return model

    model_path = os.path.join(log_dir, 'latest_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'latest_model.zip')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join(log_dir, 'best_model.zip')
    if not os.path.exists(model_path):
        files = [f for f in os.listdir(log_dir) if '.pkl' in f or '.zip' in f]
        if len(files) > 0:
            # selects the last file listed by os.listdir
            model_path = os.path.join(log_dir, np.random.choice(files))
        else:
            raise Exception('No models are saved')
    model = PPO.load(model_path)
    return model

def get_string_map(map, tiles):
    int_to_string = dict((i, s) for i, s in enumerate(tiles))
    result = []
    for y in range(map.shape[0]):
        result.append([])
        for x in range(map.shape[1]):
            result[y].append(int_to_string[int(map[y][x])])
    return result

def int_map_from_onehot(map, dims=21):
    int_map = []
    for y in map:
        for x in y:
            vals = np.argmax(x, axis=1)
            int_map.extend(list(vals))

    return np.array(int_map).reshape((dims,dims))

def make_env(env_name, representation, rank=0, log_dir=None, **kwargs):
    '''
    Return a function that will initialize the environment when called.
    '''
    max_step = kwargs.get('max_step', None)
    render = kwargs.get('render', False)
    def _thunk():
        if representation == 'wide':
            env = wrappers.ActionMapImagePCGRLWrapper(env_name, **kwargs)
        else:
            crop_size = kwargs.get('cropped_size', 22)
            env = wrappers.CroppedImagePCGRLWrapper(env_name, crop_size, **kwargs)
        # RenderMonitor must come last
        if render or log_dir is not None and len(log_dir) > 0:
            env = RenderMonitor(env, rank, log_dir, **kwargs)
        return env
    return _thunk

def make_vec_envs(env_name, representation, log_dir, n_cpu, **kwargs):
    '''
    Prepare a vectorized environment using a list of 'make_env' functions.
    '''
    if n_cpu > 1:
        env_lst = []
        for i in range(n_cpu):
            env_lst.append(make_env(env_name, representation, i, log_dir, **kwargs))
        env = SubprocVecEnv(env_lst)
    else:
        env = DummyVecEnv([make_env(env_name, representation, 0, log_dir, **kwargs)])
    return env



