import numpy as np
import utils

file_pattern = "/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/ppo_baseline/gradient_updates.npz"
data = utils.merge_npz_files(file_pattern)


print(data)