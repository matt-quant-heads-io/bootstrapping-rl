import torch
import matplotlib.pyplot as plt
import numpy as np

class GradientTracker:
    def __init__(self, exp_path = None):
        self.grad_norms = []
        self.grad_steps = 0
        self.exp_path = exp_path

    def track_gradients(self, model):
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm(2).item()  # L2 norm
        self.grad_norms.append(total_norm)
        self.grad_steps += 1

    def plot_gradients(self, label):

        plt.xlabel("Steps")
        plt.ylabel("Total gradient norms")
        plt.title(label)
        plt.plot(range(self.grad_steps), self.grad_norms, label=label)
        plt.savefig(f'{self.exp_path}/gradient_updates.png')
        np.savez(f'{self.exp_path}/gradient_updates.npz', steps=self.grad_steps, grad_norms=self.grad_norms)


