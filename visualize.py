import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import CustomPPO

from utils import make_vec_envs as mkvenv
from gym_pcgrl.envs.pcgrl_env import PcgrlEnv
import torchvision.transforms as transforms




# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleCNN, self).__init__()
        # Assuming input is one-hot encoded vector reshaped to 2D
        self.side_length = int(np.sqrt(input_dim))
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate size after conv and pooling layers
        pooled_side = self.side_length // 2 // 2  # Two pooling operations
        self.fc1 = nn.Linear(64 * pooled_side * pooled_side, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, input_dim)
        # Reshape to (batch_size, 1, side_length, side_length)
        x = x.view(-1, 1, self.side_length, self.side_length)
        
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Full-Gradient computation for one-hot encoded inputs
class FullGradVisualizer:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.gradients = []
        
    def register_hooks(self):
        def hook_fn(module, grad_input, grad_output):
            self.gradients.append(grad_input[0].detach())
        
        # Register hooks on all layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handle = module.register_backward_hook(hook_fn)
                self.handles.append(handle)
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def compute_full_grad(self, one_hot_input, target_class):
        """
        Compute full gradient for a one-hot encoded input.
        
        Args:
            one_hot_input: One-hot encoded input tensor (batch_size, input_dim)
            target_class: Target class for which to compute gradients
            
        Returns:
            full_grad: Full gradient visualization
        """
        self.gradients = []
        self.register_hooks()
        # import pdb; pdb.set_trace()
        
        # Ensure input requires gradient
        one_hot_input.requires_grad_(True)
        
        # Forward pass
        output = self.model.action_net(self.model.mlp_extractor(self.model.extract_features(one_hot_input))[0])#self.model(one_hot_input)
        
        # Backward pass
        self.model.zero_grad()
        
        # Create one-hot encoded target
        # import pdb; pdb.set_trace()
        target = torch.zeros_like(output)
        target[0, target_class] = 1
        
        # Compute gradients
        output.backward(gradient=target)
        
        # Remove hooks
        self.remove_hooks()
        
        # Get input gradient
        input_grad = one_hot_input.grad.detach()
        
        # Convert to visualization format (reshape to original image shape)
        # import pdb; pdb.set_trace()
        side_length = int(np.sqrt(one_hot_input.shape[1]))
        full_grad = input_grad.reshape(22, 22, 8).cpu().numpy()
        
        return full_grad

# Function to generate sample one-hot encoded data
def generate_one_hot_data(input_dim, num_samples=100, num_classes=10):
    """
    Generate synthetic one-hot encoded data.
    
    Args:
        input_dim: Dimension of input (should be a perfect square)
        num_samples: Number of samples to generate
        num_classes: Number of classes
    
    Returns:
        x_data: Tensor of one-hot inputs (num_samples, input_dim)
        y_data: Tensor of class labels (num_samples)
    """
    side_length = int(np.sqrt(input_dim))
    x_data = torch.zeros(num_samples, input_dim)
    y_data = torch.zeros(num_samples, dtype=torch.long)
    
    for i in range(num_samples):
        # Randomly select a class
        class_idx = np.random.randint(0, num_classes)
        y_data[i] = class_idx
        
        # Create a pattern based on class
        pattern = torch.zeros(side_length, side_length)
        
        # Different pattern for each class (simple example)
        if class_idx % 2 == 0:
            # Even classes: horizontal stripe
            row = (class_idx // 2) % side_length
            pattern[row, :] = 1
        else:
            # Odd classes: vertical stripe
            col = (class_idx // 2) % side_length
            pattern[:, col] = 1
            
        # Convert to one-hot and add some noise
        x_data[i] = pattern.reshape(-1)
    
    return x_data, y_data

# Demo function
def run_full_grad_demo():
    # Parameters
    input_dim = 22*22  # 28x28
    num_classes = 8
    kwargs = {
        'render_rank': 0,
        'render': False,
        "change_percentage": 1.0,
        "trials": 1000,
        "verbose": True,
    }
    # env = PcgrlEnv(prob="zelda", rep="narrow")#mkvenv("zelda-narrow-v0", "narrow", None, 1, **kwargs)
    env = mkvenv("zelda-narrow-v0", "narrow", None, 1, **kwargs)
    model = CustomPPO.load("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/ppo_baseline/rl_model_100000_steps.zip").policy #.features_extractor
    
    # Create model
    # model = SimpleCNN(input_dim, num_classes)
    
    # Generate dummy data
    # x_data, y_data = generate_one_hot_data(input_dim, num_samples=100, num_classes=num_classes)
    

    # Train model (simplified for demo)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion = nn.CrossEntropyLoss()
    
    # for epoch in range(5):
    #     running_loss = 0.0
    #     for i in range(0, len(x_data), 32):
    #         inputs = x_data[i:i+32]
    #         labels = y_data[i:i+32]
            
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         running_loss += loss.item()
        
    #     print(f"Epoch {epoch+1}, Loss: {running_loss / (len(x_data) // 32)}")
    
    # Create visualizer
    visualizer = FullGradVisualizer(model)
    
    # Compute full gradient for a sample input
    sample_idx = 0
    # sample_input = x_data[sample_idx:sample_idx+1]
    # sample_class = y_data[sample_idx].item()
    obs = env.reset() 
    # impo/rt 
    sample_class = model(torch.from_numpy(obs).cuda())[0].item()
    sample_input = env.envs[0].gym_env.env.render(mode='rgb_array')

    transform = transforms.Compose([
        transforms.PILToTensor()
    ])
    sample_input_tensor = transform(sample_input).float()
    

    # import pdb; pdb.set_trace()
    
    full_grad = visualizer.compute_full_grad(torch.from_numpy(obs).cuda(), sample_class)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Input (One-Hot)")
    sns.heatmap(np.array(sample_input.convert("L")), cmap='viridis')
    
    plt.subplot(1, 3, 2)
    plt.title("Full Gradient Visualization")
    import pdb; pdb.set_trace()
    sns.heatmap(torch.argmax(torch.from_numpy(full_grad), dim=2), cmap='coolwarm', center=0)
    
    plt.subplot(1, 3, 3)
    plt.title("Input * Gradient")
    sns.heatmap(np.array(sample_input.convert("L")) * full_grad, cmap='coolwarm', center=0)
    
    plt.tight_layout()
    plt.savefig('full_gradient_visualization.png')
    plt.show()
    
    return full_grad

if __name__ == "__main__":
    full_grad = run_full_grad_demo()
    print("Full gradient shape:", full_grad.shape)