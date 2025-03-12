import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import models, transforms
from PIL import Image
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

class GradCAM:
    """
    Grad-CAM implementation for CNN visualization
    """
    def __init__(self, model, target_layer):
        """
        Initialize GradCAM with a model and target layer
        
        Args:
            model: PyTorch model
            target_layer: The convolutional layer to use for Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        
        # Register hooks to get feature maps and gradients
        self.gradients = None
        self.features = None
        
        self._register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
        
    def _register_hooks(self):
        """
        Register forward and backward hooks on the target layer
        """
        # Hook for storing feature maps
        def forward_hook(module, input, output):
            self.features = output.detach()
        
        # Hook for storing gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register the hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        
        self.hooks = [forward_handle, backward_handle]
        
    def remove_hooks(self):
        """
        Remove all registered hooks
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
            
    def __call__(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_image: Input tensor (1, C, H, W)
            target_class: Target class index for Grad-CAM. If None, uses the predicted class.
            
        Returns:
            heatmap: Grad-CAM heatmap
            class_idx: Target class index used
            pred_score: Prediction score
        """
        # Forward pass
        model_output = self.model.action_net(self.model.mlp_extractor(self.model.extract_features(torch.from_numpy(obs).cuda()))[0]) #self.model(input_image)
        
        # Get target class
        if target_class is None:
            # import pdb; pdb.set_trace()
            pred_score, class_idx = torch.max(model_output, dim=1)
        else:
            class_idx = torch.tensor([target_class], device=input_image.device)
            pred_score = model_output[0, class_idx]
            
        # Create one-hot encoding for the target class
        one_hot = torch.zeros_like(model_output)
        one_hot[0, class_idx] = 1
        
        # Zero all existing gradients
        self.model.zero_grad()
        
        # Backward pass with the one-hot encoded target
        model_output.backward(gradient=one_hot, retain_graph=True)
        
        # Get the mean gradient for each feature map
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Create activation map
        activation_map = torch.sum(weights * self.features, dim=1, keepdim=True)
        
        # ReLU on the activation map
        activation_map = torch.relu(activation_map)
        
        # Normalize the activation map
        activation_map = self._normalize(activation_map)
        
        # Reshape and convert to numpy array for visualization
        heatmap = activation_map.squeeze().cpu().numpy()
        
        return heatmap, class_idx.item(), pred_score.item()
    
    def _normalize(self, tensor):
        """
        Normalize tensor to range [0, 1]
        """
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        
        if min_val == max_val:
            return torch.zeros_like(tensor)
        
        return (tensor - min_val) / (max_val - min_val)
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap (H, W)
            original_image: Original PIL Image or numpy array
            alpha: Weight for heatmap overlay
            colormap: OpenCV colormap for heatmap visualization
            
        Returns:
            overlay: Heatmap overlaid on original image
        """
        # Convert heatmap to uint8 and apply colormap
        heatmap_np = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_np, colormap)
        
        # Convert to RGB (from BGR)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy array if it's a PIL Image
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        
        # Resize heatmap to match original image size
        if colored_heatmap.shape[:2] != original_image.shape[:2]:
            colored_heatmap = cv2.resize(colored_heatmap, 
                                        (original_image.shape[1], original_image.shape[0]))
            
        # Overlay heatmap on original image
        overlay = (alpha * colored_heatmap + (1 - alpha) * original_image).astype(np.uint8)
        
        return overlay


# Example usage with pretrained ResNet
def load_image(image, resize=(224, 224)):
    original_image = image.copy()
    
    # Preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply preprocessing
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    return input_tensor, original_image


def visualize_gradcam(model, obs, img, target_layer=None, target_class=None, sample_class=None):
    """
    Visualize model decisions using Grad-CAM
    
    Args:
        model: PyTorch model
        image_path: Path to the input image
        target_layer: Layer to use for Grad-CAM (if None, tries to find the last conv layer)
        target_class: Target class for visualization (if None, uses the predicted class)
    """
    # Find the last convolutional layer if not specified
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv2d):
                target_layer = module
                print(f"Using layer: {name}")
                break
        else:
            raise ValueError("Could not find a convolutional layer in the model")
    
    # Load image
    input_tensor, original_image = load_image(img)
    
    # Get the image dimensions
    img_width, img_height = original_image.size
    
    # Create Grad-CAM object
    grad_cam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap, class_idx, pred_score = grad_cam(torch.from_numpy(obs).cuda(), target_class)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (img_width*100, img_height*100))
    
    # Create overlay
    # import pdb; pdb.set_trace()
    overlay = grad_cam.overlay_heatmap(heatmap_resized, original_image)
    
    # Get class names (if using ImageNet)
    
    # Display results
    plt.figure(figsize=(20, 20))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(np.array(original_image))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title(f"{sample_class} ({pred_score:.4f})")
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Clean up
    grad_cam.remove_hooks()
    
    return heatmap, overlay


# Example of using the Grad-CAM implementation
if __name__ == "__main__":
    # Load a pretrained model
    kwargs = {
        'render_rank': 0,
        'render': False,
        "change_percentage": 1.0,
        "trials": 1000,
        "verbose": True,
    }
    env = mkvenv("zelda-narrow-v0", "narrow", None, 1, **kwargs)
    model = CustomPPO.load("/home/jupyter-msiper/bootstrapping-rl/experiments/zelda/ppo_baseline/rl_model_100000_steps.zip").policy #.features_extractor
    # import pdb; pdb.set_trace()
    
    # Image path (replace with your image)
    obs = env.reset()
    sample_class = model(torch.from_numpy(obs).cuda())[0].item()
    img =  env.envs[0].gym_env.env.render(mode='rgb_array')#"dog.jpg"
    
    # Use the last convolutional layer in the last residual block
    target_layer = None#'pi_features_extractor.cnn.2.weight'#model.state_dict()[]
    
    # Visualize
    heatmap, overlay = visualize_gradcam(model, obs, img, target_layer, sample_class=sample_class)
    
    # Save results
    cv2.imwrite("gradcam_heatmap.jpg", cv2.cvtColor(np.uint8(255 * heatmap), cv2.COLOR_RGB2BGR))
    cv2.imwrite("gradcam_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))