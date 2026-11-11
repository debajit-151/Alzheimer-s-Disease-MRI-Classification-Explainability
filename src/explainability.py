"""
explainability.py - Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for visual explanation of deep learning decisions on Brain MRI scans.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM generates coarse localization maps highlighting the key brain regions
    (e.g., hippocampus shrinkage, ventricular enlargement) used by the CNN to assign a diagnosis.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates a 2D normalized Grad-CAM heatmap for a single input image tensor (1, C, H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        # Backward pass with respect to target class score
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weighted combination of forward activation maps
        activations = self.activations[0]
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)

        for i, w in enumerate(pooled_gradients):
            cam += w * activations[i, :, :]

        # Pass through ReLU to keep only features that have a positive influence on the class
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        if np.max(cam_np) > 0:
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np, target_class

def overlay_cam_on_image(orig_img_rgb, cam_heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap onto the original RGB image.
    orig_img_rgb: uint8 numpy array of shape (H, W, 3), range [0, 255]
    cam_heatmap: float numpy array of shape (h, w), range [0, 1]
    """
    h, w = orig_img_rgb.shape[:2]
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(cam_heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap (JET or TURBO)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap and original image
    superimposed = np.float32(heatmap_color) * alpha + np.float32(orig_img_rgb) * (1 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    return superimposed, heatmap_resized

