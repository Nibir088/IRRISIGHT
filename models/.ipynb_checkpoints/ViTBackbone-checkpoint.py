import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(42)

class ViTSegmentation(nn.Module):
    """
    Vision Transformer (ViT) based segmentation model.
    
    Args:
        backbone: Pre-trained ViT backbone
        num_classes (int): Number of output classes
        in_channels (int): Number of input channels
        img_size (int): Input image size (assumed square)
        patch_size (int): Size of patches for ViT (default: 16)
    """
    def __init__(self, backbone, num_classes, in_channels=3, img_size=224, patch_size=16):
        super(ViTSegmentation, self).__init__()
        self.backbone = backbone
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = backbone.embed_dim
        self.num_classes = num_classes
        
        # Modify the original patch embedding layer to handle new input channels
        if in_channels != 3:  # Only modify if not using standard 3 channels
            with torch.no_grad():
                # Get the original embedding projection
                orig_proj = backbone.patch_embed.proj
                
                # Create new projection layer with desired in_channels
                new_proj = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=orig_proj.out_channels,
                    kernel_size=orig_proj.kernel_size,
                    stride=orig_proj.stride,
                    padding=orig_proj.padding,
                    bias=False if orig_proj.bias is None else True
                )
                
                # Initialize weights by averaging across input channels
                orig_weight = orig_proj.weight.data
                avg_weight = orig_weight.mean(dim=1, keepdim=True)
                new_weight = avg_weight.repeat(1, in_channels, 1, 1) / np.sqrt(in_channels)
                new_proj.weight.data.copy_(new_weight)
                
                if orig_proj.bias is not None:
                    new_proj.bias.data.copy_(orig_proj.bias.data)
                    
                # Replace the projection layer
                backbone.patch_embed.proj = new_proj
        
        # Projection layers for segmentation
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, patch_size * patch_size * num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get ViT features
        features = self.backbone.forward_features(x)
        
        # Remove CLS token and process patch features
        patch_features = features[:, 1:, :]
        batch_size = patch_features.shape[0]
        
        # Generate patch predictions
        patch_preds = self.decoder(patch_features)
        
        # Reshape to final segmentation map
        patch_preds = patch_preds.reshape(
            batch_size,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.patch_size,
            self.patch_size,
            self.num_classes
        )
        
        # Rearrange dimensions
        patch_preds = patch_preds.permute(0, 5, 1, 3, 2, 4)
        patch_preds = patch_preds.reshape(
            batch_size,
            self.num_classes,
            self.img_size,
            self.img_size
        )
        
        return patch_preds

################################ Create model##############################################################
# Create model
# num_classes = 4
# img_size = 224
# backbone = timm.create_model('vit_small_patch16_224', pretrained=True)
# model = ViTSegmentation(backbone, num_classes, img_size)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)


# ################### dummy input (simulate a batch of 2 images) #########################################
# dummy_input = torch.randn(2, 3, 224, 224).to(device)

# # Forward pass
# with torch.no_grad():
#     outputs = model(dummy_input)
#     probabilities = F.softmax(outputs, dim=1)
#     predictions = torch.argmax(probabilities, dim=1)

# print(f"Input shape: {dummy_input.shape}")
# print(f"Output shape (logits): {outputs.shape}")
# print(f"Probabilities shape: {probabilities.shape}")
# print(f"Predictions shape: {predictions.shape}")
