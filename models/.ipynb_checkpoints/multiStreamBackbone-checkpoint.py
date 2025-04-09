import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.utils import make_grid
import os
from torchgeo.models import ResNet18_Weights,ResNet50_Weights



class MultiStreamAttention(nn.Module):
    def __init__(self, in_channels = 2048, K = 224):
        super(MultiStreamAttention, self).__init__()
        
        # Attention network Φ with 5x5 and 1x1 convolutions
        self.attention_fcn = nn.Sequential(
            nn.Conv2d(in_channels*2, K, kernel_size=5, padding=2),
            nn.BatchNorm2d(K),
            nn.ReLU(inplace=True),
            nn.Conv2d(K, 2, kernel_size=1)
        )
        
    def forward(self,features_list):
        """
        Args:
            features_list: List of feature maps [rgb_features, aux_features]
            Each feature map has shape [batch_size, channels, height, width]
        """
        
        batch_size, channels, height, width = features_list[0].shape
        concat_features = torch.cat(features_list, dim=1)
        
        attention_scores = self.attention_fcn(concat_features)
        attention_weights = torch.sigmoid(attention_scores)  # [B, 2, H', W']
        
        weighted_features = []
        pre_attention_outputs = []
        
        for i, features in enumerate(features_list):
            weights = attention_weights[:, i:i+1, ...]
            weighted_features.append(features * weights)
            pre_attention_outputs.append(features)
        
        # Sum the weighted features
        merged_features = sum(weighted_features)
        
        return merged_features, pre_attention_outputs, attention_weights

class SelfAttentionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super(SelfAttentionModule, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=1)
        
    def forward(self, features_list):
        # Concatenate features
        F_concat = torch.cat(features_list, dim=1)  # [B, 4096, H, W]
        batch_size, channels, height, width = F_concat.shape
        
        # Generate Q, K, V
        Q = self.query_conv(F_concat)  # [B, 2048, H, W]
        K = self.key_conv(F_concat)    # [B, 2048, H, W]
        V = self.value_conv(F_concat)  # [B, 4096, H, W]
        
        # Reshape for attention computation
        Q = Q.view(batch_size, -1, height * width)  # [B, 2048, HW]
        K = K.view(batch_size, -1, height * width)  # [B, 2048, HW]
        V = V.view(batch_size, -1, height * width)  # [B, 4096, HW]
        
        # Compute attention scores
        attention = torch.bmm(Q.permute(0, 2, 1), K)  # [B, HW, HW]
        attention = F.softmax(attention / torch.sqrt(torch.tensor(channels)), dim=-1)
        
        # Apply attention to values
        out = torch.bmm(V, attention.permute(0, 2, 1))  # [B, 4096, HW]
        out = out.view(batch_size, -1, height, width)   # [B, 4096, H, W]
        
        # Split back to two streams
        F_RGB_prime = out[:, :2048, :, :]
        F_I_prime = out[:, 2048:, :, :]
        
        # Sum the features
        F_final = F_RGB_prime + F_I_prime
        
        pre_attention_outputs = []
        for i, features in enumerate(features_list):
            pre_attention_outputs.append(features)
        
        return F_final, pre_attention_outputs, attention
    
    
class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels=2048):
        super(CrossAttentionModule, self).__init__()
        
        # Transformations for RGB stream
        self.query_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_rgb = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Transformations for Indices stream
        self.query_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_indices = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def attention(self, Q, K, V):
        batch_size, channels, height, width = Q.shape
        
        # Reshape for attention computation
        Q = Q.view(batch_size, -1, height * width)  # [B, C, HW]
        K = K.view(batch_size, -1, height * width)  # [B, C, HW]
        V = V.view(batch_size, -1, height * width)  # [B, C, HW]
        
        # Compute attention scores
        attention = torch.bmm(Q.permute(0, 2, 1), K)  # [B, HW, HW]
        attention = F.softmax(attention / torch.sqrt(torch.tensor(channels)), dim=-1)
        
        # Apply attention to values
        out = torch.bmm(V, attention.permute(0, 2, 1))  # [B, C, HW]
        return out.view(batch_size, channels, height, width)
    
    def forward(self, features_list):
        F_rgb, F_indices = features_list
        
        # RGB attending to Indices
        Q_rgb = self.query_rgb(F_rgb)
        K_ind = self.key_indices(F_indices)
        V_ind = self.value_indices(F_indices)
        F_rgb_attended = self.attention(Q_rgb, K_ind, V_ind)
        
        # Indices attending to RGB
        Q_ind = self.query_indices(F_indices)
        K_rgb = self.key_rgb(F_rgb)
        V_rgb = self.value_rgb(F_rgb)
        F_ind_attended = self.attention(Q_ind, K_rgb, V_rgb)
        
        # Combine attended features
        F_final = F_rgb_attended + F_ind_attended
        
        pre_attention_outputs = []
        for i, features in enumerate(features_list):
            pre_attention_outputs.append(features)
        
        attention_weights = torch.stack([F_rgb_attended, F_ind_attended], dim=1)
        return F_final, pre_attention_outputs, attention_weights    
    
class AttentionSegmentationModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super(AttentionSegmentationModel, self).__init__()
        self.shared_backbone = backbone
        self.channel_attention = MultiStreamAttention(in_channels=2048)
        
        # Feature normalization layers
        self.rgb_norm = nn.BatchNorm2d(2048)
        self.aux_norm = nn.BatchNorm2d(2048)
        
        # Final prediction layers
        self.conv1x1 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
        # Stream-specific prediction layers
        self.stream_predictors = nn.ModuleList([
            nn.Conv2d(2048, num_classes, kernel_size=1)
            for _ in range(2)  # One for each stream
        ])
        
    def forward(self, x):
        # Split input into RGB and auxiliary channels
        rgb_input = x[:, :3, :, :]
        aux_input = x[:, 3:, :, :]
        
        # Get features from both streams using shared backbone
        rgb_features = self.shared_backbone(rgb_input)[0]
        aux_features = self.shared_backbone(aux_input)[0]
        
                # Normalize features
        rgb_features = self.rgb_norm(rgb_features)
        aux_features = self.aux_norm(aux_features)
        
        # Apply channel attention
        merged_features, pre_attention_features, attention_weights = self.channel_attention(
            [rgb_features, aux_features]
        )
        
        # Generate predictions
        final_pred = self.conv1x1(merged_features)
        final_pred = self.upsample(final_pred)
        
        # Generate pre-attention predictions for each stream
        stream_preds = []
        for i, features in enumerate(pre_attention_features):
            pred = self.stream_predictors[i](features)
            pred = self.upsample(pred)
            stream_preds.append(pred)
            
        return final_pred, stream_preds, attention_weights

    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, use_dice=True):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.use_dice = use_dice
        self.ce_criterion = nn.CrossEntropyLoss()
        self.dice_criterion = DiceLoss() if use_dice else None
        
    def forward(self, final_pred, stream_preds, masks, attention_weights):
        # Main CE loss
        main_loss = self.ce_criterion(final_pred, masks)
        
        # Stream supervision loss
        stream_loss = sum(self.ce_criterion(pred, masks) for pred in stream_preds)
        
        # Combine losses
        total_loss = main_loss + self.alpha * stream_loss
        
        if self.use_dice:
            total_loss += self.dice_criterion(final_pred, masks)
            
        
        return total_loss

    
def modify_input_layer(original_conv, target_in_channels):
    """Modify input convolution layer for different number of input channels."""
    new_conv = nn.Conv2d(
        target_in_channels, 
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )
    
    # Use pretrained weights for first 3 channels
    new_conv.weight.data[:, :3, :, :] = original_conv.weight.data[:, :3, :, :]
    
    # Initialize additional channels with mean of RGB weights
    if target_in_channels > 3:
        avg_weights = original_conv.weight.data[:, :3, :, :].mean(dim=1, keepdim=True)
        for i in range(3, target_in_channels):
            new_conv.weight.data[:, i:i+1, :, :] = avg_weights
            
    if original_conv.bias is not None:
        new_conv.bias.data = original_conv.bias.data
        
    return new_conv

def create_backbone(in_channels=3, pretrained=True, weights = "sentinel"):
    
    if weights == "landsat":
        weights = ResNet50_Weights.LANDSAT_ETM_SR_MOCO
    elif weights == "sentinel":
        weights = ResNet50_Weights.SENTINEL2_SI_RGB_SATLAS
        
        
    """Create and modify ResNet backbone."""
    backbone = timm.create_model('resnet50', in_chans=in_channels, features_only=True, out_indices=(4,))
    
    pretrained_state_dict = weights.get_state_dict(progress=True)
    
    backbone.load_state_dict(pretrained_state_dict, strict=False)
    
    if in_channels != 3:
        backbone.conv1 = modify_input_layer(backbone.conv1, in_channels)
    
    return backbone



def train_model_with_attention(state, train_loader, test_loader, device, 
                             num_epochs=100, num_classes=4, alpha=0.5, 
                             use_dice=False, learning_rate=0.001,
                             vis_interval=5, weights = "sentinel",attention_reg=0.2,in_channels=3):
    
    backbone = create_backbone(in_channels=in_channels, pretrained=True, weights = weights)
    model = AttentionSegmentationModel(backbone, num_classes).to(device)
    

    criterion = CombinedLoss(alpha=alpha, use_dice=use_dice)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    # visualizer = AttentionVisualizer(state=state)
    epoch_stats = []
    
    for epoch in range(num_epochs):
        
        model.train()
        running_loss = 0.0
        epoch_attention_stats = {
            'rgb_mean': [], 'aux_mean': [], 
            'rgb_var': [], 'aux_var': [],
            'max_diff': [], 'entropy': []
        }
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            
            
            optimizer.zero_grad()
            final_pred, stream_preds, attention_weights = model(images)
            loss = criterion(final_pred, stream_preds, masks, attention_weights)
            
           
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            
            rgb_attention = attention_weights[:, 0]
            aux_attention = attention_weights[:, 1]
            
            epoch_attention_stats['rgb_mean'].append(rgb_attention.mean().item())
            epoch_attention_stats['aux_mean'].append(aux_attention.mean().item())
            epoch_attention_stats['rgb_var'].append(rgb_attention.var().item())
            epoch_attention_stats['aux_var'].append(aux_attention.var().item())
            epoch_attention_stats['max_diff'].append(
                (rgb_attention - aux_attention).abs().max().item()
            )
            
            # Calculate attention entropy
            entropy = -(attention_weights * torch.log(attention_weights + 1e-10)).sum(1).mean()
            epoch_attention_stats['entropy'].append(entropy.item())
            
            # # Visualize attention maps periodically
            # if epoch % vis_interval == 0 and batch_idx % 50 == 0:
            #     visualizer.visualize_attention_maps(
            #         images, masks, attention_weights, epoch, batch_idx, state
            #     )
        
    
        avg_train_loss = running_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                final_pred, stream_preds, attention_weights = model(images)
                loss = criterion(final_pred, stream_preds, masks,attention_weights)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
    
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
        
        scheduler.step(avg_val_loss)
        
        epoch_stats.append({
            k: np.mean(v) for k, v in epoch_attention_stats.items()
        })
        
        # # Plot statistics every few epochs
        # if epoch % vis_interval == 0:
        #     visualizer.plot_attention_statistics(epoch_stats)
        torch.save(model, f"/scratch/gza5dr/IrrType_LargeScale/experimentwithTorchGeo/trained_models_v1/{state}/{state}_pre_rgb_aux_{224}_{6}.pth")
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    return model, best_val_loss, epoch_stats