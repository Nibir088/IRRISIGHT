import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class CLIPSeg(nn.Module):
    def __init__(self, num_classes=2, use_text=True, freeze_model=True):
        super().__init__()
        self.use_text = use_text
        self.patch_size = 32
        self.num_classes = num_classes

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        

        if freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False

        self.vision_dim = self.model.vision_model.config.hidden_size  # 768
        self.text_dim = self.model.text_model.config.hidden_size      # 512

        if self.use_text:
            self.text_proj = nn.Linear(self.text_dim, self.vision_dim)

        # Simple per-patch decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.vision_dim+self.text_dim*use_text, 512),
            nn.ReLU(),
            nn.Linear(512, self.patch_size * self.patch_size * num_classes)
        )

    def forward(self, images: torch.Tensor, texts: list) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 224, 224]
            texts: List[str], length B
        Returns:
            logits: [B, num_classes, 224, 224]
        """
        B, _, H, W = images.shape

        # ViT feature tokens: [B, 50, D] → [B, 50, D] (exclude CLS)
        vision_out = self.model.vision_model(pixel_values=images).last_hidden_state
        image_feat = vision_out[:, 1:, :]  # [B, 49, 768]
        

        if self.use_text:
            device = next(self.model.parameters()).device
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            
            
            text_feat = self.model.text_model(**inputs).last_hidden_state[:, :1, :]  # [B, 512]
            # Expand text_feat to match image tokens
            text_feat = text_feat.expand(-1, image_feat.size(1), -1)  # [B, 49, 512]

            # Concatenate along the embedding dimension
            image_feat = torch.cat([image_feat, text_feat], dim=-1)  # [B, 49, 512+768]


        # Decode each patch token
        decoded = self.decoder(image_feat)  # [B, 49, P*P*C]
        decoded = decoded.view(B, H//self.patch_size, W//self.patch_size, self.patch_size, self.patch_size, self.num_classes)
        decoded = decoded.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, 7, 16, 7, 16]
        logits = decoded.view(B, self.num_classes, H, W)  # [B, C, 224, 224]

        return {'logits': logits}
