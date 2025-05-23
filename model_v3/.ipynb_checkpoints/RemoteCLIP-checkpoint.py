import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class RemoteCLIPSeg(nn.Module):
    def __init__(self, num_classes=2, use_text=True, freeze_model=True):
        super().__init__()
        self.use_text = use_text
        self.patch_size = 32
        self.num_classes = num_classes

        model_name = "ViT-B-32"
        self.model, _, _ = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Load RemoteCLIP checkpoint
        ckpt_path = "/project/biocomplexity/wyr6fx(Nibir)/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt)
        self.model.eval()

        if freeze_model:
            for p in self.model.parameters():
                p.requires_grad = False

        self.vision_dim = self.model.visual.class_embedding.shape[0]          # 768
        self.text_dim = self.model.text_projection.shape[1]  # 512

        # Per-pixel decoder (acts on [B, D(+T), h, w])
        self.decoder = nn.Sequential(
            nn.Linear(self.vision_dim+self.text_dim*use_text, 512),
            nn.ReLU(),
            nn.Linear(512, self.patch_size * self.patch_size * num_classes)
        )

    def forward(self, images: torch.Tensor, texts: list) -> dict:
        """
        Args:
            images: [B, 3, 224, 224]
            texts: List[str], length B

        Returns:
            logits: [B, num_classes, 224, 224]
        """
        B, _, H, W = images.shape

        # Extract patch token map: [B, 768, 7, 7]
        feat_dict = self.model.visual.forward_intermediates(images)
        image_feat = feat_dict['image_intermediates'][-1]  # [B, 768, 7, 7]

        if self.use_text:
            device = next(self.model.parameters()).device
            tokens = self.tokenizer(texts).to(device)

            text_feat = self.model.encode_text(tokens)  # [B, 512]
            text_feat = text_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, image_feat.size(2), image_feat.size(3))  # [B, 512, 7, 7]
            image_feat = torch.cat([image_feat, text_feat], dim=1)  # [B, 768+512, 7, 7]
        image_feat = image_feat.flatten(2).transpose(1, 2)  # [B, 49, 1280]
        decoded = self.decoder(image_feat)  # [B, 49, P*P*C]
        decoded = decoded.view(B, H//self.patch_size, W//self.patch_size, self.patch_size, self.patch_size, self.num_classes)
        decoded = decoded.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, C, 7, 16, 7, 16]
        logits = decoded.view(B, self.num_classes, H, W)  # [B, C, 224, 224]

        return {'logits': logits}
