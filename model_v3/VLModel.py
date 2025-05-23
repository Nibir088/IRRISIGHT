import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import open_clip
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Literal
from transformers import (
    CLIPProcessor, CLIPModel,
    Blip2Processor, Blip2Model,
    AutoProcessor, AutoModel,
    CLIPSegProcessor, CLIPSegForImageSegmentation
) 

class FiLMModulation(nn.Module):
    def __init__(self, in_channels: int, text_dim: int):
        super().__init__()
        self.gamma_fc = nn.Linear(text_dim, in_channels)
        self.beta_fc = nn.Linear(text_dim, in_channels)

    def forward(self, x: torch.Tensor, text_embed: torch.Tensor):
        # x: [B, C, H, W], text_embed: [B, D]
        gamma = self.gamma_fc(text_embed).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = self.beta_fc(text_embed).unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        return gamma * x + beta

class VLMResUNet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        vlm_type: Literal["clip", "blip2", "git"] = "clip",
        freeze_model: bool = True,
        use_text: bool = True
    ):
        super().__init__()
        self.vlm_type = vlm_type.lower()
        self.num_classes = num_classes
        self.use_text = use_text

        self.visual_proj = nn.Conv2d(input_channels, 3, kernel_size=1)

        if self.vlm_type == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.text_embed_dim = self.model.text_projection.weight.shape[1]
            self.image_feature_dim = self.model.visual_projection.out_features


        elif self.vlm_type == "blip2":
            self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.text_embed_dim = self.model.language_model.config.hidden_size
            self.image_feature_dim = self.model.vision_model.config.hidden_size
        elif self.vlm_type == "clipseg":
            self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
            self.text_embed_dim = 512  # Placeholder, unused directly
            self.image_feature_dim = 512  # Placeholder, unused directly

        elif self.vlm_type == "git":
            self.model = AutoModel.from_pretrained("microsoft/git-base")
            self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
            self.text_embed_dim = self.model.config.hidden_size
            self.image_feature_dim = self.model.config.hidden_size
        elif self.vlm_type == "remoteclip":
            self.model_name = 'ViT-B-32'
            self.model, _, _ = open_clip.create_model_and_transforms(self.model_name)
            self.tokenizer = open_clip.get_tokenizer(self.model_name)

            ckpt_path = "/project/biocomplexity/wyr6fx(Nibir)/checkpoints/models--chendelong--RemoteCLIP/snapshots/bf1d8a3ccf2ddbf7c875705e46373bfe542bce38/RemoteCLIP-ViT-B-32.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = self.model.load_state_dict(ckpt)
            # print(f"✅ RemoteCLIP loaded: {msg}")

            self.text_embed_dim = self.model.text_projection.shape[1]
            self.image_feature_dim = self.model.visual.output_dim

        else:
            raise ValueError(f"Unsupported VLM type: {vlm_type}")
            
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # print(self.image_feature_dim, self.text_embed_dim)
        self.film = FiLMModulation(in_channels=input_channels, text_dim=self.text_embed_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(self.image_feature_dim + input_channels*self.use_text, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        
        
        if self.vlm_type == 'remoteclip':
            with torch.no_grad():
                tokens = self.tokenizer(text_prompts).to(device)
                text_embed = self.model.encode_text(tokens)
                return text_embed
            
        
        
        
        
        max_len = 512 if self.vlm_type in ["blip2", "git"] else 77
        inputs = self.processor(
            text=text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        

        with torch.no_grad():
            if self.vlm_type == 'clip':
                text_embed = self.model.get_text_features(**inputs)
            elif self.vlm_type == 'blip2':
                outputs = self.model.language_model(**inputs, output_hidden_states=True)
                text_embed = outputs.hidden_states[-1][:, 0, :]
            elif self.vlm_type == 'git':
                outputs = self.model(**inputs)
                text_embed = outputs.last_hidden_state[:, 0, :]
            
            else:
                raise NotImplementedError

        return text_embed  # [B, D]

    def forward(self, visual_tensor: torch.Tensor, text_prompts: List[str]) -> torch.Tensor:
        B, _, H, W = visual_tensor.shape
        device = next(self.model.parameters()).device

        rgb_tensor = self.visual_proj(visual_tensor)
        rgb_tensor = F.interpolate(rgb_tensor, size=(224, 224), mode="bilinear").to(device)

        text_embeddings = self.encode_text(text_prompts)  # [B, D]

        with torch.no_grad():
            if self.vlm_type in ["clip"]:
                image_embed = self.model.get_image_features(pixel_values=rgb_tensor)
            elif self.vlm_type == "blip2":
                image_embed = self.model.get_image_features(pixel_values=rgb_tensor).last_hidden_state[:,0,:]
                # print(image_embed.shape)
            elif self.vlm_type == "git":
                image_embed = self.model.image_encoder(pixel_values=rgb_tensor).last_hidden_state[:, 0, :]
            elif self.vlm_type == "remoteclip":
                image_embed = self.model.encode_image(rgb_tensor)
            else:
                raise NotImplementedError
        
        
        # Image embed: [B, D] → [B, D, 1, 1] → [B, D, H, W]
        image_embed = image_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        
        
        # FiLM modulates visual_tensor using text
        if self.use_text:
            modulated_visual = self.film(visual_tensor, text_embeddings) #modulated_visual = self.film(visual_tensor, text_embeddings)  # [B, C, H, W]
            combined = torch.cat([modulated_visual, image_embed], dim=1)
            logits = self.decoder(combined)
        else:
            # modulated_visual = visual_tensor
            logits = self.decoder(image_embed)
        

        
        # print(modulated_visual.shape, image_embed.shape)
        
        return {'logits':logits}

# if __name__ == "__main__":
#     model = VLMResUNet(input_channels=13, num_classes=2, vlm_type="remoteclip", use_text=True)
#     model = model.cpu()

#     dummy_input = torch.randn(2, 13, 224, 224).to(next(model.parameters()).device)
#     dummy_texts = ["irrigated field with loamy soilImage from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.Image from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.irrigated field with loamy soilImage from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.Image from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.", "dry non-irrigated farmland"]

#     with torch.no_grad():
#         output = model(dummy_input, dummy_texts)

#     print("Logits shape:", output['logits'].shape)  # Expected: [2, 2, 128, 128]
