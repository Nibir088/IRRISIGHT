from model_v3.CLIP import CLIPSeg
# from model_v3.BLIP import BLIPSeg
from model_v3.RemoteCLIP import RemoteCLIPSeg
import torch.nn.functional as F
import torch.nn as nn
import torch


class VLMResUNet(nn.Module):
    def __init__(self, input_channels, num_classes, vlm_type="clip", freeze_model=True, use_text=True):
        super().__init__()
        self.visual_proj = nn.Conv2d(input_channels, 3, kernel_size=1)
        
        if vlm_type == 'clip':
            self.model = CLIPSeg(num_classes=num_classes, use_text=use_text, freeze_model=freeze_model)
        elif vlm_type == 'remoteclip':
            self.model = RemoteCLIPSeg(num_classes=num_classes, use_text=use_text, freeze_model=freeze_model)

    def forward(self, visual_tensor, text_prompts):
        B, C, H, W = visual_tensor.shape

        if C!=3:
            visual_tensor = self.visual_proj(visual_tensor)
        
        output = self.model(visual_tensor, text_prompts)

        return output

if __name__ == "__main__":
    model = VLMResUNet(input_channels=13, num_classes=4, vlm_type="remoteclip", use_text=False)
    model = model.cpu()

    dummy_input = torch.randn(2, 13, 224, 224)
    dummy_texts = ["irrigated field with loamy soilImage from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.Image from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.irrigated field with loamy soilImage from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.Image from Pinal county, Arizona. Average ET: 106.99 mm, precipitation: 0.00 in, ground water: 238.82 ft, and surface water: 2.90 ft. This soil unit contains the following dominant components: cuerda. The geomorphic setting includes: alluvial fans. Soil texture: Very fine sandy loam; Loam. The soil has a runoff class of low, drainage class well drained, and hydrologic group B. It is rated as no hydric. Irrigation capability: 2w. Average slope: 1.00%, elevation: 476.00 m. Soil properties: AWC=0.157, Ksat=9.000, OM=0.250, BD=1.447, water content at 1/10 bar: 0.000, at 15 bar: 9.700.", "dry non-irrigated farmland"]

    with torch.no_grad():
        output = model(dummy_input, dummy_texts)

    print("Logits shape:", output['logits'].shape)  # Expected: [2, 2, 128, 128]
