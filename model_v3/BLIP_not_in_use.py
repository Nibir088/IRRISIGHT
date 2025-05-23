# from transformers import Blip2Model, Blip2Processor
# import torch.nn.functional as F
# import torch.nn as nn
# import torch

# class BLIPSeg(nn.Module):
#     def __init__(self, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
#         self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
#         self.text_dim = self.model.language_model.config.hidden_size
#         self.image_dim = self.model.vision_model.config.hidden_size

#     def encode_text(self, text_list):
#         inputs = self.processor(text=text_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             output = self.model.language_model(**inputs, output_hidden_states=True)
#             return output.hidden_states[-1][:, 0, :]

#     def encode_image(self, image_tensor):
#         with torch.no_grad():
#             feat = self.model.get_image_features(pixel_values=image_tensor)#.last_hidden_state[:, 0, :]
#             print(feat.keys)
#         print(feat.shape)
#         B, D = feat.shape
#         _, _, H, W = image_tensor.shape
#         return feat.view(B, D, 1, 1).expand(-1, -1, H, W)
