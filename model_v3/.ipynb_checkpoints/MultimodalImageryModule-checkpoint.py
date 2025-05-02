from typing import Dict
import torch
import torch.nn as nn

class MIM(nn.Module):
    """
    Multi-Input Module (MIM) for combining RGB, vegetation indices, land mask, and crop mask.

    This module supports the inclusion of:
      - RGB image channels (3-channel input)
      - Vegetation index stack (multi-channel)
      - Land mask (single channel)
      - Crop mask (single channel)

    Args:
        use_rgb (bool): Include RGB input (expects key 'rgb' of shape [B, 3, H, W])
        use_vegetation (bool): Include agri_index stack (expects key 'agri_index' of shape [B, N, H, W])
        use_land_mask (bool): Include land mask (expects key 'land_mask' of shape [B, 1, H, W])
        use_crop_mask (bool): Include crop mask (expects key 'crop_mask' of shape [B, 1, H, W])
    """
    def __init__(
        self,
        use_rgb: bool = True,
        use_vegetation: bool = True,
        use_land_mask: bool = False,
        use_crop_mask: bool = False
    ):
        super().__init__()
        self.use_rgb = use_rgb
        self.use_vegetation = use_vegetation
        self.use_land_mask = use_land_mask
        self.use_crop_mask = use_crop_mask

        # Ensure at least one input is used
        if not any([use_rgb, use_vegetation, use_land_mask, use_crop_mask]):
            raise ValueError("At least one input source must be enabled in MIM.")

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenate selected input features along the channel dimension.

        Args:
            data (Dict[str, Tensor]): Input dictionary with expected keys as per enabled flags:
                - 'rgb': [B, 3, H, W] if use_rgb
                - 'agri_index': [B, N, H, W] if use_vegetation
                - 'land_mask': [B, 1, H, W] if use_land_mask
                - 'crop_mask': [B, 1, H, W] if use_crop_mask

        Returns:
            torch.Tensor: Concatenated tensor of shape [B, C, H, W]
        """
        features = []

        if self.use_rgb:
            if 'rgb' not in data:
                raise KeyError("Expected 'rgb' in data but not found.")
            features.append(data['rgb'])

        if self.use_vegetation:
            if 'agri_index' not in data or data['agri_index'] is None:
                raise KeyError("Expected 'agri_index' in data but not found or is None.")
            features.append(data['agri_index'])

        if self.use_land_mask:
            if 'land_mask' not in data:
                raise KeyError("Expected 'land_mask' in data but not found.")
            features.append(data['land_mask'])

        if self.use_crop_mask:
            if 'crop_mask' not in data:
                raise KeyError("Expected 'crop_mask' in data but not found.")
            features.append(data['crop_mask'])

        return torch.cat(features, dim=1)
