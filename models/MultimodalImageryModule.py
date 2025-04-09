from typing import Optional, Dict
import torch
import torch.nn as nn

class MIM(nn.Module):
    """
    Multi-Input Module for concatenating RGB and various agricultural index features.

    Args:
        Each boolean flag specifies whether a specific input should be used.
    """
    def __init__(
        self,
        use_rgb: bool = True,
        use_ndvi: bool = True,
        use_ndwi: bool = True,
        use_ndti: bool = True,
        use_evi: bool = True,
        use_gndvi: bool = True,
        use_savi: bool = True,
        use_msavi: bool = True,
        use_rvi: bool = True,
        use_cigreen: bool = True,
        use_pri: bool = True,
        use_osavi: bool = True,
        use_wdrvi: bool = True,
        use_vegetation: bool = True
    ):
        super().__init__()

        # Store all flags
        self.use_rgb = use_rgb
        self.use_ndvi = use_ndvi
        self.use_ndwi = use_ndwi
        self.use_ndti = use_ndti
        self.use_evi = use_evi
        self.use_gndvi = use_gndvi
        self.use_savi = use_savi
        self.use_msavi = use_msavi
        self.use_rvi = use_rvi
        self.use_cigreen = use_cigreen
        self.use_pri = use_pri
        self.use_osavi = use_osavi
        self.use_wdrvi = use_wdrvi
        
        self.use_vegetation = use_vegetation

        # Calculate total number of channels
        self.total_channels = 0
        self.total_channels += 3 if self.use_rgb else 0
        
        if self.use_vegetation:
            for flag in [
                self.use_ndvi, self.use_ndwi, self.use_ndti, self.use_evi,
                self.use_gndvi, self.use_savi, self.use_msavi, self.use_rvi,
                self.use_cigreen, self.use_pri, self.use_osavi, self.use_wdrvi
            ]:
                if flag:
                    self.total_channels += 1

        if self.total_channels == 0:
            raise ValueError("At least one input feature must be enabled")

    def forward(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Concatenates selected input features along the channel dimension.

        Args:
            data (dict): Dictionary of input tensors.

        Returns:
            torch.Tensor: Concatenated tensor [B, C, H, W]
        """
        features = []

        def add_feature(name: str, condition: bool):
            if condition:
                if name not in data:
                    raise ValueError(f"{name.upper()} enabled but not found in input data")
                tensor = data[name]
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(1)  # Convert [B, H, W] -> [B, 1, H, W]
                features.append(tensor)

        if self.use_rgb:
            if 'image' not in data:
                raise ValueError("RGB features enabled but 'image' not found in input data")
            features.append(data['image'])
        if self.use_vegetation:
            add_feature('ndvi', self.use_ndvi)
            add_feature('ndwi', self.use_ndwi)
            add_feature('ndti', self.use_ndti)
            add_feature('evi', self.use_evi)
            add_feature('gndvi', self.use_gndvi)
            add_feature('savi', self.use_savi)
            add_feature('msavi', self.use_msavi)
            add_feature('rvi', self.use_rvi)
            add_feature('cigreen', self.use_cigreen)
            add_feature('pri', self.use_pri)
            add_feature('osavi', self.use_osavi)
            add_feature('wdrvi', self.use_wdrvi)

        return torch.cat(features, dim=1)
