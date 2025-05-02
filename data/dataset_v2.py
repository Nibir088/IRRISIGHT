import os
import json
from typing import List, Tuple, Dict, Union
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import rasterio
import pandas as pd

class ImageProcessor:
    """
    Handles image loading, vegetation index computation, and auxiliary mask extraction
    from Sentinel-2 raster files. Assumes a fixed band order:
    
    Bands 1–9  : Spectral bands (B02 to B12) used for computing vegetation indices
    Band 10   : Land mask (categorical)
    Band 11   : Crop mask (categorical)
    Band 12   : Irrigation mask (categorical, optional)
    Band 13   : Sub-irrigation mask (categorical, optional)
    
    Outputs are returned in a structured dictionary to support multimodal training.
    """

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction to an RGB image.

        Args:
            image (np.ndarray): RGB image with pixel values in [0, 255].
            gamma (float): Gamma value for correction (default = 1.0).

        Returns:
            np.ndarray: Gamma-adjusted image in uint8 format.
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return table[image]
    
    
    @staticmethod
    def load_image_and_indices(
        path: str,
        index_names: List[str] = ['ndvi', 'gndvi'],
        gamma: bool = False,
        gamma_value: float = 1.3
    ) -> Dict[str, Union[np.ndarray, List[str], None]]:
        """
        Loads image data and computes requested vegetation indices.

        Args:
            path (str): Path to a multi-band GeoTIFF file representing a Sentinel-2 patch.
            index_names (List[str]): List of vegetation indices to compute. Available options:
                - 'ndvi', 'gndvi', 'cigreen', 'evi', 'savi', 'msavi',
                - 'ndwi', 'rvi', 'pri', 'osavi', 'wdrvi', 'ndti'
            gamma (bool): If True, applies gamma correction to the RGB image.
            gamma_value (float): Gamma correction factor (default = 1.3).

        Returns:
            Dict[str, Union[np.ndarray, List[str], None]]:
                {
                    'rgb':            RGB image (H, W, 3) normalized or gamma-corrected,
                    'agri_index':     Stack of N vegetation indices (H, W, N),
                    'agri_index_name': List of N index names corresponding to channels,
                    'land_mask':      (H, W) mask of land class,
                    'crop_mask':      (H, W) mask of crop class,
                    'irr_mask':       (H, W) irrigation label (optional),
                    'subirr_mask':    (H, W) sub-irrigation label (optional)
                }
        """
        with rasterio.open(path) as src:
            # Extract required spectral bands
            spectral = {
                "B02": src.read(1).astype(np.float32),
                "B03": src.read(2).astype(np.float32),
                "B04": src.read(3).astype(np.float32),
                "B08": src.read(4).astype(np.float32),
                "B05": src.read(5).astype(np.float32),
                "B06": src.read(6).astype(np.float32),
                "B07": src.read(7).astype(np.float32),
                "B11": src.read(8).astype(np.float32),
                "B12": src.read(9).astype(np.float32),
            }

            # RGB construction (Sentinel-2: B04=R, B03=G, B02=B)
            rgb = np.stack([spectral["B04"], spectral["B03"], spectral["B02"]], axis=-1)

            if gamma:
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10) * 255
                rgb = rgb.astype(np.uint8)
                rgb = ImageProcessor.adjust_gamma(rgb, gamma=gamma_value)
            else:
                rgb = rgb / 65535.0  # Normalize if gamma is not applied

            # Define vegetation index formulas
            index_formulas = {
                'ndvi': lambda x: (x['B08'] - x['B04']) / (x['B08'] + x['B04'] + 1e-10),
                'gndvi': lambda x: (x['B08'] - x['B03']) / (x['B08'] + x['B03'] + 1e-10),
                'cigreen': lambda x: (x['B08'] / (x['B03'] + 1e-10)) - 1,
                'evi': lambda x: 2.5 * (x['B08'] - x['B04']) / (x['B08'] + 6 * x['B04'] - 7.5 * x['B02'] + 1 + 1e-10),
                'savi': lambda x: ((x['B08'] - x['B04']) / (x['B08'] + x['B04'] + 0.5)) * 1.5,
                'msavi': lambda x: (2 * x['B08'] + 1 - np.sqrt((2 * x['B08'] + 1) ** 2 - 8 * (x['B08'] - x['B04']))) / 2,
                'ndwi': lambda x: (x['B03'] - x['B11']) / (x['B03'] + x['B11'] + 1e-10),
                'rvi': lambda x: x['B08'] / (x['B04'] + 1e-10),
                'pri': lambda x: (x['B03'] - x['B02']) / (x['B03'] + x['B02'] + 1e-10),
                'osavi': lambda x: (x['B08'] - x['B04']) / (x['B08'] + x['B04'] + 0.16),
                'wdrvi': lambda x: (0.2 * x['B08'] - x['B04']) / (0.2 * x['B08'] + x['B04'] + 1e-10),
                'ndti': lambda x: (x['B06'] - x['B07']) / (x['B06'] + x['B07'] + 1e-10),
            }

            # Compute vegetation indices
            agri_index_list, index_used = [], []
            for idx in index_names:
                if idx in index_formulas:
                    index_raw = index_formulas[idx](spectral)
                    norm_index = (index_raw - index_raw.min()) / (index_raw.max() - index_raw.min() + 1e-10)
                    agri_index_list.append(norm_index)
                    index_used.append(idx)

            agri_index_stack = np.stack(agri_index_list, axis=-1) if agri_index_list else None

            # Read masks (fixed band positions)
            land_mask = src.read(10).astype(np.uint8)
            crop_mask = src.read(11).astype(np.uint8)
            irr_mask = src.read(12).astype(np.uint8) if src.count >= 12 else None
            subirr_mask = src.read(13).astype(np.uint8) if src.count >= 13 else None

        return {
            'rgb': rgb,
            'agri_index': agri_index_stack,
            'agri_index_name': index_used,
            'land_mask': land_mask,
            'crop_mask': crop_mask,
            'irr_mask': irr_mask,
            'subirr_mask': subirr_mask
        }
class TextPromptProcessor:
    """
    A utility class for generating structured natural language prompts
    from metadata associated with Sentinel-2 image patches.

    The prompt combines soil, hydrological, and location metadata 
    into a human-readable description.

    Typical input: path to a .tif patch (e.g., "/data/patches/Arizona/2020/patch_1024_2048.tif")
    Corresponding metadata: "/data/patches/Arizona/2020/patch_metadata.csv"
    """

    @staticmethod
    def generate_text_prompt_from_patch(patch_path: str) -> str:
        """
        Generate a structured text description for a given patch.

        --------------------
        Input:
        --------------------
        patch_path : str
            Absolute or relative file path to the .tif image patch 
            (e.g., '/path/to/patch_x_y.tif')

        --------------------
        Returns:
        --------------------
        prompt : str
            A human-readable string composed from the patch metadata.
            The output contains:
              - Soil prompt (if available)
              - County and state location
              - Average July evapotranspiration (mm)
              - Precipitation (in)
              - Groundwater depth (ft)
              - Surface water level (ft)

        --------------------
        Raises:
        --------------------
        FileNotFoundError: if the metadata CSV file is missing
        ValueError: if the patch_path is not found in the metadata CSV
        """
        # Determine metadata path
        base_dir = os.path.dirname(patch_path)
        metadata_path = os.path.join(base_dir, 'patch_metadata.csv')

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Load metadata and match the current patch
        df = pd.read_csv(metadata_path)
        matched = df[df['patch_path'] == patch_path]

        if matched.empty:
            raise ValueError(f"No metadata found for patch: {patch_path}")

        row = matched.iloc[0]

        # Extract relevant metadata fields
        soil_info = row['soil_info']           # already in prompt-ready format
        county = row['county']                 # e.g., "Maricopa"
        state = row['state']                   # e.g., "Arizona"
        et = row['ET']                         # evapotranspiration (mm)
        precip = row['precipitation']          # precipitation (in)
        gw = row['groundwater']                # groundwater depth (ft)
        sw = row['surface_water']              # surface water level (ft)

        # Helper for formatting numeric fields
        def safe_format(val, unit=''):
            return f"{val:.2f}{unit}" if pd.notna(val) else "unknown"

        # Assemble prompt segments
        prompt_parts = []

        

        # Add location and hydrological context
        location_text = f"Located in {county} County, {state}."
        et_text = f"Average evapotranspiration (ET) in July: {safe_format(et, ' mm')}."
        precip_text = f"Precipitation: {safe_format(precip, ' in')}."
        gw_text = f"Groundwater depth: {safe_format(gw, ' ft')}."
        sw_text = f"Surface water level: {safe_format(sw, ' ft')}."

        # Append structured parts
        prompt_parts.extend([location_text, et_text, precip_text, gw_text, sw_text])
        
        # Include soil prompt if available
        if pd.notna(soil_info) and soil_info.strip().lower() != 'none':
            prompt_parts.append(soil_info.strip())

        # Final prompt string
        return " ".join(prompt_parts)
    
class ImageMaskDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 states: List[Tuple[str, float]],
                 image_shape: Tuple[int, int] = (224, 224),
                 transform: bool = False,
                 gamma_value: float = 1.3,
                 label_type: str = 'irrigation',
                 vision_indices: List[str] = ['image'],
                 train_type: str = 'cross-state',   # Options: 'cross-state', 'holdout', 'unsupervised'
                 split: str = 'train'):
        """
        Initialize the ImageMaskDataset for supervised or unsupervised learning using Sentinel-2 patches.

        Args:
            data_dir (str): Path to the folder containing JSON files with patch paths.
                            These files should be in the format split_by_patches_<state>.json or similar.
            states (List[Tuple[str, float]]): List of tuples where each tuple is (state_name, fraction).
                            - state_name (str): The name of the state as used in the split filenames.
                            - fraction (float): Fraction of training data to use (only applies for split='train').
            image_shape (Tuple[int, int], optional): Desired image shape (H, W) after resizing. Default is (224, 224).
            transform (bool, optional): Whether to apply torchvision transforms (Resize + ToTensor). Default is False.
            gamma_value (float, optional): Gamma correction value to apply when loading RGB. If <= 0, gamma is not applied.
            label_type (str, optional): Type of target label to use (e.g., 'irrigation'). Reserved for future use. Default is 'irrigation'.
            vision_indices (List[str], optional): List of indices to load. Can include:
                            - 'image': RGB image (constructed from Sentinel-2 bands)
                            - any of ['ndvi', 'gndvi', 'evi', 'savi', 'msavi', etc.] for vegetation index computation
            train_type (str, optional): One of ['cross-state', 'holdout', 'unsupervised']. Controls which split file is loaded.
            split (str, optional): One of ['train', 'val', 'test']. Defines the data split to use.

        Raises:
            AssertionError: If split is not one of ['train', 'val', 'test'].
            ValueError: If train_type is not one of the accepted types.
        """
        assert split in ['train', 'val', 'test'], f"Invalid split: {split}"
        self.data_dir = data_dir
        self.states = states
        self.image_shape = image_shape
        self.gamma_value = gamma_value
        self.label_type = label_type
        self.vision_indices = vision_indices
        self.train_type = train_type
        self.split = split

        # Define torchvision transform (applies ToTensor and optionally Resize)
        self.transform = transforms.Compose([transforms.ToTensor()])
        if transform:
            self.transform.transforms.append(transforms.Resize(image_shape))

        # Collect file paths for the current split
        self.data_paths = self._load_split_paths()

    def _load_split_paths(self):
        """
        Load patch paths from JSON files based on the selected train_type and split.

        For each state:
        - Loads the JSON file for that state and split.
        - Filters training data based on the specified fraction.

        Returns:
            List[str]: List of absolute patch file paths to be used in this dataset split.

        JSON file format:
        {
            "train": ["/path/to/patch1.tif", ...],
            "val": [...],
            "test": [...]
        }
        """
        data_paths = []
        for state_name, train_pct in self.states:
            if self.train_type == 'cross-state':
                file_path = f'{self.data_dir}/split_by_patches_{state_name}.json'
            elif self.train_type == 'holdout':
                file_path = f'{self.data_dir}/leaveout_patches_{state_name}.json'
            elif self.train_type == 'unsupervised':
                file_path = f'{self.data_dir}/unlabeled_split_patches_{state_name}.json'
            else:
                raise ValueError(f"Unsupported train_type: {self.train_type}")

            with open(file_path, 'r') as f:
                patch_data = json.load(f)

            split_patches = patch_data.get(self.split, [])
            if self.split == 'train':
                split_patches = split_patches[:int(len(split_patches) * train_pct)]

            data_paths.extend(split_patches)

        return data_paths

    def __len__(self) -> int:
        return len(self.data_paths)
    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor, List[str], None]]:
        """Returns image data, index stack, and all relevant masks as a dictionary."""

        path = self.data_paths[idx]
        image_dict = ImageProcessor.load_image_and_indices(
            path=path,
            index_names=self.vision_indices,
            gamma=self.gamma_value > 0,
            gamma_value=self.gamma_value
        )

        sample = {
            'image_path': path,
            'split': self.split
        }

        # ---- RGB Image ----
        # Shape: (3, H, W)
        # Only added if 'image' is requested in vision_indices
        if 'image' in self.vision_indices and image_dict['rgb'] is not None:
            rgb_tensor = self.transform(image_dict['rgb'])  # Apply ToTensor + Resize
            sample['rgb'] = rgb_tensor

        # ---- Agricultural Indices ----
        # Shape: (N, H, W)
        # Includes vegetation indices like NDVI, GNDVI, etc.
        if image_dict['agri_index'] is not None:
            sample['agri_index'] = self.transform(image_dict['agri_index'])
            sample['agri_index_names'] = image_dict['agri_index_name']
        else:
            sample['agri_index'] = None
            sample['agri_index_names'] = []

        # ---- Land Mask ----
        # Shape: (1, H, W)
        # Binary Mask:
        #   0 = Non-cropland
        #   1 = Cropland (CDL code 81 or 82)
        sample['land_mask'] = torch.from_numpy(image_dict['land_mask']).float().unsqueeze(0) / 255.0

        # ---- Crop Mask ----
        # Shape: (1, H, W)
        # Categorical USDA CDL codes (1–255) — e.g., 1 = corn, 5 = soybean, etc.
        # Refer to CDL legend: https://www.nass.usda.gov/Research_and_Science/Cropland/docs/legend.pdf
        sample['crop_mask'] = torch.from_numpy(image_dict['crop_mask']).float().unsqueeze(0) / 255.0

        # ---- Irrigation Mask ----
        # Shape: (H, W)
        # Categorical labels:
        #   0 = Non-irrigated
        #   1 = Flood irrigation
        #   2 = Sprinkler irrigation
        #   3 = Drip irrigation
        sample['irr_mask'] = (
            torch.from_numpy(image_dict['irr_mask']).long()
            if image_dict['irr_mask'] is not None else None
        )

        # ---- Sub-Irrigation Mask ----
        # Shape: (H, W)
        # Categorical labels:
        #   0 = Other / Default / Mixed
        #   1 = Center Pivot Sprinkler or Micro-Drip
        #   2 = Big Gun Sprinkler
        sample['subirr_mask'] = (
            torch.from_numpy(image_dict['subirr_mask']).long()
            if image_dict['subirr_mask'] is not None else None
        )
        if sample['irr_mask']==None:
            sample['is_labeled'] = False
        else:
            sample['is_labeled'] = True
        #  ---- Soil and Location Text Prompt ----
        # Text Prompt generation
        sample['text_prompt'] = TextPromptProcessor.generate_text_prompt_from_patch(path)

        return sample
import matplotlib.pyplot as plt

def visualize_sample(sample, index=0):
    """
    Visualize RGB image along with land mask and irrigation mask.

    Args:
        sample (dict): A dictionary returned by the ImageMaskDataset.__getitem__()
        index (int): Optional index for subplot labeling
    """
    rgb = sample.get('rgb')         # shape: (3, H, W)
    land_mask = sample.get('land_mask')  # shape: (1, H, W)
    irr_mask = sample.get('irr_mask')    # shape: (H, W) or None

    if rgb is None:
        print("⚠️ No RGB image in sample.")
        return

    rgb_np = rgb.permute(1, 2, 0).numpy()
    land_np = land_mask.squeeze().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    axs[0].imshow(rgb_np)
    axs[0].set_title("RGB Image")
    axs[0].axis('off')

    axs[1].imshow(land_np, cmap='gray')
    axs[1].set_title("Land Mask")
    axs[1].axis('off')

    if irr_mask is not None:
        axs[2].imshow(irr_mask.numpy(), cmap='viridis', vmin=0, vmax=3)
        axs[2].set_title("Irrigation Mask")
        axs[2].axis('off')
    else:
        axs[2].text(0.5, 0.5, 'No Irrigation Mask', horizontalalignment='center', verticalalignment='center')
        axs[2].axis('off')

    plt.tight_layout()
    plt.show()



# # ==== Dummy Test (you can comment/uncomment to run) ====
# if __name__ == "__main__":
#     # Dummy Setup
#     data_dir = '/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_irrigation_data/Train-Test-Split'
#     states = [('Arizona', 1.0)]
#     train_type = 'cross-state'
#     split = 'train'

#     # Initialize Dataset
#     dataset = ImageMaskDataset(
#         data_dir=data_dir,
#         states=states,
#         train_type=train_type,
#         split=split,
#         transform=True,
#         vision_indices=['ndvi', 'gndvi', 'image','ndti']
#     )

#     print(f"📦 Total samples: {len(dataset)}")

#     # Sample access
#     sample = dataset[0]
#     print(f"✅ Sample keys: {list(sample.keys())}")
#     print(f"🖼️ RGB shape: {sample['rgb'].shape if 'rgb' in sample else 'None'}")
#     print(f"🌾 Agri indices: {sample['agri_index_names']} | shape: {sample['agri_index'].shape if sample['agri_index'] is not None else 'None'}")
#     print(f"🗺️ Land mask shape: {sample['land_mask'].shape}")
#     print(f"🗺️ Crop mask shape: {sample['crop_mask'].shape}")
#     print(f"🗺️ Land mask shape: {sample['irr_mask'].shape}")
#     # print(f"🗺️ Land mask shape: {sample['subirr_mask'].shape}")
#     # ==== Example Usage ====
#     # sample = dataset[0]  # assuming you already created the dataset
#     visualize_sample(sample)

