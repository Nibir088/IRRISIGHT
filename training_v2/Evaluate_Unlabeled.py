from model_v3.TeacherModel import TeacherModel
from data.data_module_v2 import IrrigationDataModule
from omegaconf import OmegaConf
import yaml
import torch
import numpy as np
import geopandas as gpd
from shapely import wkt
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from tqdm import tqdm

# === Config ===
cfg_path = '/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_Irrigation_Mapping_Model/Output/cross-state/vision/kiim/result_stats/configs/hydra_config.yaml'
state = 'Florida'
print('************************************************')
print(state, cfg_path)
print('************************************************')

cfg = OmegaConf.load(cfg_path)
cfg.dataset.train_type = 'unsupervised'
cfg.dataset.states = [[state, 1]]

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data + Model ===
data_module = IrrigationDataModule(cfg)
data_module.setup('fit')
data_module.setup('test')

ckpt_path = '/sfs/gpfs/tardis/project/bii_nssac/people/wyr6fx/NeurIPS_Irrigation_Mapping_Model/Output/cross-state/vision/kiim/result_stats/checkpoints/epoch=17-val_iou_macro_irr=0.912.ckpt'
model = TeacherModel.load_from_checkpoint(ckpt_path, **cfg).to(device)
model.eval()

# === Load Polygons ===
gdf = gpd.read_file(f'/project/biocomplexity/wyr6fx(Nibir)/NeurIPS_irrigation_data/Agcensus/{state}_Irrigation.geojson')
gdf = gdf.to_crs("EPSG:5070")
print(gdf.head())

# === Define Raster Extent ===
xmin, ymin, xmax, ymax = gdf.total_bounds
resolution = 30
width = int(np.ceil((xmax - xmin) / resolution))
height = int(np.ceil((ymax - ymin) / resolution))
transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

# === Initialize Output Raster ===
min_pred_map = np.full((height, width), fill_value=255, dtype=np.uint8)

# === Inference Loop ===
for batch in tqdm(data_module.train_dataloader(), desc="Aggregating min predictions"):
    with torch.no_grad():
        polygons = batch['polygon']
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        preds = model(batch)['predictions'].argmax(dim=1).cpu().numpy()

    for i in range(preds.shape[0]):
        patch_mask = preds[i]
        poly = wkt.loads(polygons[i])
        bounds = poly.bounds

        h, w = patch_mask.shape
        patch_transform = from_bounds(*bounds, w, h)

        poly_raster = rasterize([(poly, 1)], out_shape=(h, w), transform=patch_transform, fill=0, dtype=np.uint8)
        patch_mask = patch_mask * poly_raster

        x0 = int((bounds[0] - xmin) / resolution)
        y0 = int((ymax - bounds[3]) / resolution)
        x1 = min(x0 + w, width)
        y1 = min(y0 + h, height)
        h_clip = y1 - y0
        w_clip = x1 - x0

        if h_clip <= 0 or w_clip <= 0:
            continue

        patch_mask = patch_mask[:h_clip, :w_clip].astype(np.uint8)
        target = min_pred_map[y0:y1, x0:x1]
        np.minimum(target, patch_mask, out=target, where=(patch_mask > 0))

# === Filter Polygons by Prediction ===
valid_rows = []
irrigated_counts = []
gdf_group = gdf.groupby(['geometry', 'County'])['irrigated_acres'].sum().reset_index()

for idx, geom in tqdm(enumerate(gdf_group.geometry), total=len(gdf_group)):
    bounds = geom.bounds
    x0 = int((bounds[0] - xmin) / resolution)
    y0 = int((ymax - bounds[3]) / resolution)
    x1 = int((bounds[2] - xmin) / resolution)
    y1 = int((ymax - bounds[1]) / resolution)

    h = y1 - y0
    w = x1 - x0
    if h <= 0 or w <= 0 or x1 > width or y1 > height:
        continue

    local_transform = from_bounds(*bounds, w, h)
    poly_mask = rasterize([(geom, 1)], out_shape=(h, w), transform=local_transform, fill=0, dtype=np.uint8)
    pred_crop = min_pred_map[y0:y1, x0:x1]

    if not np.any(poly_mask):
        continue

    irrigated_pixels = ((pred_crop == 1) * poly_mask).sum()
    if irrigated_pixels == 0:
        continue

    valid_rows.append(idx)
    irrigated_counts.append(irrigated_pixels)

# === Save Output ===
gdf_group = gdf_group.loc[valid_rows].copy()
gdf_group['irrigated_pixels'] = irrigated_counts
gdf_group['irrigation_discovered'] = gdf_group['irrigated_pixels'] * 0.2223945

print(gdf_group[['irrigated_acres', 'irrigation_discovered']].sum())

true = gdf_group['irrigated_acres'].astype(float).values
pred = gdf_group['irrigation_discovered'].astype(float).values
rmse = np.sqrt(np.mean((true - pred) ** 2))
print(f"RMSE: {rmse:.4f}")
