import yaml
from models.TeacherModel import KIIM  # Import your model
import torch
import pytorch_lightning as pl

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def verify_model_config(model, config):
    """Verify that model hyperparameters match config settings."""
    model_config = config['model']
    
    # Create verification dictionary
    verification = {
        'backbone_name': (model.hparams.backbone_name == model_config['backbone_name']),
        'encoder_name': (model.hparams.encoder_name == model_config['encoder_name']),
        'num_classes': (model.hparams.num_classes == model_config['num_classes']),
        'learning_rate': (model.hparams.learning_rate == model_config['learning_rate']),
        'use_attention': (model.hparams.use_attention == model_config['use_attention']),
        'use_projection': (model.hparams.use_projection == model_config['use_projection']),
        'use_rgb': (model.hparams.use_rgb == model_config['use_rgb']),
        'use_ndvi': (model.hparams.use_ndvi == model_config['use_ndvi']),
        'use_ndwi': (model.hparams.use_ndwi == model_config['use_ndwi']),
        'use_ndti': (model.hparams.use_ndti == model_config['use_ndti']),
        'pretrained_hidden_dim': (model.hparams.pretrained_hidden_dim == model_config['pretrained_hidden_dim']),
        'attention_hidden_dim': (model.hparams.attention_hidden_dim == model_config['attention_hidden_dim']),
        'gamma': (model.hparams.gamma == model_config['gamma']),
        'weight_decay': (model.hparams.weight_decay == model_config['weight_decay']),
    }
    
    # Verify loss config
    for loss_key in ['ce_weight', 'dice_weight', 'focal_weight', 'kg_weight']:
        verification[f'loss_config_{loss_key}'] = (
            model.hparams.loss_config[loss_key] == model_config['loss_config'][loss_key]
        )
    
    return verification

def verify_model_components(model):
    """Verify that all model components are initialized correctly."""
    component_checks = {
        'Backbone exists': hasattr(model, 'backbone'),
        'MIM exists': hasattr(model, 'mim'),
        'Projection module exists': (hasattr(model, 'projection') if model.hparams.use_projection else True),
        'Attention module exists': (hasattr(model, 'attention') if model.hparams.use_attention else True),
        'Loss functions initialized': all([
            hasattr(model, 'focal_loss'),
            hasattr(model, 'ce_loss'),
            hasattr(model, 'kg_loss'),
            hasattr(model, 'dice_loss')
        ]),
        'Metrics initialized': all([
            hasattr(model, 'train_metrics'),
            hasattr(model, 'val_metrics')
        ])
    }
    return component_checks

def create_dummy_batch(batch_size=2, image_size=(224, 224)):
    """Create a dummy batch for testing forward pass."""
    batch = {
        'image': torch.randn(batch_size, 3, *image_size),
        'ndvi': torch.randn(batch_size, *image_size),
        'ndwi': torch.randn(batch_size, *image_size),
        'ndti': torch.randn(batch_size, *image_size),
        'evi': torch.randn(batch_size, *image_size),
        'gndvi': torch.randn(batch_size, *image_size),
        'savi': torch.randn(batch_size, *image_size),
        'msavi': torch.randn(batch_size, *image_size),
        'rvi': torch.randn(batch_size, *image_size),
        'cigreen': torch.randn(batch_size, *image_size),
        'pri': torch.randn(batch_size, *image_size),
        'osavi': torch.randn(batch_size, *image_size),
        'wdrvi': torch.randn(batch_size, *image_size),
        'land_mask': torch.randint(0, 3, (batch_size, *image_size)),
        'crop_mask': torch.randn(batch_size, 21, *image_size),
        'target': torch.randint(0, 4, (batch_size, *image_size))
    }
    return batch

def test_forward_pass(model):
    """Test forward pass with dummy data."""
    try:
        batch = create_dummy_batch()
        # print(model)
        outputs = model(batch)
        print(outputs['predictions'].shape)
        # Check outputs
        expected_keys = {'predictions', 'logits'}
        if model.use_attention:
            expected_keys.add('attention')
            expected_keys.add('AM_features')
        if model.use_projection:
            expected_keys.add('CPM_logits')
        
        output_checks = {
            'Has required keys': all(key in outputs for key in expected_keys),
            'Prediction shape correct': outputs['predictions'].shape[1] == model.hparams.num_classes,
            'Valid prediction range': (0 <= outputs['predictions'].min() <= 1 and 
                                    0 <= outputs['predictions'].max() <= 1)
        }
        return True, output_checks
    except Exception as e:
        return False, str(e)

def main():
    # Load config
    config_path = "config/train.yaml"  # Update with your config path
    config = load_config(config_path)
    
    print("Step 1: Creating model from config...")
    model = KIIM(**config['model'])
    
    print("\nStep 2: Verifying model configuration...")
    config_verification = verify_model_config(model, config)
    for param, matches in config_verification.items():
        print(f"{param}: {'✓' if matches else '✗'}")
    
    print("\nStep 3: Verifying model components...")
    component_checks = verify_model_components(model)
    for component, exists in component_checks.items():
        print(f"{component}: {'✓' if exists else '✗'}")
    
    print("\nStep 4: Testing forward pass...")
    success, results = test_forward_pass(model)
    if success:
        print("Forward pass successful!")
        for check, passed in results.items():
            print(f"{check}: {'✓' if passed else '✗'}")
    else:
        print(f"Forward pass failed with error: {results}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()