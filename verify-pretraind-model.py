import torch
from model_v3.PretrainedModule_v1 import PretrainedModel  # Adjust path if needed

def test_pretrained_model():
    # Configuration
    model_name = "unet"          # Try: 'unet', 'fpn', 'deepv3+', 'vit', 'swin', 'farseg', 'kiim'
    in_channels = 13             # 3 (RGB) + 10 agri indices, adjust based on your dataset
    num_classes = 4              # Number of segmentation classes
    image_size = (224, 224)      # H, W

    # Instantiate model
    model = PretrainedModel(
        model_name=model_name,
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=16,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        activation=None,
        attention_type='self',   # Only used for KIIM
        task='segmentation'
    )

    # Dummy input
    x = torch.randn(2, in_channels, *image_size)  # Batch size 2

    # Forward pass
    output = model(x)

    # Display output shape
    if isinstance(output, dict):
        print("✅ Output keys:", output.keys())
        print("📐 Logits shape:", output['logits'].shape)
    else:
        print("❌ Unexpected output format.")

if __name__ == "__main__":
    test_pretrained_model()
