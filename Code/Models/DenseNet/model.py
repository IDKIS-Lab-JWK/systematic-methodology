"""Explicit 3D, one-channel, random-initialization DenseNet121."""

def build_model(dropout=0.0):
    import monai
    from monai.networks.nets import DenseNet121
    if monai.__version__ != "1.5.1":
        raise RuntimeError(f"CNN baseline requires monai==1.5.1, found {monai.__version__}")
    return DenseNet121(spatial_dims=3, in_channels=1, out_channels=5,
                       init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                       pretrained=False, dropout_prob=float(dropout))
