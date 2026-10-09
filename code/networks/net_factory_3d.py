def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, scaler=4, use_aspp=False, chunk_size=16,
                   use_pretrain=False, pretrained_ckpt=None):
    """Build a student/teacher network: 'unet_3D' (DyCON's 3D U-Net) or 'swin_umamba'.

    Both return (sdf_output, segmentation_logits, contrastive_features) from forward().
    """
    if net_type == "unet_3D":
        from .UNet3D_contrastive import UNet3D
        return UNet3D(in_channels=in_chns, n_classes=class_num, scale_factor=scaler, use_aspp=use_aspp)
    if net_type == "swin_umamba":
        from .SwinUMamba_DyCON import SwinUMambaDyCON
        kwargs = {'in_channels': in_chns, 'num_classes': class_num, 'scale_factor': scaler,
                  'use_pretrain': use_pretrain, 'chunk_size': chunk_size}
        if pretrained_ckpt is not None:
            kwargs['pretrained_ckpt'] = pretrained_ckpt
        return SwinUMambaDyCON(**kwargs)
    raise ValueError(f"Unknown net_type: {net_type}")
