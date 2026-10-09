def net_factory_3d(net_type="unet_3D", in_chns=1, class_num=2, scaler=4, use_aspp=False, chunk_size=16, use_pretrain=False, pretrained_ckpt=None):
    if net_type == "unet_3D":
        from .UNet3D_contrastive import UNet3D
        net = UNet3D(in_channels=in_chns, n_classes=class_num, scale_factor=scaler, use_aspp=use_aspp)
    elif net_type == "vnet":
        from .vnet_ import VNet
        net = VNet(n_channels=in_chns, n_classes=class_num, scale_factor=scaler, has_dropout=True, use_aspp=use_aspp)
    elif net_type == "swin_umamba":
        from .SwinUMamba_DyCON import SwinUMambaDyCON
        kwargs = {'in_channels': in_chns, 'num_classes': class_num, 'scale_factor': scaler, 'use_pretrain': use_pretrain, 'chunk_size': chunk_size}
        if pretrained_ckpt is not None:
            kwargs['pretrained_ckpt'] = pretrained_ckpt
        net = SwinUMambaDyCON(**kwargs)
    else:
        net = None
    return net