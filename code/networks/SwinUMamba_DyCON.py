"""
SwinUMamba adapted for DyCON semi-supervised 3D segmentation framework.

Processes 3D volumes slice-by-slice through 2D SwinUMamba, and adds a 
projection head for contrastive feature extraction.

Returns: (output_sdf, out_seg, features) to match DyCON's expected interface,
matching the signature of UNet3D and VNet in this codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .SwinUMamba import SwinUMamba, load_pretrained_ckpt


class SwinUMambaDyCON(nn.Module):
    """
    SwinUMamba backbone for DyCON.
    
    Wraps the 2D SwinUMamba to handle 3D volumetric data by processing 
    each depth slice independently, then stacking results back to 3D.
    Adds a projection head on bottleneck features for contrastive learning (FeCL/UnCL).
    
    Forward returns:
        output_sdf:  (B, num_classes, H, W, D) — tanh of logits (SDF-like output)
        out_seg:     (B, num_classes, H, W, D) — raw segmentation logits  
        features:    (B, 256, H', W', D')      — projected features for contrastive loss
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        scale_factor: int = 2,
        feat_size: list = [48, 96, 192, 384, 768],
        hidden_size: int = 768,
        use_pretrain: bool = False,
        pretrained_ckpt: str = "./data/pretrained/vmamba/vmamba_tiny_e292.pth",
        chunk_size: int = 16,
        use_checkpoint: bool = False,
    ):
        """
        Args:
            in_channels:  Number of input channels (e.g., 1 for CT).
            num_classes:  Number of segmentation classes.
            scale_factor: Controls the resolution of contrastive features.
                          Features are at 1/(scale_factor*4) of the input resolution,
                          matching UNet3D/VNet behaviour in DyCON.
            feat_size:    Channel sizes for the SwinUMamba encoder stages.
            hidden_size:  Channel size of the deepest encoder features (bottleneck).
            use_pretrain: Whether to load VMamba pretrained weights.
            pretrained_ckpt: Path to VMamba pretrained checkpoint.
            chunk_size:   Number of 2D slices to process in parallel per forward chunk.
                          Lower values save GPU memory; higher values are faster.
            use_checkpoint: Recompute the activations of each chunk in the backward pass
                          instead of storing them. Same results, far less GPU memory, slower.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint
        self.hidden_size = hidden_size

        # 2D SwinUMamba backbone (encoder + decoder)
        self.backbone = SwinUMamba(
            in_chans=in_channels,
            out_chans=num_classes,
            feat_size=feat_size,
            hidden_size=hidden_size,
            deep_supervision=False,
            spatial_dims=2,
        )

        self.tanh = nn.Tanh()

        # Projection head for contrastive features
        # Maps bottleneck features (hidden_size channels) → 256-dim embedding
        # Uses 3D convolutions since features are stacked into a 3D volume
        self.projection = nn.Sequential(
            nn.Conv3d(hidden_size, 512, kernel_size=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 256, kernel_size=1),
            nn.BatchNorm3d(256),
        )

        # Load pretrained VMamba weights into the encoder
        if use_pretrain:
            self.backbone = load_pretrained_ckpt(self.backbone, ckpt_path=pretrained_ckpt)

        # Initialize projection head
        for m in self.projection.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _forward_2d(self, x_2d):
        """
        Forward pass through SwinUMamba for a batch of 2D slices.
        
        Args:
            x_2d: (N, C, H, W) batch of 2D slices
            
        Returns:
            seg_out:    (N, num_classes, H, W) segmentation logits
            bottleneck: (N, hidden_size, bH, bW) deepest encoder features
        """
        # Stem: stride-2 conv
        x1 = self.backbone.stem(x_2d)
        
        # VSS Mamba encoder — returns multi-scale features
        # vss_outs[0]: (N, 48, H/2, W/2)   — stem-level features
        # vss_outs[1]: (N, 96, H/4, W/4)   — stage 0
        # vss_outs[2]: (N, 192, H/8, W/8)  — stage 1
        # vss_outs[3]: (N, 384, H/16, W/16) — stage 2
        # vss_outs[4]: (N, 768, H/32, W/32) — stage 3 (bottleneck)
        vss_outs = self.backbone.vssm_encoder(x1)

        # Bottleneck features for contrastive learning
        bottleneck = vss_outs[4]

        # Encoder refinement blocks
        enc1 = self.backbone.encoder1(x_2d)
        enc2 = self.backbone.encoder2(vss_outs[0])
        enc3 = self.backbone.encoder3(vss_outs[1])
        enc4 = self.backbone.encoder4(vss_outs[2])
        enc5 = self.backbone.encoder5(vss_outs[3])

        # Decoder (U-Net style upsampling with skip connections)
        dec4 = self.backbone.decoder6(vss_outs[4], enc5)
        dec3 = self.backbone.decoder5(dec4, enc4)
        dec2 = self.backbone.decoder4(dec3, enc3)
        dec1 = self.backbone.decoder3(dec2, enc2)
        dec0 = self.backbone.decoder2(dec1, enc1)
        dec_out = self.backbone.decoder1(dec0)

        # Final segmentation output
        seg_out = self.backbone.out_layers[0](dec_out)

        return seg_out, bottleneck

    def forward(self, x):
        """
        Forward pass for 3D volumetric input (slice-by-slice 2D processing).
        
        Args:
            x: (B, C, H, W, D) — 3D volume (matches DyCON's data format)
            
        Returns:
            output_sdf: (B, num_classes, H, W, D) — tanh activation of logits
            out_seg:    (B, num_classes, H, W, D) — raw logits  
            features:   (B, 256, H', W', D')      — projected contrastive features
                        where H' = H//(scale_factor*4), etc.
        """
        B, C, H, W, D = x.shape

        # Reshape 3D volume into batch of 2D slices along the depth axis
        # (B, C, H, W, D) → (B*D, C, H, W)
        x_slices = x.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

        # Process slices in chunks to manage GPU memory
        seg_chunks = []
        feat_chunks = []

        for i in range(0, B * D, self.chunk_size):
            chunk = x_slices[i:i + self.chunk_size]
            if self.use_checkpoint and torch.is_grad_enabled():
                # The random state is restored for the recomputation, so drop path matches.
                seg_c, feat_c = checkpoint(self._forward_2d, chunk, use_reentrant=False)
            else:
                seg_c, feat_c = self._forward_2d(chunk)
            seg_chunks.append(seg_c)
            feat_chunks.append(feat_c)

        seg_flat = torch.cat(seg_chunks, dim=0)      # (B*D, num_classes, H, W)
        feat_flat = torch.cat(feat_chunks, dim=0)     # (B*D, hidden_size, bH, bW)

        # Reshape segmentation output back to 3D: (B*D, nC, H, W) → (B, nC, H, W, D)
        nC = seg_flat.shape[1]
        out_seg = seg_flat.view(B, D, nC, H, W).permute(0, 2, 3, 4, 1).contiguous()

        # Reshape bottleneck features to 3D: (B*D, fC, bH, bW) → (B, fC, bH, bW, D)
        fC, bH, bW = feat_flat.shape[1], feat_flat.shape[2], feat_flat.shape[3]
        feat_3d = feat_flat.view(B, D, fC, bH, bW).permute(0, 2, 3, 4, 1).contiguous()

        # Interpolate features to the target resolution for contrastive loss
        # Target: 1/(scale_factor * 4) of original volume, matching mask_con in training
        target_h = H // (self.scale_factor * 4)
        target_w = W // (self.scale_factor * 4)
        target_d = D // (self.scale_factor * 4)
        feat_3d = F.interpolate(
            feat_3d,
            size=(target_h, target_w, target_d),
            mode='trilinear',
            align_corners=True,
        )

        # Project to 256-dim contrastive feature space
        features = self.projection(feat_3d)  # (B, 256, target_h, target_w, target_d)

        # SDF-like output
        output_sdf = self.tanh(out_seg)

        return output_sdf, out_seg, features


if __name__ == "__main__":
    """Quick sanity check with a dummy 3D volume."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SwinUMambaDyCON(
        in_channels=1,
        num_classes=2,
        scale_factor=2,
        use_pretrain=False,
        chunk_size=8,
    ).to(device)

    # PancreasCT training patch used in the thesis: (B=2, C=1, H=64, W=64, D=64)
    x = torch.randn(2, 1, 64, 64, 64).to(device)
    
    sdf_out, seg_out, features = model(x)
    print(f"output_sdf: {sdf_out.shape}")   # Expected: (2, 2, 64, 64, 64)
    print(f"out_seg:    {seg_out.shape}")     # Expected: (2, 2, 64, 64, 64)
    print(f"features:   {features.shape}")    # Expected: (2, 256, 8, 8, 8)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total_params:.2f}M")
