import torch
from torch import nn
from .mamba_simple import Mamba
from .mamba_compressor import MambaRMSNorm


class GOPCompressor(nn.Module):
    """
    Stage 1: GOP-level compression using I-frame as a query token.
    I-frame is prepended to motion vectors and processed through bidirectional Mamba.
    The updated I-frame representation becomes the GOP feature.

    Supports two motion vector modes:
    - 'raw': Use raw 2-channel motion vectors (mv_x, mv_y) with learnable projection
    - 'magnitude': Use magnitude-only (1-channel) sqrt(mv_x^2 + mv_y^2)

    Supports two GOP aggregation methods:
    - 'concat': Concatenate I-frame and MVs, process through Mamba (original method)
    - 'cross_attn': Use cross-attention with I-frame as query, MVs as key/value
    - 'add': Aggregate MVs and add to I-frame features
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_bidirectional: bool = True,
        mv_downsample_factor: int = 1,  # Default to no downsampling (1 = keep original resolution)
        mv_mode: str = "raw",  # "raw" for 2-channel MVs, "magnitude" for 1-channel magnitude
        gop_aggregation: str = "concat",  # "concat", "cross_attn", or "add"
    ):
        super().__init__()
        self.d_model = d_model
        self.mv_downsample_factor = mv_downsample_factor
        self.mv_mode = mv_mode
        self.gop_aggregation = gop_aggregation

        # Motion vector projection
        # Input channels: 2 for raw mode, 1 for magnitude mode
        mv_input_channels = 2 if mv_mode == "raw" else 1
        mv_hidden_dim = min(128, d_model // 8)  # Smaller intermediate dim to save memory
        self.mv_proj_in = nn.Linear(mv_input_channels, mv_hidden_dim, bias=True)
        self.mv_proj_gelu = nn.GELU()
        self.mv_proj_out = nn.Linear(mv_hidden_dim, d_model, bias=True)
        
        # Optional positional embedding to distinguish I-frame from motion frames
        self.frame_type_embedding = nn.Parameter(torch.randn(2, d_model) * 0.02)  # [2, C] for [iframe, motion]

        # Aggregation-specific modules
        if gop_aggregation == "cross_attn":
            # Cross-attention: I-frame as query, MVs as key/value
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = MambaRMSNorm(d_model)
        elif gop_aggregation == "add":
            # For addition, we need to aggregate MV features spatially
            # Use Mamba to process MV sequence, then pool and add to I-frame
            self.mv_mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bimamba=use_bidirectional,
                )
                for _ in range(n_layers)
            ])
            self.mv_pre_norm = MambaRMSNorm(d_model)
            self.mv_post_norm = MambaRMSNorm(d_model)

        # GOP-level Mamba blocks for temporal compression (used for concat and post-processing)
        if gop_aggregation == "concat":
            self.gop_mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bimamba=use_bidirectional,
                )
                for _ in range(n_layers)
            ])

            # Layer norms
            self.pre_norm = MambaRMSNorm(d_model)
            self.post_norm = MambaRMSNorm(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        iframe_features: torch.Tensor,  # [B, num_gops, H, W, C]
        motion_vectors: torch.Tensor,    # [T, Hmv, Wmv, 2] or [B, T, Hmv, Wmv, 2]
        gop_boundaries: list,            # List of (start, end) for each GOP
        device: torch.device = None,
    ):
        """
        Compress features within each GOP by treating I-frame as a query token.
        
        Args:
            iframe_features: I-frame visual features for each GOP
            motion_vectors: Motion vectors for all frames
            gop_boundaries: GOP boundary indices
            
        Returns:
            gop_features: Compressed features for each GOP [B, num_gops, H, W, C]
        """
        B, num_gops, H, W, C = iframe_features.shape
        
        if device is None:
            device = iframe_features.device
            
        # Handle motion vectors shape - add batch dimension if not present
        if motion_vectors.dim() == 4:  # [T, Hmv, Wmv, 2]
            motion_vectors = motion_vectors.unsqueeze(0)  # [1, T, Hmv, Wmv, 2]
            # Expand to match batch size
            motion_vectors = motion_vectors.expand(B, -1, -1, -1, -1)
            
        gop_compressed = []
        
        for gop_idx in range(num_gops):
            # Get I-frame features for this GOP
            iframe_feat = iframe_features[:, gop_idx]  # [B, H, W, C]
            
            # Get motion vectors for this GOP
            start, end = gop_boundaries[gop_idx]
            gop_mvs = motion_vectors[:, start:end]  # [B, gop_len, Hmv, Wmv, 2]
            
            if gop_mvs.shape[1] == 0:  # Handle edge case of empty GOP
                gop_compressed.append(iframe_feat)
                continue
            
            # Process motion vectors
            B, gop_len, Hmv, Wmv, _ = gop_mvs.shape
            
            # Optionally downsample motion vectors to reduce memory
            # WARNING: Motion vectors at 96x96 are already sparse - downsampling may lose important motion details
            # Only use downsampling if absolutely necessary for memory constraints
            if self.mv_downsample_factor > 1:
                # Calculate target dimensions and appropriate kernel sizes
                target_h = max(1, Hmv // self.mv_downsample_factor)
                target_w = max(1, Wmv // self.mv_downsample_factor)

                # Use adaptive average pooling for flexible downsampling
                # This handles any input size and downsample factor gracefully
                # Reshape for pooling: [B*gop_len, 2, Hmv, Wmv]
                gop_mvs_for_pool = gop_mvs.permute(0, 1, 4, 2, 3).reshape(B * gop_len, 2, Hmv, Wmv)

                # Adaptive pooling automatically determines kernel size based on input/output ratio
                gop_mvs_pooled = torch.nn.functional.adaptive_avg_pool2d(
                    gop_mvs_for_pool,
                    output_size=(target_h, target_w)
                )

                # Reshape back: [B, gop_len, target_h, target_w, 2]
                gop_mvs = gop_mvs_pooled.reshape(B, gop_len, 2, target_h, target_w).permute(0, 1, 3, 4, 2)
                Hmv, Wmv = target_h, target_w

            # Convert motion vectors based on mode
            if self.mv_mode == "magnitude":
                # Convert 2-channel MVs to magnitude: sqrt(mv_x^2 + mv_y^2)
                gop_mvs = torch.sqrt(gop_mvs[..., 0] ** 2 + gop_mvs[..., 1] ** 2).unsqueeze(-1)  # [B, gop_len, Hmv, Wmv, 1]
            # else: keep raw 2-channel motion vectors

            # Process motion vectors in smaller chunks to reduce memory usage
            # Process each frame in the GOP separately
            mv_features_list = []
            for frame_idx in range(gop_len):
                frame_mv = gop_mvs[:, frame_idx]  # [B, Hmv, Wmv, 1 or 2]
                mv_channels = frame_mv.shape[-1]
                frame_mv_flat = frame_mv.reshape(B * Hmv * Wmv, mv_channels)

                frame_features = self.mv_proj_in(frame_mv_flat)  # [B*Hmv*Wmv, mv_hidden_dim]
                frame_features = self.mv_proj_gelu(frame_features)
                frame_features = self.mv_proj_out(frame_features)  # [B*Hmv*Wmv, C]

                # Reshape to [B, C, Hmv, Wmv]
                frame_features = frame_features.reshape(B, Hmv, Wmv, -1).permute(0, 3, 1, 2)
                mv_features_list.append(frame_features)
            
            # Stack all frames: [B*gop_len, C, Hmv, Wmv]
            mv_features = torch.cat(mv_features_list, dim=0)
            
            # Keep motion vectors at native resolution!
            # Reshape: [B, gop_len, Hmv, Wmv, C]
            mv_features = mv_features.reshape(B, gop_len, C, Hmv, Wmv).permute(0, 1, 3, 4, 2)
            
            # Add frame type embeddings
            mv_features = mv_features + self.frame_type_embedding[1:2]  # Motion frame embedding
            iframe_feat_emb = iframe_feat + self.frame_type_embedding[0:1]  # I-frame embedding

            # Flatten I-frame spatial dimensions: [B, H*W, C]
            iframe_seq = iframe_feat_emb.reshape(B, H * W, C)

            # Flatten motion vectors: [B, gop_len*Hmv*Wmv, C]
            mv_seq = mv_features.reshape(B, gop_len * Hmv * Wmv, C)

            # Apply aggregation method
            if self.gop_aggregation == "concat":
                # Original method: Concatenate and process through Mamba
                # [B, H*W, C] + [B, gop_len*Hmv*Wmv, C] -> [B, H*W + gop_len*Hmv*Wmv, C]
                combined_seq = torch.cat([iframe_seq, mv_seq], dim=1)

                # Apply GOP-level temporal Mamba
                x = self.pre_norm(combined_seq)
                for mamba_layer in self.gop_mamba_layers:
                    residual = x
                    x = mamba_layer(x)
                    x = self.dropout(x)
                    x = residual + x
                x = self.post_norm(x)

                # Extract the I-frame features (first H*W positions)
                iframe_output = x[:, :H*W, :]  # [B, H*W, C]
                iframe_output = iframe_output.reshape(B, H, W, C)

            elif self.gop_aggregation == "cross_attn":
                # Cross-attention: I-frame as query, MVs as key/value
                # iframe_seq: [B, H*W, C], mv_seq: [B, gop_len*Hmv*Wmv, C]
                attn_out, _ = self.cross_attn(
                    query=iframe_seq,
                    key=mv_seq,
                    value=mv_seq,
                )
                # Residual connection and norm
                iframe_output = iframe_seq + self.dropout(attn_out)
                iframe_output = self.attn_norm(iframe_output)
                iframe_output = iframe_output.reshape(B, H, W, C)

            elif self.gop_aggregation == "add":
                # Addition: Process MVs through Mamba, pool, and add to I-frame
                # Process MV features
                x = self.mv_pre_norm(mv_seq)
                for mamba_layer in self.mv_mamba_layers:
                    residual = x
                    x = mamba_layer(x)
                    x = self.dropout(x)
                    x = residual + x
                x = self.mv_post_norm(x)

                # Global average pooling over MV sequence: [B, gop_len*Hmv*Wmv, C] -> [B, 1, C]
                mv_pooled = x.mean(dim=1, keepdim=True)  # [B, 1, C]

                # Broadcast and add to I-frame features
                # iframe_seq: [B, H*W, C], mv_pooled: [B, 1, C]
                iframe_output = iframe_seq + mv_pooled
                iframe_output = iframe_output.reshape(B, H, W, C)

            else:
                raise ValueError(f"Unknown GOP aggregation method: {self.gop_aggregation}")

            gop_compressed.append(iframe_output)
        
        # Stack all GOP features
        gop_features = torch.stack(gop_compressed, dim=1)  # [B, num_gops, H, W, C]
        
        return gop_features
    
    def forward_with_compression_ratio(
        self,
        iframe_features: torch.Tensor,
        motion_vectors: torch.Tensor,
        gop_boundaries: list,
        target_ratio: float = 0.5,
    ):
        """
        Forward with adaptive spatial downsampling based on target compression ratio.
        
        Args:
            target_ratio: Target compression ratio (0.5 means compress to half)
            
        Returns:
            Compressed GOP features with reduced spatial dimensions
        """
        # First do normal GOP compression
        gop_features = self.forward(iframe_features, motion_vectors, gop_boundaries)
        
        B, num_gops, H, W, C = gop_features.shape
        
        # Calculate target spatial dimensions
        new_H = int(H * (target_ratio ** 0.5))
        new_W = int(W * (target_ratio ** 0.5))
        
        # Reshape for interpolation: [B*num_gops, C, H, W]
        gop_features = gop_features.permute(0, 1, 4, 2, 3).reshape(B * num_gops, C, H, W)
        
        # Spatial downsampling
        gop_features = torch.nn.functional.interpolate(
            gop_features, size=(new_H, new_W), mode='bilinear', align_corners=False
        )
        
        # Reshape back: [B, num_gops, new_H, new_W, C]
        gop_features = gop_features.reshape(B, num_gops, C, new_H, new_W)
        gop_features = gop_features.permute(0, 1, 3, 4, 2)
        
        return gop_features
