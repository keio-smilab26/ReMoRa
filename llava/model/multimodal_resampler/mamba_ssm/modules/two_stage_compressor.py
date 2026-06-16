import torch
from torch import nn
import torch.nn.functional as F
from .gop_compressor import GOPCompressor
from .mamba_compressor import MambaCompressor
from .mamba_simple import Mamba
from pathlib import Path


class CNNMotionVectorRefiner(nn.Module):
    """Simple CNN-based refiner used as a lightweight baseline."""

    def __init__(self, hidden_dim: int = 64, num_blocks: int = 4) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
        self.net = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1)

    def forward(self, motion_vectors: torch.Tensor) -> torch.Tensor:
        x = self.net(motion_vectors)
        return self.head(x)


class TransformerMotionVectorRefiner(nn.Module):
    """Transformer encoder that models spatial context across the motion vector grid."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_multiplier: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Conv2d(2, hidden_dim, kernel_size=1)
        feedforward_dim = max(int(hidden_dim * ffn_multiplier), hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, motion_vectors: torch.Tensor) -> torch.Tensor:
        B, _, H, W = motion_vectors.shape
        x = self.stem(motion_vectors)
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        return self.head(x)




class MambaMotionVectorRefiner(nn.Module):
    """Mamba-based motion vector refiner using bidirectional processing."""
    def __init__(self, hidden_dim: int = 128, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()

        # Input projection: 2 channels (mv_x, mv_y) to hidden_dim
        self.stem = nn.Linear(2, hidden_dim)

        # Single bidirectional Mamba block
        self.mamba = Mamba(
            d_model=hidden_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            bimamba=True,  # Enable bidirectional processing
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection: hidden_dim back to 2 channels
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, motion_vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            motion_vectors: (B, 2, H, W) tensor of motion vectors
        Returns:
            refined_vectors: (B, 2, H, W) tensor of refined motion vectors
        """
        B, C, H, W = motion_vectors.shape

        # Reshape to (B, H*W, 2) for sequence processing
        x = motion_vectors.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Project to hidden dimension
        x = self.stem(x)

        # Apply bidirectional Mamba
        residual = x
        x = self.mamba(x)
        x = residual + x  # Residual connection

        # Normalize
        x = self.norm(x)

        # Project back to 2 channels
        x = self.head(x)

        # Reshape back to (B, 2, H, W)
        x = x.reshape(B, H, W, 2).permute(0, 3, 1, 2)

        return x


class TwoStageCompressor(nn.Module):
    """
    Two-stage video compression:
    Stage 0 (optional): Motion vector refinement
    Stage 1: GOP-level compression with motion vector fusion
    Stage 2: Global MambaCompressor across GOP features
    """

    def __init__(
        self,
        d_model: int,
        # Stage 1 params
        gop_n_layers: int = 2,
        gop_d_state: int = 16,
        gop_d_conv: int = 4,
        gop_expand: int = 2,
        gop_aggregation: str = "concat",  # "concat", "cross_attn", or "add"
        mv_downsample_factor: int = 1,  # Downsample factor for motion vectors
        mv_mode: str = "raw",  # "raw" for 2-channel MVs, "magnitude" for 1-channel magnitude
        # Stage 2 params
        global_n_layers: int = 1,
        global_d_state: int = 16,
        global_d_conv: int = 4,
        global_expand: int = 2,
        # Shared params
        use_bidirectional: bool = True,
        query_pos: str = "inter",
        fp32: bool = True,
        # MV refiner params
        use_mv_refiner: bool = False,
        mv_refiner_checkpoint: str = None,
        mv_refiner_hidden_dim: int = 64,
        mv_refiner_d_state: int = 16,
        mv_refiner_d_conv: int = 4,
        mv_refiner_expand: int = 2,
        mv_refiner_type: str = "mamba",
        mv_refiner_num_layers: int = 2,
        mv_refiner_num_heads: int = 4,
        mv_refiner_ffn_multiplier: float = 4.0,
        mv_refiner_dropout: float = 0.1,
        mv_refiner_predict_residual: bool = True,
        # Auxiliary loss params
        mv_aux_loss_weight: float = 0.0,  # Weight for MV auxiliary loss during training
    ):
        super().__init__()

        # Store auxiliary loss parameters
        self.mv_aux_loss_weight = mv_aux_loss_weight
        self.mv_refiner_predict_residual = mv_refiner_predict_residual

        # Stage 0 (optional): Motion Vector Refiner
        self.use_mv_refiner = use_mv_refiner
        self.mv_refiner = None
        self.mv_refiner_type = (mv_refiner_type or "mamba").lower()
        if use_mv_refiner:
            if self.mv_refiner_type == "mamba":
                self.mv_refiner = MambaMotionVectorRefiner(
                    hidden_dim=mv_refiner_hidden_dim,
                    d_state=mv_refiner_d_state,
                    d_conv=mv_refiner_d_conv,
                    expand=mv_refiner_expand,
                )
            elif self.mv_refiner_type == "transformer":
                self.mv_refiner = TransformerMotionVectorRefiner(
                    hidden_dim=mv_refiner_hidden_dim,
                    num_layers=mv_refiner_num_layers,
                    num_heads=mv_refiner_num_heads,
                    ffn_multiplier=mv_refiner_ffn_multiplier,
                    dropout=mv_refiner_dropout,
                )
            elif self.mv_refiner_type == "cnn":
                self.mv_refiner = CNNMotionVectorRefiner(
                    hidden_dim=mv_refiner_hidden_dim,
                    num_blocks=mv_refiner_num_layers,
                )
            else:
                raise ValueError(f"Unsupported mv_refiner_type: {mv_refiner_type}")

            # Load checkpoint if provided
            if mv_refiner_checkpoint:
                checkpoint_path = Path(mv_refiner_checkpoint)
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                    if "model_state_dict" in checkpoint:
                        self.mv_refiner.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        self.mv_refiner.load_state_dict(checkpoint)
                    print(f"Loaded MV refiner checkpoint from {checkpoint_path}")
                else:
                    print(f"Warning: MV refiner checkpoint not found at {checkpoint_path}")

            # Note: Freezing/unfreezing is handled in the training script (train.py)
            # based on mm_tunable_parts and mv_refiner_freeze settings

        # Stage 1: GOP-level compressor
        self.gop_compressor = GOPCompressor(
            d_model=d_model,
            n_layers=gop_n_layers,
            d_state=gop_d_state,
            d_conv=gop_d_conv,
            expand=gop_expand,
            use_bidirectional=use_bidirectional,
            mv_downsample_factor=mv_downsample_factor,
            mv_mode=mv_mode,  # Pass through motion vector mode
            gop_aggregation=gop_aggregation,  # Pass through GOP aggregation method
        )

        # Stage 2: Global MambaCompressor
        self.global_compressor = MambaCompressor(
            d_model=d_model,
            n_layer=global_n_layers,
            d_state=global_d_state,
            d_conv=global_d_conv,
            expand=global_expand,
            bimamba=use_bidirectional,
            query_pos=query_pos,
            fp32=fp32,
        )
        
        # Optional: learnable transition between stages
        self.stage_transition = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
    
    def forward(
        self,
        space_time_tokens: torch.Tensor,  # [B, F, H, W, C] - I-frame features
        hidden_states: torch.Tensor,       # [B, F, L, C] - query tokens
        motion_vectors: torch.Tensor = None,  # [B, T, Hmv, Wmv, 2]
        gop_boundaries: list = None,
        i_frame_indices: torch.Tensor = None,
        optical_flow: torch.Tensor = None,  # [B, T, Hmv, Wmv, 2] - optional optical flow for aux loss
        flow_visibility: torch.Tensor = None,  # [B, T, Hmv, Wmv, 1] - optional visibility mask
        return_aux_loss: bool = False,  # Whether to return auxiliary loss
    ):
        """
        Two-stage compression pipeline with optional MV refinement.

        Args:
            space_time_tokens: Visual features from I-frames
            hidden_states: Query tokens for compression
            motion_vectors: Motion vectors for all frames
            gop_boundaries: GOP boundary information
            i_frame_indices: Indices of I-frames
            optical_flow: Optional optical flow ground truth for auxiliary loss
            flow_visibility: Optional visibility mask for optical flow
            return_aux_loss: Whether to compute and return auxiliary loss

        Returns:
            If return_aux_loss is False:
                Compressed features after two-stage processing
            If return_aux_loss is True:
                (compressed_features, aux_loss_dict)
        """
        B, F, H, W, C = space_time_tokens.shape
        aux_loss_dict = {}

        # Stage 0 (optional): Motion Vector Refinement
        refined_motion_vectors = motion_vectors
        if motion_vectors is not None and self.use_mv_refiner and self.mv_refiner is not None:
            # Get original shape
            if motion_vectors.dim() == 4:  # [T, Hmv, Wmv, 2]
                motion_vectors = motion_vectors.unsqueeze(0)  # [1, T, Hmv, Wmv, 2]

            B_mv, T, Hmv, Wmv, _ = motion_vectors.shape

            # Reshape for batch processing: [B*T, 2, Hmv, Wmv]
            # contiguous: the permuted view has strides some CUDA kernels reject
            # (misaligned address) during backward
            mv_flat = motion_vectors.reshape(B_mv * T, Hmv, Wmv, 2).permute(0, 3, 1, 2).contiguous()

            # Apply refiner. The refiner builds a [B*T, hidden, Hmv, Wmv] feature
            # map over ALL frames at once; for long videos with a large hidden_dim
            # this single tensor OOMs (~17GB at hidden=256, B*T~2000). Process the
            # frames in chunks so the peak tensor is bounded by the chunk size,
            # independent of frame count / hidden_dim. During training also
            # gradient-checkpoint each chunk so intermediate activations are not
            # all retained for backward (refiner has no dropout, so recompute is
            # exact). The op is per-frame, so chunking is mathematically identical.
            n_frames = mv_flat.shape[0]
            CHUNK = 32
            # Gradient-checkpoint only when the refiner is actually trainable;
            # if it is frozen (e.g. pretrain-then-freeze runs), checkpointing is
            # wrong/wasteful, so run it under no_grad. Chunking is still needed in
            # both cases to bound the [chunk, hidden, H, W] peak tensor.
            refiner_trainable = any(p.requires_grad for p in self.mv_refiner.parameters())
            if n_frames > CHUNK:
                parts = []
                for i in range(0, n_frames, CHUNK):
                    seg = mv_flat[i:i + CHUNK]
                    if self.training and refiner_trainable:
                        seg_out = torch.utils.checkpoint.checkpoint(
                            self.mv_refiner, seg, use_reentrant=False
                        )
                    elif refiner_trainable:
                        seg_out = self.mv_refiner(seg)
                    else:
                        with torch.no_grad():
                            seg_out = self.mv_refiner(seg)
                    parts.append(seg_out)
                refined_mv_flat = torch.cat(parts, dim=0)
            elif refiner_trainable:
                refined_mv_flat = self.mv_refiner(mv_flat)
            else:
                with torch.no_grad():
                    refined_mv_flat = self.mv_refiner(mv_flat)

            # Add residual if configured
            if self.mv_refiner_predict_residual:
                refined_mv_flat = refined_mv_flat + mv_flat

            # Reshape back: [B, T, Hmv, Wmv, 2]
            refined_motion_vectors = refined_mv_flat.permute(0, 2, 3, 1).contiguous().reshape(B_mv, T, Hmv, Wmv, 2)

            # Compute auxiliary loss if requested and optical flow is provided
            if return_aux_loss and optical_flow is not None and self.mv_aux_loss_weight > 0:
                # Ensure optical flow has same shape
                if optical_flow.shape != refined_motion_vectors.shape:
                    # Reshape optical flow if needed
                    if optical_flow.dim() == 4:
                        optical_flow = optical_flow.unsqueeze(0).expand(B_mv, -1, -1, -1, -1)

                # Compute L2 loss between refined MVs and optical flow
                if flow_visibility is not None:
                    # Masked L2 loss
                    if flow_visibility.dim() == 4:
                        flow_visibility = flow_visibility.unsqueeze(0).expand(B_mv, -1, -1, -1, -1)

                    diff = (refined_motion_vectors - optical_flow) ** 2
                    weighted_diff = diff * flow_visibility
                    # Average over spatial dimensions and frames
                    mv_aux_loss = weighted_diff.sum() / (flow_visibility.sum() * 2 + 1e-6)  # *2 for x,y components
                else:
                    # Simple L2 loss
                    mv_aux_loss = F.mse_loss(refined_motion_vectors, optical_flow)

                aux_loss_dict['mv_refiner_loss'] = mv_aux_loss * self.mv_aux_loss_weight

            # Squeeze batch dimension if it was added
            if B_mv == 1 and motion_vectors.shape[0] == 1:
                refined_motion_vectors = refined_motion_vectors.squeeze(0)

        # Stage 1: GOP-level compression
        if refined_motion_vectors is not None and gop_boundaries is not None:
            # Perform GOP compression with refined motion vectors
            gop_features = self.gop_compressor(
                iframe_features=space_time_tokens,
                motion_vectors=refined_motion_vectors,
                gop_boundaries=gop_boundaries,
            )

            # Apply transition layer
            gop_features = self.stage_transition(gop_features)
        else:
            # If no motion vectors, use space_time_tokens directly
            gop_features = space_time_tokens

        # Stage 2: Global compression across GOPs
        compressed_features = self.global_compressor(
            space_time_tokens=gop_features,
            hidden_states=hidden_states,
        )

        if return_aux_loss:
            return compressed_features, aux_loss_dict
        return compressed_features
