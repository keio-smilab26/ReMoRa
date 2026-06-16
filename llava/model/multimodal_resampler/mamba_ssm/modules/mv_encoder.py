import torch
from torch import nn

# Reuse the simple Mamba used by the compressor
from .mamba_simple import Mamba


class MotionVectorEncoder(nn.Module):
    """
    Lightweight encoder for GOP motion vectors.

    - Embeds 2-channel motion vectors with a 1x1 conv (very cheap).
    - Bilinearly resizes to the visual token grid (H=W=num_patches_per_side).
    - Runs a temporal Mamba scan across frames for each spatial location.
    - Picks outputs at I-frame indices and projects to model dim for fusion.

    Shapes:
      mv_seq: [T, Hmv, Wmv, 2]
      returns: [Fi, Htok, Wtok, d_model]
    """

    def __init__(
        self,
        d_model: int,
        embed_dim: int = 64,
        d_state: int = 8,
        d_conv: int = 2,
        expand: int = 1,
        n_layer: int = 1,
    ) -> None:
        super().__init__()
        self.embed = nn.Conv2d(2, embed_dim, kernel_size=1, stride=1, padding=0)
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bimamba=False,
                )
                for _ in range(n_layer)
            ]
        )
        self.proj = nn.Linear(embed_dim, d_model, bias=True)

    def forward(
        self,
        mv_seq: torch.Tensor,
        i_indices: torch.Tensor,
        out_hw: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        """
        Args:
          mv_seq: [T, Hmv, Wmv, 2]
          i_indices: [Fi] (indices into T for I-frames)
          out_hw: output token grid size (H=W)

        Returns:
          fused: [Fi, out_hw, out_hw, d_model]
        """
        if mv_seq is None or i_indices is None or mv_seq.numel() == 0:
            return None

        if device is None:
            device = mv_seq.device
        if dtype is None:
            dtype = torch.float32

        # [T, Hmv, Wmv, 2] -> [T, 2, Hmv, Wmv]
        mv = mv_seq.to(device)
        mv = mv.permute(0, 3, 1, 2)
        T = mv.shape[0]

        # Embed per-frame: [T, 2, Hmv, Wmv] -> [T, C, Hmv, Wmv]
        mv = self.embed(mv)

        # Resize to token grid [out_hw, out_hw]
        mv = torch.nn.functional.interpolate(
            mv, size=(out_hw, out_hw), mode="bilinear", align_corners=False
        )  # [T, C, Htok, Wtok]

        # Prepare for temporal scan: collapse spatial to batch
        C, Htok, Wtok = mv.shape[1], mv.shape[2], mv.shape[3]
        mv = mv.permute(2, 3, 0, 1).contiguous()  # [Htok, Wtok, T, C]
        mv = mv.view(Htok * Wtok, T, C)  # [HW, T, C]

        x = mv
        # Run n_layer temporal Mamba scans
        for layer in self.mamba_layers:
            x = layer(x)

        # Gather I-frame outputs along time
        # x: [HW, T, C], i_indices: [Fi]
        Fi = int(i_indices.numel())
        # Clamp indices to valid range [0, T-1]
        i_indices_clamped = torch.clamp(i_indices.to(x.device, dtype=torch.long), min=0, max=T-1)
        gather_idx = i_indices_clamped.view(1, Fi, 1).expand(x.shape[0], Fi, x.shape[2])
        x_i = torch.gather(x, dim=1, index=gather_idx)  # [HW, Fi, C]

        # Reshape back to [Fi, Htok, Wtok, C]
        x_i = x_i.view(Htok, Wtok, Fi, C).permute(2, 0, 1, 3).contiguous()

        # Project to model dimension
        fused = self.proj(x_i)
        return fused  # [Fi, Htok, Wtok, d_model]

