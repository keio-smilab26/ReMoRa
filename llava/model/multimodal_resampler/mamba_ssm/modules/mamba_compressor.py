import torch

from .mamba_simple import Mamba
from torch import nn


class Attention(nn.Module):
    def __init__(
        self,
        d_model,
        expand=2,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.expand = expand

        dim = d_model * expand
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.in_proj = nn.Linear(d_model, dim, bias=True)

        self.out_proj = nn.Linear(dim, d_model, bias=True)

    def forward(self, x):
        x = self.in_proj(x)

        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = self.out_proj(x)
        return x


class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        layer_idx,
        use_norm=True,
        use_res=True,
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba=True,
        mixer_type="mamba",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_norm = use_norm
        self.use_res = use_res
        if use_norm:
            self.norm = MambaRMSNorm(d_model)
        if mixer_type == "mamba":
            self.mixer = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba=bimamba,
            )
        elif mixer_type == "attention":
            self.mixer = Attention(d_model=d_model, expand=expand)

    def forward(self, hidden_states):
        residual = hidden_states
        if self.use_norm:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        if self.use_res:
            hidden_states = residual + hidden_states
        return hidden_states


class MambaCompressor(nn.Module):
    def __init__(
        self,
        d_model,
        n_layer,
        use_norm=True,
        use_res=True,
        fp32=True,
        query_pos="inter",
        d_state=16,
        d_conv=4,
        expand=2,
        bimamba=True,
        multi_scale=True,
        mixer_type="mamba",
    ):
        super().__init__()
        self.multi_scale = multi_scale
        self.fp32 = fp32
        self.query_pos = query_pos
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    idx,
                    use_norm=use_norm,
                    use_res=use_res,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bimamba=bimamba,
                    mixer_type=mixer_type,
                )
                for idx in range(n_layer)
            ]
        )

        if fp32:
            self.layers.to(torch.float32)

    def forward(self, space_time_tokens, hidden_states):

        b, f, h, w, c = space_time_tokens.shape
        b, f, l, c = hidden_states.shape
        hidden_states = hidden_states.reshape(b, -1, c)
        n_query = hidden_states.shape[1]


        for mixer_block in self.layers:
            space_time_tokens = space_time_tokens.reshape(b, -1, c)
            if self.query_pos == "right":
                hidden_states = torch.cat((space_time_tokens, hidden_states), dim=1)
            elif self.query_pos == "inter":
                B = space_time_tokens.shape[0]
                S = space_time_tokens.shape[1]
                Q = hidden_states.shape[1]
                C = space_time_tokens.shape[2]
                total = S + Q
                import torch as _torch
                if Q == 0:
                    combined_tokens = space_time_tokens
                    mask = _torch.zeros(total, dtype=_torch.bool, device=hidden_states.device)
                    # Define idxs and space_pos for the Q=0 case
                    idxs = _torch.empty(0, dtype=_torch.long, device=hidden_states.device)
                    space_pos = _torch.arange(S, device=hidden_states.device, dtype=_torch.long)
                else:
                    # Deterministic, collision-free positions by binning S into Q+1 gaps
                    base = int(S // (Q + 1))
                    rem = int(S % (Q + 1))
                    cpu_idxs = []
                    pos_cpu = 0
                    for i_gap in range(Q):
                        s_i = base + (1 if i_gap < rem else 0)
                        pos_cpu += s_i
                        cpu_idxs.append(pos_cpu)
                        pos_cpu += 1  # account for placing a query
                    idxs = _torch.tensor(cpu_idxs, device=hidden_states.device, dtype=_torch.long)
                    # Build permutation: perm[output_pos] = input_pos
                    # Input: [space_tokens (0..S-1), query_tokens (S..S+Q-1)]
                    # Output: interleaved sequence
                    perm = _torch.empty(total, dtype=_torch.long, device=hidden_states.device)
                    
                    # Mark query positions in the output
                    is_query = _torch.zeros(total, dtype=_torch.bool, device=hidden_states.device)
                    is_query[idxs] = True
                    
                    # Space tokens: map them to their positions in the input
                    space_output_pos = _torch.arange(total, device=hidden_states.device)[~is_query]
                    space_pos = space_output_pos  # Store for later use when extracting space tokens
                    space_input_indices = _torch.arange(S, device=hidden_states.device, dtype=_torch.long)
                    perm[space_output_pos] = space_input_indices
                    
                    # Query tokens: map them to their positions in the input (S..S+Q-1)
                    query_input_indices = _torch.arange(S, S + Q, device=hidden_states.device, dtype=_torch.long)
                    perm[idxs] = query_input_indices
                    # One-run debug prints if enabled
                    import os as _os
                    if _os.environ.get("REMORA_DEBUG_INTERLEAVE", "0") == "1" and not hasattr(self, "_dbg_logged"):
                        try:
                            print(f"[Interleave DBG] B={int(B)} S={int(S)} Q={int(Q)} total={int(total)}")
                            print(f"[Interleave DBG] idxs_len={int(idxs.numel())} min={int(idxs.min().item())} max={int(idxs.max().item())}")
                            head = perm[:10].tolist()
                            tail = perm[-10:].tolist()
                            print(f"[Interleave DBG] perm_head={head} perm_tail={tail}")
                        except Exception:
                            pass
                        self._dbg_logged = True
                    tokens_concat = _torch.cat([space_time_tokens, hidden_states], dim=1)  # [B, S+Q, C]
                    combined_tokens = tokens_concat.index_select(1, perm)
                    # Build mask for downstream split
                    mask = is_query

            if self.fp32:
                dtype_prev = combined_tokens.dtype
                combined_tokens = combined_tokens.to(torch.float32)
            combined_tokens = mixer_block(combined_tokens)
            if self.fp32:
                combined_tokens = combined_tokens.to(dtype_prev)

            if self.query_pos == "right":
                hidden_states = combined_tokens[:, -n_query:, :]
                space_time_tokens = combined_tokens[:, :-n_query, :]
            elif self.query_pos == "inter":
                # Use index_select with explicit indices to avoid boolean gather asserts
                hidden_states = combined_tokens.index_select(1, idxs)
                space_time_tokens = combined_tokens.index_select(1, space_pos)

            if self.multi_scale:
                space_time_tokens = space_time_tokens.reshape(b, -1, h, w, c)
                space_time_tokens = space_time_tokens[:, ::2]

        hidden_states = hidden_states.reshape(b, -1, l, c)

        return hidden_states
