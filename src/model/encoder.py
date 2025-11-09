import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



@torch.jit.script
def _local_attention_mask(seq_len: int, local_k: int, device: Optional[torch.device] = None):
    """Mask where True = allowed; each token attends to ±local_k neighbors."""
    idx = torch.arange(seq_len, device=device)
    i = idx.unsqueeze(1)
    j = idx.unsqueeze(0)
    return (i - j).abs() <= local_k


class LocalGlobalMultiheadAttention(nn.Module):
    """
    Multi-head attention where:
      - 1 head is global (full attention)
      - others are local (±local_k)
    """
    def __init__(self, embed_dim: int, num_heads: int, local_k: int,
                 global_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.global_heads = global_heads
        self.local_heads = num_heads - global_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.local_k = local_k

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        B, n, _ = x.shape
        device = x.device

        # Compute projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (B, num_heads, n, head_dim)
        def reshape(t):
            return t.view(B, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k, v = map(reshape, (q, k, v))

        # Attention scores: (B, num_heads, n, n)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # --- Build masks ---
        # Base mask for local/global heads
        base_mask = torch.ones((n, n), dtype=torch.bool, device=device)
        if self.local_heads > 0:
            local_mask = _local_attention_mask(n, self.local_k, device=device)
            masks = []
            for h in range(self.num_heads):
                if h < self.global_heads:
                    masks.append(base_mask)
                else:
                    masks.append(base_mask & local_mask)
            base_mask = torch.stack(masks, dim=0).unsqueeze(0)  # (1, num_heads, n, n)
        else:
            base_mask = base_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, n)

        # Batch padding mask: (B, n) -> (B, 1, 1, n)
        if attn_mask is not None:
            padding_mask = attn_mask[:, None, None, :]  # True = valid token
            base_mask = base_mask & padding_mask  # broadcast to (B, num_heads, n, n)
            base_mask = base_mask.expand(B, -1, -1, -1)  # make batch dimension explicit
        else:
            base_mask = base_mask.expand(B, -1, -1, -1)

        # Apply mask
        scores = scores.masked_fill(~base_mask, float('-inf'))

        # Softmax and dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Attention output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, n, self.embed_dim)
        return self.out_proj(out)


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer encoder block with attention + FFN + dropout + residual + layernorm."""
    def __init__(self, embed_dim: int, num_heads: int, local_k: int,
                 global_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.attn = LocalGlobalMultiheadAttention(embed_dim, num_heads, local_k, global_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # Multi-head attention with residual
        attn_out = self.attn(x, attn_mask)
        x = x + self.dropout1(attn_out)
        x = self.ln1(x)

        # Feed-forward network with residual
        ffn_out = self.ffn(x)
        x = x + self.dropout2(ffn_out)
        x = self.ln2(x)
        return x


class MABeEncoder(nn.Module):
    def __init__(self,
                 input_dim=232,
                 embed_dim=256,
                 num_heads=4,
                 local_k=10,
                 n_blocks=6,
                 global_heads=1,
                 dropout=0.2,
                 n_outputs=37):
        super().__init__()
        self.embed_dim = embed_dim

        # Split input into 4 parts → FFN → concat → LayerNorm
        part_dim = input_dim // 4
        self.part_ffns = nn.ModuleList([
            nn.Linear(part_dim, embed_dim // 4) for _ in range(4)
        ])
        self.merge_norm = nn.LayerNorm(embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, local_k, global_heads, dropout)
            for _ in range(n_blocks)
        ])
        self.final_ln = nn.LayerNorm(embed_dim)

        # 🔹 Shared classifier applied to each time step
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_outputs),
        )

    def forward(self, x, attn_mask=None):
        B, n, D = x.shape
        # Split input into 4 parts and process each separately
        parts = torch.chunk(x, 4, dim=-1)  # 4 tensors of shape (B, n, D/4)
        processed_parts = [ffn(p) for ffn, p in zip(self.part_ffns, parts)]
        x = torch.cat(processed_parts, dim=-1)  # (B, n, embed_dim)
        x = self.merge_norm(x)

        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Normalize final features
        x = self.final_ln(x)

        # 🔹 Shared classifier across all time steps
        logits = self.classifier(x)  # (B, n, n_outputs)

        return logits
