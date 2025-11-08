import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional



@torch.jit.script
def _local_attention_mask(seq_len: int, local_k: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.bool):
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

        # Compute all projections in parallel
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # More efficient reshape using contiguous tensors
        def reshape(t):
            return t.view(B, n, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        q, k, v = map(reshape, (q, k, v))

        # Optimize matrix multiplication
        scores = torch.baddbmm(
            torch.empty(B, self.num_heads, n, n, dtype=q.dtype, device=device),
            q,
            k.transpose(-2, -1),
            beta=0.0,
            alpha=self.scale
        )

        # Base mask
        base_mask = torch.ones((n, n), dtype=torch.bool, device=device) if attn_mask is None else attn_mask.to(device)
        local_mask = _local_attention_mask(n, self.local_k, device=device) if self.local_heads > 0 else None

        # Create per-head mask
        masks = []
        for h in range(self.num_heads):
            if h < self.global_heads:
                masks.append(base_mask)
            else:
                masks.append(base_mask & local_mask)
        masks = torch.stack(masks, dim=0).unsqueeze(0)  # (1, num_heads, n, n)

        scores = scores.masked_fill(~masks, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
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
    """
    Encoder-only transformer for multi-label classification.
      - Input: (B, n, input_dim)
      - Output: (B, n, n_outputs)
    """
    def __init__(
        self,
        input_dim: int = 232,
        embed_dim: int = 256,
        num_heads: int = 8,
        local_k: int = 5,
        n_blocks: int = 6,
        global_heads: int = 1,
        dropout: float = 0.2,
        n_outputs: int = 37
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # Pre-attention FFN (only once)
        self.pre_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

        # Encoder blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, local_k, global_heads, dropout)
            for _ in range(n_blocks)
        ])

        self.final_ln = nn.LayerNorm(embed_dim)

        # Projection for multi-label classification
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_outputs),
        )

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        """
        x: (B, n, input_dim)
        Returns: (B, n, n_outputs)
        """
        B, n, _ = x.shape
        x = self.input_proj(x)
        x = self.pre_ffn(x)

        # No positional embeddings here
        x = self.dropout(x)

        # Encoder blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Final normalization
        x = self.final_ln(x)

        # Project to output dimension
        logits = self.classifier(x)
        return logits
