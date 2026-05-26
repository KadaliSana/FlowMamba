"""FlowMamba Encoder — context-aware Mamba-based flow embedding model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import MambaBlock


class FlowMambaEncoder(nn.Module):
    """Context-aware encoder that maps flow feature sequences to 128-d embeddings.

    Architecture:
        Input projection (input_dim → d_model)
        → N × MambaBlock (with residual + LayerNorm)
        → Temporal mean pooling
        → Projection head (d_model → embedding_dim)
        → L2 normalization

    Input:  ``(batch, seq_len, input_dim)``  — e.g., ``(B, 16, 13)``
    Output: ``(batch, embedding_dim)``        — e.g., ``(B, 128)`` L2-normalized
    """

    def __init__(
        self,
        input_dim: int = 13,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding_dim = embedding_dim

        # Project raw features into model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stack of Mamba blocks
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Final layer norm before pooling
        self.final_norm = nn.LayerNorm(d_model)

        # Projection head: d_model → embedding_dim
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of flow sequences into L2-normalized embeddings.

        Args:
            x: Flow feature sequences of shape ``(batch, seq_len, input_dim)``.
            mask: Optional boolean mask of shape ``(batch, seq_len)`` where
                ``True`` indicates a valid (non-padded) position.

        Returns:
            L2-normalized embeddings of shape ``(batch, embedding_dim)``.
        """
        # Project input features
        h = self.input_proj(x)  # (B, L, d_model)

        # Pass through Mamba blocks
        for layer in self.layers:
            h = layer(h)  # (B, L, d_model)

        h = self.final_norm(h)  # (B, L, d_model)

        # Temporal mean pooling (mask-aware)
        if mask is not None:
            # Zero out padded positions before averaging
            mask_expanded = mask.unsqueeze(-1).float()  # (B, L, 1)
            h = h * mask_expanded
            lengths = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            pooled = h.sum(dim=1) / lengths  # (B, d_model)
        else:
            pooled = h.mean(dim=1)  # (B, d_model)

        # Project to embedding space
        embedding = self.projection_head(pooled)  # (B, embedding_dim)

        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    @classmethod
    def from_config(cls, config) -> FlowMambaEncoder:
        """Construct encoder from an ``EncoderConfig`` dataclass."""
        return cls(
            input_dim=config.input_dim,
            d_model=config.d_model,
            n_layers=config.n_layers,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
        )
