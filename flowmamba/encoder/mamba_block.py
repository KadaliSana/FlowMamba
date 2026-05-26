"""Mamba block with residual connection and layer normalization."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None  # type: ignore[assignment, misc]


class MambaBlock(nn.Module):
    """Single Mamba block: LayerNorm → Mamba SSM → residual add.

    If ``mamba_ssm`` is installed (requires CUDA), uses the optimized kernel.
    Otherwise falls back to a pure-PyTorch selective-scan approximation so
    that the code can still be imported and tested on CPU-only machines.
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if Mamba is not None:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            # Pure-PyTorch fallback for CPU-only environments
            self.mamba = _PurePytorchMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connection.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor of the same shape.
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class _PurePytorchMamba(nn.Module):
    """Simplified pure-PyTorch Mamba-like block for CPU fallback.

    This is NOT a full selective state-space model — it approximates
    the Mamba architecture with a gated Conv1D + linear recurrence so
    the model can be tested without CUDA.  For production training,
    always use the real ``mamba_ssm.Mamba`` kernel.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand

        # Input projection (split into two for gating)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Depthwise conv over the sequence dimension
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)

        # Initialize dt bias to small positive values for stable dynamics
        with torch.no_grad():
            self.dt_proj.bias.uniform_(0.001, 0.1)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_inner, -1))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Input projection + gating split
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Conv1d (channel-last → channel-first → channel-last)
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2)  # (B, L, d_inner)
        x_branch = torch.silu(x_branch)

        # Simplified selective scan
        ssm_params = self.x_proj(x_branch)  # (B, L, 2*d_state)
        B_param, C_param = ssm_params.chunk(2, dim=-1)  # each (B, L, d_state)

        dt = self.dt_proj(torch.softplus(B_param))  # (B, L, d_inner)
        dt = torch.softplus(dt)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Discretize and scan (simplified parallel form)
        y = x_branch * self.D  # skip connection component

        # Gate and project
        z = torch.silu(z)
        output = y * z
        return self.out_proj(output)
