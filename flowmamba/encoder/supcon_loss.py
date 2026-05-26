"""Supervised Contrastive Loss (SupCon).

Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    For each anchor in the batch, every other sample sharing the same label
    is treated as a *positive* and all remaining samples as *negatives*.
    The loss pulls positive-pair embeddings together and pushes negative
    pairs apart using temperature-scaled cosine similarity.

    Args:
        temperature: Scaling factor for the similarity logits.
            Lower values sharpen the distribution (default ``0.07``).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            embeddings: L2-normalized embeddings of shape ``(batch_size, embedding_dim)``.
            labels: Integer class labels of shape ``(batch_size,)``.

        Returns:
            Scalar loss averaged over all valid anchors.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity matrix: (B, B)
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask: same-class pairs (excluding self)
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Remove self-contrast from both mask and logits
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask.masked_fill(self_mask, 0.0)

        # Count positives per anchor; skip anchors with no positives
        positives_count = positive_mask.sum(dim=1)  # (B,)
        valid_anchors = positives_count > 0

        if not valid_anchors.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # For numerical stability, subtract max logit per row
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Mask out self-similarity for the denominator
        logits = logits.masked_fill(self_mask, float("-inf"))

        # Log-sum-exp over all negatives + positives (denominator)
        exp_logits = torch.exp(logits)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Log-prob for positive pairs
        log_prob = logits - log_sum_exp  # (B, B)

        # Average log-prob over positive pairs for each anchor
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positives_count.clamp(min=1)

        # Average loss over valid anchors
        loss = -mean_log_prob[valid_anchors].mean()

        return loss
