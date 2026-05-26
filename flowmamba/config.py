"""Centralized configuration for the FlowMamba pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json
import yaml


@dataclass
class EncoderConfig:
    """Configuration for the FlowMambaEncoder model."""

    input_dim: int = 13  # 9 numeric features + 4 protocol one-hot
    d_model: int = 64
    n_layers: int = 4
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    embedding_dim: int = 128
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for the training loop."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    epochs: int = 100
    window_size: int = 16
    val_split: float = 0.2
    supcon_temperature: float = 0.07
    lr_scheduler: str = "cosine"  # "cosine" or "step"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10  # print every N batches
    seed: int = 42
    num_workers: int = 4
    group_by: str = "src_ip"  # field to group flows into sequences


@dataclass
class FlowMambaConfig:
    """Top-level configuration container."""

    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FlowMambaConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def from_json(cls, path: str | Path) -> FlowMambaConfig:
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> FlowMambaConfig:
        encoder_cfg = EncoderConfig(**raw.get("encoder", {}))
        training_cfg = TrainingConfig(**raw.get("training", {}))
        return cls(encoder=encoder_cfg, training=training_cfg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
