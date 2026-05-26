"""FlowMamba training loop — train the encoder with SupCon loss."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..config import FlowMambaConfig
from .dataset import FlowDataset
from .model import FlowMambaEncoder
from .supcon_loss import SupConLoss


class FlowMambaTrainer:
    """Train a FlowMambaEncoder with Supervised Contrastive Loss.

    Usage::

        config = FlowMambaConfig()
        trainer = FlowMambaTrainer(config)
        trainer.train("labeled_flows.csv")
    """

    def __init__(self, config: FlowMambaConfig | None = None) -> None:
        self.config = config or FlowMambaConfig()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Built during train()
        self.model: FlowMambaEncoder | None = None
        self.dataset: FlowDataset | None = None
        self.training_history: list[dict[str, Any]] = []

    def train(self, data_path: str | Path) -> FlowMambaEncoder:
        """Run the full training loop.

        Args:
            data_path: Path to a labeled CSV or JSON file.

        Returns:
            Trained ``FlowMambaEncoder`` model.
        """
        cfg_enc = self.config.encoder
        cfg_train = self.config.training

        # Reproducibility
        torch.manual_seed(cfg_train.seed)
        np.random.seed(cfg_train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg_train.seed)

        # ----------------------------------------------------------
        # 1. Load dataset
        # ----------------------------------------------------------
        print(f"[*] Loading dataset from {data_path}...")
        self.dataset = FlowDataset(
            data_path=data_path,
            window_size=cfg_train.window_size,
            group_by=cfg_train.group_by,
            return_mask=True,
        )
        print(
            f"[+] Loaded {len(self.dataset)} samples, "
            f"{self.dataset.num_classes} classes: "
            f"{list(self.dataset.label_to_idx.keys())}"
        )

        # ----------------------------------------------------------
        # 2. Train/val split
        # ----------------------------------------------------------
        n_total = len(self.dataset)
        n_val = max(1, int(n_total * cfg_train.val_split))
        n_train = n_total - n_val

        train_set, val_set = random_split(
            self.dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg_train.seed),
        )

        train_loader = DataLoader(
            train_set,
            batch_size=cfg_train.batch_size,
            shuffle=True,
            num_workers=cfg_train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg_train.batch_size,
            shuffle=False,
            num_workers=cfg_train.num_workers,
            pin_memory=True,
        )

        print(f"[+] Train: {n_train} samples, Val: {n_val} samples")

        # ----------------------------------------------------------
        # 3. Build model, loss, optimizer, scheduler
        # ----------------------------------------------------------
        self.model = FlowMambaEncoder.from_config(cfg_enc).to(self.device)
        criterion = SupConLoss(temperature=cfg_train.supcon_temperature)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg_train.lr,
            weight_decay=cfg_train.weight_decay,
        )

        if cfg_train.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg_train.epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )

        param_count = sum(p.numel() for p in self.model.parameters())
        print(
            f"[+] Model: {param_count:,} parameters on {self.device}"
        )

        # ----------------------------------------------------------
        # 4. Checkpoint directory
        # ----------------------------------------------------------
        ckpt_dir = Path(cfg_train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        best_val_loss = float("inf")
        self.training_history = []

        # ----------------------------------------------------------
        # 5. Training loop
        # ----------------------------------------------------------
        print(f"[*] Training for {cfg_train.epochs} epochs...")
        for epoch in range(1, cfg_train.epochs + 1):
            epoch_start = time.perf_counter()

            # --- Train ---
            train_loss = self._train_epoch(
                train_loader, criterion, optimizer, cfg_train.log_interval
            )

            # --- Validate ---
            val_loss = self._validate_epoch(val_loader, criterion)

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - epoch_start

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "elapsed_sec": round(elapsed, 2),
            }
            self.training_history.append(epoch_record)

            print(
                f"  Epoch {epoch:3d}/{cfg_train.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    ckpt_dir / "best_model.pt",
                    epoch=epoch,
                    val_loss=val_loss,
                )
                print(f"  [+] New best model saved (val_loss={val_loss:.4f})")

        # Save final checkpoint
        self._save_checkpoint(
            ckpt_dir / "final_model.pt",
            epoch=cfg_train.epochs,
            val_loss=val_loss,
        )

        # Save training history
        history_path = ckpt_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.training_history, f, indent=2)
        print(f"[+] Training complete. History saved to {history_path}")

        # Save label mapping
        label_map_path = ckpt_dir / "label_mapping.json"
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(self.dataset.label_to_idx, f, indent=2)
        print(f"[+] Label mapping saved to {label_map_path}")

        return self.model

    # ------------------------------------------------------------------
    # Epoch routines
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        criterion: SupConLoss,
        optimizer: torch.optim.Optimizer,
        log_interval: int,
    ) -> float:
        """Run one training epoch and return average loss."""
        assert self.model is not None
        self.model.train()

        total_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            features, masks, labels = batch
            features = features.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()

            embeddings = self.model(features, mask=masks)
            loss = criterion(embeddings, labels)

            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(
        self,
        loader: DataLoader,
        criterion: SupConLoss,
    ) -> float:
        """Run one validation epoch and return average loss."""
        assert self.model is not None
        self.model.eval()

        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            features, masks, labels = batch
            features = features.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.model(features, mask=masks)
            loss = criterion(embeddings, labels)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        """Save model checkpoint with metadata."""
        assert self.model is not None
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
                "label_to_idx": (
                    self.dataset.label_to_idx if self.dataset else {}
                ),
            },
            path,
        )

    @staticmethod
    def load_checkpoint(
        path: str | Path,
        device: str | torch.device | None = None,
    ) -> tuple[FlowMambaEncoder, dict[str, int], dict[str, Any]]:
        """Load a trained model from checkpoint.

        Args:
            path: Path to ``.pt`` checkpoint file.
            device: Target device (defaults to CUDA if available).

        Returns:
            Tuple of ``(model, label_to_idx, checkpoint_metadata)``.
        """
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = FlowMambaConfig._from_dict(checkpoint["config"])
        model = FlowMambaEncoder.from_config(config.encoder).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        label_to_idx = checkpoint.get("label_to_idx", {})
        metadata = {
            "epoch": checkpoint.get("epoch"),
            "val_loss": checkpoint.get("val_loss"),
        }

        return model, label_to_idx, metadata
