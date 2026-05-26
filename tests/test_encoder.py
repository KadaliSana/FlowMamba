"""Smoke test for the FlowMamba encoder — verifies shapes and training loop."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Ensure the parent package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np


def test_model_shapes():
    """Verify encoder input/output shapes."""
    from flowmamba.encoder.model import FlowMambaEncoder

    print("[*] Testing model shapes...")
    model = FlowMambaEncoder(
        input_dim=13,
        d_model=32,  # smaller for quick test
        n_layers=2,
        d_state=8,
        embedding_dim=128,
    )

    batch_size, seq_len, features = 4, 16, 13
    x = torch.randn(batch_size, seq_len, features)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 12:] = False  # test padding

    out = model(x, mask=mask)
    assert out.shape == (batch_size, 128), f"Expected (4, 128), got {out.shape}"

    # Verify L2 normalization
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Embeddings not L2-normalized: norms={norms}"

    print(f"[+] Model shape test passed: input {x.shape} → output {out.shape}")
    print(f"[+] L2 norms: {norms.tolist()}")


def test_supcon_loss():
    """Verify SupCon loss computes and gradients flow."""
    from flowmamba.encoder.supcon_loss import SupConLoss

    print("\n[*] Testing SupCon loss...")
    criterion = SupConLoss(temperature=0.07)

    embeddings = torch.randn(8, 128, requires_grad=True)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])

    loss = criterion(embeddings, labels)
    assert loss.requires_grad, "Loss should require grad"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"

    loss.backward()
    assert embeddings.grad is not None, "Gradients should flow"
    print(f"[+] SupCon loss test passed: loss={loss.item():.4f}")


def test_dataset_loading():
    """Verify dataset loading from CSV and JSON."""
    from flowmamba.encoder.dataset import FlowDataset

    print("\n[*] Testing dataset loading...")

    # Create a synthetic CSV dataset
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        csv_path = f.name
        f.write(
            "duration,orig_bytes,resp_bytes,orig_pkts,resp_pkts,"
            "rtt,jitter,packet_rate,mean_iat,proto,label,id.orig_h\n"
        )
        np.random.seed(42)
        for i in range(100):
            label = np.random.choice(["Streaming", "Gaming", "VoIP"])
            proto = np.random.choice(["tcp", "udp"])
            src_ip = f"10.0.0.{i % 5}"
            features = ",".join(
                [f"{np.random.exponential(10):.4f}" for _ in range(9)]
            )
            f.write(f"{features},{proto},{label},{src_ip}\n")

    dataset = FlowDataset(csv_path, window_size=8, group_by="src_ip")
    print(f"[+] CSV dataset: {len(dataset)} samples, {dataset.num_classes} classes")
    print(f"    Labels: {dataset.label_to_idx}")

    sample = dataset[0]
    features, mask, label = sample
    print(f"    Sample shape: features={features.shape}, mask={mask.shape}, label={label}")
    assert features.shape == (8, 13), f"Expected (8, 13), got {features.shape}"

    # Create a synthetic JSON dataset
    flows = []
    for i in range(50):
        flows.append({
            "features": {
                "duration": float(np.random.exponential(5)),
                "orig_bytes": int(np.random.randint(0, 10000)),
                "resp_bytes": int(np.random.randint(0, 10000)),
                "orig_pkts": int(np.random.randint(1, 100)),
                "resp_pkts": int(np.random.randint(1, 100)),
                "rtt": float(np.random.exponential(0.1)),
                "jitter": float(np.random.exponential(0.01)),
                "packet_rate": float(np.random.exponential(50)),
                "mean_iat": float(np.random.exponential(0.5)),
            },
            "label": np.random.choice(["Streaming", "Gaming"]),
            "metadata": {
                "id.orig_h": f"10.0.0.{i % 3}",
                "proto": np.random.choice(["tcp", "udp"]),
            },
        })

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json_path = f.name
        json.dump({"flows": flows}, f)

    dataset_json = FlowDataset(json_path, window_size=4, group_by="src_ip")
    print(f"[+] JSON dataset: {len(dataset_json)} samples, {dataset_json.num_classes} classes")

    # Clean up
    Path(csv_path).unlink(missing_ok=True)
    Path(json_path).unlink(missing_ok=True)

    return csv_path  # return for trainer test


def test_training_smoke():
    """Quick 3-epoch training smoke test on synthetic data."""
    from flowmamba.config import FlowMambaConfig
    from flowmamba.encoder.trainer import FlowMambaTrainer

    print("\n[*] Running training smoke test (3 epochs on synthetic data)...")

    # Create synthetic CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as f:
        csv_path = f.name
        f.write(
            "duration,orig_bytes,resp_bytes,orig_pkts,resp_pkts,"
            "rtt,jitter,packet_rate,mean_iat,proto,label,id.orig_h\n"
        )
        np.random.seed(123)
        for i in range(200):
            label = np.random.choice(["Streaming", "Gaming", "VoIP"])
            proto = np.random.choice(["tcp", "udp", "icmp"])
            src_ip = f"10.0.0.{i % 10}"
            features = ",".join(
                [f"{np.random.exponential(10):.4f}" for _ in range(9)]
            )
            f.write(f"{features},{proto},{label},{src_ip}\n")

    config = FlowMambaConfig()
    config.encoder.d_model = 32  # smaller for fast test
    config.encoder.n_layers = 2
    config.training.epochs = 3
    config.training.batch_size = 16
    config.training.window_size = 8
    config.training.num_workers = 0  # avoid multiprocessing in test
    config.training.checkpoint_dir = tempfile.mkdtemp()
    config.training.log_interval = 1

    trainer = FlowMambaTrainer(config)
    model = trainer.train(csv_path)

    assert model is not None, "Model should be returned"
    print(f"[+] Training smoke test passed!")
    print(f"    Final train loss: {trainer.training_history[-1]['train_loss']:.4f}")
    print(f"    Final val loss:   {trainer.training_history[-1]['val_loss']:.4f}")

    # Test checkpoint loading
    from flowmamba.encoder.trainer import FlowMambaTrainer as T
    ckpt_path = Path(config.training.checkpoint_dir) / "best_model.pt"
    loaded_model, label_map, meta = T.load_checkpoint(ckpt_path)
    print(f"[+] Checkpoint loaded: epoch={meta['epoch']}, labels={label_map}")

    # Clean up
    Path(csv_path).unlink(missing_ok=True)


if __name__ == "__main__":
    test_model_shapes()
    test_supcon_loss()
    test_dataset_loading()
    test_training_smoke()
    print("\n" + "=" * 50)
    print("[+] ALL TESTS PASSED")
    print("=" * 50)
