"""FlowDataset — PyTorch dataset for labeled network flow sequences."""

from __future__ import annotations

import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


# Feature names that the dataset expects from labeled data files.
NUMERIC_FEATURES = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "rtt",
    "jitter",
    "packet_rate",
    "mean_iat",
]

PROTOCOL_COLUMN = "proto"
LABEL_COLUMN = "label"
GROUPING_COLUMNS = {
    "src_ip": "id.orig_h",
    "dst_ip": "id.resp_h",
    "uid": "uid",
}


class FlowDataset(Dataset):
    """Dataset that loads labeled flow records and groups them into sequences.

    Each sample is a ``(sequence_tensor, label_index)`` tuple where
    ``sequence_tensor`` has shape ``(window_size, num_features)`` and
    ``label_index`` is the integer class label.

    Supports loading from:
    - **CSV** with columns for each feature + ``label`` + grouping key
    - **JSON** with ``{"flows": [{"features": {...}, "label": "...", "metadata": {...}}]}``

    Flows are grouped by a configurable key (default: source IP) and sliced
    into temporal windows.  Sequences shorter than ``window_size`` are padded
    with zeros; a boolean mask tensor is also returned when ``return_mask=True``.
    """

    def __init__(
        self,
        data_path: str | Path,
        window_size: int = 16,
        group_by: str = "src_ip",
        return_mask: bool = True,
    ) -> None:
        self.window_size = window_size
        self.return_mask = return_mask

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        # Load raw records from file
        raw_records = self._load_file(data_path)

        if not raw_records:
            raise ValueError(f"No valid records found in {data_path}")

        # Build label mapping
        unique_labels = sorted({r["label"] for r in raw_records})
        self.label_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(unique_labels)
        }
        self.idx_to_label: dict[int, str] = {
            idx: label for label, idx in self.label_to_idx.items()
        }
        self.num_classes = len(unique_labels)

        # Assign integer label indices to each record
        _assign_label_indices(raw_records, self.label_to_idx)

        # Group flows into sequences and build windowed samples
        self.samples: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._build_sequences(raw_records, group_by)

    @property
    def num_features(self) -> int:
        """Number of features per flow (9 numeric + 4 protocol one-hot = 13)."""
        return 13

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        features, mask, label_idx = self.samples[idx]
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        if self.return_mask:
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            return feature_tensor, mask_tensor, label_tensor

        return feature_tensor, label_tensor

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[dict[str, Any]]:
        """Load records from CSV or JSON file."""
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(path)
        elif suffix == ".json":
            return self._load_json(path)
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Use .csv or .json."
            )

    @staticmethod
    def _load_csv(path: Path) -> list[dict[str, Any]]:
        """Load from a CSV file with feature columns + label."""
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV file has no header row.")
            if LABEL_COLUMN not in reader.fieldnames:
                raise ValueError(
                    f"CSV must have a '{LABEL_COLUMN}' column. "
                    f"Found columns: {reader.fieldnames}"
                )
            for row_num, row in enumerate(reader, start=2):
                try:
                    record = _parse_row(row)
                    records.append(record)
                except (ValueError, KeyError) as e:
                    warnings.warn(
                        f"Skipping CSV row {row_num}: {e}",
                        RuntimeWarning,
                    )
        return records

    @staticmethod
    def _load_json(path: Path) -> list[dict[str, Any]]:
        """Load from a JSON file.

        Expected format::

            {
                "flows": [
                    {
                        "features": {
                            "duration": 1.23,
                            "orig_bytes": 456,
                            ...
                        },
                        "label": "Streaming",
                        "metadata": {"src_ip": "10.0.0.1", ...}
                    },
                    ...
                ]
            }

        Or a flat list of dicts with feature columns + label.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        records: list[dict[str, Any]] = []

        # Handle {"flows": [...]} wrapper
        if isinstance(data, dict) and "flows" in data:
            raw_flows = data["flows"]
        elif isinstance(data, list):
            raw_flows = data
        else:
            raise ValueError(
                "JSON must be a list of flow dicts or {'flows': [...]}"
            )

        for i, entry in enumerate(raw_flows):
            try:
                if "features" in entry:
                    # Nested format
                    row = {**entry["features"]}
                    row[LABEL_COLUMN] = entry[LABEL_COLUMN]
                    if "metadata" in entry:
                        row.update(entry["metadata"])
                else:
                    # Flat format
                    row = entry
                record = _parse_row(row)
                records.append(record)
            except (ValueError, KeyError) as e:
                warnings.warn(
                    f"Skipping JSON flow entry {i}: {e}",
                    RuntimeWarning,
                )

        return records

    # ------------------------------------------------------------------
    # Sequence building
    # ------------------------------------------------------------------

    def _build_sequences(
        self, records: list[dict[str, Any]], group_by: str
    ) -> None:
        """Group records by key and create windowed samples."""
        # Resolve the grouping column name
        group_col = GROUPING_COLUMNS.get(group_by, group_by)

        # Group records by the grouping key
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ungrouped: list[dict[str, Any]] = []

        for record in records:
            key = record.get("metadata", {}).get(group_col)
            if key is None:
                # Try top-level
                key = record.get(group_col)
            if key is not None:
                groups[str(key)].append(record)
            else:
                ungrouped.append(record)

        # If no grouping key found for any record, treat all as one sequence
        if not groups and ungrouped:
            groups["_all_"] = ungrouped
        elif ungrouped:
            groups["_ungrouped_"] = ungrouped

        # Create windowed samples from each group
        for group_key, group_records in groups.items():
            features_list = [r["features"] for r in group_records]
            labels_list = [r["label_idx"] for r in group_records]

            n = len(features_list)
            if n == 0:
                continue

            # Sliding window with step size 1
            if n >= self.window_size:
                for start in range(n - self.window_size + 1):
                    end = start + self.window_size
                    window_features = np.array(
                        features_list[start:end], dtype=np.float32
                    )
                    mask = np.ones(self.window_size, dtype=np.float32)

                    # Use the most common label in the window
                    window_labels = labels_list[start:end]
                    label_idx = max(set(window_labels), key=window_labels.count)

                    self.samples.append((window_features, mask, label_idx))
            else:
                # Pad short sequences
                padded = np.zeros(
                    (self.window_size, len(features_list[0])), dtype=np.float32
                )
                padded[:n] = np.array(features_list, dtype=np.float32)
                mask = np.zeros(self.window_size, dtype=np.float32)
                mask[:n] = 1.0

                window_labels = labels_list
                label_idx = max(set(window_labels), key=window_labels.count)

                self.samples.append((padded, mask, label_idx))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_row(row: dict[str, Any]) -> dict[str, Any]:
    """Parse a single record dict into features + metadata."""
    if LABEL_COLUMN not in row:
        raise KeyError(f"Missing '{LABEL_COLUMN}' field")

    label = str(row[LABEL_COLUMN]).strip()
    if not label:
        raise ValueError("Empty label")

    # Extract numeric features
    numeric_values: list[float] = []
    for feat in NUMERIC_FEATURES:
        val = row.get(feat, 0)
        try:
            numeric_values.append(max(0.0, float(val)))
        except (TypeError, ValueError):
            numeric_values.append(0.0)

    # Protocol one-hot encoding
    proto = str(row.get(PROTOCOL_COLUMN, "")).lower().strip()
    proto_onehot = [
        1.0 if proto == "tcp" else 0.0,
        1.0 if proto == "udp" else 0.0,
        1.0 if proto == "icmp" else 0.0,
        1.0 if proto not in {"tcp", "udp", "icmp"} else 0.0,
    ]

    features = numeric_values + proto_onehot  # 13-d vector

    return {
        "features": features,
        "label": label,
        "label_idx": 0,  # Will be set after label_to_idx is built
        "metadata": {k: v for k, v in row.items() if k not in NUMERIC_FEATURES and k != LABEL_COLUMN and k != PROTOCOL_COLUMN},
    }


def _assign_label_indices(
    records: list[dict[str, Any]], label_to_idx: dict[str, int]
) -> None:
    """Assign integer label indices to parsed records (in-place)."""
    for record in records:
        record["label_idx"] = label_to_idx[record["label"]]
