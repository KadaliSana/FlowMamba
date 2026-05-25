from __future__ import annotations

import json
import subprocess
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

NUMERIC_FEATURE_COUNT = 5


@dataclass(frozen=True)
class ZeekCaptureConfig:
    """Configuration for running Zeek either from interface or pcap."""

    output_dir: str
    interface: str | None = None
    pcap_path: str | None = None
    zeek_path: str = "zeek"
    local_policy: str = "local"
    no_checksums: bool = True
    timeout_seconds: int | None = 300

    def validate(self) -> None:
        if not self.interface and not self.pcap_path:
            raise ValueError("Either interface or pcap_path must be provided.")
        if self.interface and self.pcap_path:
            raise ValueError("Provide only one of interface or pcap_path.")


@dataclass(frozen=True)
class ZeekFlowRecord:
    ts: float
    uid: str
    id_orig_h: str
    id_orig_p: int
    id_resp_h: str
    id_resp_p: int
    proto: str
    duration: float
    orig_bytes: int
    resp_bytes: int
    orig_pkts: int
    resp_pkts: int
    conn_state: str


@dataclass(frozen=True)
class FlowEncoderInput:
    """Ready-to-consume normalized vectors and flow metadata."""

    feature_names: list[str]
    vectors: list[list[float]]
    metadata: list[dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def run_zeek_capture(config: ZeekCaptureConfig) -> Path:
    """Run Zeek and return path to generated conn.log."""

    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [config.zeek_path]
    if config.no_checksums:
        cmd.append("-C")

    if config.interface:
        cmd.extend(["-i", config.interface])
    elif config.pcap_path:
        cmd.extend(["-r", config.pcap_path])

    cmd.append(config.local_policy)
    cmd.append(f"Log::default_logdir={output_dir}")

    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Zeek timed out after {config.timeout_seconds} seconds."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"Zeek failed (exit {proc.returncode}). stderr:\n{proc.stderr.strip()}"
        )

    conn_log = output_dir / "conn.log"
    if not conn_log.exists():
        raise FileNotFoundError(
            f"Zeek completed but conn.log not found at {conn_log}."
        )
    return conn_log


def parse_zeek_conn_log(conn_log_path: str | Path) -> list[ZeekFlowRecord]:
    """Parse Zeek conn.log (TSV with #fields header) into flow records."""

    path = Path(conn_log_path)
    if not path.exists():
        raise FileNotFoundError(f"conn.log not found: {path}")

    fields: list[str] = []
    separator = "\t"
    unset_field = "-"
    records: list[ZeekFlowRecord] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue

            if line.startswith("#separator"):
                token = line.split(" ", 1)[-1]
                # Zeek prints escaped separator, usually "\x09" for tab.
                separator = bytes(token, "utf-8").decode("unicode_escape")
                continue
            if line.startswith("#unset_field"):
                unset_field = line.split(" ", 1)[-1]
                continue
            if line.startswith("#fields"):
                fields = line.split(separator)[1:]
                continue
            if line.startswith("#"):
                continue

            if not fields:
                raise ValueError("Malformed conn.log: missing #fields header.")

            values = line.split(separator)
            if len(values) != len(fields):
                warnings.warn(
                    "Skipping malformed conn.log row due to field count mismatch.",
                    RuntimeWarning,
                )
                continue
            row = dict(zip(fields, values))

            records.append(
                ZeekFlowRecord(
                    ts=_to_float(row.get("ts"), unset_field),
                    uid=row.get("uid", ""),
                    id_orig_h=row.get("id.orig_h", ""),
                    id_orig_p=_to_int(row.get("id.orig_p"), unset_field),
                    id_resp_h=row.get("id.resp_h", ""),
                    id_resp_p=_to_int(row.get("id.resp_p"), unset_field),
                    proto=row.get("proto", ""),
                    duration=_to_float(row.get("duration"), unset_field),
                    orig_bytes=_to_int(row.get("orig_bytes"), unset_field),
                    resp_bytes=_to_int(row.get("resp_bytes"), unset_field),
                    orig_pkts=_to_int(row.get("orig_pkts"), unset_field),
                    resp_pkts=_to_int(row.get("resp_pkts"), unset_field),
                    conn_state=row.get("conn_state", ""),
                )
            )

    return records


def build_encoder_input(records: list[ZeekFlowRecord]) -> FlowEncoderInput:
    """Convert parsed Zeek flows into normalized numerical vectors."""

    feature_names = [
        "duration",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
        "proto_tcp",
        "proto_udp",
        "proto_icmp",
        "proto_other",
    ]

    raw_vectors: list[list[float]] = []
    metadata: list[dict[str, Any]] = []

    for record in records:
        proto = (record.proto or "").lower()
        proto_flags = [
            1.0 if proto == "tcp" else 0.0,
            1.0 if proto == "udp" else 0.0,
            1.0 if proto == "icmp" else 0.0,
            1.0 if proto not in {"tcp", "udp", "icmp"} else 0.0,
        ]

        raw_vectors.append(
            [
                _clamp_non_negative(record.duration, "duration"),
                float(_clamp_non_negative(record.orig_bytes, "orig_bytes")),
                float(_clamp_non_negative(record.resp_bytes, "resp_bytes")),
                float(_clamp_non_negative(record.orig_pkts, "orig_pkts")),
                float(_clamp_non_negative(record.resp_pkts, "resp_pkts")),
                *proto_flags,
            ]
        )
        metadata.append(
            {
                "uid": record.uid,
                "src_ip": record.id_orig_h,
                "src_port": record.id_orig_p,
                "dst_ip": record.id_resp_h,
                "dst_port": record.id_resp_p,
                "conn_state": record.conn_state,
                "timestamp": record.ts,
            }
        )

    vectors = _min_max_normalize(
        raw_vectors,
        normalize_upto_index=NUMERIC_FEATURE_COUNT,
    )

    return FlowEncoderInput(
        feature_names=feature_names,
        vectors=vectors,
        metadata=metadata,
    )


def _to_float(value: str | None, unset_field: str) -> float:
    if not value or value == unset_field:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: str | None, unset_field: str) -> int:
    """Parse integer-like values; decimal strings are truncated toward zero."""
    if not value or value == unset_field:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _min_max_normalize(
    vectors: list[list[float]], normalize_upto_index: int
) -> list[list[float]]:
    """Min-max normalize numeric columns; constant columns map to 0.0."""
    if not vectors:
        return vectors
    min_len = min(len(row) for row in vectors)
    if min_len == 0:
        raise ValueError("Vectors must not contain empty rows.")
    if normalize_upto_index > min_len:
        raise ValueError(
            f"normalize_upto_index={normalize_upto_index} exceeds vector length {min_len}."
        )

    mins = [float("inf")] * normalize_upto_index
    maxs = [float("-inf")] * normalize_upto_index

    for row in vectors:
        for i in range(normalize_upto_index):
            mins[i] = min(mins[i], row[i])
            maxs[i] = max(maxs[i], row[i])

    normalized: list[list[float]] = []
    for row in vectors:
        out = list(row)
        for i in range(normalize_upto_index):
            denom = maxs[i] - mins[i]
            out[i] = 0.0 if denom == 0 else (row[i] - mins[i]) / denom
        normalized.append(out)

    return normalized


def _clamp_non_negative(value: int | float, field_name: str) -> float:
    if value < 0:
        warnings.warn(
            f"Clamping negative {field_name}={value} to 0.",
            RuntimeWarning,
        )
        return 0.0
    return float(value)
