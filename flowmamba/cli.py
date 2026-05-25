from __future__ import annotations

import argparse
from pathlib import Path

from .zeek_pipeline import (
    ZeekCaptureConfig,
    build_encoder_input,
    parse_zeek_conn_log,
    run_zeek_capture,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Baseline Zeek packet capture and flow encoder preprocessing pipeline."
    )
    parser.add_argument("--output-dir", required=True, help="Directory for Zeek logs/output.")
    parser.add_argument("--interface", help="Network interface for live capture.")
    parser.add_argument("--pcap", help="Path to pcap file for offline processing.")
    parser.add_argument("--zeek-path", default="zeek", help="Path to Zeek binary.")
    parser.add_argument(
        "--out-json",
        default="encoder_input.json",
        help="File name (inside output-dir) for encoded output JSON.",
    )
    args = parser.parse_args()

    config = ZeekCaptureConfig(
        output_dir=args.output_dir,
        interface=args.interface,
        pcap_path=args.pcap,
        zeek_path=args.zeek_path,
    )
    conn_log = run_zeek_capture(config)
    records = parse_zeek_conn_log(conn_log)
    payload = build_encoder_input(records)

    out_path = Path(args.output_dir) / args.out_json
    out_path.write_text(payload.to_json(), encoding="utf-8")
    print(f"Saved {len(payload.vectors)} flow vectors to {out_path}")


if __name__ == "__main__":
    main()
