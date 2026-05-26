from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

# Support running as a direct script and as an installed module
if __package__ is None or __package__ == "":
    parent_dir = str(Path(__file__).resolve().parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from flowmamba.zeek_pipeline import (
        ZeekCaptureConfig,
        build_encoder_input,
        parse_zeek_conn_log,
        run_zeek_capture,
    )
else:
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
    
    # Mutually exclusive group for inputs
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--interface", help="Network interface for live capture.")
    group.add_argument("--pcap", help="Path to pcap file for offline processing.")
    parser.add_argument(
        "--out-json",
        default="encoder_input.json",
        help="File name (inside output-dir) for encoded output JSON.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full stack traceback on exceptions.",
    )
    args = parser.parse_args()

    try:
        config = ZeekCaptureConfig(
            output_dir=args.output_dir,
            interface=args.interface,
            pcap_path=args.pcap,
        )

        source = f"interface '{args.interface}'" if args.interface else f"pcap file '{args.pcap}'"
        print(f"[*] Starting Zeek capture from {source}...")
        start_time = time.perf_counter()
        conn_log = run_zeek_capture(config)
        capture_duration = time.perf_counter() - start_time
        print(f"[+] Zeek capture completed in {capture_duration:.2f} seconds. Logs saved to: {conn_log}")

        print("[*] Parsing Zeek connection logs...")
        start_time = time.perf_counter()
        records = parse_zeek_conn_log(conn_log)
        parse_duration = time.perf_counter() - start_time
        print(f"[+] Parsed {len(records)} connection records in {parse_duration:.2f} seconds.")

        print("[*] Building normalized flow encoder inputs...")
        start_time = time.perf_counter()
        payload = build_encoder_input(records)
        encode_duration = time.perf_counter() - start_time
        print(f"[+] Normalized flow features generated in {encode_duration:.2f} seconds.")

        out_path = Path(args.output_dir) / args.out_json
        print(f"[*] Saving flow vectors to {out_path}...")
        out_path.write_text(payload.to_json(), encoding="utf-8")
        print(f"[+] Saved {len(payload.vectors)} flow vectors to {out_path} successfully.")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[!] Execution interrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print("Use --debug to see the full stack traceback.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
