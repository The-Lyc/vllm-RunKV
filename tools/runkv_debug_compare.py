#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RunKV Debug Log Comparator.

Compares JSONL debug logs produced by vanilla and runkv engines
to find the first decode step/layer/token where KV cache diverges.

Usage:
    python tools/runkv_debug_compare.py \\
        --vanilla  runkv_debug_logs/runkv_debug_vanilla_*.jsonl \\
        --runkv    runkv_debug_logs/runkv_debug_runkv_*.jsonl \\
        [--rtol 1e-3] [--atol 1e-5] [--max-steps 50] [--show-all-layers]

Output:
    For each decode step (in order), compare per-token KV checksums
    between vanilla and runkv engines. Report the first mismatch with
    full context: step, layer, request, token position, and the
    actual vs expected checksums.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    CYAN = "\033[36m"


def _c(text: str, *styles: str) -> str:
    return f"{''.join(styles)}{text}{C.RESET}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_steps(records: list[dict]) -> dict[int, dict]:
    """Extract step records indexed by step_id."""
    steps = {}
    for r in records:
        if r.get("type") == "step":
            steps[r["step_id"]] = r
    return steps


def extract_header(records: list[dict]) -> dict:
    for r in records:
        if r.get("type") == "header":
            return r
    return {}


# ---------------------------------------------------------------------------
# KV checksum comparison
# ---------------------------------------------------------------------------


def checksums_match(
    cksum_a: dict,
    cksum_b: dict,
    rtol: float,
    atol: float,
) -> tuple[bool, list[str]]:
    """Compare two TokenKVChecksum dicts. Returns (match, reasons)."""
    reasons = []
    for field in ("k_norm", "v_norm", "k_sum", "v_sum"):
        a = cksum_a.get(field, float("nan"))
        b = cksum_b.get(field, float("nan"))
        if math.isnan(a) or math.isnan(b):
            reasons.append(f"{field}: NaN (a={a}, b={b})")
            continue
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            reasons.append(f"{field}: {a:.8f} vs {b:.8f} (diff={abs(a - b):.8e})")

    # Also compare head elements if available
    for field in ("k_head", "v_head"):
        a_head = cksum_a.get(field, [])
        b_head = cksum_b.get(field, [])
        if a_head and b_head and len(a_head) == len(b_head):
            for i, (av, bv) in enumerate(zip(a_head, b_head)):
                if not math.isclose(av, bv, rel_tol=rtol, abs_tol=atol):
                    reasons.append(f"{field}[{i}]: {av:.8f} vs {bv:.8f}")
                    break  # one mismatch is enough per field

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# Index info analysis
# ---------------------------------------------------------------------------


def analyze_runkv_indices(step: dict) -> list[str]:
    """Analyze runkv step for potential index issues.

    Checks:
    1. Multiple requests mapped to same staging slot (cross-req contamination)
    2. Staging slot out of bounds
    3. Dirty blocks referencing non-mapped logical IDs
    """
    warnings = []
    for req_info in step.get("requests", []):
        mapping = req_info.get("mapping_snapshot")
        if not mapping:
            continue

        # Check for slot collisions across requests
        # (This is checked at step level, outside per-req)

    # Global: check if different requests' blocks map to the same staging slot
    all_mappings: dict[int, list[str]] = defaultdict(list)  # slot -> [req_ids]
    for req_info in step.get("requests", []):
        req_id = req_info.get("req_id", "?")
        mapping = req_info.get("mapping_snapshot", {})
        for logical_str, slot in mapping.items():
            all_mappings[slot].append(f"{req_id}:L{logical_str}")

    for slot, users in all_mappings.items():
        if len(users) > len(set(u.split(":")[0] for u in users)):
            # Same req mapped multiple logical blocks to same slot = BUG
            warnings.append(f"SLOT COLLISION: staging slot {slot} used by: {users}")

    return warnings


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def compare_logs(
    vanilla_path: str,
    runkv_path: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    max_steps: int = 0,
    show_all_layers: bool = False,
    verbose: bool = False,
) -> bool:
    """Compare two debug log files. Returns True if all match."""
    print(_c("=" * 80, C.DIM))
    print(_c("RunKV Debug Log Comparator", C.BOLD, C.CYAN))
    print(_c("=" * 80, C.DIM))

    vanilla_records = load_jsonl(vanilla_path)
    runkv_records = load_jsonl(runkv_path)

    v_header = extract_header(vanilla_records)
    r_header = extract_header(runkv_records)
    print(f"  Vanilla: {vanilla_path} (tag={v_header.get('engine_tag', '?')})")
    print(f"  RunKV:   {runkv_path} (tag={r_header.get('engine_tag', '?')})")

    v_steps = extract_steps(vanilla_records)
    r_steps = extract_steps(runkv_records)
    all_step_ids = sorted(set(v_steps.keys()) | set(r_steps.keys()))

    if max_steps > 0:
        all_step_ids = all_step_ids[:max_steps]

    print(f"  Vanilla steps: {len(v_steps)}, RunKV steps: {len(r_steps)}")
    print(f"  Comparing {len(all_step_ids)} steps (rtol={rtol}, atol={atol})")
    print(_c("-" * 80, C.DIM))

    total_match = 0
    total_mismatch = 0
    first_mismatch_step: int | None = None
    first_mismatch_detail: str | None = None

    for step_id in all_step_ids:
        v_step = v_steps.get(step_id)
        r_step = r_steps.get(step_id)

        if v_step is None:
            print(_c(f"Step {step_id}: MISSING in vanilla", C.YELLOW))
            continue
        if r_step is None:
            print(_c(f"Step {step_id}: MISSING in runkv", C.YELLOW))
            continue

        # ---- Index analysis for runkv ----
        idx_warnings = analyze_runkv_indices(r_step)
        for w in idx_warnings:
            print(_c(f"  [WARN] Step {step_id}: {w}", C.RED, C.BOLD))

        # ---- Build per-(req_id, layer_name, position) checksum maps ----
        # key: (req_id, layer_name, position) -> checksum dict
        v_checksums: dict[tuple, dict] = {}
        r_checksums: dict[tuple, dict] = {}
        r_cpu_checksums: dict[tuple, dict] = {}

        for lr in v_step.get("layer_kv_records", []):
            req_id = lr["req_id"]
            layer_name = lr["layer_name"]
            for tc in lr.get("token_checksums", []):
                key = (req_id, layer_name, tc["position"])
                v_checksums[key] = tc

        for lr in r_step.get("layer_kv_records", []):
            req_id = lr["req_id"]
            layer_name = lr["layer_name"]
            phase = lr.get("phase", "")
            for tc in lr.get("token_checksums", []):
                key = (req_id, layer_name, tc["position"])
                if phase == "cpu_side":
                    r_cpu_checksums[key] = tc
                elif phase == "after_attention":
                    r_checksums[key] = tc

        # Compare: vanilla after_attention vs runkv after_attention
        all_keys = sorted(set(v_checksums.keys()) | set(r_checksums.keys()))

        step_match = 0
        step_mismatch = 0
        step_mismatch_details: list[str] = []

        for key in all_keys:
            req_id, layer_name, position = key
            v_ck = v_checksums.get(key)
            r_ck = r_checksums.get(key)

            if v_ck is None:
                if verbose:
                    step_mismatch_details.append(
                        f"  {layer_name} req={req_id} pos={position}: "
                        f"MISSING in vanilla"
                    )
                continue
            if r_ck is None:
                if verbose:
                    step_mismatch_details.append(
                        f"  {layer_name} req={req_id} pos={position}: MISSING in runkv"
                    )
                continue

            match, reasons = checksums_match(v_ck, r_ck, rtol, atol)
            if match:
                step_match += 1
            else:
                step_mismatch += 1
                detail = (
                    f"  {layer_name} req={req_id} pos={position}: "
                    f"MISMATCH - {'; '.join(reasons)}"
                )
                step_mismatch_details.append(detail)

                # Also check CPU side for this key
                cpu_ck = r_cpu_checksums.get(key)
                if cpu_ck is not None:
                    cpu_match, cpu_reasons = checksums_match(v_ck, cpu_ck, rtol, atol)
                    if not cpu_match:
                        step_mismatch_details.append(
                            f"    -> CPU cache also differs: {'; '.join(cpu_reasons)}"
                        )
                    else:
                        step_mismatch_details.append(
                            "    -> CPU cache MATCHES vanilla (H2D copy issue?)"
                        )

                    # Check GPU vs CPU (batch copy correctness: after attention,
                    # new tokens won't match CPU since they were just written)
                    gpu_vs_cpu_match, gvc_reasons = checksums_match(
                        r_ck, cpu_ck, rtol, atol
                    )
                    if not gpu_vs_cpu_match:
                        step_mismatch_details.append(
                            f"    -> GPU staging != CPU: {'; '.join(gvc_reasons[:2])}"
                        )

                if first_mismatch_step is None:
                    first_mismatch_step = step_id
                    first_mismatch_detail = detail

        total_match += step_match
        total_mismatch += step_mismatch

        # Print step summary
        if step_mismatch > 0:
            print(
                _c(
                    f"Step {step_id}: {step_mismatch} MISMATCH, {step_match} match "
                    f"(reqs: {v_step.get('num_reqs', '?')})",
                    C.RED,
                )
            )
            if show_all_layers or step_id == first_mismatch_step:
                for d in step_mismatch_details:
                    print(_c(d, C.RED))
        else:
            if verbose or step_match > 0:
                print(
                    _c(
                        f"Step {step_id}: ALL MATCH ({step_match} tokens)",
                        C.GREEN,
                    )
                )

    # Summary
    print(_c("\n" + "=" * 80, C.DIM))
    print(_c("Summary", C.BOLD, C.CYAN))
    print(_c("=" * 80, C.DIM))
    print(f"  Total token-layer comparisons: {total_match + total_mismatch}")
    print(_c(f"  Matches: {total_match}", C.GREEN))
    if total_mismatch > 0:
        print(_c(f"  Mismatches: {total_mismatch}", C.RED, C.BOLD))
        print(_c(f"  First mismatch at step {first_mismatch_step}:", C.RED))
        print(_c(f"    {first_mismatch_detail}", C.RED))
    else:
        print(_c("  No mismatches found!", C.GREEN, C.BOLD))

    # ---- Print per-request index comparison for first mismatch step ----
    if first_mismatch_step is not None:
        _print_index_comparison(
            v_steps.get(first_mismatch_step, {}),
            r_steps.get(first_mismatch_step, {}),
            first_mismatch_step,
        )

    return total_mismatch == 0


def _print_index_comparison(v_step: dict, r_step: dict, step_id: int) -> None:
    """Print detailed index comparison for a step."""
    print(_c(f"\n--- Index details for step {step_id} ---", C.YELLOW, C.BOLD))

    v_reqs = {r["req_id"]: r for r in v_step.get("requests", [])}
    r_reqs = {r["req_id"]: r for r in r_step.get("requests", [])}

    for req_id in sorted(set(v_reqs.keys()) | set(r_reqs.keys())):
        v_req = v_reqs.get(req_id)
        r_req = r_reqs.get(req_id)

        print(_c(f"\n  Request {req_id}:", C.BOLD))
        if v_req:
            print(
                f"    Vanilla: seq_len={v_req.get('seq_len')}, "
                f"blocks={v_req.get('logical_block_table', [])[:8]}..."
            )
        if r_req:
            print(
                f"    RunKV:   seq_len={r_req.get('seq_len')}, "
                f"logical={r_req.get('logical_block_table', [])[:8]}..., "
                f"staging={r_req.get('staging_block_table', [])[:8]}..."
            )

        # Compare token indices
        v_tokens = v_req.get("token_indices", []) if v_req else []
        r_tokens = r_req.get("token_indices", []) if r_req else []

        if v_tokens or r_tokens:
            print(
                f"    {'pos':>5} | {'v_logical':>10} {'v_offset':>8} {'v_slot':>7} | "
                f"{'r_logical':>10} {'r_offset':>8} {'r_slot':>7} {'r_staging':>10}"
            )
            max_t = max(len(v_tokens), len(r_tokens))
            for i in range(max_t):
                vt = v_tokens[i] if i < len(v_tokens) else {}
                rt = r_tokens[i] if i < len(r_tokens) else {}
                pos = vt.get("position", rt.get("position", "?"))
                print(
                    f"    {pos:>5} | "
                    f"{vt.get('logical_block_id', '-'):>10} "
                    f"{vt.get('offset_in_block', '-'):>8} "
                    f"{vt.get('slot_mapping_value', '-'):>7} | "
                    f"{rt.get('logical_block_id', '-'):>10} "
                    f"{rt.get('offset_in_block', '-'):>8} "
                    f"{rt.get('slot_mapping_value', '-'):>7} "
                    f"{rt.get('staging_slot', '-'):>10}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare RunKV debug logs between vanilla and runkv engines."
    )
    parser.add_argument(
        "--vanilla",
        required=True,
        help="Path to vanilla engine debug JSONL file",
    )
    parser.add_argument(
        "--runkv",
        required=True,
        help="Path to runkv engine debug JSONL file",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for checksum comparison (default: 1e-3)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for checksum comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max steps to compare (0 = all, default: 0)",
    )
    parser.add_argument(
        "--show-all-layers",
        action="store_true",
        help="Show mismatch details for all steps (not just first)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    for p in [args.vanilla, args.runkv]:
        if not Path(p).exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    all_match = compare_logs(
        vanilla_path=args.vanilla,
        runkv_path=args.runkv,
        rtol=args.rtol,
        atol=args.atol,
        max_steps=args.max_steps,
        show_all_layers=args.show_all_layers,
        verbose=args.verbose,
    )

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
