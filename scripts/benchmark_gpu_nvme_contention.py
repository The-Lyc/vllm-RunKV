#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


GpuDirection = Literal["h2d", "d2h", "bidirectional"]
NvmeDirection = Literal["read", "write"]

_SIZE_UNITS = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1024,
    "KIB": 1024,
    "M": 1024**2,
    "MB": 1024**2,
    "MIB": 1024**2,
    "G": 1024**3,
    "GB": 1024**3,
    "GIB": 1024**3,
    "T": 1024**4,
    "TB": 1024**4,
    "TIB": 1024**4,
}


@dataclass
class GpuResult:
    direction: str
    duration_s: float
    buffer_mb: int
    iterations: int
    bytes_total: int
    gbps: float


@dataclass
class NvmeResult:
    direction: str
    runtime_s: float
    path: str
    size: str
    bs: str
    iodepth: int
    numjobs: int
    bytes_total: int
    gbps: float
    fio_json: dict


@dataclass
class ConcurrentResult:
    gpu_direction: str
    nvme_direction: str
    gpu: GpuResult
    nvme: NvmeResult
    gpu_ratio_vs_solo: float | None
    nvme_ratio_vs_solo: float | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether GPU H2D/D2H copies and NVMe direct IO interfere "
            "when run alone versus concurrently."
        )
    )
    parser.add_argument(
        "--file",
        default="/mnt/nvme0n1p1/runkv_pcie_contention/testfile.bin",
        help="NVMe test file path. Put this on a local NVMe mount, not /home/ceph.",
    )
    parser.add_argument("--output-dir", default="exp_results/pcie_nvme_contention")
    parser.add_argument("--runtime-s", type=float, default=8.0)
    parser.add_argument(
        "--file-size",
        default="8G",
        help="Total NVMe test file size. With --fio-numjobs > 1, each job gets a disjoint slice.",
    )
    parser.add_argument("--fio-bs", default="1M")
    parser.add_argument("--fio-iodepth", type=int, default=32)
    parser.add_argument("--fio-numjobs", type=int, default=1)
    parser.add_argument("--fio-ioengine", default="libaio")
    parser.add_argument(
        "--fio-bin",
        default=None,
        help="Path to Flexible I/O Tester. Defaults to a validated fio binary, preferring /usr/bin/fio.",
    )
    parser.add_argument("--gpu-buffer-mb", type=int, default=256)
    parser.add_argument("--gpu-device", default="cuda:0")
    parser.add_argument(
        "--gpu-directions",
        nargs="+",
        default=["h2d", "d2h"],
        choices=["h2d", "d2h", "bidirectional"],
    )
    parser.add_argument(
        "--nvme-directions",
        nargs="+",
        default=["read", "write"],
        choices=["read", "write"],
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only create/precondition the NVMe test file, then exit.",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Do not precondition the NVMe file before read tests.",
    )
    parser.add_argument(
        "--skip-concurrent",
        action="store_true",
        help="Only run solo GPU/NVMe tests.",
    )
    parser.add_argument(
        "--keep-file",
        action="store_true",
        help="Keep the NVMe test file after the benchmark.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        stdout = result.stdout[-4000:] if result.stdout else ""
        stderr = result.stderr[-4000:] if result.stderr else ""
        raise RuntimeError(
            "Command failed with code "
            f"{result.returncode}: {' '.join(cmd)}\n"
            f"--- stderr ---\n{stderr}\n"
            f"--- stdout ---\n{stdout}"
        )
    return result


def _is_fio_tester(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip().startswith("fio-")


def _resolve_fio_binary(raw_path: str | None) -> str:
    candidates: list[str] = []
    if raw_path:
        candidates.append(raw_path)
    candidates.extend(["/usr/bin/fio", shutil.which("fio") or "", "/bin/fio"])
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if _is_fio_tester(candidate):
            return candidate
    raise RuntimeError(
        "Could not find Flexible I/O Tester fio. Install fio or pass --fio-bin /path/to/fio."
    )


def _parse_size_to_bytes(value: str) -> int:
    text = value.strip().upper()
    number_part = text
    unit_part = ""
    for idx, char in enumerate(text):
        if not (char.isdigit() or char == "."):
            number_part = text[:idx]
            unit_part = text[idx:]
            break
    if not number_part:
        raise ValueError(f"Invalid size: {value!r}")
    if unit_part not in _SIZE_UNITS:
        raise ValueError(f"Unsupported size suffix in {value!r}")
    return int(float(number_part) * _SIZE_UNITS[unit_part])


def _fio_region_args(args: argparse.Namespace) -> list[str]:
    numjobs = max(int(args.fio_numjobs), 1)
    total_bytes = _parse_size_to_bytes(str(args.file_size))
    job_bytes = total_bytes // numjobs
    if job_bytes <= 0:
        raise ValueError(
            f"file-size {args.file_size!r} is too small for fio-numjobs={numjobs}"
        )
    region_args = [f"--size={job_bytes}"]
    if numjobs > 1:
        region_args.append(f"--offset_increment={job_bytes}")
    return region_args


def _fio_base_args(args: argparse.Namespace, direction: NvmeDirection, runtime_s: float) -> list[str]:
    rw = "read" if direction == "read" else "write"
    return [
        args.fio_bin,
        "--name=nvme_io",
        f"--filename={args.file}",
        f"--rw={rw}",
        f"--bs={args.fio_bs}",
        f"--iodepth={args.fio_iodepth}",
        f"--numjobs={args.fio_numjobs}",
        f"--ioengine={args.fio_ioengine}",
        "--direct=1",
        "--time_based=1",
        f"--runtime={max(runtime_s, 1.0)}",
        *_fio_region_args(args),
        "--group_reporting=1",
        "--output-format=json",
    ]


def prepare_nvme_file(args: argparse.Namespace) -> None:
    path = Path(args.file)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.fio_bin,
        "--name=prepare_nvme_file",
        f"--filename={args.file}",
        "--rw=write",
        f"--bs={args.fio_bs}",
        f"--iodepth={args.fio_iodepth}",
        "--numjobs=1",
        f"--ioengine={args.fio_ioengine}",
        "--direct=1",
        f"--size={_parse_size_to_bytes(str(args.file_size))}",
        "--group_reporting=1",
        "--output-format=json",
    ]
    print(f"[prepare] creating direct-IO test file: {path}")
    result = _run(cmd)
    summary = _parse_fio_json(result.stdout, "write", args, runtime_override=None)
    print(f"[prepare] wrote {summary.bytes_total / 1e9:.3f} GB at {summary.gbps:.3f} GB/s")


def _parse_fio_json(
    stdout: str,
    direction: NvmeDirection,
    args: argparse.Namespace,
    runtime_override: float | None,
) -> NvmeResult:
    data = json.loads(stdout)
    jobs = data.get("jobs", [])
    total_bytes = 0
    runtime_ms = 0.0
    for job in jobs:
        section = job["read" if direction == "read" else "write"]
        total_bytes += int(section.get("io_bytes", 0))
        runtime_ms = max(runtime_ms, float(section.get("runtime", 0.0)))
    runtime_s = runtime_override if runtime_override is not None else max(runtime_ms / 1000.0, 1e-9)
    gbps = total_bytes / max(runtime_s, 1e-9) / 1e9
    return NvmeResult(
        direction=direction,
        runtime_s=runtime_s,
        path=str(args.file),
        size=str(args.file_size),
        bs=str(args.fio_bs),
        iodepth=int(args.fio_iodepth),
        numjobs=int(args.fio_numjobs),
        bytes_total=total_bytes,
        gbps=gbps,
        fio_json=data,
    )


def run_nvme(args: argparse.Namespace, direction: NvmeDirection, runtime_s: float | None = None) -> NvmeResult:
    actual_runtime = float(args.runtime_s if runtime_s is None else runtime_s)
    cmd = _fio_base_args(args, direction, actual_runtime)
    print(f"[nvme:{direction}] running fio for {actual_runtime:.1f}s")
    result = _run(cmd)
    parsed = _parse_fio_json(result.stdout, direction, args, runtime_override=None)
    print(f"[nvme:{direction}] {parsed.gbps:.3f} GB/s, {parsed.bytes_total / 1e9:.3f} GB")
    return parsed


def run_gpu(args: argparse.Namespace, direction: GpuDirection, runtime_s: float | None = None) -> GpuResult:
    import torch

    actual_runtime = float(args.runtime_s if runtime_s is None else runtime_s)
    device = torch.device(args.gpu_device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU copy benchmark")
    torch.cuda.set_device(device)

    numel = int(args.gpu_buffer_mb) * 1024 * 1024
    host_src = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
    host_dst = torch.empty(numel, dtype=torch.uint8, pin_memory=True)
    gpu_buf = torch.empty(numel, dtype=torch.uint8, device=device)
    host_src.fill_(11)
    gpu_buf.fill_(17)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        gpu_buf.copy_(host_src, non_blocking=True)
        host_dst.copy_(gpu_buf, non_blocking=True)
    stream.synchronize()

    iterations = 0
    bytes_total = 0
    start = time.perf_counter()
    deadline = start + actual_runtime
    while time.perf_counter() < deadline:
        with torch.cuda.stream(stream):
            if direction in ("h2d", "bidirectional"):
                gpu_buf.copy_(host_src, non_blocking=True)
                bytes_total += host_src.numel() * host_src.element_size()
            if direction in ("d2h", "bidirectional"):
                host_dst.copy_(gpu_buf, non_blocking=True)
                bytes_total += host_dst.numel() * host_dst.element_size()
        stream.synchronize()
        iterations += 1
    elapsed = max(time.perf_counter() - start, 1e-9)
    result = GpuResult(
        direction=direction,
        duration_s=elapsed,
        buffer_mb=int(args.gpu_buffer_mb),
        iterations=iterations,
        bytes_total=int(bytes_total),
        gbps=bytes_total / elapsed / 1e9,
    )
    print(f"[gpu:{direction}] {result.gbps:.3f} GB/s, {result.bytes_total / 1e9:.3f} GB")
    return result


def run_concurrent(
    args: argparse.Namespace,
    gpu_direction: GpuDirection,
    nvme_direction: NvmeDirection,
    gpu_baseline: GpuResult | None,
    nvme_baseline: NvmeResult | None,
) -> ConcurrentResult:
    runtime_s = float(args.runtime_s)
    cmd = _fio_base_args(args, nvme_direction, runtime_s)
    print(f"[concurrent:{gpu_direction}+{nvme_direction}] starting fio and GPU copy")
    fio_proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        gpu_result = run_gpu(args, gpu_direction, runtime_s=runtime_s)
        stdout, stderr = fio_proc.communicate(timeout=max(runtime_s * 2.0, runtime_s + 30.0))
    except Exception:
        fio_proc.kill()
        fio_proc.wait(timeout=5)
        raise
    if fio_proc.returncode != 0:
        raise RuntimeError(
            f"fio failed with code {fio_proc.returncode}\n"
            f"--- stderr ---\n{stderr[-4000:] if stderr else ''}\n"
            f"--- stdout ---\n{stdout[-4000:] if stdout else ''}"
        )
    nvme_result = _parse_fio_json(stdout, nvme_direction, args, runtime_override=None)
    print(
        f"[concurrent:{gpu_direction}+{nvme_direction}] "
        f"gpu={gpu_result.gbps:.3f} GB/s nvme={nvme_result.gbps:.3f} GB/s"
    )
    return ConcurrentResult(
        gpu_direction=gpu_direction,
        nvme_direction=nvme_direction,
        gpu=gpu_result,
        nvme=nvme_result,
        gpu_ratio_vs_solo=(
            gpu_result.gbps / gpu_baseline.gbps
            if gpu_baseline is not None and gpu_baseline.gbps > 0
            else None
        ),
        nvme_ratio_vs_solo=(
            nvme_result.gbps / nvme_baseline.gbps
            if nvme_baseline is not None and nvme_baseline.gbps > 0
            else None
        ),
    )


def _write_summary(out_dir: Path, results: dict) -> None:
    md_lines = [
        "# GPU/NVMe Contention Benchmark",
        "",
        "## Solo GPU",
        "",
        "| direction | GB/s | GB | seconds | iterations |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results["solo_gpu"].values():
        md_lines.append(
            f"| {item['direction']} | {item['gbps']:.3f} | "
            f"{item['bytes_total'] / 1e9:.3f} | {item['duration_s']:.3f} | {item['iterations']} |"
        )
    md_lines.extend([
        "",
        "## Solo NVMe",
        "",
        "| direction | GB/s | GB | seconds |",
        "|---|---:|---:|---:|",
    ])
    for item in results["solo_nvme"].values():
        md_lines.append(
            f"| {item['direction']} | {item['gbps']:.3f} | "
            f"{item['bytes_total'] / 1e9:.3f} | {item['runtime_s']:.3f} |"
        )
    md_lines.extend([
        "",
        "## Concurrent",
        "",
        "| GPU | NVMe | GPU GB/s | GPU / solo | NVMe GB/s | NVMe / solo |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for item in results["concurrent"]:
        gpu_ratio = item["gpu_ratio_vs_solo"]
        nvme_ratio = item["nvme_ratio_vs_solo"]
        gpu_ratio_text = f"{gpu_ratio:.3f}" if gpu_ratio is not None else "n/a"
        nvme_ratio_text = f"{nvme_ratio:.3f}" if nvme_ratio is not None else "n/a"
        md_lines.append(
            f"| {item['gpu_direction']} | {item['nvme_direction']} | "
            f"{item['gpu']['gbps']:.3f} | "
            f"{gpu_ratio_text} | "
            f"{item['nvme']['gbps']:.3f} | "
            f"{nvme_ratio_text} |"
        )
    (out_dir / "summary.md").write_text("\n".join(md_lines) + "\n")


def main() -> int:
    args = _parse_args()
    args.fio_bin = _resolve_fio_binary(args.fio_bin)
    print(f"[fio] using {args.fio_bin}")

    out_dir = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        prepare_nvme_file(args)
    if args.prepare_only:
        return 0

    solo_gpu: dict[str, GpuResult] = {}
    solo_nvme: dict[str, NvmeResult] = {}
    concurrent: list[ConcurrentResult] = []

    for direction in args.gpu_directions:
        solo_gpu[direction] = run_gpu(args, direction)
    for direction in args.nvme_directions:
        solo_nvme[direction] = run_nvme(args, direction)

    if not args.skip_concurrent:
        for gpu_direction in args.gpu_directions:
            for nvme_direction in args.nvme_directions:
                concurrent.append(
                    run_concurrent(
                        args,
                        gpu_direction,
                        nvme_direction,
                        solo_gpu.get(gpu_direction),
                        solo_nvme.get(nvme_direction),
                    )
                )

    result_obj = {
        "args": vars(args),
        "solo_gpu": {key: asdict(value) for key, value in solo_gpu.items()},
        "solo_nvme": {
            key: {k: v for k, v in asdict(value).items() if k != "fio_json"}
            for key, value in solo_nvme.items()
        },
        "concurrent": [
            {
                "gpu_direction": item.gpu_direction,
                "nvme_direction": item.nvme_direction,
                "gpu": asdict(item.gpu),
                "nvme": {k: v for k, v in asdict(item.nvme).items() if k != "fio_json"},
                "gpu_ratio_vs_solo": item.gpu_ratio_vs_solo,
                "nvme_ratio_vs_solo": item.nvme_ratio_vs_solo,
            }
            for item in concurrent
        ],
    }
    (out_dir / "results.json").write_text(json.dumps(result_obj, indent=2) + "\n")
    _write_summary(out_dir, result_obj)
    print(f"[done] wrote {out_dir / 'results.json'}")
    print(f"[done] wrote {out_dir / 'summary.md'}")

    if not args.keep_file:
        try:
            Path(args.file).unlink()
            print(f"[cleanup] removed {args.file}")
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
