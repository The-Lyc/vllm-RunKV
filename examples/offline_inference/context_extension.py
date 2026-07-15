#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Extend an OPT checkpoint's context length by resizing its learned absolute
positional embeddings.

OPT does not use RoPE, so runtime rope scaling methods such as YARN do not
apply. This script writes a new Hugging Face checkpoint directory with:

  1. model.decoder.embed_positions.weight resized to the target length
  2. config.max_position_embeddings updated to the target length
  3. tokenizer.model_max_length updated when a tokenizer is saved

Example:
    python examples/offline_inference/context_extension.py \
        --model /data/models/opt-2.7b \
        --output /data/models/opt-2.7b-8k \
        --max-position-embeddings 8192

The default initialization method preserves all original learned position
embeddings and fills new positions by repeating the last learned position.
That is conservative for existing-context behavior, but using the extended
context still requires validation or fine-tuning for quality.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize OPT learned positional embeddings and save a new checkpoint."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Input OPT model path or Hugging Face model id.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for the extended checkpoint.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--max-position-embeddings",
        type=int,
        help="Target OPT context length, excluding OPT's internal +2 offset.",
    )
    target.add_argument(
        "--factor",
        type=float,
        help="Multiply the input config.max_position_embeddings by this factor.",
    )
    parser.add_argument(
        "--method",
        choices=("copy", "tile", "interpolate", "interpolate-preserve-prefix"),
        default="copy",
        help=(
            "How to initialize new position rows. copy preserves the original "
            "rows and repeats the last row. tile preserves the original rows "
            "and cycles them for new positions. interpolate resamples the whole "
            "table. interpolate-preserve-prefix resamples, then restores the "
            "original prefix exactly."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="dtype passed to transformers when loading the model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision for model/config/tokenizer loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to Hugging Face loaders.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or id. Defaults to --model.",
    )
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Do not load or save a tokenizer.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="max_shard_size passed to save_pretrained.",
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Save PyTorch .bin shards instead of safetensors.",
    )
    parser.add_argument(
        "--allow-non-opt",
        action="store_true",
        help="Skip the model_type == 'opt' guard.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing to a non-empty output directory.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def get_target_positions(config: Any, args: argparse.Namespace) -> int:
    old_positions = int(config.max_position_embeddings)
    if args.max_position_embeddings is not None:
        target_positions = args.max_position_embeddings
    else:
        target_positions = int(math.ceil(old_positions * args.factor))

    if target_positions <= old_positions:
        raise ValueError(
            "Target max_position_embeddings must be larger than the input "
            f"value. Got target={target_positions}, current={old_positions}."
        )
    return target_positions


def get_opt_position_embedding(model: Any) -> Any:
    base_model = getattr(model, "model", None)
    decoder = getattr(base_model, "decoder", None)
    embed_positions = getattr(decoder, "embed_positions", None)
    if embed_positions is None or not hasattr(embed_positions, "weight"):
        raise RuntimeError(
            "Could not find model.model.decoder.embed_positions. "
            "This script expects a Hugging Face OPTForCausalLM-style model."
        )
    return embed_positions


def resize_usable_positions(
    usable_weight: Any,
    target_positions: int,
    method: str,
) -> Any:
    import torch.nn.functional as F

    old_positions, hidden_size = usable_weight.shape
    if target_positions <= old_positions:
        raise ValueError("target_positions must be larger than old_positions.")

    if method == "copy":
        resized = usable_weight.new_empty(target_positions, hidden_size)
        resized[:old_positions] = usable_weight
        resized[old_positions:] = usable_weight[-1].expand(
            target_positions - old_positions, hidden_size
        )
        return resized

    if method == "tile":
        repeats = math.ceil(target_positions / old_positions)
        resized = usable_weight.repeat((repeats, 1))[:target_positions].clone()
        resized[:old_positions] = usable_weight
        return resized

    interp_source = usable_weight.float().T.unsqueeze(0)
    resized = F.interpolate(
        interp_source,
        size=target_positions,
        mode="linear",
        align_corners=True,
    ).squeeze(0).T.to(dtype=usable_weight.dtype)

    if method == "interpolate-preserve-prefix":
        resized[:old_positions] = usable_weight

    return resized.contiguous()


def extend_opt_context(args: argparse.Namespace) -> None:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Use --allow-overwrite to write there anyway."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    if getattr(config, "model_type", None) != "opt" and not args.allow_non_opt:
        model_type = getattr(config, "model_type", None)
        raise ValueError(
            f"Expected an OPT config, got model_type={model_type!r}. "
            "Use --allow-non-opt only if you know the model has OPT-compatible "
            "position embeddings."
        )

    target_positions = get_target_positions(config, args)
    old_config_positions = int(config.max_position_embeddings)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        revision=args.revision,
        torch_dtype=torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    embed_positions = get_opt_position_embedding(model)

    with torch.no_grad():
        old_weight = embed_positions.weight.detach()
        offset = int(getattr(embed_positions, "offset", 2))
        old_total_positions, hidden_size = old_weight.shape
        old_usable_positions = old_total_positions - offset
        if old_usable_positions != old_config_positions:
            print(
                "Warning: config.max_position_embeddings="
                f"{old_config_positions}, but embed_positions has "
                f"{old_usable_positions} usable rows after offset={offset}."
            )

        reserved_weight = old_weight[:offset].clone()
        usable_weight = old_weight[offset:].clone()
        resized_usable_weight = resize_usable_positions(
            usable_weight,
            target_positions,
            args.method,
        )
        resized_weight = torch.cat([reserved_weight, resized_usable_weight], dim=0)

    embed_positions.num_embeddings = resized_weight.shape[0]
    embed_positions.weight = nn.Parameter(
        resized_weight.to(device=old_weight.device),
        requires_grad=embed_positions.weight.requires_grad,
    )

    model.config.max_position_embeddings = target_positions
    if hasattr(model.config, "n_positions"):
        model.config.n_positions = target_positions

    print("Extended OPT positional embeddings")
    print(f"  input model:               {args.model}")
    print(f"  output dir:                {output_dir}")
    print(f"  old max_position_embeddings: {old_config_positions}")
    print(f"  new max_position_embeddings: {target_positions}")
    print(f"  internal offset rows:        {offset}")
    print(f"  old embed weight shape:      {tuple(old_weight.shape)}")
    print(f"  new embed weight shape:      {tuple(resized_weight.shape)}")
    print(f"  init method:                 {args.method}")

    model.save_pretrained(
        output_dir,
        safe_serialization=not args.no_safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    if not args.skip_tokenizer:
        tokenizer_source = args.tokenizer or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer.model_max_length = target_positions
        tokenizer.save_pretrained(output_dir)

    print("Saved extended checkpoint.")
    print(
        "Note: learned absolute position extension is an initialization "
        "heuristic. Validate quality before relying on the new context range."
    )


def main() -> None:
    extend_opt_context(parse_args())


if __name__ == "__main__":
    main()
