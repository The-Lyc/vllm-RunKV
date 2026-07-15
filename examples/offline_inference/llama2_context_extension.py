#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Create a Llama 2 checkpoint directory with extended RoPE context metadata.

Unlike OPT, Llama 2 uses rotary positional embeddings (RoPE), so there is no
learned position embedding weight to resize. This script copies an existing
checkpoint and updates its config/tokenizer metadata:

  1. config.max_position_embeddings is set to the target context length
  2. config.rope_scaling is set for legacy Transformers compatibility
  3. config.rope_parameters is set for vLLM/newer Transformers compatibility
  4. tokenizer.model_max_length is updated when a tokenizer is available

Example:
    python examples/offline_inference/llama2_context_extension.py \
        --model /data/models/Llama-2-7b-hf \
        --output /data/models/Llama-2-7b-hf-8k \
        --max-position-embeddings 8192

This is a config-level RoPE scaling conversion. It makes longer contexts
loadable, but long-context quality still needs validation or fine-tuning.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any


SKIP_COPY_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a Llama 2 checkpoint and add RoPE scaling metadata."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Input Llama 2 model path or Hugging Face model id.",
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
        help="Target context length, e.g. 8192 for Llama 2 7B 8k.",
    )
    target.add_argument(
        "--factor",
        type=float,
        help="Multiply the input max_position_embeddings by this factor.",
    )
    parser.add_argument(
        "--rope-type",
        choices=("linear", "dynamic"),
        default="linear",
        help=(
            "RoPE scaling strategy. linear is simple interpolation; dynamic "
            "uses dynamic NTK scaling."
        ),
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
        "--allow-non-llama",
        action="store_true",
        help="Skip the model_type == 'llama' guard.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow writing to a non-empty output directory.",
    )
    parser.add_argument(
        "--no-copy-files",
        action="store_true",
        help=(
            "Only write config/tokenizer files. By default, local checkpoints "
            "are copied and Hub ids are downloaded with huggingface_hub."
        ),
    )
    return parser.parse_args()


def load_source_dir(args: argparse.Namespace) -> Path | None:
    source = Path(args.model).expanduser()
    if source.is_dir():
        return source

    if args.no_copy_files:
        return None

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.revision,
            local_files_only=False,
        )
    )


def copy_checkpoint_files(source_dir: Path | None, output_dir: Path) -> None:
    if source_dir is None:
        return

    for source_path in source_dir.iterdir():
        if source_path.name in SKIP_COPY_NAMES:
            continue
        target_path = output_dir / source_path.name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, target_path)


def get_target_positions(
    config: Any,
    args: argparse.Namespace,
) -> tuple[int, int, float]:
    old_positions = int(config.max_position_embeddings)
    if args.max_position_embeddings is not None:
        target_positions = args.max_position_embeddings
        factor = target_positions / old_positions
    else:
        factor = args.factor
        target_positions = int(math.ceil(old_positions * factor))

    if target_positions <= old_positions:
        raise ValueError(
            "Target max_position_embeddings must be larger than the input "
            f"value. Got target={target_positions}, current={old_positions}."
        )
    if factor <= 1.0:
        raise ValueError(f"RoPE scaling factor must be > 1.0, got {factor}.")

    return old_positions, target_positions, float(factor)


def extend_llama2_context(args: argparse.Namespace) -> None:
    from transformers import AutoConfig, AutoTokenizer

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
    model_type = getattr(config, "model_type", None)
    if model_type != "llama" and not args.allow_non_llama:
        raise ValueError(
            f"Expected a Llama config, got model_type={model_type!r}. "
            "Use --allow-non-llama only for Llama-compatible RoPE models."
        )

    old_positions, target_positions, factor = get_target_positions(config, args)
    source_dir = load_source_dir(args)
    copy_checkpoint_files(source_dir, output_dir)

    rope_theta = getattr(config, "rope_theta", 10000.0)
    rope_scaling = {
        "type": args.rope_type,
        "factor": factor,
    }
    rope_parameters = {
        "rope_type": args.rope_type,
        "factor": factor,
        "rope_theta": rope_theta,
        "original_max_position_embeddings": old_positions,
    }

    config.max_position_embeddings = target_positions
    config.model_max_length = target_positions
    config.rope_scaling = rope_scaling
    config.rope_parameters = rope_parameters
    config.save_pretrained(output_dir)

    if not args.skip_tokenizer:
        tokenizer_source = args.tokenizer or args.model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer.model_max_length = target_positions
        tokenizer.save_pretrained(output_dir)

    print("Extended Llama 2 RoPE context metadata")
    print(f"  input model:                 {args.model}")
    print(f"  copied from:                 {source_dir or '(not copied)'}")
    print(f"  output dir:                  {output_dir}")
    print(f"  old max_position_embeddings: {old_positions}")
    print(f"  new max_position_embeddings: {target_positions}")
    print(f"  rope_type:                   {args.rope_type}")
    print(f"  factor:                      {factor}")
    print(f"  rope_theta:                  {rope_theta}")
    print("Saved extended checkpoint metadata.")
    print(
        "Note: RoPE scaling is a compatibility mechanism, not a guarantee of "
        "long-context quality. Validate the target context range before use."
    )


def main() -> None:
    extend_llama2_context(parse_args())


if __name__ == "__main__":
    main()
