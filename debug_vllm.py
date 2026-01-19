"""
Single-thread vLLM debugging script.
Used to directly call the vLLM API for debugging.
"""

import os
import sys
from pathlib import Path

# Important: environment variables must be set before importing vLLM.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"  # Show verbose logs.

# Set the Hugging Face cache directory to a user-writable location.
hf_cache_dir = str(Path("~/hf_cache").expanduser())
os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

# Create the cache directory (if it doesn't exist).
Path(hf_cache_dir).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("vLLM debugging script has launched")
print("=" * 60)

from vllm import LLM, SamplingParams

def main():
    """
    Main debugging entrypoint.
    """
    # Model config: prefer Qwen models.
    # Priority 1: try local Qwen models.
    local_models = [
        "~/hf_models/Qwen2.5-0.5B-Instruct",
        "~/hf_models/Qwen2.5-1.5B-Instruct",
        "~/hf_models/Qwen3-0.6B",
        "~/hf_models/Qwen2-0.5B-Instruct",
        "~/hf_models/Qwen2-1.5B-Instruct",
    ]
    
    model_name = None
    for local_path in local_models:
        expanded_path = Path(local_path).expanduser()
        if expanded_path.exists() and (expanded_path / "config.json").exists():
            model_name = str(expanded_path)
            print(f"\n✓ Found local Qwen model: {model_name}")
            break
    
    # Priority 2: if no local model is found, use a Qwen model from Hugging Face Hub.
    if model_name is None:
        print("\nNo local Qwen model found; downloading from Hugging Face Hub")
        print("The first run will download model files; this may take a few minutes...")
        # Use the smallest Qwen model for debugging.
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"Using model: {model_name}")
    
    print(f"\nModel: {model_name}")
    print("\nLoading model; this may take from a few seconds to a few minutes...")
    
    # Create the LLM instance.
    # tensor_parallel_size=1 ensures single-GPU execution.
    try:
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            max_model_len=256,               # Reduce context length to save memory.
            gpu_memory_utilization=0.7,      # Increase GPU memory utilization (from 0.3 to 0.7).
            max_num_seqs=8,                  # Reduce batch size.
            enforce_eager=True,              # Disable CUDA graphs to make debugging easier.
            trust_remote_code=True,
            disable_custom_all_reduce=True,  # Disable custom all-reduce.
        )
        print("\nModel loaded successfully")
    except Exception as e:
        print(f"\nModel load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Configure sampling parameters.
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50,  # Reduce generated tokens to save memory.
    )
    
    # Test prompts.
    prompts = [
        "Hello, my name is",
        "The future of AI is",
    ]
    
    print(f"\nStarting generation with {len(prompts)} prompts...")
    
    # Generate outputs.
    try:
        outputs = llm.generate(prompts, sampling_params)
    except Exception as e:
        print(f"\nGeneration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results.
    print("\n" + "=" * 50)
    print("Generated results:")
    print("=" * 50)
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated text: {generated_text}")
        print("-" * 50)
    
    print("\n" + "=" * 60)
    print("Debugging complete")
    print("=" * 60)


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    # Set breakpoints here for debugging.
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nUnhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
