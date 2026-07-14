<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

This is RunKV version of vLLM(Forked From vanilla vLLM).

## Startup
- Clone the resposity:
```
git clone https://github.com/The-Lyc/vllm-RunKV.git
cd vllm-RunKV
```
- create venv
```
uv venv --python 3.12 --seed
source .venv/bin/activate
```

- config cuda environment
```
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

- install torch
```
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

- install dependencies
```
python -m pip install -r requirements/build.txt
```

- compile runkv kernel
```
MAX_JOBS=16 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
python setup_runkv.py build_ext --inplace
# validate
python -c "import runkv_kernels; print(runkv_kernels.batch_copy_blocks)"
```

- compile vllm kernels
```
MAX_JOBS=16 NVCC_THREADS=1 VLLM_TARGET_DEVICE=cuda \
python -m pip install -e . --no-build-isolation -v
# validate
python -c "import vllm; import vllm._C; import vllm._moe_C; print('vLLM extensions OK')"
```

## Roadmap

| Feature                                                       | Code | Test | 
| ----------------------------------------------------------- | ---- | ---- |
| Decoupled-Paged Attention                                   | ✅    | ✅   | 
| UVA-based Copy                                              | ✅    | ✅   | 
| Compute-IO Overlapping                                      | ✅    | ✅   |
| IO & Recompute Policy                                       | 🚧    | 🚧   | 
| Reservation & Eviction Policy                               | 🚧    | 🚧   | 
| Dynamic Buffers' Size                                       | 🚧    | 🚧   | 
| Dynamic Buffers' Layout                                     | 🚧    | 🚧   | 
