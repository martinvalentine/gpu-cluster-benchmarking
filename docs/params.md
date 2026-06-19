# Configuration Parameters Reference

All settings live in `configs/models.yaml`. Cluster defaults apply to all models unless overridden per-model.

This doc is the schema reference. For `run-vllm.sh` / `run-llamacpp.sh` / `run-sglang.sh` (standalone scripts called directly), see [run.md](run.md) — those use independent env-var defaults.

---

## Cluster — vLLM Defaults

Values shown match the shipped `configs/models.yaml` (5× NVIDIA A40 48GB cluster). Adjust to your hardware.

| Parameter              | Type   | Default  | Description                                                  |
| ---------------------- | ------ | -------- | ------------------------------------------------------------ |
| `cluster.gpu_count`    | int    | `5`      | Number of GPUs. The orchestrator auto-skips models that need more. |
| `cluster.vllm.tp`      | int    | `5`      | Tensor parallel size — GPUs used for model sharding          |
| `cluster.vllm.gpu_mem_util` | float | `0.85` | Fraction of GPU VRAM to reserve for KV cache               |
| `cluster.vllm.max_model_len` | int | `16384` | Maximum context window per request (tokens)                  |
| `cluster.vllm.max_num_seqs` | int  | `1024`  | Maximum concurrent sequences the server handles              |
| `cluster.vllm.max_batched_tokens` | int | `65536` | Max tokens per batch (chunked prefill)                |
| `cluster.vllm.quant`   | string | `none`   | Quantization: `none`, `awq`, `gptq`                         |
| `cluster.vllm.block_size` | int | `16`     | PagedAttention block size in tokens (16 = best throughput)   |
| `cluster.vllm.dtype`   | string | `auto`   | Weight dtype: `auto` (bfloat16 on A40), `float16`, `bfloat16` |
| `cluster.vllm.swap_space` | int | `64`     | CPU swap space in GB (0 = disabled)                          |
| `cluster.vllm.enforce_eager` | bool | `false` | `false` = CUDA graphs (faster), `true` = eager (debugging) |
| `cluster.vllm.distributed_executor` | string | `mp` | Executor: `mp` (multiprocessing), `ray`                |
| `cluster.vllm.enable_chunked_prefill` | bool | `true` | Split long prompts across iterations |
| `cluster.vllm.enable_prefix_caching` | bool | `true` | Cache common prompt prefixes |

## Cluster — llama.cpp Defaults

| Parameter                    | Type   | Default  | Description                                                 |
| ---------------------------- | ------ | -------- | ----------------------------------------------------------- |
| `cluster.llamacpp.n_parallel` | int   | `48`     | Concurrent request slots (each gets its own KV cache)       |
| `cluster.llamacpp.ctx_size`  | int   | `16384`  | Context window per slot                                     |
| `cluster.llamacpp.n_gpu_layers` | string | `all`  | GPU offload layers: `all` (=999), or exact number           |
| `cluster.llamacpp.tensor_split` | string | `"1,1,1,1,1"` | GPU split ratios for 5-GPU cluster                 |
| `cluster.llamacpp.batch`    | int    | `8192`   | Prefill batch size (tokens per forward pass)                |
| `cluster.llamacpp.ubatch`   | int    | `2048`   | Micro-batch size (subdivides batch for GPU kernels)         |
| `cluster.llamacpp.threads`  | int    | `48`     | CPU threads for prefill/processing                          |
| `cluster.llamacpp.cache_key` | string | `q8_0`  | KV cache key quantization: `q8_0`, `f16`, `f32`            |
| `cluster.llamacpp.cache_val` | string | `turbo4` | KV cache value quantization: `turbo4`, `turbo3`, `q8_0`, `f16` |
| `cluster.llamacpp.flash_attn` | string | `on`   | Flash Attention: `on`, `off`, `auto` (required for TurboQuant) |
| `cluster.llamacpp.cache_prompt` | bool | `true` | Prompt caching — reuse KV for identical prefixes            |

## Cluster — SGLang Defaults

| Parameter                    | Type   | Default  | Description                                                 |
| ---------------------------- | ------ | -------- | ----------------------------------------------------------- |
| `cluster.sglang.mem_fraction` | string | `"0.85"` | Static memory fraction for KV cache                       |
| `cluster.sglang.max_model_len` | int  | `4096`   | Maximum context window per request                          |

---

## Per-Model Overrides — vLLM

Each overrides the corresponding cluster default for that specific model.

| YAML Key               | Overrides Cluster Default          | Description                              |
| ---------------------- | ---------------------------------- | ---------------------------------------- |
| `vllm_tp`              | `cluster.vllm.tp`                  | Tensor parallel size for this model      |
| `vllm_gpu_mem`         | `cluster.vllm.gpu_mem_util`        | GPU memory utilization for this model    |
| `vllm_max_seqs`        | `cluster.vllm.max_num_seqs`        | Max concurrent sequences for this model  |
| `vllm_max_model_len`   | `cluster.vllm.max_model_len`       | Max context window for this model        |
| `vllm_quant`           | `cluster.vllm.quant`               | Quantization override (e.g. `awq`)       |
| `vllm_max_batched_tokens` | `cluster.vllm.max_batched_tokens` | Max batched tokens for this model        |
| `vllm_block_size`      | `cluster.vllm.block_size`          | PagedAttention block size override       |
| `vllm_dtype`           | `cluster.vllm.dtype`               | Weight dtype override                    |
| `vllm_swap_space`      | `cluster.vllm.swap_space`          | CPU swap space override (GB)             |
| `vllm_distributed_executor` | `cluster.vllm.distributed_executor` | Executor override (`mp` or `ray`)     |
| `vllm_enforce_eager`   | `cluster.vllm.enforce_eager`       | CUDA graphs toggle override              |
| `vllm_enable_chunked_prefill` | `cluster.vllm.enable_chunked_prefill` | Chunked prefill override          |
| `vllm_enable_prefix_caching` | `cluster.vllm.enable_prefix_caching` | Prefix caching override            |

## Per-Model Overrides — llama.cpp

| YAML Key                 | Overrides Cluster Default              | Description                            |
| ------------------------ | -------------------------------------- | -------------------------------------- |
| `llamacpp_n_parallel`    | `cluster.llamacpp.n_parallel`          | Concurrent slots for this model        |
| `llamacpp_ctx_size`      | `cluster.llamacpp.ctx_size`            | Context window per slot                |
| `llamacpp_n_gpu_layers`  | `cluster.llamacpp.n_gpu_layers`        | GPU layers to offload                  |
| `llamacpp_tensor_split`  | `cluster.llamacpp.tensor_split`        | GPU split ratios for this model        |
| `llamacpp_batch`         | `cluster.llamacpp.batch`               | Prefill batch size                     |
| `llamacpp_ubatch`        | `cluster.llamacpp.ubatch`              | Micro-batch size                       |
| `llamacpp_threads`       | `cluster.llamacpp.threads`             | CPU threads                            |
| `llamacpp_cache_key`     | `cluster.llamacpp.cache_key`           | KV cache key quantization              |
| `llamacpp_cache_val`     | `cluster.llamacpp.cache_val`           | KV cache value quantization            |
| `llamacpp_flash_attn`    | `cluster.llamacpp.flash_attn`          | Flash Attention toggle                 |
| `llamacpp_cache_prompt`  | `cluster.llamacpp.cache_prompt`        | Prompt caching toggle                  |

## Per-Model Overrides — SGLang

| YAML Key                 | Overrides Cluster Default              | Description                            |
| ------------------------ | -------------------------------------- | -------------------------------------- |
| `sglang_mem_fraction`    | `cluster.sglang.mem_fraction`          | Static memory fraction for this model  |
| `sglang_max_model_len`   | `cluster.sglang.max_model_len`         | Max context window for this model      |

---

## Model Metadata

| Parameter    | Type   | Values / Format                    | Description                                     |
| ------------ | ------ | ---------------------------------- | ----------------------------------------------- |
| `name`       | string | unique identifier                  | Model ID used in results dirs and `--only` filter |
| `repo_id`    | string | HuggingFace `org/repo`             | Model repository for download                   |
| `local_dir`  | string | relative path under `base_dir`     | Where to store downloaded weights               |
| `format`     | string | `hf`, `gguf`                       | Weight format — determines backend compatibility |
| `include`    | string | glob pattern (e.g. `*q4_k_m*`)    | GGUF file pattern to match                      |
| `enabled`    | bool   | `true`, `false`                    | Whether to include in benchmark runs            |
| `phase`      | string | `p0`, `p1`, `p2`, `p3`, `embedding` | Benchmark phase tier (model size group)       |
| `backend`    | string | `vllm`, `llamacpp`, `sglang`      | Which serving framework to use                  |
| `proxy_name` | string | unique string                      | LiteLLM proxy route name                        |
| `endpoint`   | bool   | `true`, `false`                    | Whether to register in LiteLLM proxy            |
| `api_port`   | int    | port number                        | Override backend default port (e.g. 8003 for embedding) |

---

## Other Top-Level Keys

| Key        | Type   | Default                          | Description                                     |
| ---------- | ------ | -------------------------------- | ----------------------------------------------- |
| `base_dir` | string | `/workspace/models`              | Root directory for all downloaded model weights  |
| `dataset.sharegpt` | string | `datasets/sharegpt.json`  | Benchmark dataset path (relative to repo root)  |
| `ports.vllm` | int  | `8000`                           | vLLM server port                                |
| `ports.llamacpp` | int | `8001`                        | llama.cpp server port                           |
| `ports.sglang` | int  | `8002`                          | SGLang server port                              |
| `ports.embedding` | int | `8003`                        | Embedding server port                           |
| `ports.proxy` | int  | `4000`                           | LiteLLM proxy port                              |
