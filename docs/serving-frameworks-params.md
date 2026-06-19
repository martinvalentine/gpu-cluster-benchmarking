# Serving Framework Versions & Supported Parameters

Reference for vLLM, SGLang, and llama.cpp versions installed in this project and their supported CLI parameters.

## Version Summary

| Framework | Version | Source | Notes |
|-----------|---------|--------|-------|
| vLLM | 0.23.0 | official image | `vllm/vllm-openai` Docker image |
| SGLang | 0.5.8 | PyPI | Isolated venv (`/opt/venv-sglang`) |
| llama.cpp (TurboQuant) | 1 (7d9715f) | built from source | branch `feature-turboquant-kv-cache` |
| llama.cpp (Standard) | 1 (3ac3c20) | built from source | tag `b9570` |

### Check versions yourself (Docker)

```bash
# vLLM
docker exec <container> python3 -c "import vllm; print(vllm.__version__)"

# SGLang
docker exec <container> /opt/venv-sglang/bin/python3 -c "import sglang; print(sglang.__version__)"

# llama.cpp TurboQuant
docker exec <container> /workspace/third_party/llama-cpp-turboquant/build/bin/llama-server --version

# llama.cpp Standard
docker exec <container> /workspace/third_party/llama.cpp/build/bin/llama-server --version
```

---

## vLLM Parameters (v0.23.0)

Source: `third_party/vllm/vllm/engine/arg_utils.py` (`EngineArgs` class) + `vllm/entrypoints/openai/cli_args.py`

Usage: `vllm serve <model> [options]` or `--help=<ConfigGroup>` (e.g., `--help=ModelConfig`)

### Model Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Model | positional / `--model` | required | HuggingFace model name or local path |
| Served model name | `--served-model-name` | same as model | Name exposed in `/v1/models` |
| Model weights | `--model-weights` | none | Path to pre-downloaded weights |
| Tokenizer | `--tokenizer` | same as model | Tokenizer name or path |
| Tokenizer mode | `--tokenizer-mode` | auto | `auto`, `slow`, `mistral`, `cached` |
| Trust remote code | `--trust-remote-code` | False | Allow custom code from model repos |
| Revision | `--revision` | none | Git revision to download from |
| Code revision | `--code-revision` | none | Git revision for code files |
| HF token | `--hf-token` | none | HuggingFace auth token |
| Download dir | `--download-dir` | none | Directory to cache downloaded models |
| Load format | `--load-format` | auto | `auto`, `safetensors`, `pt`, `npz`, `gguf` |
| Config format | `--config-format` | auto | `auto`, `hf`, `safetensors` |
| Model impl | `--model-impl` | auto | Model implementation to use |
| Generation config | `--generation-config` | auto | Generation config source |

### Precision & Quantization

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Dtype | `--dtype` | auto | `auto`, `float16`, `bfloat16`, `float32` |
| Quantization | `--quantization` / `-q` | none | `awq`, `gptq`, `squeezellm`, `fp8`, `fp4`, etc. |
| KV cache dtype | `--kv-cache-dtype` | auto | `auto`, `fp8`, `fp8_e5m2`, `fp8_e4m3` |
| Calculate KV scales | `--calculate-kv-scales` | auto | Auto-calculate KV cache scales |
| Enforce eager | `--enforce-eager` | False | Disable CUDA graph capture |

### Parallelism & Distribution

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Tensor parallel size | `--tensor-parallel-size` / `-tp` | 1 | Number of GPUs for tensor parallelism |
| Pipeline parallel size | `--pipeline-parallel-size` / `-pp` | 1 | Number of GPUs for pipeline parallelism |
| Data parallel size | `--data-parallel-size` / `-dp` | 1 | Number of data parallel replicas |
| Expert parallel size | `--enable-expert-parallel` | False | Enable expert parallelism for MoE models |
| Distributed executor | `--distributed-executor-backend` | mp | `mp` (multiprocessing), `ray`, `external_launcher` |
| Worker class | `--worker-class` | auto | Worker process class |
| Disable custom all-reduce | `--disable-custom-all-reduce` | False | Disable custom all-reduce kernel |
| Master address | `--master-address` | localhost | Master node address for distributed |
| Master port | `--master-port` | 29500 | Master node port |
| Number of nodes | `--nnodes` | 1 | Number of nodes |
| Node rank | `--node-rank` | 0 | This node's rank |

### Scheduling & Batching

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Max model len | `--max-model-len` | model default | Maximum context length |
| Max num batched tokens | `--max-num-batched-tokens` | None (auto) | Max tokens per iteration |
| Max num seqs | `--max-num-seqs` | None (auto) | Max concurrent sequences |
| Max num partial prefills | `--max-num-partial-prefills` | 4 | Max concurrent partial prefills |
| Max long partial prefills | `--max-long-partial-prefills` | 2 | Max long partial prefills |
| Long prefill token threshold | `--long-prefill-token-threshold` | 30000 | Threshold for "long" prefill |
| Scheduling policy | `--scheduling-policy` | fcfs | `fcfs`, `priority`, `recompute` |
| Stream interval | `--stream-interval` | 1 | Token interval for streaming |
| Disable chunked mm input | `--disable-chunked-mm-input` | False | Disable chunked multimodal input |

### Cache Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| GPU memory utilization | `--gpu-memory-utilization` | 0.9 | Fraction of GPU memory for KV cache |
| KV cache memory bytes | `--kv-cache-memory-bytes` | None | Exact KV cache memory in bytes |
| Block size | `--block-size` | None (auto) | PagedAttention block size (16 recommended) |
| Enable prefix caching | `--enable-prefix-caching` | None (auto) | Cache common prompt prefixes |
| Prefix caching hash algo | `--prefix-caching-hash-algo` | xxhash | Hash algorithm for prefix caching |
| KV offloading size | `--kv-offloading-size` | None | KV cache offload to CPU size |
| Num GPU blocks override | `--num-gpu-blocks-override` | None | Override auto GPU block count |
| Swap space | `--swap-space` | 4 | CPU swap space in GB (per GPU) |

### Chunked Prefill

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Enable chunked prefill | `--enable-chunked-prefill` | None (auto) | Split long prefills across iterations |
| Scheduler reserve full ISL | `--scheduler-reserve-full-isl` | False | Reserve full input sequence length |

### Speculative Decoding

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Spec method | `--spec-method` | none | Speculative decoding method |
| Spec model | `--spec-model` | none | Draft model for speculation |
| Spec tokens | `--spec-tokens` | None | Number of speculative tokens |

### LoRA

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Enable LoRA | `--enable-lora` | False | Enable LoRA serving |
| Max LoRAs | `--max-loras` | 1 | Max concurrent LoRAs |
| Max LoRA rank | `--max-lora-rank` | 16 | Maximum LoRA rank |
| Max CPU LoRAs | `--max-cpu-loras` | None | Max LoRAs cached on CPU |
| LoRA dtype | `--lora-dtype` | auto | LoRA weight dtype |
| LoRA target modules | `--lora-target-modules` | None | Target modules for LoRA |

### Multimodal

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Limit MM per prompt | `--limit-mm-per-prompt` | none | Max multimodal items per prompt |
| Language model only | `--language-model-only` | False | Ignore multimodal components |
| MM processor kwargs | `--mm-processor-kwargs` | None | Extra kwargs for MM processing |
| MM processor cache gb | `--mm-processor-cache-gb` | 0 | MM processor cache size |

### API Server (Frontend)

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Host | `--host` | None (0.0.0.0) | Bind address |
| Port | `--port` | 8000 | Server port |
| API key | `--api-key` | None | Require API key auth |
| SSL keyfile | `--ssl-keyfile` | None | SSL private key |
| SSL certfile | `--ssl-certfile` | None | SSL certificate |
| CORS origins | `--allowed-origins` | `["*"]` | Allowed CORS origins |
| Chat template | `--chat-template` | None | Custom Jinja chat template |
| Response role | `--response-role` | assistant | Role name in responses |
| Max logprobs | `--max-logprobs` | None | Max logprob values returned |
| Enable auto tool choice | `--enable-auto-tool-choice` | False | Auto tool calling |
| Tool call parser | `--tool-call-parser` | None | Parser for tool calls |
| API server count | `--api-server-count` | None | Number of API server processes |
| Root path | `--root-path` | None | FastAPI root_path for proxy |
| uvicorn log level | `--uvicorn-log-level` | info | Uvicorn logging level |
| Enable request ID headers | `--enable-request-id-headers` | False | Add X-Request-Id |
| H11 max header count | `--h11-max-header-count` | 256 | Max HTTP headers per request |
| H11 max incomplete event size | `--h11-max-incomplete-event-size` | 4MB | Max incomplete HTTP event size |

### Reasoning / Thinking

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Reasoning parser | `--reasoning-parser` | none | Parser for reasoning content |
| Reasoning config | via config | auto | Controls thinking mode behavior |

### Observability

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Disable log stats | `--disable-log-stats` | False | Disable periodic stats logging |
| OTLP traces endpoint | `--otlp-traces-endpoint` | None | OpenTelemetry traces endpoint |
| KV cache metrics | `--kv-cache-metrics` | False | Enable KV cache metrics |
| Cudagraph metrics | `--cudagraph-metrics` | False | Enable CUDA graph metrics |
| MFU metrics | `--enable-mfu-metrics` | False | Enable model FLOPS utilization metrics |

### Performance Tuning

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Block size | `--block-size` | 16 | PagedAttention block size |
| Enable CUDA graph | auto | auto | CUDA graph capture for decode |
| CUDA graph capture sizes | `--cudagraph-capture-sizes` | None | Specific sizes to capture |
| Enable FlashInfer autotune | `--enable-flashinfer-autotune` | False | Auto-tune FlashInfer kernels |
| Attention backend | `--attention-backend` | auto | Attention implementation |
| MoE backend | `--moe-backend` | auto | MoE kernel backend |
| Linear backend | `--linear-backend` | auto | Linear layer backend |
| Optimization level | `--optimization-level` | o1 | `o0`-`o3` optimization levels |
| Performance mode | `--performance-mode` | none | `throughput`, `latency`, etc. |
| Ubatch size | `--ubatch-size` | 1 | Micro-batch size for scheduling |

### vLLM Example Commands

```bash
# Basic: serve a model on single GPU
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000

# 5-GPU tensor parallel for large model
vllm serve Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 5 \
    --gpu-memory-utilization 0.82 \
    --max-model-len 16384 \
    --max-num-seqs 192 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 32768 \
    --block-size 16 \
    --distributed-executor-backend mp

# AWQ quantized model
vllm serve model-awq \
    --quantization awq \
    --dtype float16

# With LoRA support
vllm serve base-model \
    --enable-lora \
    --max-loras 4 \
    --max-lora-rank 64
```

---

## SGLang Parameters (v0.5.8)

Binary: `/opt/venv-sglang/bin/python -m sglang.launch_server` (isolated venv)
Version: 0.5.8 (from PyPI)

Usage: `python -m sglang.launch_server --model <model_path> [options]`

### Model Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Model | `--model` | required | HuggingFace model name or local path |
| Tokenizer | `--tokenizer` | same as model | Tokenizer name or path |
| Tokenizer mode | `--tokenizer-mode` | auto | `auto`, `slow` |
| Trust remote code | `--trust-remote-code` | False | Allow custom code from model repos |
| Revision | `--revision` | none | Git revision to download from |
| Skip tokenizer init | `--skip-tokenizer-init` | False | Skip tokenizer (token-in / token-out mode) |
| Context length | `--context-length` | model default | Max context window |
| Model override | `--model-override` | none | Override model config (JSON) |

### Precision & Quantization

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Dtype | `--dtype` | auto | `auto`, `float16`, `bfloat16`, `float32` |
| Quantization | `--quantization` | none | `awq`, `gptq`, `fp8`, `bitsandbytes`, `gguf` |
| KV cache dtype | `--kv-cache-dtype` | auto | `auto`, `fp8_e5m2`, `fp8_e4m3` |

### Parallelism & Distribution

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Tensor parallel size | `--tp-size` | 1 | Number of GPUs for tensor parallelism |
| Pipeline parallel size | `--pp-size` | 1 | Number of GPUs for pipeline parallelism |
| Data parallel size | `--dp-size` | 1 | Number of data parallel replicas |
| Expert parallel size | `--enable-expert-parallel` | False | Expert parallelism for MoE |
| Distributed executor | `--dist-init-addr` | none | Master node address for multi-node |
| NNODES | `--nnodes` | 1 | Number of nodes |
| Node rank | `--node-rank` | 0 | This node's rank |

### Scheduling & Batching

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Max running requests | `--max-running-requests` | 256 | Max concurrent in-flight requests |
| Max queued requests | `--max-queued-requests` | None | Queue depth for waiting requests |
| Max total tokens | `--max-total-tokens` | None (auto) | Max tokens across all requests |
| Chunked prefill size | `--chunked-prefill-size` | None | Tokens per chunked prefill iteration |
| Enable chunked prefill | `--enable-chunked-prefill` | True | Split long prefills across iterations |
| Prefill threshold | `--prefill-single-batch-size` | 64 | Split large prefill into this size |
| Scheduling policy | `--schedule-policy` | lpf | `lpf` (longest prefix first), `fcfs`, `dfs-weight` |
| Disable flashinfer prefill | `--disable-flashinfer-prefill` | False | Fallback to flash attention prefill |
| Disable flashinfer decode | `--disable-flashinfer-decode` | False | Fallback to flash attention decode |

### Cache Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Memory fraction static | `--mem-fraction-static` | 0.85 | Fraction of GPU memory for static KV pool |
| Page size | `--page-size` | 16 | PagedAttention-style block size |
| Enable radix cache | `--disable-radix-cache` | (enabled) | Disable RadixAttention prefix sharing |
| Enable prefix caching | `--enable-prefix-caching` | (enabled) | Enable prefix caching |
| Disable disk cache | `--disable-disk-cache` | False | Disable on-disk KV cache |
| Host KV cache size | `--host-kv-cache-size` | 0 | CPU-side KV cache size in GB |
| KV cache storage | `--kv-cache-storage` | none | Tiered cache backend (e.g., `sglang.s3` for S3) |

### Speculative Decoding

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Spec algorithm | `--speculative-algorithm` | none | `EAGLE`, `EAGLE3`, `MTP`, `ngram` |
| Spec draft model | `--speculative-draft-model` | none | Draft model path or HF ID |
| Spec draft num tokens | `--speculative-num-draft-tokens` | 5 | Draft tokens per step |
| Spec draft steps | `--speculative-num-steps` | 5 | Draft steps per verification |
| Spec top k | `--speculative-eagle-topk` | 8 | Top-k for EAGLE tree draft |

### LoRA

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Enable LoRA | `--enable-lora` | False | Enable LoRA serving |
| Max LoRAs | `--max-loras` | 1 | Max concurrent LoRAs |
| LoRA backend | `--lora-backend` | triton | `triton`, `flashinfer` |
| LoRA target modules | `--lora-target-modules` | all-linear | Comma-separated target module list |

### Multimodal

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Limit MM per prompt | `--limit-mm-per-prompt` | 1 | Max multimodal items per prompt |
| MM attention backend | `--mm-attention-backend` | sdpa | MM attention implementation |

### API Server (Frontend)

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Host | `--host` | 127.0.0.1 | Bind address |
| Port | `--port` | 30000 | Server port |
| API key | `--api-key` | None | Require API key auth |
| SSL keyfile | `--ssl-keyfile` | None | SSL private key |
| SSL certfile | `--ssl-certfile` | None | SSL certificate |
| CORS origins | `--cors-allowed-origins` | `*` | Allowed CORS origins |
| Chat template | `--chat-template` | from model | Custom Jinja chat template |
| Tool call parser | `--tool-call-parser` | None | Parser for tool calls |
| Reasoning parser | `--reasoning-parser` | None | Extract reasoning content |
| Skip warmup | `--skip-warmup` | False | Skip model warmup on startup |
| Warmup requests | `--warmup-requests` | None | Run N warmup requests at startup |

### Observability

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Enable metrics | `--enable-metrics` | False | Expose Prometheus `/metrics` endpoint |
| Metrics port | `--metrics-port` | None | Separate port for metrics endpoint |
| OTLP traces endpoint | `--otlp-traces-endpoint` | None | OpenTelemetry OTLP endpoint |
| Log level | `--log-level` | info | `debug`, `info`, `warning`, `error` |
| Log requests | `--log-requests` | False | Log every incoming request |
| Log responses | `--log-responses` | False | Log every response payload |
| Request timeout | `--request-timeout` | None | Per-request timeout in seconds |

### Attention Backend

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Attention backend | `--attention-backend` | flashinfer | `flashinfer`, `flashattention`, `triton`, `torch_native` |
| Sampling backend | `--sampling-backend` | flashinfer | `flashinfer`, `torch` |
| Grammar backend | `--grammar-backend` | outlines | `outlines`, `xgrammar`, `llguidance` |

### Performance Tuning

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Enable torch compile | `--enable-torch-compile` | False | Apply torch.compile to model |
| Torch compile backend | `--torch-compile-max-bs` | 32 | Max batch size captured for torch.compile |
| CUDA graph max bs | `--cuda-graph-max-bs` | None | Max batch size for CUDA graph capture |
| Enable CUDA graph | `--disable-cuda-graph` | (enabled) | Disable CUDA graph capture |
| Flashinfer autotune | `--enable-flashinfer-autotune` | False | Auto-tune FlashInfer kernels |
| Num continuous decode steps | `--num-continuous-decode-steps` | 1 | Batched decode iterations |
| MoE runner backend | `--moe-runner-backend` | auto | MoE kernel backend |

### SGLang Example Commands

```bash
# Basic: serve a model on single GPU
python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct --port 8002

# 5-GPU tensor parallel for large model
python -m sglang.launch_server --model Qwen/Qwen2.5-32B-Instruct \
    --tp-size 5 \
    --mem-fraction-static 0.85 \
    --context-length 16384 \
    --max-running-requests 256 \
    --max-queued-requests 1024 \
    --disable-radix-cache

# AWQ quantized model
python -m sglang.launch_server --model model-awq \
    --quantization awq \
    --dtype float16

# EAGLE speculative decoding
python -m sglang.launch_server --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model lmsys/sglang-EAGLE-LLaMA3-Instruct-7B \
    --speculative-num-draft-tokens 5
```

---

## llama.cpp Parameters (TurboQuant build)

Binary: `third_party/llama-cpp-turboquant/build/bin/llama-server`
Version: 1 (7d9715f), branch: `feature-turboquant-kv-cache`

Usage: `llama-server -m <model.gguf> [options]`

### Common / Core Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Model | `-m`, `--model` | required | Path to GGUF model file |
| Context size | `-c`, `--ctx-size` | 0 (from model) | Total prompt context window |
| Predict tokens | `-n`, `--n-predict` | -1 (infinity) | Max tokens to generate |
| GPU layers | `-ngl`, `--gpu-layers` | auto | Layers to offload to VRAM (`all` = 999) |
| Threads | `-t`, `--threads` | -1 (auto) | CPU threads for generation |
| Threads batch | `-tb`, `--threads-batch` | same as -t | CPU threads for batch processing |
| Batch size | `-b`, `--batch-size` | 2048 | Logical max batch size for prefill |
| UBatch size | `-ub`, `--ubatch-size` | 512 | Physical max batch size |
| Flash Attention | `-fa`, `--flash-attn` | auto | `on`, `off`, `auto` |
| Verbose | `-v`, `--verbose` | False | Log all debug messages |
| Log verbosity | `--log-verbosity` | 3 (info) | 0=generic, 1=error, 2=warn, 3=info, 4=debug |
| Log file | `--log-file` | none | Redirect logs to file |

### Multi-GPU & Tensor Split

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Split mode | `-sm`, `--split-mode` | layer | `none`, `layer`, `row`, `tensor` (experimental) |
| Tensor split | `-ts`, `--tensor-split` | auto | GPU weight distribution, e.g. `1,1,1,1,1` |
| Main GPU | `-mg`, `--main-gpu` | 0 | Primary GPU index |
| Device | `--device` | auto | Comma-separated GPU list |
| List devices | `--list-devices` | — | Print available devices and exit |

Split modes:
- `none`: Single GPU only
- `layer` (default): Split layers across GPUs (pipelined)
- `row`: Split weight rows across GPUs (parallelized)
- `tensor`: Split weights and KV across GPUs (EXPERIMENTAL)

### KV Cache Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Cache type K | `-ctk`, `--cache-type-k` | f16 | Key cache data type |
| Cache type V | `-ctv`, `--cache-type-v` | f16 | Value cache data type |
| KV offload | `-kvo`, `--kv-offload` | enabled | Offload KV cache to GPU |
| KV unified | `--kv-unified` | auto | Single unified KV buffer across sequences |
| Cache RAM | `--cache-ram` | 8192 MiB | Max RAM cache size |
| Cache idle slots | `--cache-idle-slots` | enabled | Save/clear idle slots on new task |
| Defrag threshold | `-dt`, `--defrag-thold` | — | DEPRECATED |

Allowed KV cache types: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`, `turbo2`, `turbo3`, `turbo4`

TurboQuant types (unique to this build):
- `turbo4`: ~0.5 bytes/value (~75% VRAM savings vs f16)
- `turbo3`: ~0.375 bytes/value (~81% VRAM savings vs f16)
- `turbo2`: ~0.25 bytes/value (~87% VRAM savings vs f16)

### Server Configuration

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Host | `--host` | 127.0.0.1 | Bind address (or .sock for Unix socket) |
| Port | `--port` | 8080 | Server port |
| Parallel slots | `-np`, `--parallel` | -1 (auto) | Concurrent request slots |
| Cont batching | `-cb`, `--cont-batching` | enabled | Continuous/dynamic batching |
| Timeout | `--timeout` | 600 | Read/write timeout in seconds |
| Threads HTTP | `--threads-http` | -1 (auto) | HTTP request processing threads |
| Reuse port | `--reuse-port` | False | Allow multiple sockets on same port |
| API prefix | `--api-prefix` | none | URL prefix for all endpoints |
| Warmup | `--warmup` | enabled | Run empty warmup on startup |
| Sleep idle | `--sleep-idle-seconds` | -1 (disabled) | Seconds before sleeping when idle |

### Prompt Caching & Context

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Cache prompt | `--cache-prompt` | enabled | Cache prompt computations |
| Cache reuse | `--cache-reuse` | 0 | Min chunk size for KV shifting reuse |
| Context shift | `--context-shift` | disabled | Context shift on infinite generation |
| Keep tokens | `--keep` | 0 (-1=all) | Tokens to keep from initial prompt |
| SWA full | `--swa-full` | False | Full-size sliding window attention cache |
| CTX checkpoints | `--ctx-checkpoints` | 32 | Max context checkpoints per slot |
| Checkpoint every N tokens | `--checkpoint-every-n-tokens` | 8192 | Checkpoint frequency |

### Embedding & Pooling

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Embedding mode | `--embedding` | False | Restrict to embedding use case |
| Reranking | `--rerank` | False | Enable reranking endpoint |
| Pooling | `--pooling` | none | `none`, `mean`, `cls`, `last`, `rank` |

### Model Loading

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Model URL | `--model-url` | none | Download model from URL |
| HF repo | `-hf`, `--hf-repo` | none | HuggingFace repo (auto-downloads) |
| HF file | `-hff`, `--hf-file` | none | Specific file from HF repo |
| HF token | `-hft`, `--hf-token` | none | HuggingFace auth token |
| Docker repo | `--docker-repo` | none | Docker Hub model repo |
| Mmap | `--mmap` | enabled | Memory-map model |
| Mlock | `--mlock` | False | Lock model in RAM |
| Repack | `--repack` | enabled | Weight repacking |
| Check tensors | `--check-tensors` | False | Validate tensor data |
| Override KV | `--override-kv` | none | Override model metadata |
| Override tensor | `-ot`, `--override-tensor` | none | Override tensor buffer type |

### Adaptation (RoPE / YaRN / Scaling)

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| RoPE scaling | `--rope-scaling` | linear | `none`, `linear`, `yarn` |
| RoPE scale | `--rope-scale` | — | Context scaling factor |
| RoPE freq base | `--rope-freq-base` | from model | Base frequency |
| RoPE freq scale | `--rope-freq-scale` | — | Frequency scaling factor |
| YaRN orig ctx | `--yarn-orig-ctx` | 0 | Original context size for YaRN |
| YaRN ext factor | `--yarn-ext-factor` | -1.0 | Extrapolation mix factor |
| YaRN attn factor | `--yarn-attn-factor` | -1.0 | Attention magnitude scale |
| YaRN beta slow | `--yarn-beta-slow` | -1.0 | High correction dim |
| YaRN beta fast | `--yarn-beta-fast` | -1.0 | Low correction dim |

### Sampling Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Temperature | `--temp` / `--temperature` | 0.80 | Sampling temperature |
| Top K | `--top-k` | 40 | Top-k sampling (0 = disabled) |
| Top P | `--top-p` | 0.95 | Nucleus sampling (1.0 = disabled) |
| Min P | `--min-p` | 0.05 | Min-p sampling (0.0 = disabled) |
| Top N Sigma | `--top-nsigma` | -1.0 | Top-n-sigma sampling |
| Typical P | `--typical-p` | 1.0 | Locally typical sampling |
| Mirostat | `--mirostat` | 0 | 0=off, 1=v1, 2=v2 |
| Mirostat LR | `--mirostat-lr` | 0.10 | Mirostat learning rate |
| Mirostat entropy | `--mirostat-ent` | 5.0 | Target entropy |
| Repeat penalty | `--repeat-penalty` | 1.0 | Repeat penalty (1.0 = disabled) |
| Presence penalty | `--presence-penalty` | 0.0 | Alpha presence penalty |
| Frequency penalty | `--frequency-penalty` | 0.0 | Alpha frequency penalty |
| Repeat last N | `--repeat-last-n` | 64 | Tokens to consider for penalty |
| DRY multiplier | `--dry-multiplier` | 0.0 | DRY sampling multiplier |
| DRY base | `--dry-base` | 1.75 | DRY base value |
| DRY allowed length | `--dry-allowed-length` | 2 | DRY allowed length |
| DRY penalty last N | `--dry-penalty-last-n` | -1 | DRY penalty window |
| DRY sequence breaker | `--dry-sequence-breaker` | `\n: "*"` | Sequence breakers |
| Dynatemp range | `--dynatemp-range` | 0.0 | Dynamic temperature range |
| Dynatemp exp | `--dynatemp-exp` | 1.0 | Dynamic temperature exponent |
| XTC probability | `--xtc-probability` | 0.0 | XTC probability |
| XTC threshold | `--xtc-threshold` | 0.1 | XTC threshold |
| Adaptive target | `--adaptive-target` | -1.0 | Adaptive-p target probability |
| Adaptive decay | `--adaptive-decay` | 0.90 | Adaptive-p decay rate |
| Seed | `-s`, `--seed` | -1 (random) | RNG seed |
| Samplers | `--samplers` | penalties;dry;...;temperature | Sampler pipeline |
| Ignore EOS | `--ignore-eos` | False | Continue past EOS |
| Logit bias | `-l`, `--logit-bias` | none | Modify token likelihoods |
| Grammar | `--grammar` | none | BNF grammar constraint |
| JSON schema | `-j`, `--json-schema` | none | JSON schema constraint |

### Speculative Decoding

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Spec type | `--spec-type` | none | `draft-simple`, `draft-eagle3`, `draft-mtp`, `ngram-*` |
| Draft model | `-md`, `--model-draft` | none | Draft model path |
| Draft HF repo | `-hfd`, `--hf-repo-draft` | none | Draft model from HF |
| Draft max tokens | `--spec-draft-n-max` | 16 | Max draft tokens |
| Draft min tokens | `--spec-draft-n-min` | 0 | Min draft tokens |
| Draft split prob | `--spec-draft-p-split` | 0.10 | Split probability |
| Draft min prob | `--spec-draft-p-min` | 0.75 | Min greedy probability |
| Draft NGL | `--spec-draft-ngl` | auto | Draft model GPU layers |
| Draft device | `--spec-draft-device` | auto | Draft model device |
| Draft cache K | `--cache-type-k-draft` | f16 | Draft KV cache key type |
| Draft cache V | `--cache-type-v-draft` | f16 | Draft KV cache value type |

### Multimodal (Vision)

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| MM projector | `-mm`, `--mmproj` | none | Multimodal projector file |
| MM projector URL | `--mmproj-url` | none | URL for projector |
| MM projector auto | `--mmproj-auto` | enabled | Auto-detect projector |
| MM projector offload | `--mmproj-offload` | enabled | Offload projector to GPU |
| Image min tokens | `--image-min-tokens` | from model | Min tokens per image |
| Image max tokens | `--image-max-tokens` | from model | Max tokens per image |

### LoRA & Adapters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| LoRA | `--lora` | none | LoRA adapter path (comma-sep for multiple) |
| LoRA scaled | `--lora-scaled` | none | LoRA with custom scaling |
| LoRA init without apply | `--lora-init-without-apply` | False | Load LoRA without applying |
| Control vector | `--control-vector` | none | Control vector path |
| Control vector scaled | `--control-vector-scaled` | none | Control vector with scaling |
| Control vector layer range | `--control-vector-layer-range` | all | Layer range for control vector |

### Server Endpoints

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Metrics | `--metrics` | disabled | Prometheus `/metrics` endpoint |
| Props | `--props` | disabled | Global properties POST endpoint |
| Slots | `--slots` | enabled | Slots monitoring endpoint |
| Slot save path | `--slot-save-path` | none | Save slot KV cache to disk |
| UI | `--ui` | enabled | Web UI |
| API key | `--api-key` | none | Auth key (comma-sep for multiple) |
| API key file | `--api-key-file` | none | File containing API keys |
| SSL key file | `--ssl-key-file` | none | SSL private key |
| SSL cert file | `--ssl-cert-file` | none | SSL certificate |
| Jinja | `--jinja` | enabled | Jinja template engine |
| Chat template | `--chat-template` | from model | Custom chat template |
| Chat template file | `--chat-template-file` | none | Template from file |
| Chat template kwargs | `--chat-template-kwargs` | none | Extra template params (JSON) |
| Reasoning format | `--reasoning-format` | auto | `none`, `deepseek`, `deepseek-legacy` |
| Reasoning | `--reasoning` | auto | `on`, `off`, `auto` |
| Reasoning budget | `--reasoning-budget` | -1 (unlimited) | Token budget for thinking |
| Skip chat parsing | `--skip-chat-parsing` | False | Pure content parser |
| Prefill assistant | `--prefill-assistant` | enabled | Prefill last assistant message |
| Slot prompt similarity | `--slot-prompt-similarity` | 0.10 | Prompt match threshold for slot reuse |
| Alias | `-a`, `--alias` | none | Model name aliases |
| Tags | `--tags` | none | Model tags |
| Tools | `--tools` | none | Built-in tools (experimental) |
| Models dir | `--models-dir` | none | Directory for router server models |
| Models max | `--models-max` | 4 | Max simultaneous models |
| Models autoload | `--models-autoload` | enabled | Auto-load models |

### CPU / NUMA Optimization

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| CPU mask | `-C`, `--cpu-mask` | "" | CPU affinity mask (hex) |
| CPU range | `-Cr`, `--cpu-range` | — | CPU range for affinity |
| CPU strict | `--cpu-strict` | 0 | Strict CPU placement |
| Priority | `--prio` | 0 | -1=low, 0=normal, 1=medium, 2=high, 3=realtime |
| Poll | `--poll` | 50 | Polling level (0=no polling) |
| NUMA | `--numa` | none | `distribute`, `isolate`, `numactl` |
| CPU MoE | `--cmoe`, `--cpu-moe` | False | Keep MoE weights on CPU |
| CPU MoE layers | `-ncmoe`, `--n-cpu-moe` | 0 | First N layers MoE on CPU |
| Op offload | `--op-offload` | enabled | Offload host tensor ops to device |
| No host | `--no-host` | False | Bypass host buffer |
| Direct IO | `--direct-io` | disabled | Use DirectIO |

### llama.cpp Example Commands

```bash
# Basic: serve a GGUF model
llama-server -m qwen2.5-7b-q4_k_m.gguf --host 0.0.0.0 --port 8001

# 5-GPU with TurboQuant KV cache
llama-server \
    -m qwen2.5-32b-q4_k_m.gguf \
    --host 0.0.0.0 --port 8001 \
    -np 12 \
    -c 16384 \
    -ngl all \
    -ts 1,1,1,1,1 \
    -b 8192 -ub 2048 \
    -t 48 \
    -ctk q8_0 -ctv turbo4 \
    -fa on \
    --cache-prompt \
    --metrics

# High concurrency for small model
llama-server \
    -m qwen2.5-0.5b-q4_k_m.gguf \
    -np 128 \
    -c 16384 \
    -ngl all \
    -ts 1,1,1,1,1 \
    -ctv turbo4 \
    -fa on
```

---

## Key Differences: vLLM vs SGLang vs llama.cpp

| Aspect | vLLM | SGLang | llama.cpp |
|--------|------|--------|-----------|
| Model format | HuggingFace (safetensors) | HuggingFace (safetensors) | GGUF |
| Parallelism | Tensor parallel (NCCL) | Tensor + pipeline + data parallel | Tensor split (layer/row) |
| KV cache | PagedAttention (GPU) | Static pool + RadixAttention tree | Contiguous buffer (GPU/CPU) |
| KV quantization | fp8 only | fp8 only | f16, q8_0, q4_0, turbo2/3/4 |
| Concurrent requests | `max-num-seqs` (scheduler) | `max-running-requests` + queue | `-np` (fixed slots) |
| Batching | Continuous batching | Continuous batching | Continuous batching |
| Prefix sharing | `--enable-prefix-caching` (per-request blocks) | RadixAttention (shared tree, free) | `--cache-prompt` (per slot) |
| Chunked prefill | Built-in | Built-in | N/A |
| Attention backends | FlashInfer, FlashAttention, XFormers | FlashInfer, FlashAttention, Triton | Custom (llama.cpp) |
| Speculative decoding | Draft model | EAGLE / EAGLE3 / MTP / ngram | Draft model + ngram |
| LoRA | Runtime switching | Runtime (triton / flashinfer) | Static load |
| Multimodal | Native | Native | Via mmproj file |
| Metrics | Prometheus | Prometheus | Prometheus |
| Tooling maturity | Production-grade, large community | Production-grade, growing | Single-binary, broad model coverage |
