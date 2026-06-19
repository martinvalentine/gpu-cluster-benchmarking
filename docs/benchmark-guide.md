# Benchmark Guide

- Date: 2026-06-18
- Status: Active
- Scope: Hosting via Docker + benchmarking + visualization

## Prerequisites

```bash
# 1. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repo
git clone --recurse-submodules <repo-url> && cd gpu-cluster-benchmarking

# 3. Sync Python deps
uv sync --group common --group benchmark --group litellm --group monitoring

# 4. Download models
uv run python scripts/download-models.py

# 5. Prepare dataset
./scripts/prepare-dataset.sh
# Outputs: datasets/sharegpt.json (50K Vietnamese samples)

# 6. Increase file descriptor limit (required for high CCU tests)
ulimit -n 65536
```

---

## 1. Hosting via Docker

All backends use the `harmony-bench:cu129` Docker image (the one built from `docker/Dockerfile.vllm-sglang-llama`). Replace with your own tag if you publish to a private registry.

### 1.1 Pull the Image

```bash
docker pull harmony-bench:cu129
```

### 1.2 Build for Blackwell GPU

The image must include Blackwell CUDA architecture (sm_120), otherwise llama.cpp silently falls back to CPU with 50-200 tok/s instead of 2000+ tok/s.

Check your GPU compute capability:
```bash
python3 -c "import torch; print(f'CC: {torch.cuda.get_device_capability()}')"
# Blackwell: (12, 0)
# Ada Lovelace: (8, 6)
```

Build with multiple architectures to support both:
```bash
docker build \
  -f docker/Dockerfile.vllm-sglang-llama \
  -t harmony-bench:cu129 \
  --build-arg CUDA_ARCH="8.6 12.0" \
  --build-arg CMAKE_JOBS=$(nproc) \
  .
```

Verify the build includes your architecture:
```bash
docker run --rm --gpus all IMAGE llama-server --version 2>&1 | grep "ARCHS"
# Should show 860,1200 for Ada + Blackwell
# Should show BLACKWELL_NATIVE_FP4=1 for Blackwell
```

There are two hosting modes:

- **Foreground (--rm)** — server logs visible, Ctrl+C to stop. Preferred for testing. Need a second terminal for benchmarks.
- **Background** — container runs idle, `docker exec` to start servers. Good for automation.

### 1.3 Host vLLM (port 8000)

> **Important:** For high-CCU benchmarks (≥256), add `--max-num-seqs 1024`. vLLM defaults to 256 max sequences — at 900+ concurrent requests, it drops excess connections with "Server disconnected" errors.

With `--served-model-name`, the server reports `Qwen/Qwen2.5-7B-Instruct` — benchmarks work without `-m`.

**Foreground mode** (one command, logs visible):
```bash
docker run --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/datasets:/workspace/datasets \
  --name harmony-bench-vllm \
  harmony-bench:cu129 \
  vllm serve /workspace/models/hf/qwen2.5-7b \
    --port 8000 --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1024 \
    --served-model-name Qwen/Qwen2.5-7B-Instruct
```

**Background mode** (container runs, exec to start):
```bash
docker run -d --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/datasets:/workspace/datasets \
  --name harmony-bench-vllm \
  harmony-bench:cu129 sleep infinity

docker exec -d harmony-bench-vllm vllm serve /workspace/models/hf/qwen2.5-7b \
  --port 8000 --host 0.0.0.0 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 1024 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct
```

Without `--served-model-name`, the server reports `/workspace/models/hf/qwen2.5-7b` (local path). Benchmarks then need `-m Qwen/Qwen2.5-7B-Instruct` for tokenization.

**Wait for ready:**
```bash
docker exec harmony-bench-vllm bash -c \
  'while ! curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 2; done && echo "vLLM ready"'
```

**Test:**
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
```

### 1.4 Host llama.cpp (port 8001)

> **Important — Blackwell GPU:** The image MUST be compiled with `CUDA_ARCH=12.0` (or `"8.6 12.0"`) to use the GPU. Without this, llama.cpp silently runs on CPU even with `-ngl all`. Check the server log for `ARCHS=860,1200 | BLACKWELL_NATIVE_FP4=1` to confirm GPU offloading is active.

> **Important — Context allocation changes with `--kv-unified`:** Without `--kv-unified`, the `-c` context is divided evenly among slots (`per_slot = c / np`). With `--kv-unified`, each slot can use the **full** context (`c`), so `per_slot = c` regardless of `np`. The KV cache is shared across all slots.
>
> Examples without `--kv-unified` (context divided):
> | `-c` | `-np` | per-slot | pp=2048 fits? | tg=128 fits? |
> |:----:|:-----:|:--------:|:-------------:|:------------:|
> | 16384 | 48 | 341 | no | — |
> | 16384 | 8 | 2048 | yes | no |
> | 300000 | 128 | 2343 | yes | yes |
>
> With `--kv-unified`, each slot gets the full context — just ensure the total KV cache (`c`) is large enough for all concurrent requests combined: `c >= sum of (pp + tg) across all active slots`. For safety, `c >= np × (pp + tg)` still works as a conservative formula.

> **Important — `--kv-unified` required for prompt cache under load:** Without `--kv-unified`, llama-server silently disables `--cache-idle-slots` with the warning: `--cache-idle-slots requires --kv-unified, disabling`. This means the prompt cache cannot evict idle slots, causing a **slot-scheduler deadlock** when many concurrent requests (≥51) trigger simultaneous prompt cache save/lookup operations. Always add `--kv-unified` for CCU ≥ 51.
>
> **Even with `--kv-unified`, high CCU (≥51) may still deadlock** if the batch pipeline is resource-starved. The default `n_threads_batch` is only 8, which can stall when 51+ large-prefill requests arrive simultaneously. The fix: increase `-tb` (batch threads) and raise `--batch-size`/`--ubatch-size` to match the concurrent prefill load. See the parameter table below.

**Key parameters for high-CCU performance:**

| Flag | Default | Recommended for CCU ≥ 51 | Description |
|------|:-------:|:------------------------:|-------------|
| `-t N` | 8 | **32** | CPU threads for compute (applies to prompt eval and generation within each batch) |
| `-tb N` | 8 | **32** | Batch threads — the number of threads used for batched prompt processing. **Critical for high CCU.** Default (8) can deadlock under 51+ concurrent prefills. | 
| `--batch-size N` | 2048 | **4096** | Maximum batch size for prompt processing (tokens processed per step). Larger batches improve throughput at high CCU but use more VRAM. |
| `--ubatch-size N` | 512 | **2048** | Micro-batch size — splits batch processing into smaller chunks for better GPU utilization. Increasing helps when 51+ prompts are processed concurrently. |

> **Rule of thumb:** Set `-tb` to match or exceed your CPU thread count (up to `nproc`). Set `--batch-size` to at least `max_ccu × pp` (e.g. `51 × 2048 = ~104K`), capped by available VRAM. Set `--ubatch-size` to `--batch-size / 2` or `max_ccu × tg` whichever is larger.

**Recommended flags by workload (relative ranking, absolute t/s depends on model size):**

| Goal | Config | Notes |
|:-----|:-------|:------|
| Max gen throughput | `-ctk q8_0 -ctv q8_0` | Fastest decode, highest VRAM |
| Max prefill throughput | `-ctk q8_0 -ctv turbo3` | Fastest prompt processing |
| Good balance | `-ctk q8_0 -ctv turbo4` | Recommended starting point |

**Common base flags (add your ctv choice):**
```bash
-ngl all -fa on -ctk q8_0 --kv-unified --cache-prompt \
  -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048
```

**ctv benchmark results (Blackwell RTX PRO 6000, Qwen2.5-7B Q4_K_M):**

> Values below are from 0.5B runs — 7B will be proportionally slower (expect ~4-5× lower throughput). Update these once you have 7B data.

| -ctv | Prefill (tok/s) | Gen (tok/s) | Notes |
|------|:---------------:|:-----------:|-------|
| f16 | — | — | Crashes with `-fa on` |
| q8_0 | — | — | High VRAM usage |
| turbo2 | — | — | — |
| turbo3 | — | — | — |
| turbo4 | — | — | Recommended starting point |

**Low-CCU foreground (np ≤ 48, e.g. testing):**
```bash
docker run --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  --name harmony-bench-llama \
  harmony-bench:cu129 \
  llama-server \
    -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --port 8001 --host 0.0.0.0 \
    -ngl all -c 16384 -np 16 -fa on \
    -ctk q8_0 -ctv q8_0 --kv-unified --cache-prompt
```

**High-CCU foreground (np=256, CCU ladder up to 256, turbo4 for stability):**
```bash
docker run --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  --name harmony-bench-llama \
  harmony-bench:cu129 \
  llama-server \
    -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --port 8001 --host 0.0.0.0 \
    -ngl all -c 557056 -np 256 \
    -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048 \
    -fa on --kv-unified \
    -ctk q8_0 -ctv turbo4 --cache-prompt
```

> `c = 256 × (2048 + 128) = 557056` — unified KV cache, each slot can use up to full 557K context. With `--kv-unified`, the formula ensures the KV cache pool is large enough for all concurrent requests combined.

**Low-CCU background:**
```bash
docker exec -d harmony-bench-llama llama-server \
  -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8001 --host 0.0.0.0 \
  -ngl all -c 16384 -np 16 -fa on \
  -ctk q8_0 -ctv q8_0 --kv-unified --cache-prompt
```

**High-CCU background:**
```bash
docker exec -d harmony-bench-llama llama-server \
  -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8001 --host 0.0.0.0 \
  -ngl all -c 557056 -np 256 \
  -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048 \
  -fa on --kv-unified \
  -ctk q8_0 -ctv turbo4 --cache-prompt
```

**Wait for ready:**
```bash
docker exec harmony-bench-llama bash -c \
  'while ! curl -sf http://localhost:8001/health >/dev/null 2>&1; do sleep 2; done && echo "llama.cpp ready"'
```

**Test:**
```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

### 1.5 Host SGLang (port 8002)

**Foreground mode:**
```bash
docker run --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/datasets:/workspace/datasets \
  --name harmony-bench-sglang \
  harmony-bench:cu129 \
  /opt/venv-sglang/bin/python -m sglang.launch_server \
    --model /workspace/models/hf/qwen2.5-7b \
    --port 8002 --host 0.0.0.0 \
    --mem-fraction-static 0.85 \
    --context-length 32768 \
    --max-running-requests 2048 \
    --max-queued-requests 4096 \
    --trust-remote-code \
    --disable-radix-cache \
    --tp-size 1
```

**Background mode:**
```bash
docker exec -d harmony-bench-sglang /opt/venv-sglang/bin/python -m sglang.launch_server \
  --model /workspace/models/hf/qwen2.5-7b \
  --port 8002 --host 0.0.0.0 \
  --mem-fraction-static 0.85 \
  --context-length 32768 \
  --max-running-requests 2048 \
  --max-queued-requests 4096 \
  --trust-remote-code \
  --disable-radix-cache \
  --tp-size 1

**Wait for ready:**
```bash
docker exec harmony-bench-sglang bash -c \
  'while ! curl -sf http://localhost:8002/v1/models >/dev/null 2>&1; do sleep 2; done && echo "SGLang ready"'
```

**Test:**
```bash
curl -s http://localhost:8002/v1/models | python3 -m json.tool
```

---

## 2. Benchmark Scripts (llama-benchy sweeps)

Each backend has a script that runs sweeps using llama-benchy:
- **Concurrency ladder**: fixed pp=2048, varies CCU (find throughput knee)
- **Prompt length ladder**: fixed CCU=1, varies pp (measure KV cache pressure)
- **Cross-sweep**: CCU ladder at each prompt size (KV cache pressure vs CCU)

### 2.1 vLLM — llama-benchy sweeps

**Script:** `vllm_bench.sh` | **URL:** port 8000 | **Results:** `results/vllm/`

| Flag | Default | Description |
|------|---------|-------------|
| `-u, --url URL` | `http://localhost:8000/v1` | Endpoint URL (with `/v1`) |
| `-m, --model NAME` | auto-detected | HF model name for tokenization (required if server returns non-HF name) |
| `-o, --output DIR` | `./results/vllm` | Results directory |
| `--format FMT` | `json` | Output format: json, csv, md |
| `--ccu-mode MODE` | `mul` | CCU step mode: mul, add |
| `--ccu-start N` | 1 | Starting CCU |
| `--ccu-max N` | 256 (mul) / 64 (add) | Maximum CCU |
| `--ccu-step N` | 2 (mul) / 4 (add) | Step size |
| `--prompt-start N` | 1 | Starting prompt tokens |
| `--prompt-max N` | 16384 | Max prompt tokens (auto-capped to model's `max_model_len`) |
| `--cross-sweep` | off | Run CCU ladder at each prompt size (replaces separate sweeps) |
| `--early-exit` | off | Stop CCU ladder at first hard error per prompt size |
| `--tg N` | 128 | Token generation count |
| `--runs N` | 3 | Runs per test |
| `--native` | off | Also run native vllm bench serve dataset benchmark |
| `--full` | off | Alias for `--native` |
| `--native-max-conc N` | 128 | Native benchmark max concurrency |
| `--native-num-prompts N` | 512 | Native benchmark num prompts |

> **Auto model detection:** Script fetches `${BASE_URL}/models` and checks if the served name is in HF format (`org/model`). If so, it uses that name for tokenization. If the server returns a non-HF name (e.g. GGUF filename), you must pass `--model` explicitly. Also auto-caps `--prompt-max` to the model's `max_model_len` from the endpoint to avoid sending oversized prompts.

```bash
# Default: concurrency ladder (pp=2048, ccu=1-256) + prompt ladder (pp=1-16384)
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1

# Cross-sweep: CCU ladder at each prompt size (KV cache pressure vs CCU)
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 \
  --cross-sweep --early-exit \
  --ccu-mode add --ccu-start 1 --ccu-max 2001 --ccu-step 100 \
  --prompt-start 2048 --prompt-max 16384

# Full sweep: concurrency ladder + prompt ladder + native dataset benchmark
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 --full

# Custom concurrency range
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 \
  --ccu-mode mul --ccu-start 1 --ccu-max 512 --ccu-step 2 --runs 3

# Explicit model (when server returns non-HF name)
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 \
  -m Qwen/Qwen2.5-7B-Instruct
```

### 2.2 llama.cpp — llama-benchy sweeps

**Script:** `llamacpp_bench.sh` | **URL:** port 8001 | **Results:** `results/llamacpp/`

The llama.cpp server returns GGUF filenames (e.g. `qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`) which llama-benchy cannot auto-detect. You **must** pass `--model` explicitly.

| Flag | Default | Description |
|------|---------|-------------|
| `-u, --url URL` | `http://localhost:8001/v1` | Endpoint URL (with `/v1`) |
| `-m, --model NAME` | (auto from server) | **Required.** HF model name for tokenization |
| `-o, --output DIR` | `./results/llamacpp` | Results directory |
| `--format FMT` | `json` | Output format: json, csv, md |
| `--ccu-mode MODE` | `mul` | CCU step mode: mul, add |
| `--ccu-start N` | 1 | Starting CCU |
| `--ccu-max N` | 256 (mul) / 64 (add) | Maximum CCU |
| `--ccu-step N` | 2 (mul) / 4 (add) | Step size |
| `--prompt-start N` | 1 | Starting prompt tokens |
| `--prompt-max N` | 16384 | Maximum prompt tokens |
| `--tg N` | 128 | Token generation count |
| `--runs N` | 3 | Runs per test |

```bash
# Default: concurrency ladder (pp=2048, ccu=1-256) + prompt ladder (pp=1-16384)
./scripts/benchmark/llamacpp_bench.sh -u http://localhost:8001/v1 \
  --model Qwen/Qwen2.5-7B-Instruct

# Custom CCU range with explicit model
./scripts/benchmark/llamacpp_bench.sh -u http://localhost:8001/v1 \
  --ccu-mode add --ccu-start 1 --ccu-max 256 --ccu-step 10 \
  --model Qwen/Qwen2.5-7B-Instruct
```

> **Note:** `llamacpp_bench.sh` does not support `--cross-sweep` or `--native`. Use `vllm_bench.sh` or `sglang_bench.sh` for cross-sweep mode.

### 2.3 SGLang — llama-benchy sweeps

**Script:** `sglang_bench.sh` | **URL:** port 8002 | **Results:** `results/sglang/`

Same feature set as `vllm_bench.sh` (identical code structure):

| Flag | Default | Description |
|------|---------|-------------|
| `-u, --url URL` | `http://localhost:8002/v1` | Endpoint URL (with `/v1`) |
| `-m, --model NAME` | auto-detected | HF model name for tokenization (required if server returns non-HF name) |
| `-o, --output DIR` | `./results/sglang` | Results directory |
| `--format FMT` | `json` | Output format: json, csv, md |
| `--ccu-mode MODE` | `mul` | CCU step mode: mul, add |
| `--ccu-start N` | 1 | Starting CCU |
| `--ccu-max N` | 256 (mul) / 64 (add) | Maximum CCU |
| `--ccu-step N` | 2 (mul) / 4 (add) | Step size |
| `--prompt-start N` | 1 | Starting prompt tokens |
| `--prompt-max N` | 16384 | Maximum prompt tokens |
| `--cross-sweep` | off | Run CCU ladder at each prompt size (replaces separate sweeps) |
| `--early-exit` | off | Stop CCU ladder at first hard error per prompt size |
| `--tg N` | 128 | Token generation count |
| `--runs N` | 3 | Runs per test |
| `--native` | off | Also run native sglang.bench_serving dataset benchmark |
| `--full` | off | Alias for `--native` |
| `--native-max-conc N` | 128 | Native benchmark max concurrency |
| `--native-num-prompts N` | 512 | Native benchmark num prompts |

```bash
# Default sweeps
./scripts/benchmark/sglang_bench.sh -u http://localhost:8002/v1

# Cross-sweep
./scripts/benchmark/sglang_bench.sh -u http://localhost:8002/v1 \
  --cross-sweep --early-exit \
  --ccu-mode add --ccu-start 1 --ccu-max 2001 --ccu-step 100 \
  --prompt-start 2048 --prompt-max 16384

# Full sweep: concurrency ladder + prompt ladder + native dataset benchmark
./scripts/benchmark/sglang_bench.sh -u http://localhost:8002/v1 --full

# Explicit model
./scripts/benchmark/sglang_bench.sh -u http://localhost:8002/v1 \
  -m Qwen/Qwen2.5-7B-Instruct
```

### 2.4 Unified dispatcher

**Script:** `bench.sh`

Routes to the per-backend script. All additional flags are forwarded as-is.

```bash
# Route by -b backend flag
./scripts/benchmark/bench.sh -b vllm
./scripts/benchmark/bench.sh -b llamacpp --model Qwen/Qwen2.5-7B-Instruct
./scripts/benchmark/bench.sh -b sglang --full
./scripts/benchmark/bench.sh -b vllm --cross-sweep
```

---

## 3. np Sweep — Find Optimal Parallel Slots

`scripts/benchmark/find_best_np_llama.sh` sweeps llama-server's `-np` to find the throughput knee — the point where adding more concurrent slots gives diminishing returns.

### Usage

```bash
./scripts/benchmark/find_best_np_llama.sh [options]
```

> **Important — Add `--kv-unified`, `-tb`, `--batch-size` for np ≥ 32:** The sweep script uses `--cache-prompt` by default but does NOT pass `--kv-unified`, `-tb`, `--batch-size`, or `--ubatch-size`. For reliable results at high np, add all flags via the `--fa` workaround (e.g. `--fa "on --kv-unified -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048"`) until the script is updated.

| Flag | Default | Description |
|------|---------|-------------|
| `--image` | `harmony-bench:cu129` | Docker image |
| `--model` | GGUF path | Model inside container |
| `--ctx` | 16384 | Total context (`-c`). Per-slot = ctx / np |
| `--np-list` | `8 16 32 48 64 96 128 192 256` | Space-separated np values |
| `--fa` | `on` | Flash Attention (on/off/auto) |
| `--kv-unified` | off | Unified KV cache (required for np ≥ 32 to avoid deadlock) |
| `-t` | auto | CPU threads for compute (passed to `-t`). Higher = faster batch processing |
| `-tb` | auto | Batch threads (passed to `-tb`). Default 8 can deadlock at high np — increase to 32+ |
| `--batch-size` | server default | Max batch size for prompt processing (passed to `--batch-size`) |
| `--ubatch-size` | server default | Micro-batch size (passed to `--ubatch-size`) |
| `--ctk` | `q8_0` | Cache type K |
| `--ctv` | `turbo4` | Cache type V |
| `--prompt-tokens` | 512 | Synthetic prompt length |
| `--gen-tokens` | 128 | Tokens to generate per request |
| `--auto-ctx` | off | Auto-calc `-c` per rung: `np * (prompt + gen)`, capped at `np * 32768` |
| `--model-ctx` | 32768 | Model's training context ceiling for `--auto-ctx` |
| `--full-sweep` | off | Run all np candidates, skip knee-detection early exit |
| `--request-timeout` | 120 | Max seconds per curl request |

### Important: Per-slot context

Without `--kv-unified`, llama.cpp divides total context evenly among slots: `per_slot = ctx / np`. With `--kv-unified`, each slot gets the full context.

The sweep script currently starts servers **without** `--kv-unified`, so the divided context formula applies:

```
ctx=16384, np=16 → per_slot=1024  (enough for 512-token prompt)
ctx=16384, np=48 → per_slot=341   (too small for 512-token prompt → 400 error)
ctx=16384, np=256 → per_slot=64   (too small for anything)
```

Always ensure `prompt_tokens <= ctx / np` for the highest np in your sweep. If using `--auto-ctx`, this is handled automatically.

### Examples

```bash
# Fast knee detection (small prompts, dense np ladder)
./scripts/benchmark/find_best_np_llama.sh --np-list "16 32 48 64 96 128" \
  --prompt-tokens 64 --gen-tokens 64

# Auto-context: calculates -c per rung automatically
./scripts/benchmark/find_best_np_llama.sh --np-list "16 32 48 64" \
  --prompt-tokens 2000 --gen-tokens 1000 --auto-ctx --request-timeout 600

# Full sweep with Blackwell-optimized cache settings (low CCU, safe without extras)
./scripts/benchmark/find_best_np_llama.sh --np-list "16 32 48 64 96 128 192 256" \
  --fa on --ctk q8_0 --ctv q8_0 \
  --prompt-tokens 64 --gen-tokens 64 --auto-ctx --full-sweep

# High-CCU sweep with --kv-unified and thread/batch tuning (workaround until baked into script)
./scripts/benchmark/find_best_np_llama.sh --np-list "32 48 64 96 128" \
  --fa "on --kv-unified -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048" \
  --ctk q8_0 --ctv turbo4 \
  --prompt-tokens 2000 --gen-tokens 128 --auto-ctx \
  --full-sweep --request-timeout 600
```

### Sample Results (Blackwell RTX PRO 6000, Qwen2.5-7B Q4_K_M, ctv=q8_0)

> Values below are from 0.5B runs. 7B aggregate throughput will be lower but the knee pattern (diminishing returns after a certain np) is similar.

| np | Aggregate tok/s | Gain | Cumulative |
|:--:|:--------------:|:----:|:----------:|
| 16 | — | — | |
| 32 | — | — | |
| 48 | — | — | knee (estimate) |
| 64 | — | — | diminishing returns |

---

## 4. Native Benchmark Scripts

Run framework-native dataset benchmarks via the `--full` flag on each per-backend script:

- vLLM: `vllm_bench.sh --full` (Section 2.1)
- llama.cpp: `llamacpp_bench.sh` (Section 2.2) — sweeps only
- SGLang: `sglang_bench.sh --full` (Section 2.3)

`--full` runs the framework's native benchmark (`vllm bench serve` / `sglang.bench_serving`) plus the llama-benchy sweeps, in one command.

---

## 5. Convert to Summary Table & PNG Visualization

All results are parsed with the unified parser:

**`parse_bench.py`** — generates CSV + TSV + Markdown table:

```bash
# Parse a single session directory
.venv/bin/python scripts/parse_bench.py results/vllm/<session>/

# Scan ALL sessions under results/
.venv/bin/python scripts/parse_bench.py results/ --all

# Markdown table only (no CSV/TSV)
.venv/bin/python scripts/parse_bench.py results/vllm/<session>/ --md-only
```

**Outputs per session:**

| Script | File | Format | Content |
|--------|------|--------|---------|
| parse_bench.py | `*_summary.csv` | CSV | Detailed per-result metrics |
| parse_bench.py | `*_report.tsv` | TSV | Pivoted comparison table (backend × phase × CCU) |
| parse_bench.py | `*_table.md` | Markdown | Aggregate throughput + per-request + TTFT tables |

### 5.1 Visualize cross-sweep results

If you ran with `--cross-sweep`, generate charts:

```bash
.venv/bin/python scripts/visualize_cross_sweep.py results/vllm/<session>/
```

**Output charts:**

| Chart | Content |
|-------|---------|
| `ccu_ladder_pp*.png` | CCU vs aggregate throughput + per-request tok/s + TTFR |
| `ccu_vs_prompt.png` | Bar chart: max CCU achieved at each prompt size |
| `throughput_heatmap.png` | Heatmap: prompt × CCU → throughput (requires 2+ prompt sizes) |

### 5.2 Quick bash pipeline

```bash
# ── vLLM: run sweeps + native dataset → parse → visualize ──
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 --full
SESSION=results/vllm/$(ls -t results/vllm/ | head -1)
.venv/bin/python scripts/parse_bench.py "$SESSION"
.venv/bin/python scripts/visualize_cross_sweep.py "$SESSION"
echo "Results in $SESSION/"
echo "  Report: cat $SESSION/*_report.tsv"
echo "  Charts: ls $SESSION/*.png"

# ── SGLang: run sweeps + native dataset → parse → visualize ──
./scripts/benchmark/sglang_bench.sh -u http://localhost:8002/v1 --full
SESSION=results/sglang/$(ls -t results/sglang/ | head -1)
.venv/bin/python scripts/parse_bench.py "$SESSION"
.venv/bin/python scripts/visualize_cross_sweep.py "$SESSION"
```

---

## 6. What Each Metric Means

| Metric | Definition | Good / Bad |
|--------|-----------|------------|
| **TTFT** | Time to first token (prefill latency) | `< 500ms` good, `> 5s` bad |
| **TPOT** | Time per output token (decode speed) | `< 20ms` (=50+ tok/s) good |
| **ITL** | Inter-token latency (decode stability) | P99/Mean `< 1.5` = stable |
| **tok/s (aggregate)** | Total generation throughput across all users | Higher = better |
| **tok/s/req** | Per-user throughput (aggregate / CCU) | Shows saturation knee |
| **CCU max** | Max concurrency before failure | Higher = more capacity |
| **Prompt tok/s** | Prompt processing throughput | Higher = faster prefill |

### Metric Availability by Benchmark Type

Not all metrics are available from every benchmark. Use `--full` (or `--native`) to get the complete set:

| Metric | llama-benchy only | + --full (native) |
|--------|:---:|:---:|
| output_tok_s | yes | yes |
| tok_s_per_req | yes | yes |
| mean_ttft_ms | yes (mean only) | yes |
| p99_ttft_ms | - | yes (vLLM) |
| mean_tpot_ms | - | yes |
| p99_tpot_ms | - | yes (vLLM) |
| mean_itl_ms | - | yes |
| p99_itl_ms | - | yes (vLLM) |
| successful_requests | - | yes |
| token counts | - | yes |

- **llama-benchy** provides aggregate throughput and mean TTFT — sufficient for CCU/prompt ladder sweeps and finding throughput knees.
- **Native benchmarks** (`vllm bench serve` / `sglang.bench_serving`) provide per-request data with percentiles, TPOT, ITL, and token counts — needed for latency distribution analysis.
- Run `vllm_bench.sh --full` or `sglang_bench.sh --full` to get both in one pass.

---

## 7. Docker Build Notes

### CUDA Architecture for llama.cpp

The image builds llama-cpp-turboquant from source. The `CUDA_ARCH` build arg controls which NVIDIA architectures are supported:

| GPU Generation | Compute Capability | `CUDA_ARCH` value |
|----------------|:------------------:|-------------------|
| Ada Lovelace (RTX 40xx, A40, A100) | sm_86 | `8.6` |
| **Blackwell (RTX PRO 6000, B100, B200)** | **sm_120** | **`12.0`** |
| Hopper (H100, H200) | sm_90 | `9.0` |
| Multiple | — | `"8.6 12.0"` (space-separated) |

Without the correct architecture, `-ngl all` is accepted silently but the model runs entirely on CPU at ~50-200 tok/s.

### Verify the build includes your GPU:

```bash
docker run --rm --gpus all IMAGE llama-server --help 2>&1 | grep "ARCHS"
# Expected for Blackwell: CUDA : ARCHS = 860,1200 | BLACKWELL_NATIVE_FP4 = 1
```

---

## 8. Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Server disconnected` | vLLM drops excess connections. Default `--max-num-seqs` is 256. | Add `--max-num-seqs 1024` to vLLM startup. |
| `Too many open files` | OS file descriptor limit (default 1024). llama-benchy exhausts sockets at high CCU. | Run `ulimit -n 65536` before benchmarking. |
| `Model not in HF format` | vLLM serves model from local path, not `/namespace/model` name. | Add `--served-model-name <hf_id>` to vLLM startup. |
| `GPU OOM` | VRAM exhausted by KV cache or model weights. | Lower `--gpu-memory-utilization` or `--max-model-len`. |
| `Slow gen (~50 tok/s) despite -ngl all` | Image compiled for wrong CUDA architecture (e.g., sm_86 only on Blackwell). Check for `ARCHS=860,1200` in logs. | Rebuild with `--build-arg CUDA_ARCH="8.6 12.0"`. |
| `request (N tokens) exceeds context size` | Per-slot context = ctx / np. Prompt is larger than per-slot limit. | Increase `-c` or reduce `-np`. Or use `--auto-ctx`. |
| `GPU util < 5% with 7B model` | Model is too small to saturate the GPU on high-end hardware. Normal for small models — tests scheduling overhead. | This is expected for smaller models. Throughput knee shows the CPU scheduling limit. |
| `llama-benchy: model not in HF format` | llama.cpp server returns GGUF filename, not HF model ID. | Pass `--model Qwen/Qwen2.5-7B-Instruct` to benchmark script. |
| `Container exited early` | Server failed during initialization. Common causes: wrong CUDA arch, invalid ctv, port conflict. | Check `server_np_*.log` for error messages. |
| `llama-benchy hangs at batch size ≥ 51 (GPU stays at 0-1%)` | **Two possible causes:** (1) Missing `--kv-unified` — prompt cache deadlock. (2) Even with `--kv-unified`, default `-tb 8` (batch threads) is too low for 51+ concurrent prefills, causing batch pipeline stall. | Fix both: (1) Add `--kv-unified`. (2) Increase `-tb` to 32, raise `--batch-size` to 4096, `--ubatch-size` to 2048. See Section 1.4 parameter table. |
| `--cache-idle-slots requires --kv-unified, disabling` warning | You used `--cache-prompt` (or prompt cache is default-on) but omitted `--kv-unified`. This disables idle-slot eviction, causing deadlock at CCU ≥ 51. | Add `--kv-unified` to the server command. Do not suppress the warning. |
| `Prompt eval speed 10-100x slower than llama-bench single-shot` | Large `-c` with many `-np` slots creates a fragmented KV cache. Each prefill operation must index into scattered 2560-token windows within a huge allocation, wasting memory bandwidth. | Use exact-fit context: `c = np × (pp + tg)`. Don't inflate `-c` beyond what the test needs. See Section 1.4. |
| `Server hangs for ALL chat completions after reaching high CCU` | Batch pipeline thread starvation. Default `-tb 8` means only 8 threads handle all batch processing. When 51+ concurrent streaming requests land, threads contend for slot locks, KV cache locks, and the inference queue — no thread makes progress, GPU stays at 0%. | Increase `-tb` to 32 (or `nproc`), `--batch-size` to 4096, `--ubatch-size` to 2048. Also switch from turbo3 to turbo4 if the issue persists. |

---

## 9. Quick Reference

```bash
# ── Setup ──
ulimit -n 65536

# ── Build for Blackwell ──
docker build -f docker/Dockerfile.vllm-sglang-llama \
  -t harmony-bench:cu129 \
  --build-arg CUDA_ARCH="8.6 12.0" --build-arg CMAKE_JOBS=$(nproc) .

# ── Host (foreground — low CCU, np ≤ 16) ──
docker run --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  --name harmony-bench-llama \
  harmony-bench:cu129 \
  llama-server \
    -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --port 8001 --host 0.0.0.0 \
    -ngl all -c 16384 -np 16 -fa on \
    -ctk q8_0 -ctv q8_0 --kv-unified --cache-prompt

# ── Host (foreground — high CCU, np=256, turbo4 with thread/batch tuning) ──
docker run --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  --name harmony-bench-llama \
  harmony-bench:cu129 \
  llama-server \
    -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --port 8001 --host 0.0.0.0 \
    -ngl all -c 557056 -np 256 \
    -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048 \
    -fa on --kv-unified \
    -ctk q8_0 -ctv turbo4 --cache-prompt

# ── Host (background — docker exec) ──
docker run -d --rm --gpus all --ipc=host --network host \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/scripts:/workspace/scripts \
  -v $(pwd)/results:/workspace/results \
  -v $(pwd)/datasets:/workspace/datasets \
  --name harmony-bench-vllm harmony-bench:cu129 sleep infinity

docker exec -d harmony-bench-llama llama-server \
  -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8001 --host 0.0.0.0 \
  -ngl all -c 16384 -np 16 -fa on \
  -ctk q8_0 -ctv q8_0 --kv-unified --cache-prompt

# ── Host (background — high CCU) ──
docker exec -d harmony-bench-llama llama-server \
  -m /workspace/models/gguf/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --port 8001 --host 0.0.0.0 \
  -ngl all -c 557056 -np 256 \
  -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048 \
  -fa on --kv-unified \
  -ctk q8_0 -ctv turbo4 --cache-prompt

# ── Stop ──
docker stop harmony-bench-vllm
docker stop harmony-bench-llama
docker stop harmony-bench-sglang

# ── Bench (llama-benchy) ──
./scripts/benchmark/llamacpp_bench.sh -u http://localhost:8001/v1 \
  --model Qwen/Qwen2.5-7B-Instruct

# ── np Sweep (find best slot count — low CCU, safe without --kv-unified) ──
./scripts/benchmark/find_best_np_llama.sh --np-list "16 32 48 64 96 128" \
  --prompt-tokens 64 --gen-tokens 64 --auto-ctx

# ── np Sweep (high CCU — --fa workaround with thread/batch tuning) ──
./scripts/benchmark/find_best_np_llama.sh --np-list "32 48 64 96 128" \
  --fa "on --kv-unified -t 32 -tb 32 --batch-size 4096 --ubatch-size 2048" \
  --ctk q8_0 --ctv turbo4 \
  --prompt-tokens 2000 --gen-tokens 128 --auto-ctx \
  --full-sweep --request-timeout 600

# ── Bench (cross-sweep) ──
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 \
  --cross-sweep --early-exit \
  --ccu-mode add --ccu-start 1 --ccu-max 2001 --ccu-step 100 \
  --prompt-start 2048 --prompt-max 16384

# ── Bench (native dataset) ──
./scripts/benchmark/vllm_bench.sh -u http://localhost:8000/v1 --full

# ── Parse & Visualize ──
.venv/bin/python scripts/parse_bench.py results/vllm/<session>/
.venv/bin/python scripts/visualize_cross_sweep.py results/vllm/<session>/

# ── Stop ──
docker stop harmony-bench-vllm
docker stop harmony-bench-llama
docker stop harmony-bench-sglang
```
