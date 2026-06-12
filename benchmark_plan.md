

**KẾ HOẠCH BENCHMARK & TRIỂN KHAI**

**LLM SELF-HOSTED TOÀN DIỆN**

6× NVIDIA A40 · vLLM · llama.cpp · SGLang · RunPod

*Phiên bản 2.0 · Production LLMOps · Benchmark với Caching & Load Testing*

| Phạm vi hạ tầng | 6× NVIDIA A40 (384 GB VRAM tổng) · RunPod GPU Pod |
| :---- | :---- |
| Framework đánh giá | vLLM v0.8+ · llama.cpp (CUDA) · SGLang v0.5+ |
| Mô hình kiểm thử | Qwen2.5-0.6B / 7B / 14B / 32B, Llama-3.1-8B |
| Caching Layer | Redis Semantic Cache · vLLM Prefix Cache · SGLang RadixAttention |
| Load Testing | vllm benchmark\_serving · sglang.bench\_serving · llama-benchy · LiteLLM |

  **PHẦN 0 — KẾT QUẢ BENCHMARK MẪU & GIẢI THÍCH CHỈ SỐ**


## **0.1  Kết quả benchmark thực tế (baseline đã đo)**

Đây là output benchmark thực tế từ vLLM benchmark\_serving.py, làm baseline so sánh. Hiểu đúng từng chỉ số là bắt buộc trước khi tối ưu.

| Chỉ số | Giá trị đo được | Ý nghĩa & Nhận xét |
| ----- | :---: | ----- |
| Successful requests | 10 | 10/10 thành công — OK |
| Failed requests | 0 | Không có lỗi — hệ thống ổn định |
| Benchmark duration (s) | 38.33 s | Tổng thời gian benchmark |
| Total input tokens | 1,404 | Tổng token đầu vào |
| Total generated tokens | 1,919 | Tổng token đầu ra |
| Request throughput (req/s) | 0.26 req/s | ⚠ Thấp — chỉ 10 req / 38s do concurrent=10 mà queue |
| Output token throughput (tok/s) | 50.06 tok/s | Tốc độ sinh token trung bình theo thời gian |
| Peak output tok throughput (tok/s) | 493.00 tok/s | Đỉnh thực tế — GPU đang decode nhanh |
| Peak concurrent requests | 10.00 | Tất cả 10 req đồng thời |
| Total token throughput (tok/s) | 86.69 tok/s | In+Out / duration |
| Mean TTFT (ms) | 31,170.69 ms | ⚠ 31 giây\! Dấu hiệu prefill queue dài cần tối ưu |
| Median TTFT (ms) | 31,171.05 ms | Mean ≈ Median → queue đều, không outlier |
| P99 TTFT (ms) | 31,171.58 ms | Gần flat → tất cả request đợi như nhau |
| Mean TPOT (ms) | 9.87 ms | Tốc độ decode \~101 tok/s — tốt với A40 |
| Median TPOT (ms) | 9.93 ms | Ổn định sau khi có token đầu tiên |
| P99 TPOT (ms) | 10.45 ms | P99/Mean ratio \~1.06 — rất ổn định |
| Mean ITL (ms) | 9.54 ms | Inter-token latency trung bình — tốt |
| Median ITL (ms) | 9.19 ms | Ổn định |
| P99 ITL (ms) | 11.08 ms | P99 chỉ \+16% so Mean — decode rất đều |

## **0.2  Phân tích vấn đề & hướng tối ưu**

**PHÂN TÍCH: TTFT 31 giây với 10 concurrent requests**

Root cause: Tất cả 10 request được gửi đồng thời (burst). vLLM phải prefill tuần tự từng request trong KV cache.

Giải pháp: (1) Bật \--enable-prefix-caching để tái dùng prefix. (2) Dùng \--chunked-prefill để prefill song song.

Giải pháp: (3) Bật Redis semantic cache để request tương tự không cần prefill lại.

Decode tốt: TPOT \~9.87ms tương đương \~101 tok/s/req — GPU A40 hoạt động hiệu quả sau prefill.

  **PHẦN A — THIẾT LẬP HẠ TẦNG RUNPOD**


## **A.1  Cấu hình Pod RunPod**

| Thành phần | Thông số kỹ thuật | Mục tiêu / Ghi chú |
| ----- | ----- | ----- |
| GPU | 6× NVIDIA A40 64 GB \= 384 GB VRAM tổng | Kiểm tra NVLink/PCIe: nvidia-smi topo \-m |
| CPU | ≥ 64 cores Intel Xeon / AMD EPYC | I/O, tiền xử lý, CPU-bound của llama.cpp |
| RAM | ≥ 256 GB DDR4/DDR5 ECC | Buffer weights khi swap, CPU offload |
| Storage | ≥ 500 GB NVMe SSD (đọc ≥ 3 GB/s) | Tải model nhanh, lưu kết quả benchmark |
| Kết nối GPU | NVLink Gen3 hoặc PCIe Gen4 ×16 | \~600 GB/s NVLink → giảm độ trễ TP |
| Network | ≥ 10 Gbps internal pod network | Cần thiết cho distributed inference |

## **A.2  Script khởi động Pod (On-Start Script)**

Dán toàn bộ script này vào ô 'On-start Script' khi tạo Pod RunPod. Script cài đầy đủ 3 framework, Redis cho caching, và các công cụ monitoring.

\#\!/bin/bash

set \-euo pipefail

\# ── 1\. System packages ──────────────────────────────────────────

apt-get update \-y && apt-get install \-y \\

  htop nvtop iotop wget curl git tmux jq redis-server

\# ── 2\. Khởi động Redis (semantic cache backend) ──────────────────

\# Redis lắng nghe trên port 6379, dùng cho LiteLLM & RedisSemanticCache

redis-server \--daemonize yes \--maxmemory 8gb \--maxmemory-policy allkeys-lru

\# ── 3\. Python environment ────────────────────────────────────────

pip install \--upgrade pip

\# ── 3\. Khởi dựng serving engines từ nguồn opensource ─────────────  
\# Clone và compile llama-cpp-turboquant (C++ Server từ source)  
git clone https://github.com/TheTom/llama-cpp-turboquant.git /workspace/llama-cpp-turboquant  
cd /workspace/llama-cpp-turboquant  
git checkout feature/turboquant-kv-cache  
cmake \-B build \-DGGML\_CUDA=ON \-DCMAKE\_BUILD\_TYPE=Release \-DCMAKE\_CUDA\_ARCHITECTURES=86  
cmake \--build build \--config Release \-j$(nproc)  
ln \-s /workspace/llama-cpp-turboquant/build/bin/llama-server /usr/local/bin/llama-server  
cd /workspace

\# Clone và cài đặt vLLM từ source  
git clone https://github.com/vllm-project/vllm.git /workspace/vllm  
cd /workspace/vllm  
MAX\_JOBS=4 pip install \-e . \--no-cache-dir  
cd /workspace

\# Clone và cài đặt SGLang từ source  
git clone https://github.com/sgl-project/sglang.git /workspace/sglang  
cd /workspace/sglang/python  
pip install \-e .\[all\] \--no-cache-dir  
cd /workspace

\# ── 4\. Benchmark tools ───────────────────────────────────────────

\# vLLM benchmark script (chuẩn để đo TTFT/TPOT/ITL)

git clone \--depth=1 https://github.com/vllm-project/vllm.git /workspace/vllm\_bench

pip install aiohttp locust pandas numpy tqdm transformers

\# llama-benchy: benchmark llama.cpp với concurrency & depth

pip install llama-benchy

\# LiteLLM: load balancer \+ proxy với caching

pip install 'litellm\[proxy\]\>=1.40.0' redis

\# ── 5\. Monitoring ────────────────────────────────────────────────

pip install prometheus-client psutil gputil

\# ── 6\. Verify ────────────────────────────────────────────────────

nvidia-smi && nvidia-smi topo \-m

python3 \-c "import torch; print(f'CUDA: {torch.cuda.device\_count()} GPUs')"

redis-cli ping  \# Phải in PONG

echo 'RunPod setup complete\!'

## **A.3  Chuẩn bị Model Weights**

Download model về /workspace/models/ trước khi chạy bất kỳ serving nào. Dùng huggingface-cli để tải.

\# Cài HuggingFace CLI

pip install huggingface\_hub

\# Tạo thư mục chuẩn

mkdir \-p /workspace/models/hf /workspace/models/gguf

\# Tải Qwen2.5-32B dạng AWQ (cho vLLM & SGLang)

huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ \\

  \--local-dir /workspace/models/hf/qwen2.5-32b-awq

\# Tải Qwen2.5-32B dạng GGUF Q4\_K\_M (cho llama.cpp)

huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF \\

  \--include 'qwen2.5-32b-instruct-q4\_k\_m.gguf' \\

  \--local-dir /workspace/models/gguf

\# Tải Qwen2.5-0.6B (Phase P0 \- Ultra-Light Load)  
huggingface-cli download Qwen/Qwen2.5-0.6B-Instruct \\  
  \--local-dir /workspace/models/hf/qwen2.5-0.6b \\  
  \--local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-0.6B-Instruct-GGUF \\  
  \--include '\*q4\_k\_m.gguf' \\  
  \--local-dir /workspace/models/gguf \\  
  \--local-dir-use-symlinks False

\# Tải Llama-3.1-8B (Phase P1 \- Light Load)

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \\

  \--local-dir /workspace/models/hf/llama3.1-8b

| Model | Params | vLLM/SGLang Format | llama.cpp Format | VRAM (TP=6) |
| ----- | :---: | ----- | ----- | :---: |
| Qwen2.5-0.6B | 0.6B | Float16/BF16 | GGUF Q4\_K\_M | \~0.2 GB |
| Llama-3.1-8B / Qwen2.5-7B | 7–9B | Float16/BF16 | GGUF Q4\_K\_M | \~3 GB |
| Qwen2.5-14B | 14B | Float16/BF16 | GGUF Q4\_K\_M | \~5 GB |
| Qwen2.5-32B (AWQ) | 32B | AWQ Int4 / GPTQ | GGUF Q4\_K\_M | \~20 GB |

  **PHẦN B — KHỞI ĐỘNG 3 FRAMEWORK VỚI CƠ CHẾ CACHING**


## **B.1  vLLM — Khởi động với Prefix Caching & Chunked Prefill**

vLLM dùng PagedAttention và Continuous Batching. Bật prefix caching để tái dùng KV cache cho các prompt có prefix chung (giảm TTFT đáng kể). Chunked Prefill giúp xử lý nhiều request song song ngay cả khi prefill lớn.

\# ── Khởi động vLLM API Server (port 8000\) ───────────────────────

\# Chạy trong tmux: tmux new \-s vllm

vllm serve \\

  \--model /workspace/models/hf/qwen2.5-32b-awq \\

  \--tensor-parallel-size 6 \\

  \# Dùng 6 GPU chia tensor song song

  \--gpu-memory-utilization 0.87 \\

  \# 87% VRAM cho KV cache, 13% buffer tránh OOM

  \--max-model-len 4096 \\

  \# Context window tối đa — tăng nếu muốn long context

  \--max-num-seqs 256 \\

  \# Số sequence đồng thời tối đa trong một batch

  \--quantization awq \\

  \# Dùng AWQ int4 — tiết kiệm VRAM, tốc độ gần FP16

  \--enable-prefix-caching \\

  \# BẬT: Cache KV prefix. Cùng system prompt → TTFT giảm \~80%

  \--enable-chunked-prefill \\

  \# BẬT: Chia prefill thành chunk nhỏ → giảm TTFT khi nhiều user

  \--max-num-batched-tokens 8192 \\

  \# Tổng token tối đa mỗi batch chunked prefill

  \--enable-metrics \\

  \--metrics-port 9090 \\

  \# Prometheus metrics tại :9090/metrics

  \--swap-space 4 \\

  \# 4 GB CPU RAM làm swap KV cache khi GPU đầy

  \--host 0.0.0.0 \--port 8000 \\

  \--trust-remote-code

**GIẢI THÍCH: Tại sao TTFT baseline \= 31,170ms?**

10 request burst → vLLM phải prefill 10 prompts tuần tự vào KV cache.

Với prefix caching: request 2-10 tái dùng KV của prefix chung → TTFT giảm còn \~200ms.

Với chunked prefill: prefill được chia nhỏ, xen kẽ với decode → TTFT trung bình giảm \~60%.

## **B.2  llama.cpp — Khởi động với CUDA & Parallel Slots**

llama.cpp dùng GGUF quantization chạy trực tiếp trên GPU. \--n\_parallel cho phép nhiều request xử lý song song trong cùng batch KV cache. Flash Attention tích hợp giảm memory bandwidth.

\# ── Khởi động llama.cpp Server (port 8001\) ──────────────────────

\# Chạy trong tmux: tmux new \-s llamacpp

llama-server \\  
  \-m /workspace/models/gguf/qwen2.5-32b-instruct-q4\_k\_m.gguf \\  
  \-ngl \-1 \\  
  \# \-ngl \-1: Offload tất cả layers lên GPU (A40 \= 64 layers)

  \-t 60 \\  
  \# \-t 60: Số threads xử lý

  \-b 2048 \\  
  \# \-b 2048: Batch size xử lý prefill

  \-ub 512 \\  
  \# \-ub 512: Micro-batch size

  \-c 4096 \\  
  \# \-c 4096: Tổng context window

  \-np 8 \\  
  \# \-np 8: Chạy 8 slots song song

  \-ctk q8\_0 \\  
  \# \-ctk q8\_0: Cache key dạng 8-bit

  \-ctv turbo4 \\  
  \# \-ctv turbo4: Cache value dạng TurboQuant 4-bit (tiết kiệm 50% VRAM)

  \-fa on \\  
  \# \-fa on: Bật Flash Attention (bắt buộc cho TurboQuant)

  \--host 0.0.0.0 \--port 8001

**LƯU Ý llama.cpp: Cơ chế \--cache-prompt**

\--cache-prompt true: llama.cpp lưu KV state của prompt vào disk/memory.

Request tiếp theo có cùng prefix → skip prefill → TTFT giảm từ \~31s xuống \~500ms.

Tắt cache để đo baseline sạch: \--cache-prompt false

## **B.3  SGLang — Khởi động với RadixAttention & Cache-Aware LB**

SGLang dùng RadixAttention — tự động cache KV theo cấu trúc cây (radix tree). Mọi request đều được cache tự động, không cần flag đặc biệt. Đây là điểm mạnh nhất của SGLang so với vLLM.

\# ── Khởi động SGLang Server (port 8002\) ─────────────────────────

\# Chạy trong tmux: tmux new \-s sglang

sglang launch\_server \\

  \--model-path /workspace/models/hf/qwen2.5-32b-awq \\

  \--tp 6 \\

  \# Tensor Parallel trên 6 GPU

  \--mem-fraction-static 0.87 \\

  \# 87% GPU memory cho KV cache (tương tự vLLM)

  \--max-total-tokens 1048576 \\

  \# Tổng token pool cho toàn bộ KV cache

  \--chunked-prefill-size 8192 \\

  \# Chunked prefill: 8192 token mỗi chunk

  \--attention-backend flashinfer \\

  \# FlashInfer: attention kernel tốt nhất hiện tại (2025)

  \--quantization awq \\

  \--max-running-requests 256 \\

  \# Tối đa 256 request đồng thời

  \--enable-torch-compile \\

  \# torch.compile tăng decode throughput \~15%

  \--disable-radix-cache false \\

  \# RadixAttention BẬT (default) — đây là caching cốt lõi của SGLang

  \# KHÔNG tắt trừ khi muốn đo baseline không cache

  \--host 0.0.0.0 \--port 8002 \\

  \--trust-remote-code

  **PHẦN C — CƠ CHẾ CACHING 3 TẦNG (Giảm TTFT & Token Cost)**


## **C.1  Tổng quan kiến trúc caching**

| Tầng | Công nghệ | Cơ chế hoạt động | Hiệu quả kỳ vọng |
| ----- | ----- | ----- | ----- |
| L1 — Semantic | Redis \+ LiteLLM | Hash embedding query → lookup cache Redis trước khi gửi GPU | Giảm 30–60% chi phí token, TTFT ≈ 5ms |
| L2 — Prefix KV | vLLM / SGLang | Cache KV tensor của prefix dùng chung (system prompt) | TTFT giảm 70–90% với prompt dài |
| L3 — Prompt | llama.cpp | Lưu KV state toàn bộ prompt ra disk/memory | Identical request: TTFT \~0 |

## **C.2  Thiết lập Redis Semantic Cache với LiteLLM**

LiteLLM Proxy đặt trước tất cả 3 serving engine. Nó kiểm tra Redis trước mỗi request. Nếu câu hỏi tương tự đã được trả lời, trả về ngay mà không tốn GPU.

\# ── File: /workspace/litellm\_config.yaml ────────────────────────

model\_list:

  \# vLLM endpoint

  \- model\_name: qwen32b-vllm

    litellm\_params:

      model: openai/qwen2.5-32b

      api\_base: http://localhost:8000/v1

      api\_key: EMPTY

  \# llama.cpp endpoint

  \- model\_name: qwen32b-llamacpp

    litellm\_params:

      model: openai/qwen2.5-32b

      api\_base: http://localhost:8001/v1

      api\_key: EMPTY

  \# SGLang endpoint

  \- model\_name: qwen32b-sglang

    litellm\_params:

      model: openai/qwen2.5-32b

      api\_base: http://localhost:8002/v1

      api\_key: EMPTY

\# ── Semantic Cache với Redis ─────────────────────────────────────

cache:

  type: redis-semantic       \# Dùng embedding similarity (cosine)

  host: localhost

  port: 6379

  similarity\_threshold: 0.95 \# 95% tương đồng → cache hit

  \# Giảm về 0.90 nếu muốn cache aggressive hơn

  ttl: 3600                  \# Cache sống 1 giờ

\# ── Load Balancer routing ────────────────────────────────────────

router\_settings:

  routing\_strategy: least-busy

  \# Phân tải về server ít request nhất

  \# Thay bằng 'latency-based' để route về server có TTFT thấp nhất

litellm\_settings:

  cache: true

  success\_callback: \["prometheus"\]

  failure\_callback: \["prometheus"\]

\# ── Khởi động LiteLLM Proxy (port 4000\) ─────────────────────────

litellm \--config /workspace/litellm\_config.yaml \\

  \--port 4000 \\

  \--detailed\_debug

\# Test semantic cache:

\# Request 1: Gửi lần đầu (cache miss → GPU xử lý)

curl http://localhost:4000/v1/chat/completions \\

  \-H 'Content-Type: application/json' \\

  \-d '{"model":"qwen32b-vllm","messages":\[{"role":"user",

       "content":"Explain TCP vs UDP"}\],"max\_tokens":256}'

\# Request 2: Câu hỏi gần giống (cache hit → \< 5ms)

curl http://localhost:4000/v1/chat/completions \\

  \-H 'Content-Type: application/json' \\

  \-d '{"model":"qwen32b-vllm","messages":\[{"role":"user",

       "content":"What is the difference between TCP and UDP"}\],

       "max\_tokens":256}'

  **PHẦN D — LỆNH BENCHMARK CHI TIẾT (Đo đủ TTFT/TPOT/ITL/Throughput)**


## **D.1  vLLM benchmark\_serving.py — Cú pháp chuẩn**

Script benchmark\_serving.py của vLLM là chuẩn công nghiệp. Nó đo đầy đủ: TTFT, TPOT, ITL, throughput, thành công/thất bại. Dùng cho cả 3 framework vì tất cả đều OpenAI-compatible.

\# ── Bước 1: Clone vLLM để lấy benchmark script ──────────────────

cd /workspace/vllm\_bench   \# đã clone ở A.2

\# ── Bước 2: Tải & chuẩn bị dataset Tiếng Việt (bkai vi-alpaca) ─────  
python3 \-c '  
import os, json  
try:  
    os.system("pip install datasets pyarrow \-q")  
    from datasets import load\_dataset  
    print("Loading vi-alpaca from Hugging Face...")  
    ds \= load\_dataset("bkai-foundation-models/vi-alpaca", split="train")  
    sharegpt \= \[{"conversations": \[{"from": "human", "value": item\["instruction"\] \+ ("\\n" \+ item\["input"\] if item.get("input") else "")}, {"from": "gpt", "value": item\["output"\]}\]} for item in ds\]  
    os.makedirs("/workspace/datasets", exist\_ok=True)  
    with open("/workspace/datasets/sharegpt.json", "w", encoding="utf-8") as f:  
        json.dump(sharegpt, f, ensure\_ascii=False, indent=2)  
    print("Success preparing Vietnamese dataset\!")  
except Exception as e:  
    print(f"Error: {e}. Falling back to English ShareGPT.")  
    os.system("wget \-q \-O /workspace/datasets/sharegpt.json \\'https://huggingface.co/datasets/anon8231489123/ShareGPT\_Vicuna\_unfiltered/resolve/main/ShareGPT\_V3\_unfiltered\_cleaned\_split.json\\'")  
'

\# ── Phase P0: Ultra-Light Load — 0.6B model ─────────────────────  
for CONCURRENCY in 1 32 64 128 256; do  
  echo "=== Phase P0: concurrency=${CONCURRENCY} \==="  
  python benchmarks/benchmark\_serving.py \\  
    \--backend openai-chat \\  
    \--endpoint /v1/chat/completions \\  
    \--base-url http://localhost:8000 \\  
    \--model /workspace/models/hf/qwen2.5-0.6b \\  
    \--dataset-name sharegpt \\  
    \--dataset-path /workspace/datasets/sharegpt.json \\  
    \--num-prompts 200 \\  
    \--max-concurrency ${CONCURRENCY} \\  
    \--request-rate inf \\  
    \--percentile-metrics ttft,tpot,itl,e2el \\  
    \--save-result \\  
    \--result-dir /workspace/results/p0\_vllm \\  
    \--result-filename "p0\_conc${CONCURRENCY}.json"  
done

\# ── Phase P1: Light Load — 9B model, các mức concurrency ─────────

\# Mục tiêu: Đo TTFT tối ưu, thiết lập baseline TPS

for CONCURRENCY in 1 32 64 128; do

  echo "=== Phase P1: concurrency=${CONCURRENCY} \==="

  python benchmarks/benchmark\_serving.py \\

    \--backend openai-chat \\

    \--endpoint /v1/chat/completions \\

    \--base-url http://localhost:8000 \\

    \# Đổi port: 8001 cho llama.cpp, 8002 cho SGLang

    \--model /workspace/models/hf/llama3.1-8b \\

    \--dataset-name sharegpt \\

    \--dataset-path /workspace/datasets/sharegpt.json \\

    \--num-prompts 200 \\

    \# Tổng số request gửi. Luôn dùng num-prompts \>= 5 \* concurrency

    \--max-concurrency ${CONCURRENCY} \\

    \# Giới hạn concurrent in-flight requests

    \--request-rate inf \\

    \# inf \= gửi tất cả ngay (burst). Dùng số thực (vd: 10\) cho rate control

    \--percentile-metrics ttft,tpot,itl,e2el \\

    \# Báo cáo P50, P95, P99 cho tất cả metrics

    \--save-result \\

    \--result-dir /workspace/results/p1\_vllm \\

    \--result-filename "p1\_conc${CONCURRENCY}.json"

done

\# ── Phase P2: Medium Load — 14B model ───────────────────────────

for CONCURRENCY in 1 16 32 64; do

  echo "=== Phase P2: concurrency=${CONCURRENCY} \==="

  python benchmarks/benchmark\_serving.py \\

    \--backend openai-chat \\

    \--base-url http://localhost:8000 \\

    \--model /workspace/models/hf/qwen2.5-14b \\

    \--dataset-name sharegpt \\

    \--dataset-path /workspace/datasets/sharegpt.json \\

    \--num-prompts $((${CONCURRENCY} \* 8)) \\

    \# num-prompts \= 8x concurrency → đủ warm-up steady state

    \--max-concurrency ${CONCURRENCY} \\

    \--request-rate inf \\

    \--percentile-metrics ttft,tpot,itl,e2el \\

    \--save-result \\

    \--result-dir /workspace/results/p2\_vllm \\

    \--result-filename "p2\_conc${CONCURRENCY}.json"

done

\# ── Phase P3: Heavy Stress — 32B AWQ, tải thực tế ────────────────

\# Đánh giá giới hạn KV Cache và P99 latency

for CONCURRENCY in 1 4 8 16; do

  echo "=== Phase P3: concurrency=${CONCURRENCY} \==="

  python benchmarks/benchmark\_serving.py \\

    \--backend openai-chat \\

    \--base-url http://localhost:8000 \\

    \--model /workspace/models/hf/qwen2.5-32b-awq \\

    \--dataset-name sharegpt \\

    \--dataset-path /workspace/datasets/sharegpt.json \\

    \--num-prompts $((${CONCURRENCY} \* 10)) \\

    \--max-concurrency ${CONCURRENCY} \\

    \--request-rate inf \\

    \--burstiness 1.0 \\

    \# burstiness: 1.0=Poisson, \<1=bursty, \>1=uniform

    \--percentile-metrics ttft,tpot,itl,e2el \\

    \--save-result \\

    \--result-dir /workspace/results/p3\_vllm \\

    \--result-filename "p3\_conc${CONCURRENCY}.json"

done

## **D.2  SGLang bench\_serving — Cú pháp chuẩn**

SGLang có module benchmark riêng: python \-m sglang.bench\_serving. Giao diện tương tự vLLM nhưng có một số tham số khác biệt. Quy tắc: num-prompts \>= 5 × max-concurrency để đo steady-state.

\# ── SGLang benchmark — Phase P0 Ultra-Light Load ─────────────────  
for CONCURRENCY in 1 32 64 128 256; do  
  echo "=== SGLang P0: concurrency=${CONCURRENCY} \==="  
  python3 \-m sglang.bench\_serving \\  
    \--backend sglang \\  
    \--base-url http://localhost:8002 \\  
    \--model /workspace/models/hf/qwen2.5-0.6b \\  
    \--dataset-name sharegpt \\  
    \--dataset-path /workspace/datasets/sharegpt.json \\  
    \--num-prompts $((${CONCURRENCY} \* 5)) \\  
    \--max-concurrency ${CONCURRENCY} \\  
    \--request-rate inf \\  
    \--output-file /workspace/results/p0\_sglang/sglang\_p0\_conc${CONCURRENCY}.jsonl  
done

\# ── SGLang benchmark — Phase P1 Light Load ───────────────────────

for CONCURRENCY in 1 32 64 128; do

  echo "=== SGLang P1: concurrency=${CONCURRENCY} \==="

  python3 \-m sglang.bench\_serving \\

    \--backend sglang \\

    \--base-url http://localhost:8002 \\

    \--model /workspace/models/hf/llama3.1-8b \\

    \--dataset-name sharegpt \\

    \--dataset-path /workspace/datasets/sharegpt.json \\

    \--num-prompts $((${CONCURRENCY} \* 5)) \\

    \# SGLang: num-prompts \>= 5 \* max-concurrency (steady state)

    \--max-concurrency ${CONCURRENCY} \\

    \--request-rate inf \\

    \--output-file /workspace/results/p1\_sglang/sglang\_p1\_conc${CONCURRENCY}.jsonl

    \# JSONL: mỗi dòng là 1 request metric

done

\# ── SGLang benchmark — Phase P3 Heavy Stress ─────────────────────

\# Dùng random dataset để kiểm soát chính xác input/output length

for CONCURRENCY in 1 4 8 16; do

  python3 \-m sglang.bench\_serving \\

    \--backend sglang \\

    \--base-url http://localhost:8002 \\

    \--dataset-name random \\

    \--random-input 512 \\

    \# random-input: số token input trung bình

    \--random-output 256 \\

    \# random-output: số token output tối đa

    \--random-range-ratio 0.5 \\

    \# Biến động ±50% quanh giá trị trung bình

    \--num-prompts $((${CONCURRENCY} \* 5)) \\

    \--max-concurrency ${CONCURRENCY} \\

    \--output-file /workspace/results/p3\_sglang/sglang\_p3\_conc${CONCURRENCY}.jsonl

done

## **D.3  llama-benchy — Benchmark llama.cpp với Context Depth**

llama-benchy là công cụ chuyên dụng cho llama.cpp. Nó đo PP (Prompt Processing \= prefill speed) và TG (Token Generation \= decode speed) ở nhiều độ sâu context (--depth) và mức concurrency khác nhau.

\# ── Cài llama-benchy ─────────────────────────────────────────────

pip install llama-benchy

\# ── Benchmark llama.cpp: tất cả phase ───────────────────────────

\# ── Benchmark llama.cpp: Phase P0 0.6B ───────────────────────────  
llama-benchy \\  
  \--base-url http://localhost:8001/v1 \\  
  \--model qwen2.5-0.6b \\  
  \--pp 128 256 512 \\  
  \--tg 64 128 256 \\  
  \--depth 0 512 2048 \\  
  \--concurrency 1 4 8 \\  
  \--runs 3 \\  
  \--output /workspace/results/llamacpp\_bench\_0.6b.json \\  
  \--format json

llama-benchy \\

  \--base-url http://localhost:8001/v1 \\

  \--model qwen2.5-32b \\

  \# Tên model như khai báo trong llama.cpp server

  \--pp 128 256 512 \\

  \# Prompt Processing: test 3 độ dài prompt (128/256/512 tokens)

  \--tg 64 128 256 \\

  \# Token Generation: đo ở 3 độ dài output (64/128/256 tokens)

  \--depth 0 512 2048 \\

  \# Context depth: 0=fresh, 512=half-full, 2048=near full

  \# depth ảnh hưởng lớn đến TTFT vì KV cache size tăng

  \--concurrency 1 4 8 \\

  \# Phase P1=1, P2=4, P3=8 concurrent clients

  \--runs 3 \\

  \# Lặp 3 lần → báo mean ± std (ổn định hơn)

  \--output /workspace/results/llamacpp\_bench.json \\

  \--format json

  \# Lưu JSON để import vào bảng theo dõi

**GIẢI THÍCH CÁC CHỈ SỐ LLAMA-BENCHY**

PP (tok/s): tốc độ prefill. A40 với Q4\_K\_M 32B nên đạt \~300-500 tok/s.

TG (tok/s): tốc độ decode per-user. 1 user: \~20-30 tok/s. 8 users: \~15-20 tok/s aggregate.

TTFR: Time to First Response (data chunk từ server). Gần giống TTFT của vLLM.

est\_ppt: Estimated Prompt Processing Time \= thời gian thực tế prefill.

depth: Context depth ảnh hưởng TTFT vì attention phải scan KV cache dài hơn.

## **D.4  LiteLLM Load Test — Đo hiệu năng qua Proxy với Caching**

Dùng LiteLLM load\_test để đo toàn bộ stack: Gateway → Cache → Serving Engine. Đây là benchmark phản ánh trải nghiệm người dùng thực tế nhất.

\# ── Tài liệu tham khảo: https://docs.litellm.ai/docs/load\_test ──

\# ── Cách 1: locust (LiteLLM built-in load test) ──────────────────

\# File: /workspace/litellm\_loadtest.py

from locust import HttpUser, task, between

import json, random

PROMPTS \= \[

  "Explain the difference between TCP and UDP in detail",

  "What is the CAP theorem in distributed systems?",

  "Describe the attention mechanism in transformers",

  "How does PagedAttention work in vLLM?",

  "Explain gradient descent and its variants",

\]

class LLMUser(HttpUser):

  wait\_time \= between(0.1, 1.0)

  \# Mỗi user đợi 0.1-1s giữa các request (realistic behavior)

  @task(3)  \# weight 3: hay gặp nhất

  def chat\_cached(self):

    \# Dùng 1 trong 5 prompt cố định → cache hit rate cao

    payload \= {

      "model": "qwen32b-vllm",

      "messages": \[{"role": "user", "content": random.choice(PROMPTS)}\],

      "max\_tokens": 256,

      "stream": True,

    }

    with self.client.post("/v1/chat/completions",

      json=payload, stream=True, catch\_response=True) as resp:

      if resp.status\_code \!= 200:

        resp.failure(f"HTTP {resp.status\_code}")

  @task(1)  \# weight 1: ít hơn, request mới

  def chat\_unique(self):

    \# Request unique → cache miss, đo latency thực từ GPU

    import uuid

    payload \= {

      "model": "qwen32b-vllm",

      "messages": \[{"role": "user",

                     "content": f"Unique request {uuid.uuid4()}"}\],

      "max\_tokens": 64,

    }

    self.client.post("/v1/chat/completions", json=payload)

\# ── Chạy locust ──────────────────────────────────────────────────

\# Terminal 1: locust server

locust \-f /workspace/litellm\_loadtest.py \\

  \--host http://localhost:4000 \\

  \--headless \\

  \--users 50 \\

  \# Tổng 50 virtual users

  \--spawn-rate 5 \\

  \# Thêm 5 users/giây cho đến khi đủ 50

  \--run-time 5m \\

  \--csv /workspace/results/litellm\_locust

  \# Lưu: litellm\_locust\_stats.csv, litellm\_locust\_history.csv

\# ── Cách 2: Python async script đo đầy đủ TTFT/tok/s ───────────

\# File: /workspace/benchmark\_litellm.py

import asyncio, aiohttp, time, json

from statistics import mean, median, quantiles

BASE\_URL \= 'http://localhost:4000'  \# LiteLLM proxy

MODEL \= 'qwen32b-vllm'

async def single\_request(session, prompt, max\_tokens=256):

  start \= time.perf\_counter()

  first\_token\_time \= None

  token\_times \= \[\]

  payload \= {

    'model': MODEL,

    'messages': \[{'role':'user','content': prompt}\],

    'max\_tokens': max\_tokens,

    'stream': True,

  }

  async with session.post(f'{BASE\_URL}/v1/chat/completions',

      json=payload) as resp:

    async for line in resp.content:

      now \= time.perf\_counter()

      if first\_token\_time is None:

        first\_token\_time \= now  \# TTFT đo tại đây

      token\_times.append(now)

  end \= time.perf\_counter()

  return {

    'ttft\_ms': (first\_token\_time \- start) \* 1000,

    'total\_ms': (end \- start) \* 1000,

    'token\_count': len(token\_times),

    'tpot\_ms': mean(\[b-a for a,b in zip(token\_times,token\_times\[1:\])\]) \* 1000

               if len(token\_times) \> 1 else 0,

  }

async def run\_benchmark(concurrency=10, num\_requests=100):

  prompts \= \['Explain TCP vs UDP'\] \* 60 \+ \[f'Unique {i}' for i in range(40)\]

  \# 60% repeated (cache hit) \+ 40% unique (cache miss)

  semaphore \= asyncio.Semaphore(concurrency)

  results \= \[\]

  async with aiohttp.ClientSession() as session:

    async def bounded(p):

      async with semaphore:

        return await single\_request(session, p)

    tasks \= \[bounded(p) for p in prompts\[:num\_requests\]\]

    results \= await asyncio.gather(\*tasks, return\_exceptions=True)

  ok \= \[r for r in results if isinstance(r, dict)\]

  ttfts \= \[r\['ttft\_ms'\] for r in ok\]

  tpots \= \[r\['tpot\_ms'\] for r in ok if r\['tpot\_ms'\] \> 0\]

  print(f'Requests: {len(ok)}/{num\_requests}')

  qs \= quantiles(ttfts, n=100)

  print(f'TTFT  Mean={mean(ttfts):.1f}ms  Median={median(ttfts):.1f}ms  P99={qs\[98\]:.1f}ms')

  if tpots: print(f'TPOT  Mean={mean(tpots):.2f}ms  P99={quantiles(tpots,n=100)\[98\]:.2f}ms')

asyncio.run(run\_benchmark(concurrency=10, num\_requests=100))

  **PHẦN E — BẢNG THEO DÕI KẾT QUẢ BENCHMARK**


## **E.1  Bảng theo dõi đầy đủ — Mẫu điền kết quả**

Điền kết quả vào bảng này sau mỗi lần chạy benchmark. Cột 'Baseline (đo thực tế)' đã được điền với dữ liệu thực tế cung cấp.

| Chỉ số | Baseline(vLLM 10req) | vLLM(với cache) | llama.cpp(với cache) | SGLang(RadixAttn) | Ghi chú / Target |
| ----- | :---: | :---: | :---: | :---: | ----- |
| Successful requests | 10 | \_\_\_ / \_\_\_ | \_\_\_ / \_\_\_ | \_\_\_ / \_\_\_ | Luôn \= num-prompts |
| Failed requests | 0 | \_\_\_ | \_\_\_ | \_\_\_ | Target \= 0 |
| Benchmark duration (s) | 38.33 | \_\_\_ | \_\_\_ | \_\_\_ | Phụ thuộc num-prompts |
| Total input tokens | 1,404 | \_\_\_ | \_\_\_ | \_\_\_ |  |
| Total generated tokens | 1,919 | \_\_\_ | \_\_\_ | \_\_\_ |  |
| Request throughput (req/s) | 0.26 | \_\_\_ | \_\_\_ | \_\_\_ | Tăng với concurrency |
| Output tok throughput (tok/s) | 50.06 | \_\_\_ | \_\_\_ | \_\_\_ | Target \> 200 tok/s |
| Peak output tok/s | 493.00 | \_\_\_ | \_\_\_ | \_\_\_ | GPU capacity thực |
| Peak concurrent requests | 10.00 | \_\_\_ | \_\_\_ | \_\_\_ |  |
| Total token throughput (tok/s) | 86.69 | \_\_\_ | \_\_\_ | \_\_\_ | In+Out / duration |
| Mean TTFT (ms) | 31,170.7 | \_\_\_ | \_\_\_ | \_\_\_ | Target \< 2,000ms (cache) |
| Median TTFT (ms) | 31,171.1 | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P99 TTFT (ms) | 31,171.6 | \_\_\_ | \_\_\_ | \_\_\_ | Target \< 5,000ms |
| Mean TPOT (ms) | 9.87 | \_\_\_ | \_\_\_ | \_\_\_ | \~10ms \= 100 tok/s. Tốt\! |
| Median TPOT (ms) | 9.93 | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P99 TPOT (ms) | 10.45 | \_\_\_ | \_\_\_ | \_\_\_ | P99/Mean \< 1.2 \= ổn |
| Mean ITL (ms) | 9.54 | \_\_\_ | \_\_\_ | \_\_\_ | Inter-token latency |
| Median ITL (ms) | 9.19 | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P99 ITL (ms) | 11.08 | \_\_\_ | \_\_\_ | \_\_\_ | Target \< 15ms |
| Cache hit rate (%) | N/A | \_\_\_ | \_\_\_ | \_\_\_ | LiteLLM metrics |
| GPU VRAM sử dụng (GB) | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | nvidia-smi dmon |

## **E.2  Bảng so sánh theo Phase & Model**

| Phase | Model | Conc. | vLLMTTFT ms | llama.cppTTFT ms | SGLangTTFT ms | vLLMtok/s | Ghi chú |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| P0 Ultra-Light | 0.6B Float16 | 1 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | Baseline 1 user |
| P0 Ultra-Light | 0.6B Float16 | 32 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | Multi-user |
| P0 Ultra-Light | 0.6B Float16 | 64 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P0 Ultra-Light | 0.6B Float16 | 128 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P0 Ultra-Light | 0.6B Float16 | 256 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | Extreme concurrency |
| P1 Light | 9B Float16 | 1 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | Baseline 1 user |
| P1 Light | 9B Float16 | 32 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | Multi-user |
| P1 Light | 9B Float16 | 64 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P1 Light | 9B Float16 | 128 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | Stress test |
| P2 Medium | 14B Float16 | 1 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P2 Medium | 14B Float16 | 32 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | Peak throughput |
| P3 Heavy | 32B AWQ | 1 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ |  |
| P3 Heavy | 32B AWQ | 8 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | KV limit test |
| P3 Heavy | 32B AWQ | 16 | \_\_\_ | \_\_\_ | \_\_\_ | \_\_\_ | OOM boundary |

  **PHẦN F — SCRIPT THU THẬP & PHÂN TÍCH KẾT QUẢ**


## **F.1  Script tự động chạy toàn bộ benchmark**

Script run\_all\_benchmarks.sh chạy tuần tự tất cả phase cho cả 3 framework, lưu kết quả JSON và tạo báo cáo CSV tổng hợp.

\#\!/bin/bash

\# File: /workspace/run\_all\_benchmarks.sh

\# Chạy: bash run\_all\_benchmarks.sh 2\>&1 | tee benchmark\_log.txt

set \-euo pipefail

BENCH\_DIR="/workspace/vllm\_bench"

DATA="/workspace/datasets/sharegpt.json"

RESULTS="/workspace/results"

mkdir \-p $RESULTS/{p0,p1,p2,p3}/{vllm,sglang,llamacpp}

\# ── Hàm helper ──────────────────────────────────────────────────

run\_vllm\_bench() {

  local PHASE=$1 PORT=$2 MODEL=$3 CONC=$4

  local NREQ=$((CONC \* 8))

  echo "\[$(date \+%H:%M:%S)\] vLLM ${PHASE} conc=${CONC}"

  python $BENCH\_DIR/benchmarks/benchmark\_serving.py \\

    \--backend openai-chat \\

    \--base-url http://localhost:${PORT} \\

    \--model ${MODEL} \\

    \--dataset-name sharegpt \\

    \--dataset-path ${DATA} \\

    \--num-prompts ${NREQ} \\

    \--max-concurrency ${CONC} \\

    \--request-rate inf \\

    \--percentile-metrics ttft,tpot,itl,e2el \\

    \--save-result \\

    \--result-dir $RESULTS/${PHASE}/vllm \\

    \--result-filename "conc${CONC}.json" 2\>&1

}

run\_sglang\_bench() {

  local PHASE=$1 PORT=$2 MODEL=$3 CONC=$4

  local NREQ=$((CONC \* 5))

  echo "\[$(date \+%H:%M:%S)\] SGLang ${PHASE} conc=${CONC}"

  python3 \-m sglang.bench\_serving \\

    \--backend sglang \\

    \--base-url http://localhost:${PORT} \\

    \--model ${MODEL} \\

    \--dataset-name sharegpt \\

    \--dataset-path ${DATA} \\

    \--num-prompts ${NREQ} \\

    \--max-concurrency ${CONC} \\

    \--output-file $RESULTS/${PHASE}/sglang/conc${CONC}.jsonl 2\>&1

}

\# ── Phase P0: Ultra-Light Load — Qwen2.5-0.6B ───────────────────  
echo '====== PHASE P0: ULTRA-LIGHT LOAD (0.6B model) \======'  
for CONC in 1 32 64 128 256; do  
  run\_vllm\_bench p0 8000 /workspace/models/hf/qwen2.5-0.6b $CONC  
  run\_sglang\_bench p0 8002 /workspace/models/hf/qwen2.5-0.6b $CONC  
  sleep 5  
done

\# ── Phase P1: Light Load — Llama-3.1-8B ─────────────────────────

echo '====== PHASE P1: LIGHT LOAD (8B model) \======'

for CONC in 1 32 64 128; do

  run\_vllm\_bench p1 8000 /workspace/models/hf/llama3.1-8b $CONC

  run\_sglang\_bench p1 8002 /workspace/models/hf/llama3.1-8b $CONC

  sleep 10  \# Cool-down giữa các lần đo

done

\# ── Phase P2: Medium Load — Qwen2.5-14B ─────────────────────────

echo '====== PHASE P2: MEDIUM LOAD (14B model) \======'

for CONC in 1 16 32 64; do

  run\_vllm\_bench p2 8000 /workspace/models/hf/qwen2.5-14b $CONC

  run\_sglang\_bench p2 8002 /workspace/models/hf/qwen2.5-14b $CONC

  sleep 15

done

\# ── Phase P3: Heavy Stress — Qwen2.5-32B AWQ ────────────────────

echo '====== PHASE P3: HEAVY STRESS (32B AWQ) \======'

for CONC in 1 4 8 16; do

  run\_vllm\_bench p3 8000 /workspace/models/hf/qwen2.5-32b-awq $CONC

  run\_sglang\_bench p3 8002 /workspace/models/hf/qwen2.5-32b-awq $CONC

  sleep 20

done

echo 'All benchmarks complete\! Results in /workspace/results/'

## **F.2  Script phân tích & tổng hợp kết quả JSON**

\# File: /workspace/parse\_results.py

import json, os, csv

from pathlib import Path

RESULTS\_DIR \= Path('/workspace/results')

OUTPUT\_CSV  \= '/workspace/results/benchmark\_summary.csv'

rows \= \[\]

for json\_file in sorted(RESULTS\_DIR.rglob('\*.json')):

  try:

    data \= json.loads(json\_file.read\_text())

  except Exception:

    continue

  \# Trích xuất path để biết phase/framework

  parts \= json\_file.parts

  phase \= \[p for p in parts if p.startswith('p')\]\[0\] if any(p.startswith('p') for p in parts) else '?'

  framework \= \[p for p in parts if p in ('vllm','sglang','llamacpp')\]

  fw \= framework\[0\] if framework else '?'

  rows.append({

    'phase': phase,

    'framework': fw,

    'file': json\_file.name,

    'successful\_requests': data.get('completed', data.get('successful\_requests', 0)),

    'duration\_s': round(data.get('duration', 0), 2),

    'req\_throughput': round(data.get('request\_throughput', 0), 3),

    'output\_tok\_s': round(data.get('output\_throughput', 0), 2),

    'total\_tok\_s': round(data.get('total\_token\_throughput', 0), 2),

    'mean\_ttft\_ms': round(data.get('mean\_ttft\_ms', 0), 2),

    'median\_ttft\_ms': round(data.get('median\_ttft\_ms', 0), 2),

    'p99\_ttft\_ms': round(data.get('p99\_ttft\_ms', 0), 2),

    'mean\_tpot\_ms': round(data.get('mean\_tpot\_ms', 0), 2),

    'p99\_tpot\_ms': round(data.get('p99\_tpot\_ms', 0), 2),

    'mean\_itl\_ms': round(data.get('mean\_itl\_ms', 0), 2),

    'p99\_itl\_ms': round(data.get('p99\_itl\_ms', 0), 2),

  })

with open(OUTPUT\_CSV, 'w', newline='') as f:

  writer \= csv.DictWriter(f, fieldnames=rows\[0\].keys())

  writer.writeheader()

  writer.writerows(rows)

print(f'Parsed {len(rows)} results → {OUTPUT\_CSV}')

\# Import CSV vào Excel/Google Sheets để visualize

  **PHẦN G — KIẾN TRÚC PRODUCTION & BEST PRACTICES**


## **G.1  Layered Architecture Blueprint**

Kiến trúc phân lớp 4 tầng: mỗi tầng có trách nhiệm rõ ràng, tách biệt mối quan tâm, dễ scale độc lập.

| Tầng | Công nghệ | Mục đích thực thi | Cấu hình RunPod |
| ----- | ----- | ----- | ----- |
| L1: Gateway | Nginx / Traefik | Rate limiting (100 req/min/IP), Auth JWT, TLS termination | Pod port 443 → L1 → port 4000 |
| L2: Cache | LiteLLM \+ Redis | Semantic cache: cosine similarity 0.95+. Giảm 30-60% GPU load | Redis port 6379, LiteLLM port 4000 |
| L3: Reliability | LiteLLM Fallback | Circuit breaker: vLLM → SGLang → llama.cpp khi OOM/error | fallback\_models config trong litellm\_config.yaml |
| L4: Serving | vLLM \+ SGLang (TP=6) | GPU inference với Prefix Caching & RadixAttention | vLLM :8000, SGLang :8002, llama.cpp :8001 |

## **G.2  Monitoring: Lấy metrics trong lúc benchmark**

\# ── Script monitor GPU realtime ─────────────────────────────────

\# Chạy trong tmux riêng: tmux new \-s monitor

watch \-n 2 'nvidia-smi \--query-gpu=index,name,memory.used,memory.total,\\

  utilization.gpu,temperature.gpu \--format=csv,noheader'

\# ── Thu thập VRAM usage trong khi benchmark chạy ─────────────────

nvidia-smi dmon \-s mu \-d 5 \-f /workspace/results/gpu\_monitor.csv &

\# \-s mu: memory & utilization; \-d 5: mỗi 5 giây

\# ── vLLM Prometheus metrics (khi \--enable-metrics bật) ───────────

\# Xem tại: http://localhost:9090/metrics

\# Các metric quan trọng:

\# vllm:num\_requests\_running — số request đang xử lý

\# vllm:gpu\_cache\_usage\_perc — % KV cache đã dùng

\# vllm:num\_preemptions\_total — số lần preempt (KV cache full)

\# vllm:time\_to\_first\_token\_seconds — histogram TTFT

curl http://localhost:9090/metrics | grep vllm:gpu\_cache\_usage\_perc

## **G.3  Quick Decision Guide — Chọn Framework nào?**

| Tiêu chí | vLLM | llama.cpp | SGLang |
| ----- | :---: | :---: | ----- |
| Throughput cao (nhiều user) | BEST | OK | VERY GOOD |
| TTFT thấp (1 user) | GOOD | GOOD | BEST |
| VRAM tiết kiệm | OK | BEST | OK |
| RAG / Prefix sharing | GOOD | OK | BEST |
| Dễ triển khai | GOOD | BEST | GOOD |
| Model support | BEST | VERY GOOD | GOOD |
| KHUYẾN NGHỊ | Production APImulti-user | Edge / Dev / Low VRAM | RAG / Chatbot low-latency |

  **PHẦN H — CHECKLIST TRIỂN KHAI & TROUBLESHOOTING**


## **H.1  Checklist trước khi chạy benchmark**

* GPU topology: chạy nvidia-smi topo \-m — kiểm tra NVLink hoặc P2P enabled

* VRAM trống: nvidia-smi — tất cả GPU phải có \<1GB used trước khi load model

* Redis running: redis-cli ping phải trả về PONG

* LiteLLM proxy healthy: curl http://localhost:4000/health

* vLLM server healthy: curl http://localhost:8000/v1/models

* SGLang server healthy: curl http://localhost:8002/v1/models

* Dataset tồn tại: ls \-la /workspace/datasets/sharegpt.json

* Results dir tạo sẵn: mkdir \-p /workspace/results

## **H.2  Troubleshooting TTFT cao (\>10 giây)**

* **Nguyên nhân 1 — Prefill queue:** Tất cả request prefill tuần tự. FIX: Bật \--enable-chunked-prefill và \--max-num-batched-tokens 8192

* **Nguyên nhân 2 — Không có prefix cache:** FIX: Bật \--enable-prefix-caching (vLLM) hoặc \--cache-prompt true (llama.cpp)

* **Nguyên nhân 3 — Model quá lớn:** 32B model với 10 concurrent → prefill \~31s là bình thường. FIX: Dùng model nhỏ hơn hoặc giảm concurrency

* **Nguyên nhân 4 — VRAM fragment:** FIX: Restart server, giảm \--gpu-memory-utilization xuống 0.80

## **H.3  Troubleshooting OOM (Out of Memory)**

* **vLLM OOM:** Giảm \--gpu-memory-utilization từ 0.87 xuống 0.80. Giảm \--max-num-seqs xuống 128\.

* **llama.cpp OOM:** Giảm \--n\_ctx hoặc \--n\_parallel. Tăng swap: thêm \--mmap true

* **SGLang OOM:** Giảm \--mem-fraction-static hoặc \--max-total-tokens. Kiểm tra nvidia-smi khi server đang chạy.

*LLM Self-Hosted Benchmark & Deployment Plan v2.0 · 6× NVIDIA A40 · RunPod · vLLM · llama.cpp · SGLang*


