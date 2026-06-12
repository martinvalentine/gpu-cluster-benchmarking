

**KẾ HOẠCH TRIỂN KHAI SELF-HOST LLM**

**TRÊN RUNPOD PODS**

*Hướng dẫn từng bước chính xác theo tài liệu RunPod chính chủ*

6× NVIDIA A40 · vLLM · llama.cpp · SGLang · Redis Cache · LiteLLM Proxy

| Nền tảng | RunPod Pods — Secure Cloud (T3/T4 data centers) |
| :---- | :---- |
| **GPU target** | 6× NVIDIA A40 64GB \= 384 GB VRAM tổng |
| **Cloud type** | Secure Cloud — độ ổn định cao, IP ổn định |
| **Storage** | Volume Disk 500 GB (persistent per Pod lease) \+ Container disk 50 GB |
| **Ports expose** | HTTP: 8000, 8001, 8002, 4000, 9090 | TCP: 22 (SSH) |
| **Nguồn tài liệu** | https://docs.runpod.io/pods — chính chủ RunPod |

  **PHẦN 0 — TỔNG QUAN RUNPOD PODS (Từ docs.runpod.io)**


## **0.1  Pods là gì? (theo docs.runpod.io/pods/overview)**

Theo tài liệu RunPod chính thức: Pods cung cấp truy cập tức thì vào tài nguyên GPU và CPU mạnh mẽ cho AI Training, Fine-tuning, rendering, và các workload compute-intensive. Bạn có toàn quyền kiểm soát môi trường tính toán, cho phép tùy chỉnh phần mềm, storage và networking theo đúng yêu cầu.

### **0.1.1  Hai loại Cloud của RunPod**

| Loại | Secure Cloud | Community Cloud |
| ----- | ----- | ----- |
| **Hạ tầng** | T3/T4 data centers — độ tin cậy cao | Peer-to-peer (không nhận host mới) |
| **Độ ổn định** | Redundancy cao, IP ổn định | Variable, IP có thể thay đổi khi restart |
| **Phù hợp** | Production, sensitive data — CHỌN CÁI NÀY | Cost-sensitive, dev/test |
| **Giá** | Standard | Thấp hơn |

**LỰA CHỌN CHO DỰ ÁN NÀY: Secure Cloud**

Vì: IP ổn định (không thay đổi khi restart), reliability cao, phù hợp production LLM serving.

Community Cloud IP có thể thay đổi khi Pod restart → không phù hợp cho serving liên tục.

### **0.1.2  Storage trong RunPod Pods (theo docs.runpod.io/pods/storage/types)**

| Loại | Persistence | Mount path | Phù hợp / Giá |
| ----- | ----- | ----- | ----- |
| **Container disk** | MẤT khi stop/restart | System-managed | OS, temp, cache — $0.10/GB/tháng |
| Volume disk | Giữ đến khi terminate Pod | /workspace | Model weights, datasets — $0.10/GB (running) / $0.20/GB (stopped) — KHUYẾN NGHỊ |
| Network Volume | VĨNH VIỄN (độc lập Pod) | /workspace (thay thế volume disk) | Shared data, portable — $0.07/GB/tháng (Không dùng cho dự án này) |

**QUAN TRỌNG: Volume Disk phải được cấu hình dung lượng KHI TẠO Pod — không thể tăng thêm sau.**

Khi tạo Pod: Volume Disk được mount tự động tại /workspace.

Model weights (hàng trăm GB) phải lưu ở Volume Disk (/workspace) để persistent qua các lần Stop/Start Pod.

### **0.1.3  Expose Ports (theo docs.runpod.io/pods/configuration/expose-ports)**

RunPod cho phép expose ports theo 2 cách: HTTP Proxy và TCP Direct. Cần hiểu rõ để cấu hình đúng cho vLLM/SGLang/llama.cpp.

| Phương thức | URL format | Lưu ý quan trọng |
| ----- | ----- | ----- |
| **HTTP Proxy** | https://\[POD\_ID\]-\[PORT\].proxy.runpod.net | Timeout 100 giây (Cloudflare). Dùng cho API calls ngắn. |
| **TCP Direct** | \[PUBLIC\_IP\]:\[EXTERNAL\_PORT\] | Không có timeout. Dùng cho SSH, streaming dài. |

**Cloudflare 100-second timeout cho HTTP Proxy\!**

LLM inference đôi khi mất \>100s → dùng TCP hoặc thiết kế non-blocking API.

SSH qua TCP: port 22 bên trong Pod → mapped ra external port khác nhau mỗi lần reset.

Kiểm tra port mapping: Connect tab → Direct TCP Ports trong RunPod console.

  **PHẦN 1 — CHUẨN BỊ TRƯỚC KHI TẠO POD**


## **Bước 1.1  Tạo tài khoản RunPod & nạp credit**

Thực hiện trên máy local (không cần Pod).

1. Truy cập [https://www.runpod.io](https://www.runpod.io) → Click 'Sign Up'  
2. Điền email, password → Xác nhận email  
3. Vào Billing → Add Credits → Nạp tối thiểu $50 cho lần đầu test (6× A40 tốn \~$3-5/giờ)  
4. Vào Settings → Add payment method để auto-recharge tránh gián đoạn

## **Bước 1.2  Tạo SSH Key & Thêm vào RunPod Account**

Theo [docs.runpod.io/pods/configuration/use-ssh](http://docs.runpod.io/pods/configuration/use-ssh) — SSH key authentication là phương thức an toàn và được khuyến nghị.

\# Trên máy LOCAL (không phải Pod)

\# Tạo SSH key pair

ssh-keygen \-t ed25519 \-C "your\_email@example.com"

\# Nhấn Enter 2 lần để dùng default path và không đặt passphrase

\# Xem public key để copy vào RunPod

cat \~/.ssh/id\_ed25519.pub

\# Output ví dụ: ssh-ed25519 AAAAC3NzaC1lZDI1... your\_email@example.com

5. Copy toàn bộ output của lệnh cat (bắt đầu bằng 'ssh-ed25519')  
6. Vào [https://www.console.runpod.io/user/settings](https://www.console.runpod.io/user/settings)  
7. Tìm phần 'SSH Public Keys' → Paste public key vào → Save

**Copy ĐÚNG public key — bắt đầu bằng 'ssh-ed25519', KHÔNG phải fingerprint SHA256.**

Mỗi key một dòng nếu có nhiều key.

Nếu dùng Windows: dùng WSL hoặc Git Bash để chạy lệnh ssh-keygen.

## **Bước 1.3  Tạo API Key RunPod**

API key cần để quản lý Pod qua CLI hoặc REST API (tùy chọn, nhưng rất hữu ích).

8. Vào [https://www.console.runpod.io/user/settings](https://www.console.runpod.io/user/settings) → Phần 'API Keys'  
9. Click 'Create API Key' → Đặt tên 'llm-benchmark-key' → Copy và lưu ngay (chỉ hiện 1 lần)  
10. Cài RunPod CLI trên máy local (tùy chọn):

\# Cài runpodctl (tùy chọn \- dùng để quản lý Pod qua terminal local)

\# macOS/Linux:

wget \-qO- cli.runpod.net | sudo bash

\# Cấu hình API key:

runpodctl config \--apiKey YOUR\_API\_KEY\_HERE

\# Kiểm tra:

runpodctl pod list

## **Bước 1.4  Hoạch định dung lượng Volume Disk (Lưu model weights)**

Volume Disk là persistent storage gắn liền với vòng đời của Pod. Model weights (Qwen2.5-32B \~20GB, Llama-3.1-8B \~16GB...) phải lưu ở đây (/workspace) để tránh tải lại khi Stop/Restart Pod. Giá: $0.10/GB/tháng (khi running) hoặc $0.20/GB/tháng (khi stopped).

11. Chúng ta sẽ cấu hình dung lượng Volume Disk trực tiếp trong giao diện Deploy Pod ở Phần 3\.  
12. Cấu hình:

| Cấu hình | Volume Disk (gắn trực tiếp khi tạo Pod) |
| :---- | :---- |
| Dung lượng | 500 GB (tối thiểu cho 32B model \+ datasets \+ kết quả benchmark) |
| Data Center | Chọn region có sẵn GPU NVIDIA A40 (ví dụ: US-TX, EU) |
| Giá ước tính | 500 × $0.10 \= $50/tháng (khi running) hoặc $0.20 \= $100/tháng (khi stopped) |

Khi Pod được tạo: /workspace sẽ tự động mount vào Volume Disk này.

  **PHẦN 2 — TẠO CUSTOM POD TEMPLATE (Docker Image)**


Theo [docs.runpod.io/pods/templates/create-custom-template](http://docs.runpod.io/pods/templates/create-custom-template): Template là Docker image đã cài sẵn dependencies. Tạo một lần, dùng nhiều lần. Không phải cài lại mỗi lần tạo Pod.

## **Bước 2.1  Dockerfile cho LLM Serving Stack**

Tạo thư mục dự án trên máy local, viết Dockerfile extending runpod/pytorch base image.

\# Trên máy LOCAL:

mkdir llm-serving-pod && cd llm-serving-pod

touch Dockerfile start.sh requirements.txt

\# ── File: Dockerfile ─────────────────────────────────────────────

\# Base image RunPod PyTorch: đã có CUDA 12.1, PyTorch 2.0, Ubuntu 22.04

FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04

\# Biến môi trường cơ bản

ENV PYTHONUNBUFFERED=1

ENV DEBIAN\_FRONTEND=noninteractive

ENV HF\_HOME=/workspace/hf\_cache

\# HF\_HOME trỏ vào /workspace (Volume Disk) → model cache persistent

WORKDIR /app

\# ── System packages ─────────────────────────────────────────────

RUN apt-get update \-y && apt-get install \-y \\

    htop nvtop iotop wget curl git tmux jq redis-server \\

    openssh-server && \\

    rm \-rf /var/lib/apt/lists/\*

\# ── Clone and install frameworks from source ────────────────────  
\# 1\. llama-cpp-turboquant (C++ server compiled from source)  
RUN git clone https://github.com/TheTom/llama-cpp-turboquant.git /app/llama-cpp-turboquant && \\  
    cd /app/llama-cpp-turboquant && \\  
    git checkout feature/turboquant-kv-cache && \\  
    cmake \-B build \-DGGML\_CUDA=ON \-DCMAKE\_BUILD\_TYPE=Release \-DCMAKE\_CUDA\_ARCHITECTURES=86 && \\  
    cmake \--build build \--config Release \-j$(nproc) && \\  
    ln \-s /app/llama-cpp-turboquant/build/bin/llama-server /usr/local/bin/llama-server

\# 2\. vLLM (compiled from source)  
RUN git clone https://github.com/vllm-project/vllm.git /app/vllm && \\  
    cd /app/vllm && \\  
    MAX\_JOBS=4 pip install \-e . \--no-cache-dir

\# 3\. SGLang (compiled from source)  
RUN git clone https://github.com/sgl-project/sglang.git /app/sglang && \\  
    cd /app/sglang/python && \\  
    pip install \-e .\[all\] \--no-cache-dir

\# ── Other auxiliary python packages ─────────────────────────────  
COPY requirements.txt /app/  
RUN pip install \--no-cache-dir \--upgrade pip && \\  
    pip install \--no-cache-dir \-r /app/requirements.txt

\# ── Copy startup script ──────────────────────────────────────────

COPY start.sh /app/start.sh

RUN chmod \+x /app/start.sh

\# ── SSH: expose port 22 ─────────────────────────────────────────

EXPOSE 22

\# Dùng Option 2: Chạy app sau khi base services (SSH/Jupyter) start

CMD \["/app/start.sh"\]

## **Bước 2.2  requirements.txt**

\# ── File: requirements.txt ───────────────────────────────────────  
\# LiteLLM: Load balancer \+ Semantic Cache proxy  
litellm\[proxy\]\>=1.40.0

\# Benchmark tools  
aiohttp  
locust  
pandas  
numpy  
tqdm  
transformers\>=4.40.0  
huggingface\_hub

\# Monitoring  
prometheus-client  
psutil  
gputil

\# Cache  
redis  
llama-benchy

## **Bước 2.3  start.sh — Script khởi động container**

Theo template doc: start.sh chạy base services (/start.sh) trong background, sau đó chạy ứng dụng. Đây là Option 2 từ RunPod template guide.

\#\!/bin/bash

\# ── File: start.sh ───────────────────────────────────────────────

\# Theo RunPod docs: Option 2 — chạy app sau khi base services start

set \-e

echo '=========================================='

echo ' LLM Serving Stack \- RunPod Startup'

echo '=========================================='

\# ── 1\. Khởi động SSH (base image service) ────────────────────────

\# Inject SSH key từ RunPod environment variable PUBLIC\_KEY

mkdir \-p \~/.ssh && chmod 700 \~/.ssh

echo "$PUBLIC\_KEY" \>\> \~/.ssh/authorized\_keys

chmod 600 \~/.ssh/authorized\_keys

service ssh start

echo '\[OK\] SSH started'

\# ── 2\. Tạo thư mục workspace trên Volume Disk ─────────────────

\# /workspace là mount point của Volume Disk

mkdir \-p /workspace/{models/hf,models/gguf,datasets,results,logs}

mkdir \-p /workspace/hf\_cache

echo '\[OK\] Workspace directories created'

\# ── 3\. Khởi động Redis (semantic cache backend) ──────────────────

redis-server \--daemonize yes \\

  \--maxmemory 8gb \\

  \--maxmemory-policy allkeys-lru \\

  \--logfile /workspace/logs/redis.log

sleep 1

redis-cli ping && echo '\[OK\] Redis started' || echo '\[ERROR\] Redis failed'

\# ── 4\. Download and convert Vietnamese dataset benchmark (nếu chưa có) ──  
if \[ \! \-f /workspace/datasets/sharegpt.json \]; then  
  echo 'Downloading and preparing Vietnamese vi-alpaca dataset...'  
  python3 \-c '  
import os, json  
try:  
    os.system("pip install datasets pyarrow \-q")  
    from datasets import load\_dataset  
    print("Loading vi-alpaca dataset from Hugging Face...")  
    ds \= load\_dataset("bkai-foundation-models/vi-alpaca", split="train")  
    sharegpt \= \[\]  
    for item in ds:  
        prompt \= item\["instruction"\]  
        if item.get("input"):  
            prompt \+= "\\n" \+ item\["input"\]  
        sharegpt.append({  
            "conversations": \[  
                {"from": "human", "value": prompt},  
                {"from": "gpt", "value": item\["output"\]}  
            \]  
        })  
    os.makedirs("/workspace/datasets", exist\_ok=True)  
    with open("/workspace/datasets/sharegpt.json", "w", encoding="utf-8") as f:  
        json.dump(sharegpt, f, ensure\_ascii=False, indent=2)  
    print("Success preparing Vietnamese dataset\!")  
except Exception as e:  
    print(f"Error loading Vietnamese dataset: {e}. Falling back to English ShareGPT.")  
    exit(1)  
' || {  
    echo 'Fallback: Downloading English ShareGPT dataset...'  
    wget \-q \-O /workspace/datasets/sharegpt.json \\  
      'https://huggingface.co/datasets/anon8231489123/ShareGPT\_Vicuna\_unfiltered/resolve/main/ShareGPT\_V3\_unfiltered\_cleaned\_split.json'  
  }  
fi

\# ── 5\. Clone vLLM benchmark scripts ──────────────────────────────

if \[ \! \-d /workspace/vllm\_bench \]; then

  git clone \--depth=1 https://github.com/vllm-project/vllm.git \\

    /workspace/vllm\_bench && echo '\[OK\] vLLM benchmark scripts cloned'

fi

\# ── 6\. Verify GPU ────────────────────────────────────────────────

echo '--- GPU Topology \---'

nvidia-smi \--query-gpu=index,name,memory.total \--format=csv,noheader

nvidia-smi topo \-m

\# ── 7\. Giữ container sống (RunPod cần process chạy) ─────────────

echo '=========================================='

echo ' Setup complete\! Connect via SSH to start serving.'

echo '=========================================='

sleep infinity

## **Bước 2.4  Build Docker image & Push lên Docker Hub**

\# Trên máy LOCAL (cần Docker cài sẵn)

\# Login Docker Hub

docker login

\# Build image (lần đầu \~15-20 phút vì install vLLM, SGLang)

docker build \-t YOUR\_DOCKERHUB\_USERNAME/llm-serving:v1.0 .

\# Test local (không có GPU, chỉ kiểm tra dependencies)

docker run \--rm YOUR\_DOCKERHUB\_USERNAME/llm-serving:v1.0 python \-c \\

  'import vllm; import sglang; import litellm; print("All imports OK")'

\# Push lên Docker Hub

docker push YOUR\_DOCKERHUB\_USERNAME/llm-serving:v1.0

\# Ghi nhớ image name: YOUR\_DOCKERHUB\_USERNAME/llm-serving:v1.0

\# Dùng image này khi tạo Pod ở Bước 3

**Thay YOUR\_DOCKERHUB\_USERNAME bằng username Docker Hub thực của bạn.**

Image phải PUBLIC trên Docker Hub (hoặc cấu hình registry auth trong RunPod Settings).

Private registry: vào RunPod Settings → Container Registry Auth → Add credentials.

## **Bước 2.5  Tạo Pod Template trong RunPod Console**

Theo [docs.runpod.io/pods/templates/overview](http://docs.runpod.io/pods/templates/overview): Template lưu config Pod để deploy nhanh sau này.

13. Vào [https://www.console.runpod.io/user/templates](https://www.console.runpod.io/user/templates) → Click 'New Template'  
14. Điền thông tin:

| Template Name | LLM-Serving-6xA40 |
| :---- | :---- |
| **Container Image** | YOUR\_DOCKERHUB\_USERNAME/llm-serving:v1.0 |
| **Container Disk** | 50 GB (cho OS, pip cache, tmp) |
| Volume Disk | 500 GB (cho model weights, datasets, benchmarks) |
| **Expose HTTP Ports** | 8000, 8001, 8002, 4000, 9090 |
| **Expose TCP Ports** | 22 (SSH) |
| **Docker Command** | (để trống — dùng CMD trong Dockerfile) |

15. Thêm Environment Variables (click 'Add Environment Variable'):

| Key | Value |
| ----- | ----- |
| HF\_TOKEN | hf\_xxxxxxxxxxxx (Hugging Face token để tải model) |
| HF\_HOME | /workspace/hf\_cache |
| VLLM\_PORT | 8000 |
| SGLANG\_PORT | 8002 |
| LLAMACPP\_PORT | 8001 |
| LITELLM\_PORT | 4000 |

16. Click 'Save Template' → Ghi lại Template ID

**HF\_TOKEN là Hugging Face Access Token (tạo tại https://huggingface.co/settings/tokens).**

Cần HF\_TOKEN để tải Llama-3.1-8B (gated model — phải accept license trước).

Dùng Runpod Secrets cho HF\_TOKEN để bảo mật: Settings → Secrets → Tạo secret 'HF\_TOKEN'.

Reference trong template: HF\_TOKEN \= {{ RUNPOD\_SECRET\_HF\_TOKEN }}

  **PHẦN 3 — DEPLOY POD 6× NVIDIA A40**


Đây là bước thực tế tạo Pod trên RunPod. Theo docs.runpod.io/pods/manage-pods.

## **Bước 3.1  Deploy qua Web UI (Khuyến nghị cho lần đầu)**

17. Vào [https://www.console.runpod.io/pods](https://www.console.runpod.io/pods) → Click 'Deploy'  
18. Ở phần đầu: Cấu hình dung lượng Volume Disk

**QUAN TRỌNG — Cấu hình Volume Disk khi chọn GPU:**

Tại phần Storage, thiết lập dung lượng Volume Disk là 500 GB.

Sau khi tạo: /workspace trong Pod sẽ tự động được gán vào 500 GB Volume Disk.

19. Chọn GPU Type: Tìm 'NVIDIA A40' trong danh sách  
20. Đặt GPU Count: 6 (hoặc bắt đầu với 1 để test rẻ hơn)  
21. Cloud Type: Chọn 'Secure Cloud'  
22. Đặt tên Pod: llm-benchmark-6xa40  
23. Click 'Select Template' → Chọn template 'LLM-Serving-6xA40' đã tạo  
24. Xem lại config:

| Pod name | llm-benchmark-6xa40 |
| :---- | :---- |
| **GPU** | 6× NVIDIA A40 64GB |
| **Cloud** | Secure Cloud |
| **Container disk** | 50 GB |
| Volume Disk | 500 GB Volume Disk → mount tại /workspace |
| **HTTP ports** | 8000, 8001, 8002, 4000, 9090 |
| **TCP ports** | 22 (SSH) |
| **Template** | LLM-Serving-6xA40 |

25. Click 'Deploy On-Demand' → Xác nhận  
26. Chờ 3-5 phút để Pod khởi động (status chuyển từ 'Initializing' sang 'Running')

## **Bước 3.2  Deploy qua REST API (Tự động hóa)**

Khi đã có template ID, dùng API để deploy nhanh. Theo [docs.runpod.io/api-reference/pods/POST/pods](http://docs.runpod.io/api-reference/pods/POST/pods).

\# Thay các giá trị: YOUR\_API\_KEY, YOUR\_TEMPLATE\_ID  
curl \--request POST \\  
  \--url https://rest.runpod.io/v1/pods \\  
  \--header 'Authorization: Bearer YOUR\_RUNPOD\_API\_KEY' \\  
  \--header 'Content-Type: application/json' \\  
  \--data '{  
    "name": "llm-benchmark-6xa40",  
    "templateId": "YOUR\_TEMPLATE\_ID",  
    "gpuTypeIds": \["NVIDIA A40"\],  
    "gpuCount": 6,  
    "cloudType": "SECURE",  
    "containerDiskInGb": 50,  
    "volumeInGb": 500,  
    "ports": "8000/http,8001/http,8002/http,4000/http,9090/http,22/tcp"  
  }'

\# Response trả về: podId, status, publicIp, ...

\# Lưu podId để stop/start sau này

## **Bước 3.3  Kiểm tra Pod đã sẵn sàng**

27. Vào RunPod console → Pods page → Tìm Pod vừa tạo  
28. Pod có chấm xanh (Running) → Click Pod để xem chi tiết  
29. Tab 'Telemetry': Nếu thấy GPU metrics → Pod đã sẵn sàng  
30. Tab 'Logs' → 'Container Logs': Xem output của start.sh

**Pod trạng thái 'Running' (chấm xanh) CHƯA chắc services đã sẵn sàng.**

Cách chính xác nhất: Kiểm tra Telemetry tab — nếu có GPU metrics thì Pod live.

Sau khi Pod running: Vào 'Connect' tab để lấy SSH command và HTTP proxy URLs.

## **Bước 3.4  Kết nối SSH vào Pod**

Theo [docs.runpod.io/pods/configuration/use-ssh](http://docs.runpod.io/pods/configuration/use-ssh) — Lấy SSH command từ Connect tab.

31. Click Pod → Tab 'Connect' → Copy lệnh SSH

\# Ví dụ SSH command (lấy từ Connect tab trong console):

ssh 8y5rumuyb50m78-6441103b@ssh.runpod.io \-i \~/.ssh/id\_ed25519

\# Hoặc SSH qua TCP trực tiếp (nếu expose TCP port 22):

ssh root@213.173.109.39 \-p 17445 \-i \~/.ssh/id\_ed25519

\# IP và port lấy từ: Connect → Direct TCP Ports

\# Sau khi kết nối thành công, kiểm tra GPU:

nvidia-smi

nvidia-smi topo \-m  \# Kiểm tra NVLink topology

\# Kiểm tra workspace (Volume Disk mount):

df \-h /workspace

ls /workspace/

\# Kiểm tra Redis:

redis-cli ping  \# Phải ra PONG

  **PHẦN 4 — DOWNLOAD MODEL WEIGHTS (Trong Pod qua SSH)**


Tất cả lệnh từ đây thực hiện TRONG Pod qua SSH. Models lưu vào /workspace (Volume Disk) để persistent.

## **Bước 4.1  Cấu hình Hugging Face CLI**

\# Trong Pod (qua SSH):

\# Login Hugging Face (dùng token đã set trong env var)

huggingface-cli login \--token $HF\_TOKEN

\# Xác nhận login:

huggingface-cli whoami

## **Bước 4.2  Download Models theo Phase**

\# ── Phase P0: Qwen2.5-0.6B (Float16) cho Ultra-Light Load test ──  
huggingface-cli download Qwen/Qwen2.5-0.6B-Instruct \\  
  \--local-dir /workspace/models/hf/qwen2.5-0.6b \\  
  \--local-dir-use-symlinks False

huggingface-cli download Qwen/Qwen2.5-0.6B-Instruct-GGUF \\  
  \--include '\*q4\_k\_m.gguf' \\  
  \--local-dir /workspace/models/gguf \\  
  \--local-dir-use-symlinks False

\# ── Phase P1: Llama-3.1-8B (Float16) cho Light Load test ────────

\# Cần accept license tại: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \\

  \--local-dir /workspace/models/hf/llama3.1-8b \\

  \--local-dir-use-symlinks False

\# \~16 GB — mất 5-10 phút

\# ── Phase P2: Qwen2.5-14B (Float16) cho Medium Load test ─────────

huggingface-cli download Qwen/Qwen2.5-14B-Instruct \\

  \--local-dir /workspace/models/hf/qwen2.5-14b \\

  \--local-dir-use-symlinks False

\# \~28 GB

\# ── Phase P3: Qwen2.5-32B AWQ (Int4) cho Heavy Stress test ───────

huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ \\

  \--local-dir /workspace/models/hf/qwen2.5-32b-awq \\

  \--local-dir-use-symlinks False

\# \~18 GB (AWQ compressed)

\# ── GGUF cho llama.cpp ────────────────────────────────────────────

huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF \\

  \--include 'qwen2.5-32b-instruct-q4\_k\_m\*.gguf' \\

  \--local-dir /workspace/models/gguf \\

  \--local-dir-use-symlinks False

\# \~20 GB

\# Kiểm tra tổng dung lượng:

du \-sh /workspace/models/

**Llama-3.1-8B là gated model — phải accept license trên HuggingFace.co trước.**

Download chạy trong tmux để không bị ngắt khi SSH disconnect:

  tmux new \-s download

  \# Chạy lệnh download

  \# Ctrl+B rồi D để detach, tmux attach \-t download để quay lại

  **PHẦN 5 — KHỞI ĐỘNG 3 SERVING FRAMEWORK**


Mỗi framework chạy trong tmux session riêng. Dùng tmux để process sống khi SSH disconnect.

## **Bước 5.1  Khởi động vLLM (port 8000\)**

\# Trong Pod — tạo tmux session riêng cho vLLM

tmux new \-s vllm

\# Khởi động vLLM API Server

vllm serve \\

  \--model /workspace/models/hf/qwen2.5-32b-awq \\

  \--tensor-parallel-size 6 \\

  \--gpu-memory-utilization 0.87 \\

  \--max-model-len 4096 \\

  \--max-num-seqs 256 \\

  \--quantization awq \\

  \--enable-prefix-caching \\

  \--enable-chunked-prefill \\

  \--max-num-batched-tokens 8192 \\

  \--enable-metrics \\

  \--metrics-port 9090 \\

  \--swap-space 4 \\

  \--host 0.0.0.0 \\

  \--port 8000 \\

  \--trust-remote-code 2\>&1 | tee /workspace/logs/vllm.log

\# Chờ thấy 'Application startup complete' → Ctrl+B rồi D để detach

\# Kiểm tra từ terminal khác:

curl http://localhost:8000/v1/models

Truy cập qua RunPod HTTP Proxy URL (từ Connect tab):

\# URL format: https://\[POD\_ID\]-8000.proxy.runpod.net/v1/models

\# Ví dụ: https://abc123xyz-8000.proxy.runpod.net/v1/models

curl https://YOUR\_POD\_ID-8000.proxy.runpod.net/v1/models

## **Bước 5.2  Khởi động llama.cpp (port 8001\)**

\# Tạo tmux session mới

tmux new \-s llamacpp

llama-server \\  
  \-m /workspace/models/gguf/qwen2.5-32b-instruct-q4\_k\_m.gguf \\  
  \-ngl \-1 \\  
  \-t 60 \\  
  \-b 2048 \\  
  \-ub 512 \\  
  \-c 4096 \\  
  \-np 8 \\  
  \-ctk q8\_0 \\  
  \-ctv turbo4 \\  
  \-fa on \\  
  \--host 0.0.0.0 \\  
  \--port 8001 2\>&1 | tee /workspace/logs/llamacpp.log

\# Ctrl+B rồi D để detach

## **Bước 5.3  Khởi động SGLang (port 8002\)**

\# Tạo tmux session mới

tmux new \-s sglang

sglang launch\_server \\

  \--model-path /workspace/models/hf/qwen2.5-32b-awq \\

  \--tp 6 \\

  \--mem-fraction-static 0.87 \\

  \--max-total-tokens 1048576 \\

  \--chunked-prefill-size 8192 \\

  \--attention-backend flashinfer \\

  \--quantization awq \\

  \--max-running-requests 256 \\

  \--enable-torch-compile \\

  \--host 0.0.0.0 \\

  \--port 8002 \\

  \--trust-remote-code 2\>&1 | tee /workspace/logs/sglang.log

\# Ctrl+B rồi D để detach

## **Bước 5.4  Khởi động LiteLLM Proxy với Redis Cache (port 4000\)**

\# Tạo file cấu hình LiteLLM

cat \> /workspace/litellm\_config.yaml \<\< 'EOF'

model\_list:

  \- model\_name: qwen32b-vllm

    litellm\_params:

      model: openai/qwen2.5-32b

      api\_base: http://localhost:8000/v1

      api\_key: EMPTY

  \- model\_name: qwen32b-llamacpp

    litellm\_params:

      model: openai/qwen2.5-32b

      api\_base: http://localhost:8001/v1

      api\_key: EMPTY

  \- model\_name: qwen32b-sglang

    litellm\_params:

      model: openai/qwen2.5-32b

      api\_base: http://localhost:8002/v1

      api\_key: EMPTY

cache:

  type: redis-semantic

  host: localhost

  port: 6379

  similarity\_threshold: 0.95

  ttl: 3600

router\_settings:

  routing\_strategy: least-busy

litellm\_settings:

  cache: true

EOF

\# Khởi động LiteLLM Proxy

tmux new \-s litellm

litellm \--config /workspace/litellm\_config.yaml \--port 4000 \\

  2\>&1 | tee /workspace/logs/litellm.log

\# Ctrl+B D để detach

\# Test:

curl http://localhost:4000/health

## **Bước 5.5  Kiểm tra tất cả services**

\# Kiểm tra tất cả services đang chạy:

echo '=== vLLM \==='

curl \-s http://localhost:8000/v1/models | python3 \-m json.tool

echo '=== llama.cpp \==='

curl \-s http://localhost:8001/v1/models | python3 \-m json.tool

echo '=== SGLang \==='

curl \-s http://localhost:8002/v1/models | python3 \-m json.tool

echo '=== LiteLLM \==='

curl \-s http://localhost:4000/health | python3 \-m json.tool

echo '=== Redis \==='

redis-cli ping

echo '=== GPU VRAM \==='

nvidia-smi \--query-gpu=index,memory.used,memory.total \--format=csv

echo '=== tmux sessions \==='

tmux ls

  **PHẦN 6 — CHẠY BENCHMARK (Trong Pod qua SSH)**


## **Bước 6.1  Access URLs — Cách gọi API từ bên ngoài Pod**

RunPod expose HTTP services qua proxy URL format: https://\[POD\_ID\]-\[PORT\].proxy.runpod.net

| Service | URL bên ngoài Pod | URL bên trong Pod |
| ----- | ----- | ----- |
| **vLLM API** | https://\[POD\_ID\]-8000.proxy.runpod.net | http://localhost:8000 |
| **llama.cpp API** | https://\[POD\_ID\]-8001.proxy.runpod.net | http://localhost:8001 |
| **SGLang API** | https://\[POD\_ID\]-8002.proxy.runpod.net | http://localhost:8002 |
| **LiteLLM Proxy** | https://\[POD\_ID\]-4000.proxy.runpod.net | http://localhost:4000 |
| **Prometheus** | https://\[POD\_ID\]-9090.proxy.runpod.net | http://localhost:9090 |

**Benchmark nên chạy TỪ BÊN TRONG Pod (localhost) để tránh HTTP proxy 100s timeout.**

Dùng SSH để vào Pod → chạy benchmark script từ terminal trong Pod.

Nếu muốn benchmark từ bên ngoài: dùng TCP direct connection (không qua Cloudflare).

## **Bước 6.2  Chạy benchmark vLLM (từ trong Pod)**

\# Trong Pod — tạo tmux session benchmark

tmux new \-s bench

\# Phase P0: Ultra-Light Load — 0.6B model  
mkdir \-p /workspace/results/p0\_vllm  
for CONC in 1 32 64 128 256; do  
  echo "=== vLLM P0 concurrency=${CONC} \==="  
  python benchmarks/benchmark\_serving.py \\  
    \--backend openai-chat \\  
    \--base-url http://localhost:8000 \\  
    \--model /workspace/models/hf/qwen2.5-0.6b \\  
    \--dataset-name sharegpt \\  
    \--dataset-path /workspace/datasets/sharegpt.json \\  
    \--num-prompts $((CONC \* 8)) \\  
    \--max-concurrency ${CONC} \\  
    \--request-rate inf \\  
    \--percentile-metrics ttft,tpot,itl,e2el \\  
    \--save-result \\  
    \--result-dir /workspace/results/p0\_vllm \\  
    \--result-filename "conc${CONC}.json"  
  sleep 5  
done  
echo 'P0 vLLM complete\!'

\# Phase P1: Light Load — 8B model

cd /workspace/vllm\_bench

mkdir \-p /workspace/results/p1\_vllm

for CONC in 1 32 64 128; do

  echo "=== vLLM P1 concurrency=${CONC} \==="

  python benchmarks/benchmark\_serving.py \\

    \--backend openai-chat \\

    \--base-url http://localhost:8000 \\

    \--model /workspace/models/hf/llama3.1-8b \\

    \--dataset-name sharegpt \\

    \--dataset-path /workspace/datasets/sharegpt.json \\

    \--num-prompts $((CONC \* 8)) \\

    \--max-concurrency ${CONC} \\

    \--request-rate inf \\

    \--percentile-metrics ttft,tpot,itl,e2el \\

    \--save-result \\

    \--result-dir /workspace/results/p1\_vllm \\

    \--result-filename "conc${CONC}.json"

  sleep 10

done

echo 'P1 vLLM complete\!'

## **Bước 6.3  Chạy benchmark SGLang**

\# Phase P0: Ultra-Light Load — 0.6B model  
mkdir \-p /workspace/results/p0\_sglang  
for CONC in 1 32 64 128 256; do  
  echo "=== SGLang P0 concurrency=${CONC} \==="  
  python3 \-m sglang.bench\_serving \\  
    \--backend sglang \\  
    \--base-url http://localhost:8002 \\  
    \--dataset-name sharegpt \\  
    \--dataset-path /workspace/datasets/sharegpt.json \\  
    \--num-prompts $((CONC \* 5)) \\  
    \--max-concurrency ${CONC} \\  
    \--request-rate inf \\  
    \--output-file /workspace/results/p0\_sglang/conc${CONC}.jsonl  
  sleep 5  
done  
echo 'P0 SGLang complete\!'

mkdir \-p /workspace/results/p1\_sglang

for CONC in 1 32 64 128; do

  echo "=== SGLang P1 concurrency=${CONC} \==="

  python3 \-m sglang.bench\_serving \\

    \--backend sglang \\

    \--base-url http://localhost:8002 \\

    \--dataset-name sharegpt \\

    \--dataset-path /workspace/datasets/sharegpt.json \\

    \--num-prompts $((CONC \* 5)) \\

    \--max-concurrency ${CONC} \\

    \--request-rate inf \\

    \--output-file /workspace/results/p1\_sglang/conc${CONC}.jsonl

  sleep 10

done

## **Bước 6.4  Chạy benchmark llama.cpp**

mkdir \-p /workspace/results/llamacpp

\# Phase P0: Ultra-Light Load — 0.6B model  
llama-benchy \\  
  \--base-url http://localhost:8001/v1 \\  
  \--model qwen2.5-0.6b \\  
  \--pp 128 256 512 \\  
  \--tg 64 128 256 \\  
  \--depth 0 512 2048 \\  
  \--concurrency 1 4 8 \\  
  \--runs 3 \\  
  \--output /workspace/results/llamacpp/bench\_0.6b.json \\  
  \--format json

llama-benchy \\

  \--base-url http://localhost:8001/v1 \\

  \--model qwen2.5-32b \\

  \--pp 128 256 512 \\

  \--tg 64 128 256 \\

  \--depth 0 512 2048 \\

  \--concurrency 1 4 8 \\

  \--runs 3 \\

  \--output /workspace/results/llamacpp/bench.json \\

  \--format json

  **PHẦN 7 — QUẢN LÝ POD & TỐI ƯU CHI PHÍ**


## **Bước 7.1  Stop Pod khi không dùng (Tiết kiệm chi phí)**

Theo docs.runpod.io/pods/manage-pods: Stop Pod giải phóng GPU. Data tại /workspace (Volume Disk) vẫn an toàn. Khi stopped chỉ bị tính phí Volume Disk ($0.20/GB/tháng).

\# Cách 1: Web UI

\# Pods page → Expand Pod → Click Stop button (icon vuông)

\# Cách 2: CLI

runpodctl pod stop $RUNPOD\_POD\_ID

\# Cách 3: REST API

curl \--request POST \\

  \--url "https://rest.runpod.io/v1/pods/YOUR\_POD\_ID/stop" \\

  \--header 'Authorization: Bearer YOUR\_API\_KEY'

\# Schedule auto-stop sau 4 giờ (chạy trong Pod):

sleep 4h && runpodctl pod stop $RUNPOD\_POD\_ID &

## **Bước 7.2  Xuất kết quả benchmark ra local**

Trước khi terminate Pod, export kết quả về máy local. Theo docs: dùng SCP qua TCP SSH.

\# Từ máy LOCAL — copy kết quả từ Pod về local

\# Port SSH lấy từ: Connect tab → Direct TCP Ports

scp \-P SSH\_PORT \-r root@POD\_PUBLIC\_IP:/workspace/results ./benchmark\_results/

\# Ví dụ cụ thể:

scp \-P 17445 \-r root@213.173.109.39:/workspace/results ./llm\_benchmark\_results/

\# Hoặc dùng Cloud Sync (RunPod UI):

\# Pod page → Cloud Sync → Chọn AWS S3 / GCS / Azure

\# Source: /workspace/results

\# Destination: s3://your-bucket/benchmark-results/

## **Bước 7.3  Ước tính chi phí**

| Tài nguyên | Giá ước tính | Ghi chú |
| ----- | ----- | ----- |
| **6× A40 64GB (Secure Cloud)** | \~$3-5/giờ | Giá chính xác: Pods page → Filter A40 → xem giá |
| Volume Disk 500 GB | $50-100/tháng | $0.10/GB/tháng khi Pod chạy, $0.20/GB/tháng khi Pod dừng |
| **Container disk 50 GB (running)** | \~$0.2/ngày | Mất khi terminate Pod |
| **Benchmark 8 giờ** | \~$24-40 | Stop Pod ngay khi xong |
| **Savings Plan** | Giảm \~30-40% | Xem: docs.runpod.io/pods/pricing |

**Stop Pod ngay sau khi xong benchmark để không tốn tiền GPU giờ.**

Terminate Pod nếu không cần nữa — ⚠ LƯU Ý: Toàn bộ data tại Volume Disk (/workspace) sẽ BỊ XÓA khi terminate Pod.

Kiểm tra billing: console.runpod.io → Billing → Usage để theo dõi chi phí thực tế.

## **Bước 7.4  Terminate Pod (sau khi xong tất cả)**

\# Web UI: Pods page → Expand Pod → Stop trước → Terminate (trash icon)

\# CLI:

runpodctl pod delete YOUR\_POD\_ID

\# REST API:

curl \--request DELETE \\

  \--url "https://rest.runpod.io/v1/pods/YOUR\_POD\_ID" \\

  \--header 'Authorization: Bearer YOUR\_API\_KEY'

\# LƯU Ý: Toàn bộ data tại /workspace (Volume Disk) sẽ BỊ XÓA hoàn toàn sau khi terminate Pod.

\# Hãy tải các kết quả benchmark quan trọng về máy local trước khi terminate.

\# Xóa Volume Disk: Storage → Volume Disks → Delete (nếu không cần nữa)

  **PHẦN 8 — CHECKLIST TRIỂN KHAI & TROUBLESHOOTING**


## **Checklist triển khai theo thứ tự**

| \# | Bước | Kiểm tra / Kết quả mong đợi |
| :---: | ----- | ----- |
| ☐ | Bước 1.1: Tạo tài khoản & nạp credit | Credit balance \> $50 |
| ☐ | Bước 1.2: Thêm SSH key | Key hiện trong Settings → SSH Public Keys |
| ☐ | Bước 1.3: Tạo API key | Key lưu an toàn |
| ☐ | Bước 1.4: Hoạch định Volume Disk 500 GB | Thiết lập dung lượng Volume Disk trong Template & Deploy UI |
| ☐ | Bước 2.1-2.3: Viết Dockerfile \+ start.sh | Files tạo xong trên máy local |
| ☐ | Bước 2.4: Build & push Docker image | docker push thành công, image public |
| ☐ | Bước 2.5: Tạo Template trong RunPod | Template ID ghi lại |
| ☐ | Bước 3.1: Deploy Pod 6× A40 | Pod status: Running (chấm xanh) |
| ☐ | Bước 3.3: Kiểm tra Telemetry | GPU metrics hiện trong Telemetry tab |
| ☐ | Bước 3.4: SSH vào Pod | nvidia-smi thấy 6 GPU A40 |
| ☐ | Bước 4: Download models | ls /workspace/models/ thấy đủ thư mục |
| ☐ | Bước 5: Khởi động vLLM/llama.cpp/SGLang | curl localhost:8000/v1/models trả về JSON |
| ☐ | Bước 5.4: Khởi động LiteLLM \+ Redis | curl localhost:4000/health → OK |
| ☐ | Bước 6: Chạy benchmark | Results JSON trong /workspace/results/ |
| ☐ | Bước 7.2: Export kết quả | scp thành công về máy local |
| ☐ | Bước 7.1: Stop Pod | GPU released, không tốn tiền GPU |

## **Troubleshooting thường gặp**

| Vấn đề | Giải pháp (theo RunPod docs) |
| ----- | ----- |
| **Pod stuck 'Initializing'** | Check Container Logs → Thường do start.sh lỗi. Thêm 'sleep infinity' vào cuối CMD. |
| **SSH hỏi password** | SSH key chưa đúng. Kiểm tra: Settings → SSH Keys. Copy đúng public key (bắt đầu ssh-ed25519). |
| **HTTP 524 timeout** | Cloudflare 100s limit. Chạy benchmark từ BÊN TRONG Pod (localhost), không qua proxy URL. |
| **OCI runtime create failed** | CUDA mismatch. Deploy → Additional filters → CUDA Versions → Chọn 12.1. |
| Volume Disk hết dung lượng | Do tải quá nhiều model hoặc logs quá nặng. Xóa bớt cache HF ở /workspace/hf\_cache hoặc files logs cũ. |
| **GPU count \= 0 sau restart** | Capacity thay đổi. Theo docs: Zero GPU issue. Stop → Start lại hoặc chọn GPU khác. |
| **vLLM OOM error** | Giảm \--gpu-memory-utilization từ 0.87 → 0.80. Hoặc giảm \--max-num-seqs. |
| **Port không accessible** | Service phải bind 0.0.0.0 (không phải localhost). Kiểm tra \--host 0.0.0.0. |

*RunPod LLM Self-Host Deployment Plan · Dựa trên docs.runpod.io · 6× NVIDIA A40 · vLLM · llama.cpp · SGLang*


