# Environment Setup

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Clone with submodules

```bash
git clone --recurse-submodules <repo-url>
cd gpu-cluster-benchmarking
```

Or if already cloned:

```bash
git submodule update --init --depth 1
```

## Install dependencies

```bash
uv sync --group common --group benchmark --group litellm --group monitoring
```

## Start Redis

```bash
redis-server --daemonize yes
```

Verify:

```bash
redis-cli ping
# Should return PONG
```
