"""Locust load test for LiteLLM proxy with semantic cache.

Simulates realistic user traffic with a mix of repeated prompts (cache hits)
and unique prompts (cache misses). Measures throughput, latency distribution,
and error rate under concurrent load.

Usage:
    locust -f bench-litellm-locust.py --host http://localhost:4000 --headless \
        --users 50 --spawn-rate 5 --run-time 5m --csv results/locust
"""

import random
import uuid

from locust import HttpUser, between, task


PROMPTS = [
    "Explain the difference between TCP and UDP in detail",
    "What is the CAP theorem in distributed systems?",
    "Describe the attention mechanism in transformers",
    "How does PagedAttention work in vLLM?",
    "Explain gradient descent and its variants",
    "What are the key differences between SQL and NoSQL databases?",
    "Describe how garbage collection works in Java",
    "Explain the Raft consensus algorithm",
    "What is the purpose of a load balancer?",
    "How does HTTPS encryption work?",
]


class CachedUser(HttpUser):
    """Simulates a user whose queries are likely cacheable."""

    wait_time = between(0.1, 1.0)

    @task(3)
    def chat_cached(self):
        payload = {
            "model": "qwen35b-llamacpp",
            "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "max_tokens": 256,
            "stream": True,
        }
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(1)
    def chat_unique(self):
        payload = {
            "model": "qwen35b-llamacpp",
            "messages": [
                {"role": "user", "content": f"Unique request {uuid.uuid4()}"}
            ],
            "max_tokens": 64,
        }
        self.client.post("/v1/chat/completions", json=payload)
