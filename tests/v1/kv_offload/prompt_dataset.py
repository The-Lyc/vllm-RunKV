# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared prompt dataset for RunKV / dynamic replay E2E tests."""

from __future__ import annotations

SHORT_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
    "Explain quantum computing in simple terms:",
    "Write a short poem about the ocean:",
    "What is the meaning of life?",
    "Once upon a time in a land far away,",
    "The best programming language is",
    "How do you make a perfect cup of coffee?",
    "Describe the process of photosynthesis:",
    "What are the benefits of regular exercise?",
    "In the year 2050, technology will",
    "The most important invention in history is",
    "A recipe for happiness includes",
    "The future of artificial intelligence is",
    "Climate change affects the planet by",
    "The secret to success is",
    "My favorite book is about",
    "Space exploration is important because",
    "The internet has changed society by",
    "A healthy diet should include",
]

# Backward-compatible alias used by the concurrent scripts today.
SAMPLE_PROMPTS = SHORT_PROMPTS

MEDIUM_PROMPTS = [
    (
        "Please provide a detailed explanation of how neural networks learn "
        "through backpropagation, including the mathematical foundations and "
        "practical considerations:"
    ),
    (
        "Write a comprehensive essay about the history of computing, from "
        "early mechanical calculators to modern quantum computers, discussing "
        "key innovations and pioneers:"
    ),
    (
        "Explain the principles of thermodynamics and their applications in "
        "engineering, including real-world examples from power plants, "
        "refrigeration, and heat engines:"
    ),
    (
        "Describe the evolution of programming languages from assembly to "
        "modern high-level languages, including their design philosophies and "
        "use cases:"
    ),
    (
        "Discuss the impact of social media on modern communication, "
        "relationships, and mental health, with both positive and negative "
        "aspects:"
    ),
]

VERY_LONG_PROMPTS = [
    """You are an expert software architect reviewing a complex distributed system.
The system consists of multiple microservices communicating via message queues and
REST APIs. The main components include:

1. User Authentication Service: Handles user login, registration, and token
management. Uses JWT tokens with RSA-256 signing. Stores user credentials in
PostgreSQL with bcrypt hashing.

2. Product Catalog Service: Manages product information, categories, and
inventory. Uses Elasticsearch for full-text search and Redis for caching
frequently accessed items.

3. Order Processing Service: Handles order creation, payment processing, and
fulfillment. Integrates with external payment gateways (Stripe, PayPal) and
shipping providers (FedEx, UPS).

4. Notification Service: Sends emails, SMS, and push notifications. Uses
RabbitMQ for message queuing and supports templated messages with
internationalization.

5. Analytics Service: Collects and processes user behavior data. Uses Apache
Kafka for event streaming and ClickHouse for analytical queries.

6. API Gateway: Routes requests to appropriate services, handles rate limiting,
and manages API versioning. Implements circuit breaker patterns for fault
tolerance.

The system currently handles 10,000 requests per second during peak hours and
stores 50TB of data across all services. Recent performance issues have been
reported:
- Order processing latency increased from 200ms to 800ms
- Search queries timing out during high traffic
- Memory usage spiking on the notification service

Please analyze the architecture and provide detailed recommendations for:""",
    """Abstract: Recent advances in large language models (LLMs) have demonstrated
remarkable capabilities in natural language understanding and generation.
However, the computational requirements for training and inference remain a
significant challenge. This paper presents a comprehensive survey of
optimization techniques for LLM deployment.

Introduction:
The transformer architecture, introduced by Vaswani et al. (2017), has become
the foundation for modern language models. Models like GPT-4, Claude, and LLaMA
have shown impressive performance across diverse tasks including text
generation, code completion, and reasoning. However, these models contain
billions of parameters, requiring substantial computational resources.

Key challenges in LLM deployment include:
1. Memory bandwidth limitations during inference
2. KV cache management for long sequences
3. Batch processing efficiency under varying sequence lengths
4. Quantization effects on model quality
5. Distributed inference across multiple devices

Related Work:
Previous approaches to efficient LLM inference include:
- PagedAttention (vLLM): Manages KV cache as virtual memory pages
- FlashAttention: Optimizes attention computation through tiling
- Speculative decoding: Uses smaller models to predict multiple tokens
- Continuous batching: Dynamically adjusts batch composition
- Model parallelism: Distributes model across multiple GPUs

Our contribution focuses on KV cache offloading to CPU memory, enabling larger
batch sizes and longer sequences without GPU memory constraints.

Methodology:
We propose a novel approach called RunKV that addresses memory limitations
through intelligent cache management. The key innovations include:

Based on this context, please provide a detailed technical analysis of:""",
    """Code Review Request: High-Performance Key-Value Cache Implementation

File: kv_cache_manager.py

```python
import torch
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict
import numpy as np

@dataclass
class CacheBlock:
    block_id: int
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    ref_count: int = 0
    last_access: float = 0.0
    is_pinned: bool = False

class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self._lock = threading.RLock()
        self._blocks: Dict[int, CacheBlock] = {}
        self._free_blocks: List[int] = list(range(num_blocks))
        self._lru_order = OrderedDict()
        self._initialize_blocks()

    def _initialize_blocks(self):
        for block_id in range(self.num_blocks):
            key_cache = torch.zeros(
                (self.block_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device="cuda",
            )
            value_cache = torch.zeros(
                (self.block_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device="cuda",
            )
            self._blocks[block_id] = CacheBlock(
                block_id=block_id,
                key_cache=key_cache,
                value_cache=value_cache,
            )

    def allocate_block(self) -> Optional[int]:
        with self._lock:
            if self._free_blocks:
                block_id = self._free_blocks.pop()
                self._blocks[block_id].ref_count = 1
                return block_id
            return self._evict_and_allocate()
```

Please review this code and provide detailed feedback on:""",
    """Conversation History:

User: I'm building a real-time chat application and need advice on the architecture.

Assistant: I'd be happy to help! Real-time chat applications have several
architectural considerations. What scale are you targeting - hundreds,
thousands, or millions of concurrent users?

User: We're expecting around 100,000 concurrent users at peak, with about
1 million registered users total.

Assistant: That's a solid medium-scale application. For 100K concurrent users,
you'll want to consider: WebSocket connections for real-time messaging, a
message broker like Redis Pub/Sub or Kafka for distributing messages, and a
database strategy that separates hot data from cold storage.

User: We're currently using PostgreSQL for everything. Is that going to be a problem?

Assistant: PostgreSQL is excellent for user data and message history, but for
real-time message delivery, you'll want to add a caching layer. Redis is
commonly used here. For the message flow: User A sends message -> Your server ->
Redis Pub/Sub -> Server instances -> User B's WebSocket.

User: What about message persistence? We need to store all messages for
compliance reasons.

Assistant: For compliance, you'll want a robust storage strategy. Consider:
1) Write messages to Kafka first (durable log),
2) Consume from Kafka to PostgreSQL for permanent storage,
3) Use Redis only for recent messages (last 24 hours).
This gives you durability, real-time access, and audit trails.

User: We also need end-to-end encryption. How does that affect the architecture?

Assistant: E2E encryption changes things significantly. The server cannot read
message content, only route encrypted payloads. You'll need: client-side key
management, key exchange protocols (like Signal Protocol), and encrypted
payload storage. The server stores ciphertext only.

User: One more thing - we need to support file sharing up to 100MB.

Assistant: For large files with E2E encryption, I recommend:
1) Encrypt file client-side,
2) Upload encrypted blob to object storage (S3/GCS),
3) Share only the encrypted download link and decryption key via chat.
Never pass large files through WebSocket.

Based on this entire conversation, please provide a comprehensive architecture
document that includes:""",
    """Document Analysis Request

You are analyzing multiple related documents for a legal case. Please review
the following excerpts and provide a comprehensive analysis.

Document 1 - Contract Agreement (Excerpt):
"Section 4.2 - Performance Standards: The Service Provider agrees to maintain
system uptime of 99.9% measured on a monthly basis. Scheduled maintenance
windows (not to exceed 4 hours per month) shall be excluded from uptime
calculations. Any downtime exceeding the guaranteed uptime shall result in
service credits as specified in Schedule B.

Section 4.3 - Data Security: The Service Provider shall implement
industry-standard security measures including but not limited to: AES-256
encryption for data at rest, TLS 1.3 for data in transit, multi-factor
authentication for administrative access, and annual third-party security
audits."

Document 2 - Incident Report (Excerpt):
"On March 15, 2025, at approximately 14:32 UTC, the primary database cluster
experienced a cascading failure due to a misconfigured replication setting.
The incident resulted in complete service unavailability lasting 6 hours and
47 minutes. Root cause analysis revealed that a configuration change deployed
on March 14 introduced a race condition in the failover logic.

Impacted customers: 2,847 enterprise accounts
Data loss: None confirmed, investigation ongoing
Financial impact: Estimated $2.3M in SLA credits"

Document 3 - Email Correspondence (Excerpt):
"From: ops-team@serviceprovider.com
To: engineering-leads@serviceprovider.com
Date: March 14, 2025, 18:45 UTC
Subject: Re: Urgent - Deployment approval needed

I understand the pressure to ship this before quarter-end, but I have concerns
about the testing coverage. We only ran integration tests on the staging
environment for 2 hours. Our standard procedure requires 24-hour soak testing
for database configuration changes. Given the timeline, I'll approve with the
caveat that we monitor closely tomorrow."

Document 4 - Customer Communication:
"Dear Valued Customer, We sincerely apologize for the service disruption
experienced on March 15. We take our reliability commitments seriously and are
implementing additional safeguards including enhanced deployment procedures and
expanded monitoring coverage."

Based on these documents, please analyze:""",
]

EXTRA_LONG_PROMPTS = [
    """You are an evaluation assistant for a large-scale LLM serving system. Read the full background carefully and produce the final answer strictly according to the required format at the end. You must obey all constraints, must not invent missing facts, and must not reveal your chain-of-thought. Only provide the final answer.

Background:
A research team is evaluating an online multi-tenant inference system running on a single GPU server. The GPU has 24 GB of HBM, the host has 256 GB of DRAM, and the PCIe link has a theoretical peak bandwidth of 28 GB/s. The system serves three request classes: Class A requests are latency-sensitive, Class B requests are throughput-oriented, and Class C requests are offline batch jobs. The system uses a hierarchical KV cache policy: hot layers are kept in GPU HBM, warm layers are stored in pinned host memory, and cold data may be restored from SSD when necessary. The team has implemented layer-granularity HBM buffer reuse, asynchronous H2D/D2H copy, prefetching, and limited recompute.

The model has 40 transformer layers, hidden size 5120, and KV cache stored in fp16. For each token, the combined size of K and V for one layer is 20 KB. Therefore, one token consumes 40 × 20 KB = 800 KB of KV cache across the full model. If a request has context length 1024, its full KV cache footprint is approximately 800 MB. If the context length is 2048, the footprint is about 1.6 GB. If the context length is 4096, the footprint is about 3.2 GB.

The system currently has 6 active requests, labeled R1 through R6, with the following attributes:
R1: Class A, already in decode, current context length 1536, expected remaining output 128 tokens, priority 10.
R2: Class B, near the end of prefill, current context length 4096, priority 6.
R3: Class A, already in decode, current context length 896, expected remaining output 64 tokens, priority 9.
R4: Class C, in offline batch processing, current context length 2048, no interactive response needed, priority 2.
R5: Class B, already in decode, current context length 3072, expected remaining output 256 tokens, priority 5.
R6: Class A, just finished prefill, current context length 1024, about to enter decode, priority 8.

Measured empirical observations:
1. If the most recent 8 layers of a decode request are fully resident on the GPU, the attention-related stage of the next token is reduced by 22% on average.
2. If host-to-device on-demand fetching occurs, every 1 GB of cold KV transferred adds an average of 42 ms of extra latency.
3. If the system chooses recompute for a layer instead of fetching that layer’s KV, the average extra compute cost is 2.8 ms per layer, while the transfer overhead for that layer can be avoided.
4. The system currently has only 6 HBM staging buffers, and each buffer can hold all KV for one request at one layer at a time.
5. At this moment, the GPU already keeps the most recent 10 layers of R1, the most recent 8 layers of R3, and the most recent 4 layers of R6 resident.
6. R2, R4, and R5 currently have no layers resident on the GPU.
7. To preserve fairness, the system enforces the following limits: at any moment, Class C may use at most 1 buffer; all Class B requests together may use at most 2 buffers; all remaining buffers should preferentially go to Class A.
8. The SLA target for Class A is that the extra queuing plus data-preparation time per token should not exceed 35 ms. The target for Class B is to maximize average throughput. The target for Class C is only opportunistic resource utilization and not real-time responsiveness.
9. The team proposes a heuristic scheduling policy: first guarantee locality for the highest-priority Class A requests that are already in decode; second, support Class A requests that are about to enter decode; third, improve throughput for Class B; finally, consider Class C.
10. If two requests have equal priority, prefer the one with shorter context length because its per-token service time is usually shorter.
11. The system should avoid buffer thrashing: unless the gain is clearly significant, do not frequently evict layers that are already hitting in GPU residency.
12. In this scheduling window, you only need to decide how the 6 buffers should be assigned across requests, and whether some missing layers should use recompute instead of fetching. You do not need to produce an exact per-layer mapping.

Additional notes:
- You do not need to compute an exact byte-level schedule.
- You should make a reasonable, explainable scheduling recommendation consistent with the stated goals.
- “Assigning a buffer to a request” means giving that request priority for upcoming residency or prefetch opportunities in this scheduling window.
- For Class A requests already in decode, prioritizing next-token tail latency is more important than long-run average throughput.
- For Class B requests, higher single-step latency is acceptable, but overall efficiency should be preserved.
- For Class C requests, it is acceptable to defer service as long as it does not noticeably interfere with Class A or Class B.

Task:
Produce exactly the following 4 parts, using the required format.

[Part 1]
List the ownership of the 6 buffers in priority order using exactly this format:
Buffer1 -> RequestID
Buffer2 -> RequestID
...
Buffer6 -> RequestID

[Part 2]
State which requests should be handled mainly by “prefer fetch”, which should use “selective recompute”, and which should receive “no buffer for now”. Each category must list request IDs.

[Part 3]
Provide one short explanation of no more than 220 words summarizing the core scheduling logic. It must explicitly mention all of the following: Class A SLA, Class B/Class C limits, existing residency state, and avoidance of thrashing.

[Part 4]
For each request from R1 to R6, output exactly one sentence in this format:
R1: ...
R2: ...
R3: ...
R4: ...
R5: ...
R6: ...

Extra constraints:
- Do not use a table.
- Do not use a code block.
- Do not restate the problem.
- Do not reveal intermediate reasoning.
- If a request receives no buffer, explicitly state why.
- Your answer must be self-consistent and must not violate the rule that Class C uses at most 1 buffer and all Class B requests together use at most 2 buffers.
""",
]

PROMPT_POOLS = {
    "short": SHORT_PROMPTS,
    "medium": MEDIUM_PROMPTS,
    "very-long": VERY_LONG_PROMPTS,
    "extra-long": EXTRA_LONG_PROMPTS,
}
