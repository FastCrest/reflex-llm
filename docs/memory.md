# Memory System

## The Core Constraint

Jetson Orin Nano Super has 8 GB LPDDR5 shared between CPU, GPU, DLA, camera, and OS. After firmware carveouts and OS, ~5.5–6 GB is available. The model, KV cache, scratch buffers, and CUDA runtime must all fit in this space.

## Memory Budget

`MemoryBudget` (`jllm_memory.h`) tracks every allocation category:

```
╔══════════════════════════════════╗
║   JLLM Memory Budget             ║
╠══════════════════════════════════╣
║ Total DRAM:      7633 MB         ║  ← /proc/meminfo MemTotal
║ OS + kernel:    - 500 MB         ║  ← MemTotal - MemAvailable - CMA
║ CMA reserved:  - 768 MB         ║  ← /proc/meminfo CmaTotal
║ CUDA context:  - 300 MB         ║  ← estimate (updated after cudaSetDevice)
║ Model weights: -1800 MB         ║  ← actual GGUF file size after mmap
║ KV cache:      - 200 MB         ║  ← KVCachePool allocated size
║ Scratch:       -  64 MB         ║  ← ScratchPool allocated size
║ Safety margin: - 256 MB         ║  ← headroom to prevent OOM
╠══════════════════════════════════╣
║ FREE:           3745 MB         ║
╚══════════════════════════════════╝
```

### Source: `src/memory/budget.cpp`

`probe_system_memory()` reads `/proc/meminfo` fields:
- `MemTotal` — total Linux-visible RAM
- `MemAvailable` — realistic available (accounts for cache, buffers)
- `CmaTotal` — contiguous memory allocator reservation

### Auto Context Calculation

The engine calculates maximum context length from remaining memory:

```
max_context = max_kv_mb × 1024 × 1024 / kv_per_token_bytes

kv_per_token_bytes = 2 × n_layers × n_kv_heads × head_dim × kv_type_bytes

Example (Llama 3.2 3B, INT8 KV):
  max_kv_mb = 7633 - 500 - 768 - 300 - 1800 - 64 - 256 = 3945 MB
  kv_per_token = 2 × 26 × 8 × 128 × 1 = 53,248 bytes
  max_context = 3945 × 1024 × 1024 / 53,248 = ~77,700 tokens
  (capped at model's max_seq_len)
```

## OOM Guard

`OOMGuard` (`jllm_memory.h`) checks real memory before every KV cache extension.

### How it works

Before generating each token:
1. Read `MemAvailable` from `/proc/meminfo` (real kernel value, not cached)
2. Compare against `kv_per_token_bytes + safety_margin`
3. If insufficient: stop generation gracefully, report in stats

```cpp
bool OOMGuard::can_extend(int64_t additional_bytes) const {
    int64_t free = real_free_mb();  // reads /proc/meminfo
    int64_t needed_mb = additional_bytes / (1024 * 1024) + 1;
    return free > (needed_mb + safety_mb_);
}
```

### Emergency Free

`emergency_free()` drops filesystem caches and triggers compaction:
- `echo 3 > /proc/sys/vm/drop_caches`
- `echo 1 > /proc/sys/vm/compact_memory`

Only called as a last resort — requires root.

## KV Cache Pool

`KVCachePool` (`src/memory/kv_cache.cpp`) manages key/value tensors for all layers.

### Two-Tier Design

```
┌─────────────────────────────────────────────┐
│  Fast Pool (cudaMallocHost)                  │
│  Pinned DRAM — GPU reads at full bandwidth  │
│  Recent tokens live here                     │
│  Size: n_layers × entry_bytes × max_context │
├─────────────────────────────────────────────┤
│  Overflow Pool (malloc)                      │
│  Unpinned DRAM — GPU reads via page faults  │
│  Old tokens evicted here                     │
│  Size: n_layers × entry_bytes × overflow    │
└─────────────────────────────────────────────┘
```

### Why This Works on Jetson

On discrete GPUs, "CPU offload" means PCIe transfer (~64 GB/s, high latency). On Jetson, CPU and GPU share the same physical DRAM. The "fast pool" uses `cudaMallocHost` (pinned — no page faults, ~102 GB/s). The "overflow pool" uses `malloc` (pageable — GPU accesses via page faults, ~50 GB/s, but still the same DRAM).

### Eviction

When the fast pool is full, oldest tokens are moved to overflow:
```
Before:  Fast: [tok 0][tok 1][tok 2]...[tok 2047]  (full)
After:   Fast: [tok 512][tok 513]...[tok 2047]      (moved 0-511 to overflow)
         Overflow: [tok 0][tok 1]...[tok 511]
```

This is a `memcpy` within the same DRAM — fast.

### Memory Layout

Per layer, per token:
```
entry_bytes = 2 × n_kv_heads × head_dim × kv_type_bytes

For Llama 3.2 3B (8 KV heads, 128 dim, INT8):
  entry = 2 × 8 × 128 × 1 = 2,048 bytes per token per layer
  
26 layers × 2,048 bytes × 2,048 tokens = 109 MB for 2K context
26 layers × 2,048 bytes × 4,096 tokens = 218 MB for 4K context
```

## Scratch Pool

`ScratchPool` (`src/memory/pool.cpp`) provides temporary buffers for intermediate tensors.

### Bump Allocator

```
Pre-allocated backing:  [────────────────────── 64 MB ──────────────────────]
                         ^offset=0

After get(1024):        [used│──────────────── remaining ───────────────────]
                              ^offset=1024

After get(2048):        [used│used│────────── remaining ────────────────────]
                                   ^offset=3072

After reset():          [────────────────────── 64 MB ──────────────────────]
                         ^offset=0 (all memory reusable)
```

- `get(size)` — returns pointer, advances offset. Aligns to 256 bytes.
- `reset()` — resets offset to 0. Called at start of each decode step.
- Zero `malloc`/`free` during inference — just pointer arithmetic.

### Sizing

Scratch size is calculated at load time:
```
scratch = hidden_dim × 8 (attention intermediates)
        + n_heads × head_dim × 3 (Q, K, V projections)
        + intermediate_dim × 4 (FFN intermediates)
        + vocab_size × sizeof(float) + vocab_size × sizeof(half) (logits)
        minimum 64 MB
```

## Unified Memory Patterns

### Model Weights: mmap + cudaHostRegister

```
File on disk (GGUF)
      │
      ▼ mmap(PROT_READ, MAP_PRIVATE)
DRAM pages (copy-on-write, demand-paged)
      │
      ▼ cudaHostRegister(ptr, size, cudaHostRegisterReadOnly)
GPU can read these pages directly (no copy, no page faults)
      │
      ▼ madvise(MADV_RANDOM)
Kernel knows access pattern is random (inference reads scattered weights)
```

### KV Cache: cudaMallocHost

```
cudaMallocHost(&ptr, size)
  → Allocates pinned DRAM
  → Registered with CUDA runtime
  → GPU reads at full 102 GB/s bandwidth
  → CPU can read/write directly (for debugging, export)
  → Never swapped to disk
```

### Scratch: cudaMallocHost

Same as KV cache — pinned, GPU-accessible, CPU-accessible. Reused every step.
