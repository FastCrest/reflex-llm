# Performance

## Hardware Specs (Orin Nano Super)

| Spec | Value |
|------|-------|
| GPU TOPS (INT8) | 67 |
| DLA TOPS (INT8) | ~10 |
| Total TOPS | ~77 |
| Memory | 8 GB LPDDR5 |
| Bandwidth | 102 GB/s |
| CUDA cores | 1024 (16 SMs × 64) |
| Tensor Cores | 32 |
| Power modes | 7W / 10W / 15W / 25W |
| Ridge point | 0.66 OP/byte |

## Why LLM Decode is Bandwidth-Bound

LLM autoregressive decode generates one token at a time. Each token requires reading the **entire** weight matrix from DRAM:

```
Llama 3.2 3B INT4 — one decode step:

  Weight read:    1.5 GB from DRAM
  Compute:        6 GFLOP (3B × 2 ops)
  Time to read:   1.5 GB / 102 GB/s = 14.7 ms
  Time to compute: 6 GFLOP / 67 TOPS = 0.09 ms

  → 99.4% of time is reading weights
  → Compute utilization: 0.6%
  → Theoretical max: ~68 tokens/sec (bandwidth-limited)
```

Smaller model = fewer bytes to read = more tokens/sec. Quantization directly translates to throughput.

## Expected Performance

### TinyLlama 1.1B (Q4_K_M, 669 MB) — Test Model

| Metric | 25W mode | 15W mode |
|--------|----------|----------|
| Prompt eval (512 tok) | ~200 tok/s | ~120 tok/s |
| Decode (128 tok) | ~65 tok/s | ~40 tok/s |
| Peak memory | ~2.5 GB | ~2.5 GB |
| Peak temperature | <60°C | <55°C |

### Llama 3.2 3B (Q4_K_M, 1.8 GB) — Target Model

| Metric | 25W mode | 15W mode |
|--------|----------|----------|
| Prompt eval (512 tok) | ~65 tok/s | ~40 tok/s |
| Decode (128 tok) | ~25 tok/s | ~15 tok/s |
| Peak memory | ~4 GB | ~4 GB |
| Peak temperature | <70°C | <60°C |

### Phi-4 Mini 3.8B (Q4_K_M, 2.3 GB) — Stress Test

| Metric | 25W mode |
|--------|----------|
| Prompt eval (512 tok) | ~50 tok/s |
| Decode (128 tok) | ~20 tok/s |
| Peak memory | ~4.5 GB |

*All values are estimates. Actual performance depends on thermal design, context length, and prompt content.*

## Profiling

### Nsight Systems — Timeline Profile

```bash
./scripts/profile.sh models/tinyllama.gguf
# Creates: profile_YYYYMMDD_HHMMSS.nsys-rep
```

Shows:
- Kernel-by-kernel timeline
- Memory transfer timing
- CPU-GPU synchronization points
- Kernel launch overhead

### Key Metrics to Watch

```bash
nsys stats profile.nsys-rep
```

Expected output:
```
Kernel                              Time%    Calls
────────────────────────────────────────────────────
jllm::gemv_q4_kernel               38.2%    312
jllm::flash_attention_decode_kernel 28.1%    156
jllm::fused_rmsnorm_residual_kernel 11.4%    312
jllm::swiglu_kernel                  7.8%    156
jllm::rope_kernel                    4.2%    312
jllm::vec_add_kernel                 3.1%    312
other                                7.2%    ...
```

### Benchmark Script

```bash
./scripts/bench.sh models/tinyllama.gguf
```

Records:
- System state (power mode, GPU freq, RAM, temperature)
- Short generation (128 tokens): tok/s
- Long generation (256 tokens): tok/s
- Memory profile during inference (RSS every second)
- Temperature during inference

## Optimization Targets

### Level 1: Fused Kernels (Done)

| Operation | Without fusion | With fusion | Saving |
|-----------|---------------|-------------|--------|
| RMSNorm + residual | 3 kernels, 6 DRAM ops | 1 kernel, 3 DRAM ops | 2× |
| SwiGLU | 2 kernels, 4 DRAM ops | 1 kernel, 2 DRAM ops | 2× |
| Dequant + GEMV | 2 kernels, 2× bandwidth | 1 kernel, 1× bandwidth | 3.5× |

### Level 2: Tile Size Tuning (Done)

| Parameter | Desktop GPU (H100) | Orin Nano | Why different |
|-----------|-------------------|-----------|---------------|
| GEMV block | 256 threads | 128 threads | Fewer SMs, less occupancy pressure |
| Attention tile | 128 KV tokens | 64 KV tokens | 48 KB shared (not 228 KB) |
| GEMM tile | 128×128 | 64×64 | Less shared memory per SM |

### Level 3: CUDA Graphs (Implemented)

Captures decode step as a graph, replays with single `cudaGraphLaunch()`.
Saves ~1 ms per step from kernel launch overhead.

### Level 4: Future Optimizations

| Optimization | Expected gain | Effort |
|-------------|---------------|--------|
| Tensor Core WMMA for prefill | 2–3× prefill speed | 2 days |
| Persistent kernels (stay resident on SM) | 10–20% decode | 3 days |
| Custom memory allocator (bypass CUDA) | 5% less overhead | 2 days |
| INT4 KV cache (not just INT8) | 2× more context | 1 day |
| Speculative decoding with draft model | 1.5–3× tok/s | 3 days |

## Roofline Analysis

```
TFLOPS
  │
  │                    ──── 67 TOPS peak (INT8)
  │                 ╱
  │              ╱
  │           ╱
  │        ╱  ← slope = 102 GB/s bandwidth
  │     ╱
  │  ╱
  └───────────────── Arithmetic Intensity (OP/byte)
        ↑
  ridge = 0.66

  LLM decode AI ≈ 0.5 OP/byte → severely left of ridge → bandwidth-bound
  Prefill GEMM AI ≈ 50+ OP/byte → right of ridge → compute-bound
```

All decode optimizations must reduce bandwidth (quantization, fusion, caching).
Prefill optimizations should maximize Tensor Core utilization.

## Power Efficiency

| Model | Tokens/sec | Power (W) | Tokens/Joule |
|-------|-----------|-----------|-------------|
| TinyLlama 1.1B @ 25W | ~65 | 25 | 2.6 |
| TinyLlama 1.1B @ 7W | ~20 | 7 | 2.9 |
| Llama 3.2 3B @ 25W | ~25 | 25 | 1.0 |
| Llama 3.2 3B @ 7W | ~8 | 7 | 1.1 |

*7W mode is more power-efficient (tokens/joule) despite being slower.*
