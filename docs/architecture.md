# Architecture

## Overview

`reflex-llm` is a memory-first LLM inference runtime targeting NVIDIA Jetson
devices. It runs autoregressive transformer models from GGUF format files and
owns the runtime layer: model loading, tokenizer state, memory budget, KV-cache
lifecycle, sampling, CLI/server surfaces, and application integration.

CUDA kernels are being extracted into `reflex-infer`. In the long-term design,
`reflex-llm` depends on `reflex-infer` for optimized Jetson kernels instead of
owning all low-level CUDA implementations directly.

Initial target: Jetson Orin Nano 8GB with Qwen 3.5B or Qwen3 4B class Q4
models. Later targets include Orin NX 16GB, AGX Orin 64GB, Thor, and additional
model families such as Phi-4.

```
User prompt (text)
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    reflex-llm CLI / HTTP Server              │
│  main.cpp: parse args, probe system, pre-check memory       │
│  http_server.cpp: /v1/chat/completions, /health, /v1/models │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                          Engine                              │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Tokenizer│  │  Prefill │  │  Decode  │  │  Sampling  │  │
│  │ encode() │→ │ N layers │→ │ 1 token  │→ │ top-k/p    │  │
│  │ decode() │  │ per tok  │  │ per step │  │ temperature│  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│                       │              │                        │
│              ┌────────┴──────────────┘                        │
│              ▼                                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            transformer_layer() × N_layers               │ │
│  │                                                         │ │
│  │  RMSNorm → Q/K/V proj → RoPE → KV store → Attention    │ │
│  │  → Output proj → Residual → RMSNorm → Gate/Up proj     │ │
│  │  → SwiGLU → Down proj → Residual                       │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│          CUDA Kernels / reflex-infer Bridges                  │
│                                                              │
│  gemv_q4        INT4 dequant-fused GEMV (38% of decode)     │
│  attention      Flash attention decode, online softmax       │
│  fused_norm     RMSNorm + residual add (2× less traffic)    │
│  rope           Rotary position embedding (fused Q+K)        │
│  convert        FP16↔INT8 + SwiGLU activation               │
│  softmax        Numerically stable logit softmax             │
│  vec_add        Residual connection add                      │
│  fp16_to_fp32   Logit conversion for sampling                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     Memory Manager                           │
│                                                              │
│  MemoryBudget   Track every MB (OS, CMA, CUDA, model, KV)   │
│  OOMGuard       Check /proc/meminfo before every KV extend   │
│  KVCachePool    Pinned fast pool + unpinned overflow          │
│  ScratchPool    Bump allocator, zero malloc during inference  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Jetson HAL                              │
│                                                              │
│  PowerState     nvpmodel query (7W/10W/15W/25W)              │
│  ThermalState   Thermal zones, adaptive backoff              │
│  JetsonInfo     One-time system probe (L4T, CUDA, SMs, RAM) │
│  LiveStats      tegrastats-style real-time metrics           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Jetson Orin Nano / Orin-family Hardware          │
│                                                              │
│  1024 CUDA cores (8 SMs × 128)  │  102 GB/s LPDDR5         │
│  32 Tensor Cores                  │  8 GB unified memory     │
│  67 TOPS INT8 (GPU)              │  7–25W power modes       │
│  ~10 TOPS INT8 (DLA)             │  SM 8.7 (Ampere)         │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Memory-First

On Jetson, CPU and GPU share the same 8 GB LPDDR5. Every byte for the model is a byte not available for KV cache, OS, or camera. The runtime tracks every allocation.

- `MemoryBudget` knows where every MB goes before inference starts
- Context length is auto-calculated from remaining memory after model load
- `OOMGuard` checks real `/proc/meminfo` before every KV cache extension
- Generation stops gracefully on memory pressure (no crash, no OOM killer)

### 2. Jetson-First

No code paths for x86, desktop GPUs, or multi-GPU in the first runtime path.
The default profile is tuned for the validated Orin Nano SM 8.7 target, while
runtime probes report the actual SKU details at startup. Future Jetson SKUs
should be selected through explicit hardware profiles, not hidden heuristics.

### 3. Pre-Allocated Decode Pools

Most high-volume memory is prepared at load time:
- Model weights: mmap'd and CUDA-mapped
- KV cache: pre-allocated pool
- Scratch buffers: bump allocator with pre-allocated backing

The current final-logits fast path still has a small temporary CUDA allocation;
removing that is tracked as throughput cleanup rather than a correctness gate.

### 4. Unified Memory Exploitation

Jetson's CPU and GPU share the same DRAM. The runtime exploits this:
- Model weights: `mmap` + `cudaHostRegister` → GPU reads file directly
- KV cache: `cudaMallocHost` → both CPU and GPU access without copy
- KV overflow: regular `malloc` → GPU reads via page faults (slower but works)
- No `cudaMemcpy` for data sharing between CPU and GPU

### 5. Power/Thermal Awareness

The runtime adapts to Jetson's power constraints:
- Reads `nvpmodel` state at startup
- Monitors thermal zones during generation
- Backs off (inserts delays) before hardware thermal throttling triggers
- Reports power mode, temperature, and utilization in health endpoint

### 6. Kernel/Runtime Boundary

The runtime must not bake every kernel decision directly into the transformer
loop. The intended boundary is:

- `reflex-llm` describes model shape, quantization layout, KV-cache layout,
  workspace, stream, and target hardware profile.
- `reflex-infer` selects and launches the best supported kernel for that
  operator and target.
- Unsupported model/hardware/quantization combinations must fall back to a
  known-correct runtime path.

This is the boundary that lets `reflex-infer` grow into a Jetson-only kernel
library without turning `reflex-llm` into a monolithic benchmark project.

## Data Flow — One Token

```
1. Engine receives last generated token ID
2. Embedding lookup: tok_embd[token_id] → hidden state x (cudaMemcpy from mmap)
3. For each layer (0 to N-1):
   a. RMSNorm(x) → normed
   b. Q = gemv_q4(W_q, normed)
   c. K = gemv_q4(W_k, normed)
   d. V = gemv_q4(W_v, normed)
   e. RoPE(Q, K, position)
   f. Store K,V into KV cache pool (INT8 quantized if enabled)
   g. attn_out = flash_attention(Q, K_cache, V_cache)
   h. attn_proj = gemv_q4(W_o, attn_out)
   i. x2 = x + attn_proj                    ← first residual
   j. normed2 = RMSNorm(x2)
   k. gate = gemv_q4(W_gate, normed2)
   l. up = gemv_q4(W_up, normed2)
   m. swiglu_out = silu(gate) × up
   n. ffn_out = gemv_q4(W_down, swiglu_out)
   o. x = x2 + ffn_out                      ← second residual
4. Final RMSNorm(x) → normed
5. logits = gemv_q4(W_output, normed)        → FP16
6. fp16_to_fp32(logits)                      → FP32
7. cudaMemcpy logits to CPU
8. sample_token(logits, top_k, top_p, temperature)
9. Return token ID, call streaming callback
```

## File Organization

```
include/
  jllm.h            Master header + Orin constants
  jllm_memory.h     MemoryBudget, OOMGuard, KVCachePool, ScratchPool
  jllm_jetson.h     PowerState, ThermalState, LiveStats, JetsonInfo
  jllm_kernels.h    Kernel API declarations + SM 8.7 tile/block constants
  jllm_engine.h     ModelConfig, LayerWeights, ModelWeights, Tokenizer, Engine

src/memory/         Memory management (budget, KV cache, scratch pool)
src/jetson/         Hardware abstraction (power, thermal, sysinfo)
src/kernels/        Local CUDA kernels and ABI bridges into reflex-infer
src/engine/         Model loading, forward pass, sampling, tokenizer
src/server/         HTTP API server
src/main.cpp        CLI entry point
src/main_server.cpp Server entry point

scripts/            Setup, benchmarking, profiling
tests/              Unit and integration tests
docs/               This documentation
```
