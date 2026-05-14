# CUDA Kernels

All production decode kernels are tuned for Jetson Orin SM 8.7: 48 KB shared
memory, 128-thread blocks, and the 8-SM Orin Nano Super profile validated on
L4T R36 / CUDA 12.6. Runtime hardware probing is used for display and sizing;
the kernels themselves avoid SKU-specific assumptions where possible.

Fast paths are enabled by default after the Week 1 Qwen3 validation. Each path
can be disabled independently for debugging:

```
JLLM_FAST_GEMV=0
JLLM_FAST_NORM=0
JLLM_FAST_ATTN=0
```

## gemv_q4 — GGML K-Quant Dequant-Fused GEMV

**File:** `src/kernels/gemv_q4.cu`
**Validated:** Q4_K, Q5_K, Q6_K tensors inside Qwen3-4B Q4_K_M.

### What it does

Computes `y[M] = W[M×K] × x[K]` where W is a GGML K-quant tensor. Qwen3
Q4_K_M mixes Q4_K, Q5_K, and Q6_K tensors, so the dispatcher selects the
matching kernel by GGML tensor type.

Weights stay in the mmap'd GGUF file. The loader registers that mmap with CUDA
mapped-host access and gives the kernels a device-visible alias; raw CPU mmap
pointers are never dereferenced by CUDA kernels.

### Why fused dequant matters

Without fusion: read INT4 weights → write FP16 weights to DRAM → read FP16 weights → compute.
With fusion: read INT4 weights → dequantize in registers → compute. Never writes FP16 weights to DRAM.

Bandwidth: K/2 bytes (INT4) vs K×2 bytes (FP16) = **3.5× reduction**.

### Orin tuning

```
Grid:   (ceil(M / 4), 1)
Block:  128 threads = 4 warps
        Each warp handles one output row (M dimension)
        32 lanes stride across K dimension (coalesced uint32 loads)

Reduction: warp shuffle (__shfl_xor_sync) — no shared memory needed
Dequant:   8 INT4 values from one uint32, multiply by group scale
```

### Key code path

```
1. Resolve host GGUF tensor pointer to CUDA mapped device pointer
2. One warp computes one output row
3. Dequantize K-quant blocks in registers
4. Dot product with x
5. Warp shuffle reduce
6. Lane 0 writes y[row]
```

## fused_norm — RMSNorm + Residual Add

**File:** `src/kernels/fused_norm.cu`
**Validated:** layer RMSNorm, Qwen3 Q/K per-head RMSNorm, final RMSNorm.

### What it does

Computes `output = RMSNorm(x) × weight` in one kernel.

Without fusion: 3 kernels, 6 DRAM accesses.
With fusion: 1 kernel, 3 DRAM accesses (read x, read weight, write output).

### Algorithm

```
Pass 1: Load x, compute sum of squares (variance)
  - Each thread handles hidden_dim/blockDim elements
  - Warp shuffle reduce for partial sums
  - Cross-warp reduce via shared memory (4 floats for 4 warps)
  - Compute rrms = rsqrt(variance/dim + eps)

Pass 2: Normalize and scale
  - normed = x * rrms * weight
  - Write output
```

### Shared memory usage

The current kernel does not cache the full hidden vector in shared memory. It
reads the input once for the sum-of-squares reduction and once for the final
scale/write. This avoided an earlier shared-memory layout bug that produced
alternating zeros in Qwen3 RMSNorm output.

## attention — Flash Attention Decode

**File:** `src/kernels/attention.cu`
**Validated:** Qwen3 single-token decode attention with GQA.

### What it does

Single-query attention for decode (one new token). Computes:
```
output = softmax(Q × K^T / sqrt(d)) × V
```

without materializing the full seq×seq attention matrix.

### Algorithm (online softmax)

```
For each KV tile (64 tokens):
  1. Compute Q×K^T for tile (each thread handles some time steps)
  2. Find tile max (warp reduce + block reduce via shared memory)
  3. Update running max, correct previous accumulators by exp(old_max - new_max)
  4. Exponentiate scores, accumulate sum
  5. Accumulate P × V into s_out[head_dim] in shared memory
Final: output = s_out / running_sum
```

### Orin tuning

```
Grid:   (n_heads, 1)  — one block per query head
Block:  128 threads
Shared: ATTN_TILE_KV (64) + head_dim floats for scores + output accumulator
Tile:   64 KV tokens per iteration

KV layout: [seq_len, n_kv_heads, head_dim]
GQA: kv_head = head / (n_heads / n_kv_heads)
INT8 KV: dequantize on-the-fly in the dot product loop
```

### Memory access pattern

- Q: read once from global, stays in L1 (small: 128 × 2 = 256 bytes)
- K: read tile by tile, 64 × 128 × element_size per tile
- V: read tile by tile, same pattern
- Scores: shared memory only (never written to DRAM)
- Output: one write at the end

## rope — Rotary Position Embedding

**File:** `src/kernels/rope.cu`
**Time share:** ~4% of decode time.

### What it does

Applies rotary position encoding in-place to Q and K:
```
q'[2i]   = q[2i] × cos(θ) - q[2i+1] × sin(θ)
q'[2i+1] = q[2i] × sin(θ) + q[2i+1] × cos(θ)
where θ = position / (theta_base ^ (2i / head_dim))
```

### Orin tuning

```
One thread per dimension pair (both Q and K in same launch)
Total threads: (n_heads + n_kv_heads) × head_dim/2
cos/sin computed on-the-fly (cheaper than loading from table on bandwidth-limited Orin)
```

## convert — FP16↔INT8 + SwiGLU

**File:** `src/kernels/convert.cu`

### fp16_to_int8

Per-row absmax quantization for KV cache:
```
scale = max(|row|) / 127
int8_val = round(fp16_val / scale)
```

### fused_swiglu

Computes `output = silu(gate) × up` where `silu(x) = x / (1 + exp(-x))`.
One thread per element. Fusing avoids writing intermediate silu result to DRAM.

## softmax — Logit Softmax

**File:** `src/kernels/softmax.cu`

Used only for final logit→probability conversion (vocab_size elements). Three passes:
1. Find max (numerically stable)
2. Exponentiate and sum
3. Normalize

Single block, 256 threads. Vocab sizes up to 128K.

## Utility Kernels (in decode.cu)

### vec_add

`out[i] = a[i] + b[i]` — used for residual connections between attention and FFN.

### fp16_to_fp32

Converts FP16 logits to FP32 on GPU before D2H copy for sampling.

## Performance Characteristics (Orin Nano Super)

| Kernel | Bottleneck | Registers | Shared mem |
|--------|-----------|-----------|------------|
| gemv_q4/q5/q6 K | Memory bandwidth | 40-48 | 0 |
| fused_norm | Memory bandwidth | 26 | 128 bytes |
| attention | Memory bandwidth | 40 | (64 + head_dim) × 4 |
| rope | Compute (trig) | 13 | 0 |
| softmax | Memory bandwidth | 23 | ~36 bytes |
| swiglu | Memory bandwidth | 14 | 0 |
| fp16_to_int8 | Memory bandwidth | 14 | 4 bytes |
