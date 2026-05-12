# CUDA Kernels

All kernels are tuned for Orin SM 8.7: 48 KB shared memory, 128-thread blocks, 16 SMs.

## gemv_q4 — INT4 Dequant-Fused GEMV

**File:** `src/kernels/gemv_q4.cu`
**Time share:** ~38% of decode time — the #1 optimization target.

### What it does

Computes `y[M] = W[M×K] × x[K]` where W is 4-bit quantized (2 weights per byte) with FP16 per-group scales.

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
1. Each lane loads W_packed[lane], W_packed[lane+32], ... (coalesced)
2. Dequantize 8 values per uint32 (shift + mask + scale)
3. Dot product with x[k0..k0+7] (x stays in L1/L2 cache)
4. Warp shuffle reduce (5 rounds: offset 16,8,4,2,1)
5. Lane 0 writes y[row]
```

## fused_norm — RMSNorm + Residual Add

**File:** `src/kernels/fused_norm.cu`
**Time share:** ~11% of decode time.

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

`hidden_dim × sizeof(float)` for intermediate values. For hidden_dim=2048: 8 KB. For hidden_dim=3072: 12 KB. Both fit in 48 KB.

## attention — Flash Attention Decode

**File:** `src/kernels/attention.cu`
**Time share:** ~28% of decode time.

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
| gemv_q4 | Memory bandwidth | 34 | 0 |
| fused_norm | Memory bandwidth | 26 | hidden_dim × 4 |
| attention | Memory bandwidth | 40 | (64 + head_dim) × 4 |
| rope | Compute (trig) | 13 | 0 |
| softmax | Memory bandwidth | 23 | ~36 bytes |
| swiglu | Memory bandwidth | 14 | 0 |
| fp16_to_int8 | Memory bandwidth | 14 | 4 bytes |
