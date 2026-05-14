# Qwen3 architecture vs. our runtime

A walkthrough of how Qwen3 is built, how our runtime processes a GGUF
file, and the specific places where Qwen3's choices differed from our
initial assumptions and surfaced bugs during the Week-1 bring-up of
`#1`. Written 2026-05-14 after the Q6_K + QK-norm + memory-mapping
debug session.

## Why this matters

When a runtime is "code-complete but unvalidated", every choice the
runtime made was implicitly an assumption about the model. Different
model families bake in different choices. Most of the alpha-bring-up
bugs we hit were architecture-mismatch bugs, not generic CUDA bugs.
Knowing where Qwen3 diverges from the "Llama-shaped" baseline tells
you exactly which assumptions you cannot keep.

## Qwen3 4B — the architecture in detail

Qwen3-4B is a decoder-only transformer. It looks Llama-shaped at first
glance but has several specific choices that bite a runtime that
assumes Llama defaults.

### Shape numbers

| Field             | Value      | Notes |
| ----------------- | ---------- | --- |
| `n_layers`        | 36         | |
| `n_heads`         | 32         | query heads |
| `n_kv_heads`      | 8          | **GQA, 4-way grouping** |
| `head_dim`        | 128        | from `attention.key_length`; do not derive from `hidden_dim / n_heads` |
| `hidden_dim`      | 2560       | model residual width |
| `q_proj_dim`      | 4096       | `n_heads × head_dim` |
| `kv_proj_dim`     | 1024       | `n_kv_heads × head_dim` |
| `intermediate_dim`| 9216       | FFN width |
| `vocab_size`      | 151,936    | huge (Llama is 32k, Llama-3 is 128k) |
| `max_seq_len`     | 40,960     | model max context |
| `rope_theta`      | 1,000,000  | **not 10,000** — for long-context |
| `tie_word_embd`   | **true**   | output projection re-uses `token_embd.weight` |

### Tensor list (per layer)

For each of the 36 transformer blocks Qwen3 ships:

```
blk.N.attn_norm.weight       (F32)   pre-attention RMSNorm γ
blk.N.attn_q.weight          (Q4_K)  query projection
blk.N.attn_q_norm.weight     (F32)   **QK-norm γ for queries**     ← Qwen3-specific
blk.N.attn_k.weight          (Q4_K)  key projection
blk.N.attn_k_norm.weight     (F32)   **QK-norm γ for keys**        ← Qwen3-specific
blk.N.attn_v.weight          (Q4_K)  value projection
blk.N.attn_output.weight     (Q4_K)  output projection
blk.N.ffn_norm.weight        (F32)   pre-FFN RMSNorm γ
blk.N.ffn_gate.weight        (Q4_K)  SwiGLU gate
blk.N.ffn_up.weight          (Q4_K)  SwiGLU up
blk.N.ffn_down.weight        (Q4_K)  SwiGLU down
```

Plus globally:

```
token_embd.weight            (Q6_K)  embedding matrix — note: Q6_K, not Q4_K
output_norm.weight           (F32)   final pre-logits RMSNorm γ
(output.weight is omitted; tied to token_embd.weight)
```

That's `11 × 36 + 2 = 398` tensor entries, matching the log line:

```
[gguf] Version 3, 398 tensors, 28 metadata entries
```

### Per-token forward pass

```
                ┌──────────────────────────────────────────────────┐
   token_id ──► │ embedding lookup (Q6_K dequant)                  │ ──► x (FP16, hidden_dim)
                └──────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────────┐
                │  for layer in 0..35:                             │
                │    ─── attention block ───                       │
                │    h  = RMSNorm(x)                               │
                │    Q  = h @ Wq                                   │
                │    K  = h @ Wk                                   │
                │    V  = h @ Wv                                   │
                │    Q  = RMSNorm_qk(Q, attn_q_norm)  ◄── Qwen3 specific
                │    K  = RMSNorm_qk(K, attn_k_norm)  ◄── Qwen3 specific
                │    Q,K = RoPE(Q, K, pos, theta=1e6)              │
                │    write K, V into kv_cache[layer][pos]          │
                │    a  = softmax(Q @ K^T / sqrt(head_dim)) @ V    │
                │    a  = a @ Wo                                   │
                │    x  = x + a                       (residual)   │
                │                                                  │
                │    ─── feed-forward block ───                    │
                │    h  = RMSNorm(x)                               │
                │    g  = silu(h @ W_gate) * (h @ W_up)            │
                │    f  = g @ W_down                               │
                │    x  = x + f                       (residual)   │
                └──────────────────────────────────────────────────┘
                                       │
                                       ▼
                ┌──────────────────────────────────────────────────┐
                │ x  = RMSNorm(x, output_norm)                     │
                │ logits = x @ token_embd.weight^T  (tied weights) │
                │ sample(logits) → next_token                      │
                └──────────────────────────────────────────────────┘
```

The two boxes labelled "Qwen3 specific" are the QK-norm step — Qwen3
applies RMSNorm to Q and K *separately* after the projection and
*before* RoPE. Llama, Mistral, Phi do not have this step.

## How our runtime processes a GGUF, end to end

This is what `Engine::load()` and `Engine::generate()` actually do when
you point them at `Qwen3-4B-Q4_K_M.gguf`. Each numbered step maps to
a specific source file.

### 1. Open the GGUF

`src/engine/model.cpp::load_gguf_config`

- `fopen()` the file
- Read 24-byte header: magic `GGUF` + version (3) + `tensor_count`
  (398) + `kv_count` (28)
- Walk the 28 KV pairs looking for `qwen3.block_count`, `.attention.head_count`,
  `.attention.head_count_kv`, `.embedding_length`, `.rope.freq_base`, etc.
- Populate `ModelConfig`

This step is **fast** — it only reads the header section, not the
weights. So when our config print runs, the file isn't even fully
read yet.

### 2. Parse tensor info (offsets + shapes)

`src/engine/model.cpp::parse_tensor_infos`

- For each of the 398 tensors, read: name (variable-length string),
  `n_dims`, `shape[n_dims]`, `dtype`, `offset`.
- Returns `data_offset` (where the actual weight bytes start in the
  file). For Qwen3-4B this is **5,956,352** — that's the size of the
  header + metadata + tensor info section.

This is where we derive `vocab_size` if it wasn't in metadata: read
the shape of `token_embd.weight` (always `[hidden_dim, vocab_size]`
in GGUF's column-major layout) and use the non-`hidden_dim` axis.

### 3. Map the weight bytes

`src/engine/model.cpp::load_gguf_weights`

- `mmap()` the entire file `PROT_READ | MAP_PRIVATE`. The file isn't
  copied into RAM yet — Linux's page cache will pull pages on demand.
- `cudaHostRegister(..., cudaHostRegisterMapped | cudaHostRegisterReadOnly)`
  to pin the pages and get a CUDA-visible device alias for the mmap'd
  weights. CUDA kernels use that device alias; raw CPU mmap pointers are
  not dereferenced on-device.

After this, `Engine` holds a single `void* weights_` pointing at the
mapped+pinned blob.

### 4. Walk tensor infos and bind weights to logical slots

`src/engine/model.cpp::load_and_map_weights`

For each tensor, compute `ptr = (char*)blob + data_offset + tensor_offset`
and bind it to the right field of `ModelWeights` / `LayerWeights[N]`:

```
"token_embd.weight"           → mw->tok_embd
"output_norm.weight"          → mw->output_norm
"output.weight"               → mw->output            (absent for Qwen3!)
"blk.N.attn_norm.weight"      → mw->layers[N].rms_attn
"blk.N.attn_q.weight"         → mw->layers[N].wq
"blk.N.attn_q_norm.weight"    → mw->layers[N].q_norm  ◄── Qwen3-specific slot
"blk.N.attn_k_norm.weight"    → mw->layers[N].k_norm  ◄── Qwen3-specific slot
... etc
```

We track `mapped` count vs total tensors. For Qwen3-4B you should see
`Mapped 398 / 398 tensors to weight structs`.

If `output.weight` was missing, we alias to `token_embd.weight` — see
`if (!mw->output && mw->tok_embd) mw->output = mw->tok_embd;`.

### 5. Allocate runtime buffers

- **`MemoryBudget::probe_system_memory()`** reads `/proc/meminfo` and
  reserves CMA, CUDA context, safety margin, and the just-loaded model
  weights. What's left is the "free pool" the KV cache and scratch
  arenas can draw from.
- **`KVCachePool::init`** allocates `n_layers × n_kv_heads × head_dim ×
  context × 2 (K+V) × kv_bytes` of unified memory (`cudaMallocManaged`).
  For Qwen3-4B with FP16 KV and 4096 context:
  `2 × 36 × 8 × 128 × 4096 × 2` = 576 MiB. Plus a 144 MiB
  overflow pool for the extra 1024-token overflow context.
- **`ScratchPool::init`** grabs 64 MB of unified memory as a bump
  allocator for per-layer activation buffers (Q, K, V, attn output,
  gate/up/down FFN intermediates, etc.). Reset on each token.

### 6. Build the tokenizer

`src/engine/tokenizer.cpp`

- Read `tokenizer.ggml.tokens` array from the GGUF metadata — 151,936
  strings.
- Read `tokenizer.ggml.bos_token_id` (151643) and `.eos_token_id`
  (151645).
- Index every token string in a hash table (`token_to_id_`).
- Pre-sort by length so encode() can do longest-match-first BPE
  approximation.

### 7. Generate

`src/engine/decode.cu::Engine::generate`

For each prompt token, *prefill*:
1. Reset the scratch pool.
2. Dequant the embedding row from `token_embd.weight` into the
   first scratch buffer. For Qwen3-4B this is **Q6_K** (not Q4_K) —
   handled by `dequant_q6k_row`.
3. Run all 36 `transformer_layer(layer, pos, x)` calls.

Then for each output token, *decode*:
1. Same as above for one token.
2. Final `output_norm` RMSNorm.
3. `gemv_quant_f32(logits_fp32, output, output_type, normed, vocab_size, hidden_dim)`
   — multiplies the final normed hidden state by the tied embedding
   matrix to get 151,936 FP32 logits.
4. Sample (top-k / top-p / temperature).
5. Append to `recent_tokens_`, set `last_token_`, loop.

### 8. transformer_layer (the most complex single step)

```cpp
half* normed = scratch_.get(H);
fused_rmsnorm_residual(normed, x, nullptr, lw.rms_attn, ...);

gemv_quant(q_buf, lw.wq, lw.type_wq, normed, n_heads*head_dim, H, ...);
gemv_quant(k_buf, lw.wk, lw.type_wk, normed, n_kv_heads*head_dim, H, ...);
gemv_quant(v_buf, lw.wv, lw.type_wv, normed, n_kv_heads*head_dim, H, ...);

// Qwen3 only: RMSNorm Q and K per-head against attn_q_norm / attn_k_norm
fused_rmsnorm_residual(q_buf, q_buf, nullptr, lw.q_norm, n_heads,    head_dim, ...);
fused_rmsnorm_residual(k_buf, k_buf, nullptr, lw.k_norm, n_kv_heads, head_dim, ...);

rope_inplace(q_buf, k_buf, n_heads, n_kv_heads, head_dim, pos, rope_theta, rope_neox);

kv_cache_.store(layer, pos, k_buf, v_buf);
flash_attention(attn_out, q_buf, kv_cache_, layer, pos, n_heads, n_kv_heads, head_dim);
gemv_quant(attn_proj, lw.wo, lw.type_wo, attn_out, H, n_heads*head_dim, ...);

vec_add(x2, x, attn_proj);                           // residual

fused_rmsnorm_residual(normed2, x2, nullptr, lw.rms_ffn, ...);
gemv_quant(gate_buf, lw.w_gate, lw.type_w_gate, normed2, I, H, ...);
gemv_quant(up_buf,   lw.w_up,   lw.type_w_up,   normed2, I, H, ...);
swiglu(swiglu_out, gate_buf, up_buf, I);             // silu(gate) * up
gemv_quant(ffn_out, lw.w_down, lw.type_w_down, swiglu_out, H, I, ...);

vec_add(x, x2, ffn_out);                             // residual into x in place
```

## Where Qwen3 differs from our assumptions — bug ↔ fix map

These are the actual bugs we hit on first run, mapped to the Qwen3
choice that broke our default.

| Bug we hit | Qwen3 choice that broke us | Fix |
| --- | --- | --- |
| Auto-context picked 40,960 → oversized KV allocation | `max_seq_len = 40,960` (large) | Cap auto-context to a Jetson-safe default, currently 4096 for 128-dim K/V heads |
| `output.weight` missing → null pointer in final GEMV | `tie_word_embeddings: true` | Alias `output` to `token_embd` when not found (`5abc01b`) |
| `vocab_size = 2560` reported (= hidden_dim) | GGUF stores `token_embd.weight` as `[hidden_dim, vocab_size]` (column-major) | Read the non-hidden_dim axis (`202b6ca`) |
| Q/K/V projection widths wrong | Qwen3 stores `attention.key_length = 128`, so Q is 4096-wide and K/V are 1024-wide even though hidden_dim is 2560 | Use `head_dim` metadata for attention projection widths (`d96d082`) |
| Token salad despite "valid" embeddings, magnitudes 13/59 | `token_embd.weight` is **Q6_K** in Q4_K_M builds, not Q4_K | Add `dequant_q6k_row` (`afffb28`) |
| RMSNorm output = constant garbage regardless of input | `cudaMallocHost` on Tegra isn't auto-GPU-visible | Switch scratch + KV pools to `cudaMallocManaged` (`1f4bcac`) |
| Garbled attention even after RMSNorm fixed | Qwen3 has **QK-norm**: RMSNorm on Q and K per-head before RoPE | Add the QK-norm step (`258aa2d`) |
| Subtle attention errors that grew across layers | KV cache fast-pool stride was wrong | Fix layer stride (`fb8fbae`) |
| Custom kernels miscomputed | Multiple — kernel-level work for Week 3-5 | Re-enable validated GPU GEMV/RMSNorm/attention paths by default (`88cab24`) |

## Mental model going forward

When you bring a new model family up on this runtime, walk the list:

1. **Architecture name** in GGUF metadata (`general.architecture`)?
2. **Tensor naming convention** matches what `load_and_map_weights`
   expects? Especially: are there extra weights like `attn_q_norm`?
3. **Embedding type**: Q4_K? Q6_K? FP16? FP32?
4. **Output weight tied** to embedding or separate?
5. **RoPE theta** — 10k? 1M? Something else?
6. **Head dim** — 128 or unusual?
7. **GQA grouping** — what's `n_heads / n_kv_heads`?
8. **Per-layer extras** — QK-norm? attention biases? something we haven't seen?

If the runtime's `Mapped N/M` print shows a lot less than M, you're
silently missing tensors. That used to be a `WARNING` we glossed over;
it's the most useful diagnostic for "this model has structure we
don't know about."

## Current validation state

Week 1 is complete for Qwen3-4B Q4_K_M: the runtime generates coherent
output on Jetson, and the validated GPU GEMV, RMSNorm, and decode attention
paths are enabled by default. CPU/reference fallbacks remain available through
`JLLM_FAST_GEMV=0`, `JLLM_FAST_NORM=0`, and `JLLM_FAST_ATTN=0`.

The next roadmap items are llama.cpp baseline benchmarking, long-run stability,
and additional throughput tuning beyond the correctness baseline.
