# Engine

## Lifecycle

```
Engine engine;
engine.load("model.gguf", params);   // parse GGUF, mmap, allocate pools
engine.generate("Hello", params, cb); // tokenize → prefill → decode → stream
engine.unload();                      // free everything
```

## load()

1. Probe system memory (`probe_system_memory()`)
2. Parse GGUF config (`load_gguf_config()`)
3. Load and map weights (`load_and_map_weights()`) — mmap + cudaHostRegister + tensor name matching
4. Auto-calculate max context from remaining memory
5. Allocate KV cache pool (pinned fast + unpinned overflow)
6. Allocate scratch pool (bump allocator)
7. Create CUDA stream
8. Load tokenizer from GGUF
9. Print memory budget

## Transformer Layer

`transformer_layer(layer, pos, x)` — 12 operations per layer:

```
Input: x [hidden_dim] — hidden state from previous layer

┌─ Attention Block ────────────────────────────────────┐
│  1. normed = RMSNorm(x) × attn_weight               │
│  2. Q = gemv_q4(W_q, normed)                         │
│  3. K = gemv_q4(W_k, normed)                         │
│  4. V = gemv_q4(W_v, normed)                         │
│  5. RoPE(Q, K, position)                              │
│  6. KV cache store (INT8 quantize if enabled)         │
│  7. attn_out = flash_attention(Q, K_cache, V_cache)   │
│  8. attn_proj = gemv_q4(W_o, attn_out)                │
│  9. x2 = x + attn_proj              ← residual #1    │
└──────────────────────────────────────────────────────┘

┌─ FFN Block ──────────────────────────────────────────┐
│  10. normed2 = RMSNorm(x2) × ffn_weight              │
│  11. gate = gemv_q4(W_gate, normed2)                  │
│  12. up = gemv_q4(W_up, normed2)                      │
│  13. swiglu_out = silu(gate) × up                     │
│  14. ffn_out = gemv_q4(W_down, swiglu_out)            │
│  15. x = x2 + ffn_out               ← residual #2    │
└──────────────────────────────────────────────────────┘

Output: x [hidden_dim] — input to next layer
```

All intermediate buffers allocated from `ScratchPool` (reset each decode step).

## Decode Step

`decode_step(pos)` — one token generation:

1. Get hidden state buffer from scratch
2. Embedding lookup: `cudaMemcpyAsync(x, tok_embd + token_id × hidden_dim)`
3. Run `transformer_layer()` for all N layers
4. Final RMSNorm
5. Logit projection: `gemv_q4(W_output, normed)` → FP16
6. Convert FP16 → FP32 on GPU (`fp16_to_fp32` kernel)
7. Copy FP32 logits to CPU (`cudaMemcpy D2H`)
8. Sample: `sample_token(logits, vocab_size, params)`
9. Update recent tokens (for repeat penalty)
10. Return token ID

## Generation Loop

`generate(prompt, params, callback)`:

```
Tokenize prompt → token IDs
│
├── Prefill phase:
│   For each prompt token:
│     scratch.reset()
│     embedding lookup + all transformer layers
│     (builds KV cache, no sampling)
│
├── Decode phase:
│   For each output token (up to max_tokens):
│     check_memory_and_thermal()  ← OOM guard + thermal
│     scratch.reset()
│     token = decode_step(pos)
│     callback(detokenized_text, is_eos)
│     if EOS: break
│
└── Return GenStats
```

## CUDA Graphs

`build_cuda_graph(pos)` captures the GPU-side decode step as a CUDA graph:

- All transformer layers + final norm + logit projection captured
- Replayed with `cudaGraphLaunch()` for subsequent tokens
- Reduces kernel launch overhead from ~1ms to ~5μs per token
- Not captured: embedding lookup (host→device), sampling (host-side)
- Graph must be rebuilt if KV cache structure changes

## Sampling

`sample_token()` (`src/engine/sample.cpp`) — CPU-side token selection:

1. Apply repeat penalty (penalize recent tokens)
2. Apply temperature (divide logits by T)
3. If T=0: greedy (argmax)
4. Softmax on CPU (logits are small: vocab_size × 4 bytes)
5. Top-K filter (keep K highest, partial sort)
6. Top-P filter (keep until cumulative probability > P)
7. Random sample from filtered distribution

## Tokenizer

`Tokenizer` (`src/engine/tokenizer.cpp`):

### Encoding

Uses hash map `token_to_id_` for O(max_token_len) per position:
1. At each position, try longest match first (decreasing length)
2. Hash map lookup for each candidate substring
3. If no match: byte fallback (`<0x41>` → byte token)

### Decoding

Direct lookup: `vocab[token_id]`. Handles byte tokens (`<0xNN>` → actual byte).

## Stop Mechanism

`engine.stop()` sets `stop_flag_ = true`. The decode loop checks this every iteration and breaks cleanly. Used for:
- SIGINT handler (Ctrl+C in CLI)
- HTTP request cancellation
- Timeout

## GenStats

Returned after generation:

| Field | Description |
|-------|-------------|
| `prompt_tokens` | Number of prompt tokens processed |
| `completion_tokens` | Number of tokens generated |
| `prompt_ms` | Time for prefill phase |
| `decode_ms` | Time for decode phase |
| `prompt_tok_per_sec` | Prefill throughput |
| `decode_tok_per_sec` | Decode throughput |
| `peak_memory_mb` | Maximum memory usage observed |
| `peak_thermal_c` | Maximum GPU temperature observed |
| `oom_stops` | Times OOM guard stopped generation |
| `thermal_pauses` | Times thermal backoff triggered |
