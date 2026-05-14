# Week 1 Validation

Date: 2026-05-14
Hardware: Jetson Orin Nano Super 8 GB
Software: L4T 36.4, CUDA 12.6
Power: 25 W mode, GPU locked at 918 MHz with `jetson_clocks`

## Model

`/opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf`

Observed model load:

```
[engine] Qwen3 4B Instruct Awq: 36 layers, 32 heads (8 KV),
dim=2560, head_dim=128, vocab=151936, rms_eps=1e-06, rope=neox
[model] Mapped 398 / 398 tensors to weight structs
[model] output.weight not found; tying to token_embd.weight
[model] Materialized up to 145 RMSNorm tensors on device
[engine] Context: 4096 tokens
```

## Command

```bash
./build/jetson-llm \
  -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
  -p "Hello, this is testing." \
  -n 64 -v
```

Fast CUDA paths are enabled by default:

| Path | Default | Fallback |
|------|---------|----------|
| K-quant GEMV | CUDA | `JLLM_FAST_GEMV=0` |
| RMSNorm | CUDA | `JLLM_FAST_NORM=0` |
| Decode attention | CUDA | `JLLM_FAST_ATTN=0` |

## Result

Output was coherent and on-topic:

```
Hello! It seems like you're testing the system. How can I assist you today?
```

Stats:

| Metric | Value |
|--------|-------|
| Prompt tokens | 18 |
| Prompt eval | 12,377 ms, 1.5 tok/s |
| Decode tokens | 19 |
| Decode | 13,109 ms, 1.4 tok/s |
| Model file | 2381 MB |
| KV cache | 720 MB |
| Scratch | 64 MB |
| Peak memory | 3425 MB |
| Peak thermal | 50.0°C |

## Fixed During Bring-Up

- GGUF vocab dimension detection for Qwen3.
- Qwen BPE merges and special token handling.
- Qwen3 chat template with no-think default.
- Qwen3 architecture mapping: 128-dim attention heads, Q/K RMSNorm, NeoX RoPE.
- Tied output embedding fallback.
- KV cache layout for `[seq_len, n_kv_heads, head_dim]`.
- CUDA-visible mapped GGUF weights for K-quant GEMV.
- GPU RMSNorm and decode attention paths.
- L4T R36 Jetson power and GPU frequency reporting.

## Remaining Work

- Baseline against llama.cpp with the same Qwen3 GGUF and prompt.
- Long-run stability: 1000+ token generations and repeated iterations.
- Server parity: apply the same Qwen chat-template behavior in HTTP requests.
- Throughput tuning beyond the correctness baseline.
