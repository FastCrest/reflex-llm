# jetson-llm Roadmap

## Current State: v0.1-alpha.2 (Week 1 validated on Jetson)

```
Jetson Orin Nano Super 8 GB | L4T 36.4 | CUDA 12.6 | Qwen3-4B-Q4_K_M.gguf
```

Validated on hardware:

- Coherent Qwen3 instruct generation from a real GGUF model.
- Default CUDA decode paths for K-quant GEMV, RMSNorm, and single-token attention.
- Qwen3 model support: BPE merges, special tokens, chat template with no-think default,
  128-dim attention heads, Q/K RMSNorm, NeoX RoPE, tied output embeddings.
- L4T R36 power reporting: 25 W mode and GPU frequency reported correctly.

Current measured smoke test:

| Item | Result |
|------|--------|
| Model | `Qwen3-4B-Q4_K_M.gguf` |
| Command | `./build/jetson-llm -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf -p "Hello, this is testing." -n 64 -v` |
| Output | `Hello! It seems like you're testing the system. How can I assist you today?` |
| Prompt | 18 tokens, 12.377 s, 1.5 tok/s |
| Decode | 19 tokens, 13.109 s, 1.4 tok/s |
| Memory | 3425 MB peak, 720 MB KV, 64 MB scratch |
| Thermal | 50.0 deg C peak |
| Power | 25 W mode, GPU locked at 918 MHz |

Evidence:

- Validation log: [`docs/validation-week1.md`](docs/validation-week1.md)
- Fixed/closed: GitHub issue #1 (first coherent tokens), issue #8 (Jetson power reporting)
- Still open: llama.cpp baseline, longer stability tests, server parity, multi-model matrix

---

## v0.1 — First Tokens (Target: Week 1)

**Goal:** generate coherent text from a real GGUF model on Jetson hardware.

Status: complete for Qwen3 CLI bring-up.

- [x] Build on Jetson with CMake, NVCC, SM 8.7.
- [x] Load real GGUF model: `/opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf`.
- [x] Parse GGUF metadata, tokenizer, BPE merges, special tokens, and 398/398 tensors.
- [x] Apply Qwen chat template by default and suppress Qwen3 thinking tokens by default.
- [x] Generate coherent English output from CLI.
- [x] Avoid the previous failure modes: garbage tokens, NaNs in layer 0, CUDA illegal access,
  segfault, and all-zero/invalid logits.
- [x] Use safe context auto-cap for 8 GB Orin Nano: 4096 tokens for Qwen3-4B in the
  validated memory state.
- [ ] Capture a final standalone `ctest --output-on-failure` log on the Jetson after
  the latest CUDA default-path changes.

Validated command:

```bash
./build/jetson-llm \
  -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
  -p "Hello, this is testing." \
  -n 64 -v
```

**Deliverable:** first coherent generation on Jetson is documented in
[`docs/validation-week1.md`](docs/validation-week1.md).

---

## v0.2 — Benchmark Baseline (Target: Week 2)

**Goal:** measure performance and compare against llama.cpp.

Status: next active milestone. llama.cpp source was used as a reference during
fixes, but a same-device llama.cpp benchmark has not been captured yet.

- [ ] Build llama.cpp on the same Jetson:

```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

- [ ] Run the current runtime benchmark with debug logging off:

```bash
cd ~/genie-ai-runtime
./build/jetson-llm \
  -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
  -p "Hello, this is testing." \
  -n 64 -v
```

- [ ] Run the same model and comparable prompt on llama.cpp:

```bash
cd ~/llama.cpp
./build/bin/llama-cli \
  -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
  -p "Hello, this is testing." \
  -n 64 -ngl 99

./build/bin/llama-bench \
  -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
  -ngl 99 -p 18 -n 64
```

- [ ] Record baseline comparison:

| Runtime | Model | Prompt tok/s | Decode tok/s | Peak RAM | Peak temp | Notes |
|---------|-------|--------------|--------------|----------|-----------|-------|
| jetson-llm | Qwen3-4B Q4_K_M | 1.5 | 1.4 | 3425 MB | 50.0 deg C | 25 W, GPU 918 MHz, correctness smoke test |
| llama.cpp | Qwen3-4B Q4_K_M | TBD | TBD | TBD | TBD | Same Jetson, same model |

- [ ] Run profile script / Nsight Systems and identify top 3 bottlenecks:

| Rank | Kernel / path | Time share | Notes |
|------|---------------|------------|-------|
| #1 | TBD | TBD | likely K-quant GEMV / output projection |
| #2 | TBD | TBD | likely attention or final vocab projection |
| #3 | TBD | TBD | likely RMSNorm / sampling / host sync |

- [ ] Test context lengths: 512, 1024, 2048, 4096.
- [ ] Test power modes: 7 W, 15 W, 25 W; record tok/s and tokens/joule.
- [ ] Later: repeat on TinyLlama 1.1B, Llama 3.2 3B, and other Tier 1 models.

**Deliverable:** performance comparison table (jetson-llm vs llama.cpp).

---

## v0.3 — Kernel Optimization (Target: Week 3–4)

**Goal:** >20% faster decode than stock llama.cpp on Llama 3.2 3B.

Status: started early for correctness and basic acceleration. Benchmark-winning
tuning still depends on v0.2 llama.cpp and Nsight data.

- [x] Map GGUF weights into CUDA-visible host memory for K-quant GEMV.
- [x] Enable default CUDA GEMV for Q4_K, Q5_K, and Q6_K tensors.
- [x] Enable default CUDA RMSNorm path with CPU fallback via `JLLM_FAST_NORM=0`.
- [x] Enable default CUDA decode-attention path with CPU fallback via `JLLM_FAST_ATTN=0`.
- [x] Keep per-kernel fallback switches:
  `JLLM_FAST_GEMV=0`, `JLLM_FAST_NORM=0`, `JLLM_FAST_ATTN=0`.
- [x] Remove the need for debug env vars for normal coherent output.
- [ ] Tune GEMV block size, elements per thread, vectorized loads, and output projection.
- [ ] Tune attention tile size and compare FP16 vs INT8 KV behavior.
- [ ] Profile register count, occupancy, bandwidth, and host synchronization.
- [ ] Validate CUDA graph capture/replay on the current decode loop.
- [ ] Re-run against llama.cpp after profiling and record speedup:

| Metric | Before CUDA defaults | Current smoke test | Target after tuning |
|--------|----------------------|--------------------|---------------------|
| Qwen3-4B decode | garbage / invalid output | 1.4 tok/s | TBD after llama.cpp baseline |
| Correctness | failing | coherent | coherent |
| CUDA errors | seen during bring-up | none in final smoke test | none |

**Deliverable:** >20% decode speedup over v0.2, profiling evidence.

---

## v0.4 — Memory Stability (Target: Week 5)

**Goal:** stable operation for 1000+ tokens with no memory growth.

Status: not yet tested beyond short smoke runs. Run this after v0.2 baseline.

- [ ] Generate 1000 tokens continuously and capture tegrastats:

```bash
cd ~/genie-ai-runtime
tegrastats --interval 1000 | tee stability-qwen3-1000.log &
TEGRASTATS_PID=$!
./build/jetson-llm \
  -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
  -p "Write a concise technical overview of Jetson Orin Nano inference." \
  -n 1000 -v
kill "$TEGRASTATS_PID"
```

- [ ] Verify no memory growth:
  delta between early steady state and token 1000 should be under 5 MB.
- [ ] Test KV cache eviction:
  run Qwen3-4B with `-c 512 -n 1000` and verify overflow/eviction behavior.
- [ ] Test OOM guard:
  attempt a larger model or an intentionally oversized context and verify graceful stop.
- [ ] Test thermal stability:
  30 minutes continuous generation at 25 W; stay below 85 deg C with active cooling.
- [ ] Stress test rapid start/stop:

```bash
for i in {1..100}; do
  ./build/jetson-llm \
    -m /opt/geniepod/models/Qwen3-4B-Q4_K_M.gguf \
    -p "Hi" -n 10 >/tmp/jllm-stress.log 2>&1 || exit 1
done
```

**Deliverable:** stability report — memory graph, thermal graph, OOM test results.

---

## v0.5 — Server + Streaming (Target: Week 6)

**Goal:** production-ready HTTP API with streaming.

Status: server binary builds, but the final Qwen CLI behavior has not yet been
validated through HTTP.

- [ ] Verify server uses the same Qwen chat template/no-think behavior as CLI.
- [ ] Implement or validate streaming SSE in `http_server.cpp`:
  `data: {"choices":[{"delta":{"content":"token"}}]}\n\n`, final `data: [DONE]\n\n`.
- [ ] Add request timeout, default 60 seconds.
- [ ] Test with real clients:
  curl streaming, Python OpenAI SDK streaming, browser `ReadableStream`.
- [ ] Add systemd service:
  restart always, network dependency, environment file for model path and port.
- [ ] Auto-start on boot and auto-restart on crash.
- [ ] Enhance `/health` with uptime, total requests, average tok/s, and KV cache tokens used.

**Deliverable:** streaming API working with OpenAI SDK, systemd service running.

---

## v0.6 — Multi-Model Support (Target: Week 7–8)

**Goal:** test and document performance across all Tier 1–2 models.

Status: one Tier 2 model family validated; broader matrix remains.

| Model | Quant/file size | Decode tok/s | Prompt tok/s | Peak RAM | Max ctx | Status |
|-------|-----------------|--------------|--------------|----------|---------|--------|
| Qwen3 4B Instruct AWQ | 2381 MB `Q4_K_M` | 1.4 | 1.5 | 3425 MB | 4096 | validated CLI |
| TinyLlama 1.1B | 669 MB `Q4_K_M` | TBD | TBD | TBD | TBD | not yet tested |
| Llama 3.2 1B | 750 MB `Q4_K_M` | TBD | TBD | TBD | TBD | not yet tested |
| Qwen3 1.7B | ~1.0 GB `Q4_K_M` | TBD | TBD | TBD | TBD | not yet tested |
| Llama 3.2 3B | ~1.8 GB `Q4_K_M` | TBD | TBD | TBD | TBD | not yet tested |
| Phi-4 Mini 3.8B | ~2.3 GB `Q4_K_M` | TBD | TBD | TBD | TBD | not yet tested |
| Gemma 3 4B | ~2.5 GB `Q4_K_M` | TBD | TBD | TBD | TBD | not yet tested |
| Llama 3.3 8B | ~4.6 GB `Q4_K_M` | TBD | TBD | TBD | TBD | stress / OOM guard |

- [x] Qwen tokenizer: BPE merges and special tokens.
- [ ] Llama tokenizer family.
- [ ] Gemma tokenizer family.
- [ ] Phi tokenizer family.
- [ ] Record tensor naming, RoPE theta, head_dim, and GQA differences per model.
- [ ] Update README and `docs/performance.md` after each verified model.

**Deliverable:** performance table with 8+ models, all verified working.

---

## v0.7 — Speculative Decoding (Target: Week 9–10)

**Goal:** 1.5–2× faster decode using draft model.

```
□ Implement speculative decode:
    Draft model: TinyLlama 1.1B (fast, ~65 tok/s)
    Target model: Llama 3.2 3B (slow, ~25 tok/s)

    Algorithm:
    1. Draft generates N candidate tokens (N=4–8)
    2. Target verifies all N in one forward pass
    3. Accept matching tokens, reject from first mismatch
    4. Expected: accept 60–80% → 1.5–2× speedup

□ Memory budget for both models:
    Draft:  ~670 MB (TinyLlama 1.1B Q4_K_M)
    Target: ~1.8 GB (Llama 3.2 3B Q4_K_M)
    KV caches × 2: ~200 MB
    Total:  ~2.7 GB → fits in 5.5 GB available ✓

□ Implement draft-verify loop in engine
□ Share tokenizer between draft and target
□ Measure acceptance rate at draft lengths 2, 4, 6, 8
□ Measure effective tok/s vs single-model decode
□ Add --draft-model CLI flag
```

**Deliverable:** speculative decoding working, measured speedup.

---

## v0.8 — Multi-Turn Conversation (Target: Week 11)

**Goal:** KV cache persistence across turns.

```
□ Implement conversation state:
    Keep KV cache between turns (don't re-prefill system prompt)
    Only prefill new user message each turn

□ System prompt support:
    Pre-load system prompt into KV cache at startup
    New turns only process user message + generate response

□ Context window management:
    When KV cache exceeds limit:
    Option A: truncate oldest messages (sliding window)
    Option B: summarize old context (future)

□ Conversation API:
    POST /v1/chat/completions with messages array
    Server maintains conversation_id → KV cache state

□ Test: 10-turn conversation
    Verify later turns reference earlier context
    Verify memory stable across turns
    Measure: time-to-first-token for turn 1 vs turn 10
```

**Deliverable:** multi-turn chat working, KV reuse verified.

---

## v1.0 — Production Release (Target: Week 12)

**Goal:** stable, documented, deployable.

```
□ 24-hour stability test:
    Continuous requests every 5 seconds for 24 hours
    No OOM, no crash, no thermal shutdown
    Memory delta < 10 MB over 24 hours

□ Documentation complete:
    □ README: features, quickstart, performance table
    □ docs/: all 10 documents up to date
    □ TESTING.md: reflects actual test results
    □ CHANGELOG.md: all versions documented

□ Packaging:
    □ Single tar.gz with binary + scripts + docs
    □ Or: Dockerfile for Jetson (l4t-base image)

□ Release:
    □ Tag v1.0.0
    □ GitHub release with binary + docs
    □ Blog post: "Building a Memory-First LLM Runtime for Jetson"
```

**Deliverable:** tagged v1.0.0 release, 24-hour stability proven.

---

## Future (Post v1.0)

| Feature | Description | Effort |
|---------|-------------|--------|
| **INT4 KV cache** | Halve KV memory → 2× more context | 1 week |
| **Tensor Core prefill** | WMMA for batch matmul during prefill | 2 weeks |
| **Vision-language models** | Gemma 3 4B image+text | 2 weeks |
| **DLA offload** | Run some layers on DLA, free GPU for others | 2 weeks |
| **Multiple concurrent models** | Hot-swap models without restart | 1 week |
| **WebSocket API** | Real-time bidirectional streaming | 1 week |
| **ONNX Runtime fallback** | Support non-GGUF models | 2 weeks |
| **Jetson Orin NX 16 GB** | Extend to larger Jetson for 7B models | 1 week |
| **Benchmark dashboard** | Live Grafana/Prometheus metrics | 1 week |
| **Model quantization on-device** | Quantize FP16→INT4 directly on Jetson | 2 weeks |

---

## Timeline Summary

```
Week 1:    v0.1  First Tokens         [x] Qwen3 coherent CLI output on Jetson
Week 2:    v0.2  Benchmark            [ ] llama.cpp baseline + Nsight profile
Week 3-4:  v0.3  Kernel Optimization  [~] GPU GEMV/RMSNorm/attention landed; tuning remains
Week 5:    v0.4  Memory Stability     [ ] 1000 tokens + eviction + OOM + thermal
Week 6:    v0.5  Server + Streaming   [ ] Qwen server parity + SSE + systemd
Week 7-8:  v0.6  Multi-Model          [~] Qwen3-4B validated; 7+ models remain
Week 9-10: v0.7  Speculative Decode   [ ] draft+target -> 1.5-2x speedup
Week 11:   v0.8  Multi-Turn           [ ] KV persistence -> conversation state
Week 12:   v1.0  Production Release   [ ] 24-hour test -> package -> release
```
