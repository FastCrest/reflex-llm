# Changelog

## v0.1.0-alpha.1 — 2026-05-13

Initial seeded release. The codebase is lifted verbatim from the
[ai-hpc/ai-hardware-engineer-roadmap / Projects / jetson-llm-runtime](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Projects/jetson-llm-runtime)
framework (31 files, ~4500 LOC). Status: code-complete, compiles cleanly,
needs hardware validation on Orin Nano.

### Added

- `src/memory/` — `MemoryBudget`, `OOMGuard`, `KVCachePool`, `ScratchPool`.
  Every allocation accounted for before inference starts so the runtime
  can coexist with `whisper-server` + Piper + Home Assistant on a
  7.6 GB iGPU.
- `src/jetson/` — Orin HAL: `PowerState` (nvpmodel 7–25 W), `ThermalState`
  (adaptive backoff), `JetsonInfo` (probed once at startup), `LiveStats`.
- `src/kernels/` — SM 8.7 CUDA kernels: `gemv_q4` (INT4 dequant-fused
  matrix-vector), flash `attention`, `fused_norm`, `rope`, `softmax`,
  `convert`, `vec_add`. Tuned for 48 KB shared memory and 16 SMs.
- `src/engine/` — Transformer forward pass: GGUF loader, tokenizer
  (encode/decode), top-k/top-p/temperature sampling, layer orchestration.
- `src/server/` — OpenAI-compatible HTTP: `/v1/chat/completions`, `/health`,
  `/v1/models`. Drop-in target for swapping out `llama-server` from
  GenieClaw.
- CMake build, with explicit aarch64-Jetson-only guard (rejects x86,
  discrete GPUs, Windows, macOS).
- `scripts/setup_jetson.sh`, `scripts/bench.sh`, `scripts/profile.sh`,
  `scripts/test_plan.sh`.
- Tests: `tests/test_kernels.cu`, `tests/test_memory.cpp`,
  `tests/test_model_load.cpp`.
- Documentation: architecture, build, engine, GGUF, jetson-HAL, kernels,
  memory, performance, server, testing.

### Known limits

- No on-device runtime validation yet (this is what alpha.1 → alpha.2
  exists to do).
- Internal symbols use `jllm`/`jllm_*` namespacing inherited from the
  seed framework. Renaming to a GenieClaw-aligned `geniert_*` namespace
  is deferred to v0.2 to keep the alpha churn-free.
- GenieClaw integration is read-only for now — both runtimes will run
  side-by-side starting in alpha.8 before any default flip.
