# genie-ai-runtime

Jetson Orin-tuned LLM inference runtime — memory-first, power-aware,
zero-allocation. Built to serve [`GenieClaw`](https://github.com/GeniePod/genie-claw)
on a 7.6 GB iGPU without crowding out whisper.cpp + Piper + Home Assistant.

**Target hardware:** Jetson Orin Nano Super 8 GB (SM 8.7, 102 GB/s, 67 TOPS GPU)
**Not supported:** x86, discrete GPUs, Windows, macOS — Jetson only.

## Status

`v0.1.0-alpha.1` — code-complete from the seed framework at
[ai-hpc/ai-hardware-engineer-roadmap / Projects / jetson-llm-runtime](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Projects/jetson-llm-runtime).
Compiles cleanly; pending hardware validation per `ROADMAP.md` Week 1–2.

## Why

Existing runtimes are not designed for 8 GB unified memory shared with
voice STT, TTS, denoise, and a Home Assistant container:

- **llama.cpp** — portable, generic CUDA kernels, no Jetson memory
  awareness. Current default in GenieClaw; the runtime this project aims
  to replace.
- **TensorRT-LLM** — fast but datacenter-shaped (A100/H100), too heavy
  for Orin Nano's iGPU budget.
- **genie-ai-runtime** — memory-first, power-aware, Orin-tuned CUDA
  kernels (SM 8.7), zero-allocation steady-state inference. Single
  binary, single GGUF model file, single shared-memory budget that
  fits alongside `whisper-server` and `genie-core`.

## Architecture (modules)

| Module | Header | Responsibility |
| --- | --- | --- |
| `src/memory/` | `jllm_memory.h` | `MemoryBudget`, `OOMGuard`, `KVCachePool`, `ScratchPool` — every allocation accounted for before inference starts |
| `src/jetson/` | `jllm_jetson.h` | `PowerState` (nvpmodel 7–25 W), `ThermalState` (adaptive backoff), `JetsonInfo`, `LiveStats` |
| `src/kernels/` | `jllm_kernels.h` | Orin SM 8.7 CUDA: `gemv_q4` (INT4 dequant-fused), flash `attention`, `fused_norm`, `rope`, `softmax`, `convert` |
| `src/engine/` | `jllm_engine.h` | GGUF model load, transformer forward pass, tokenizer, top-k/top-p/temperature sampling |
| `src/server/` | — | OpenAI-compatible HTTP `/v1/chat/completions`, `/health`, `/v1/models` |

Master header: [`include/jllm.h`](include/jllm.h).

## Integration with GenieClaw

`genie-ai-runtime` ships an HTTP server (`src/main_server.cpp`) whose
`/v1/chat/completions` shape matches `llama-server`'s `/completion`
endpoint closely enough that GenieClaw can swap backends by changing
one config value (`llm_model_path` plus a future
`llm_backend = "llama-server" | "genie-ai-runtime"` toggle).

The plan is to land genie-ai-runtime as **opt-in** in GenieClaw's
alpha.8 cycle, run both backends in parallel on the same Jetson for a
week, A/B compare on the issue #19 latency banner, and flip the
default once parity is confirmed.

## Build

See [`docs/build.md`](docs/build.md). Quick path on a Jetson:

```
# Prereqs: CUDA toolkit, CMake ≥ 3.18, gcc/g++ aarch64-linux-gnu
git clone https://github.com/GeniePod/genie-ai-runtime.git
cd genie-ai-runtime
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Outputs:
- `build/jllm` — single-prompt CLI
- `build/jllm-server` — HTTP server (drop-in replacement target for
  `llama-server`)

## Run

```
# CLI
./build/jllm --model /path/to/model.gguf --prompt "Hello"

# Server (HTTP, OpenAI-compatible)
./build/jllm-server --model /path/to/model.gguf --port 8080
```

## Test

See [`TESTING.md`](TESTING.md) for the full test plan. Quick check:

```
cd build && ctest --output-on-failure
```

## Roadmap

Full plan in [`ROADMAP.md`](ROADMAP.md). Short version:

| Phase | Weeks | Goal |
| --- | --- | --- |
| Early validation | 1–2 | First coherent tokens on Jetson, baseline vs. llama.cpp |
| Core optimization | 3–5 | Kernel tuning ≥ 20% faster decode, 1000+ token stability |
| Production features | 6–8 | HTTP streaming, multi-model compat, systemd unit |
| Advanced capabilities | 9–11 | Speculative decoding, persistent KV cache |
| **v1.0** | 12 | 24-hour stability test, packaging, docs complete |

## License

MIT — see [`LICENSE`](LICENSE).

This is a permissive choice on purpose: `genie-ai-runtime` is
infrastructure that other projects (including non-AGPL ones) should be
able to embed cheaply. The integrating product, GenieClaw, stays AGPL-3.0.

## Related

- [GenieClaw](https://github.com/GeniePod/genie-claw) — the local home AI
  assistant this runtime is built for. Alpha.7 currently uses llama.cpp's
  `llama-server`; genie-ai-runtime is the planned replacement.
- [Original framework / roadmap context](https://github.com/ai-hpc/ai-hardware-engineer-roadmap/tree/main/Projects/jetson-llm-runtime).
