# reflex-llm

Jetson-first LLM runtime for local inference.

`reflex-llm` owns model loading, tokenizer/runtime state, memory budgeting,
KV-cache lifecycle, sampling, CLI/server surfaces, and integration with local
applications. CUDA kernel specialization is being split into `reflex-infer`,
which should become the reusable Jetson kernel library.

Status: rebrand in progress from the original `genie-ai-runtime` codebase.
The internal C++ namespace and environment variables still use `jllm` /
`JLLM_*` while the public repo, binaries, and architecture are moved to the
Reflex naming.

## Target

Initial target:

- Hardware: Jetson Orin Nano 8GB.
- Model: Qwen 3.5B or Qwen3 4B class GGUF model.
- Precision: GGUF `Q4_K_M` first, matching llama.cpp baselines.
- Runtime mode: standalone CUDA first.
- Baselines: `llama.cpp` and `tensorrt-edge-llm`.

Future targets:

- Jetson Orin NX 16GB.
- Jetson AGX Orin 64GB.
- Jetson Thor.
- More model families, including Phi-4 and other compact edge models.

## Two-Repo Split

The intended architecture is:

- `reflex-llm`: full LLM runtime and product integration layer.
- `reflex-infer`: Jetson-optimized CUDA kernel library, similar in spirit to
  FlashInfer but scoped to NVIDIA Jetson devices.

`reflex-llm` uses `reflex-infer` as the external kernel dependency. The local
`src/kernels/` files preserve the runtime ABI and forward extracted operators
into `reflex-infer`.

The split is documented in [docs/reflex-infer-integration.md](docs/reflex-infer-integration.md).

## Build

Prereqs on Jetson:

- JetPack 6 / L4T R36.x.
- CUDA 12.x from JetPack.
- CMake 3.20 or newer.
- C++17 compiler.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Outputs:

- `build/reflex-llm`: CLI.
- `libreflex_llm_core.a`: embeddable runtime library.

The OpenAI-compatible HTTP server is opt-in:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DREFLEX_LLM_BUILD_SERVER=ON
cmake --build build -j$(nproc)
```

Server output:

- `build/reflex-llm-server`

`reflex-infer` is required for extracted GEMM and attention kernels. For local
two-repo development keep it as a sibling checkout:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DREFLEX_LLM_USE_REFLEX_INFER=ON
cmake --build build -j$(nproc)
```

By default CMake looks for `../reflex-infer`. Override with
`-DREFLEX_INFER_SOURCE_DIR=/path/to/reflex-infer`, or install `reflex-infer`
as a CMake package.

The GGUF K-quant Q4 GEMV/GEMM wrappers and decode/prefill attention wrappers
now route into `reflex-infer`.

## Run

Single prompt:

```bash
./build/reflex-llm -m /path/to/model.gguf -p "Hello"
```

Interactive chat:

```bash
./build/reflex-llm -m /path/to/model.gguf -i --chat
```

Server:

```bash
./build/reflex-llm-server -m /path/to/model.gguf -p 8080
```

## Current Modules

| Module | Responsibility |
| --- | --- |
| `src/engine/` | GGUF loading, transformer forward pass, tokenizer, sampling |
| `src/memory/` | memory budget, OOM guard, KV-cache pool, scratch pool |
| `src/jetson/` | power, thermal, system probe, live stats |
| `src/kernels/` | local CUDA kernels plus ABI bridges into `reflex-infer` |
| `src/persistence/` | persistent KV-cache file format |
| `src/server/` | optional OpenAI-compatible HTTP server |

Master header: [include/jllm.h](include/jllm.h)

## Documentation

| Doc | Purpose |
| --- | --- |
| [docs/reflex-infer-integration.md](docs/reflex-infer-integration.md) | two-repo architecture and extraction contract |
| [docs/architecture.md](docs/architecture.md) | runtime architecture |
| [docs/kernels.md](docs/kernels.md) | current in-repo CUDA kernel notes |
| [docs/memory.md](docs/memory.md) | memory and KV-cache design |
| [docs/jetson-hal.md](docs/jetson-hal.md) | power and thermal handling |
| [docs/qwen3-vs-our-runtime.md](docs/qwen3-vs-our-runtime.md) | Qwen bring-up notes |
| [docs/performance.md](docs/performance.md) | historical performance notes |

## Tests

```bash
cd build
ctest --output-on-failure
```

This repository is Jetson-only. The top-level CMake configuration rejects
non-aarch64 hosts.

## License

Business Source License 1.1. See [LICENSE](LICENSE).
