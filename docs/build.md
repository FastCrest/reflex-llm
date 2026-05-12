# Build System

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Platform | Jetson Orin (aarch64) | Orin Nano Super 8GB |
| JetPack | 5.1+ | 6.1 (R36.4) |
| CUDA | 11.4+ | 12.6 |
| CMake | 3.20+ | 3.24+ |
| GCC | 11+ | 11.4 |
| nvcc | matches CUDA | 12.6.68 |

**Cannot cross-compile on x86.** CMakeLists.txt enforces `aarch64` with a fatal error.

## Quick Build

```bash
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="87" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Build Targets

| Target | Binary | Description |
|--------|--------|-------------|
| `jetson-llm` | `build/jetson-llm` | CLI inference |
| `jetson-llm-server` | `build/jetson-llm-server` | HTTP API server |
| `test_memory` | `build/test_memory` | Memory subsystem tests |
| `test_kernels` | `build/test_kernels` | CUDA kernel tests |
| `test_model_load` | `build/test_model_load` | GGUF loading tests |

## CMake Configuration

### Key Variables

| Variable | Value | Why |
|----------|-------|-----|
| `CMAKE_CUDA_ARCHITECTURES` | `87` | SM 8.7 = Orin Nano/NX/AGX |
| `CMAKE_BUILD_TYPE` | `Release` | -O3 optimizations |
| `CMAKE_CXX_STANDARD` | `17` | Required for structured bindings, constexpr if |
| `CMAKE_CUDA_STANDARD` | `17` | Match C++ standard |

### Compiler Flags

```
CXX:  -O3 -march=armv8.2-a+fp16 -ffast-math -Wno-format-truncation -Wno-unused-result
CUDA: -O3 --use_fast_math --ptxas-options=-v --diag-suppress=177
```

- `-march=armv8.2-a+fp16` — enables FP16 NEON intrinsics on ARM
- `--use_fast_math` — fast but less precise GPU math (acceptable for inference)
- `--ptxas-options=-v` — shows register/shared memory usage per kernel
- `-Wno-format-truncation` — suppresses snprintf truncation warnings
- `-Wno-unused-result` — suppresses fscanf return value warnings

### Dependencies

Found automatically via CMake:
- `CUDAToolkit` — provides `CUDA::cudart`, `CUDA::cublas`, include paths
- `Threads` — pthreads

No external libraries required. All HTTP, JSON, and GGUF parsing is built-in.

## Library Architecture

```
libjetson_llm_core.a (static library)
  ├── src/memory/    (budget, kv_cache, pool)     ← .cpp → g++
  ├── src/jetson/    (power, thermal, sysinfo)    ← .cpp → g++
  ├── src/kernels/   (6 CUDA kernels)             ← .cu  → nvcc
  └── src/engine/    (model, decode, sample, tok)  ← .cpp/.cu → g++/nvcc

jetson-llm          → links libjetson_llm_core.a
jetson-llm-server   → links libjetson_llm_core.a + http_server.cpp
test_*              → links libjetson_llm_core.a
```

## File Types

| Extension | Compiler | Why |
|-----------|----------|-----|
| `.cpp` | g++ | No CUDA kernels, no `__global__`, no `half` arithmetic |
| `.cu` | nvcc | Contains `__global__` kernels, uses `__half2float`, CUDA math |

`decode.cu` is `.cu` because it defines `vec_add_kernel` and `fp16_to_fp32_kernel`.
All other engine files are `.cpp` (they call kernel functions but don't define them).
CUDA headers (`cuda_runtime.h`, `cuda_fp16.h`) are visible to `.cpp` via `CUDAToolkit_INCLUDE_DIRS`.

## Clean Rebuild

```bash
rm -rf build
cmake -B build -DCMAKE_CUDA_ARCHITECTURES="87" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `fatal error: cuda_runtime.h` | Install CUDA toolkit: `sudo apt install cuda-toolkit-12-6` |
| `aarch64 ONLY` error | Build on Jetson, not x86 |
| `no kernel image for sm_87` | Ensure `-DCMAKE_CUDA_ARCHITECTURES="87"` |
| `nvcc not found` | Add to PATH: `export PATH=/usr/local/cuda/bin:$PATH` |
| Linker errors | Clean build: `rm -rf build` and reconfigure |
