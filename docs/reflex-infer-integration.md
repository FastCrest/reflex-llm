# reflex-llm and reflex-infer split

## Decision

Use two repositories:

- `reflex-llm`: LLM runtime, model loading, tokenizer, sampling, server, memory
  policy, persistence, and app integration.
- `reflex-infer`: Jetson-optimized CUDA kernel library.

`reflex-llm` should consume `reflex-infer` as an external dependency through
CMake. During early development, a git submodule or sibling checkout is fine.
The long-term shape should be a versioned CMake package.

## Why Not Keep One Repo

The runtime and kernel library have different rates of change.

`reflex-llm` needs stable application behavior:

- GGUF parsing.
- prompt/chat template handling;
- sampling;
- HTTP/CLI compatibility;
- memory budgeting;
- persistence;
- model-specific correctness.

`reflex-infer` needs aggressive kernel iteration:

- hardware-specialized kernels;
- microbenchmarks;
- Nsight Compute feedback;
- device capability dispatch;
- quantization-specific layout work;
- future Orin NX, AGX Orin, and Thor support.

Splitting them lets kernel experiments move quickly without destabilizing the
runtime surface.

## Dependency Direction

Only one direction is allowed:

```text
reflex-llm  --->  reflex-infer
```

`reflex-infer` must not depend on `reflex-llm`.

The kernel library may define small structs for tensor metadata, quantization
layout, hardware profile, and workspace requirements, but it should not know
about HTTP requests, GGUF files, tokenizer state, chat templates, or
conversation persistence.

## Recommended Integration Mode

Phase 1:

- Keep current legacy kernels in `reflex-llm/src/kernels`.
- Add standalone `reflex-infer` microbenchmarks and API prototypes.
- Compare kernel outputs against the runtime's current path.

Phase 2:

- Add `REFLEX_LLM_USE_REFLEX_INFER=ON`.
- Support sibling checkout:

```text
/home/speedy/learning/reflex-llm
/home/speedy/learning/reflex-infer
```

- CMake should locate `../reflex-infer` first, then fall back to an installed
  `reflex-infer` package.
- `reflex-infer` currently exports a minimal `reflex::infer` interface target
  and `include/reflex/infer.h` capability API. Real fast-path kernels are still
  disabled until they move out of `reflex-llm` and pass parity tests.

Phase 3:

- Move Q4 GEMV/MMQ, decode attention, RoPE, norm, and KV conversion behind a
  `reflex-infer` dispatch layer.
- Keep the old in-repo kernels as fallback until the external path is stable.

Phase 4:

- Remove or archive legacy in-repo kernels after correctness and performance
  parity is proven.

## Public Kernel API Shape

The API should be explicit about hardware, model shape, and quantization.

Suggested concepts:

```cpp
namespace reflex::infer {

enum class DeviceClass {
    OrinNano8GB,
    OrinNX16GB,
    AGXOrin64GB,
    Thor,
};

enum class ModelFamily {
    Qwen,
    Phi,
    Llama,
    Gemma,
    Unknown,
};

enum class QuantFormat {
    Q4_0,
    Q4_K_M,
    AWQ4,
    GPTQ4,
    NF4,
    FP16,
};

struct HardwareProfile {
    DeviceClass device;
    int sm;
    int sm_count;
    size_t shared_mem_per_block;
    size_t total_memory_bytes;
    float nominal_dram_gbps;
};

struct ModelShape {
    ModelFamily family;
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    int max_context;
};

struct KernelSupport {
    bool q4_gemv;
    bool q4_mmq_prefill;
    bool decode_attention;
    bool rope;
    bool rmsnorm;
    bool kv_convert;
};

KernelSupport query_support(const HardwareProfile&, const ModelShape&, QuantFormat);

}  // namespace reflex::infer
```

Exact names can change, but the important design point is capability discovery.
`reflex-llm` should ask what is supported instead of assuming a kernel exists.

## Operator Boundary

First operators to extract:

1. Q4 dequant + GEMV.
2. Q4_K MMQ prefill.
3. Decode attention.
4. RoPE.
5. RMSNorm.
6. FP16 to INT8 KV conversion.

Do not extract GGUF parsing, tokenizer, sampling, or HTTP code into
`reflex-infer`.

## Hardware Abstraction

`reflex-infer` should be Jetson-only, but not Orin-Nano-only.

Use explicit profiles:

- `orin_nano_8gb_sm87`
- `orin_nx_16gb_sm87`
- `agx_orin_64gb_sm87`
- `thor_blackwell`

The library should compile only the kernels that match the selected CUDA
architectures. Runtime dispatch should still verify the probed GPU before
launching a profile-specific kernel.

## Model Abstraction

Start with Qwen GGUF `Q4_K_M`. Do not hard-code Qwen or one quantization
format forever.

The first reusable shape abstraction should cover:

- hidden size;
- intermediate size;
- layer count;
- attention heads;
- KV heads;
- head dimension;
- RoPE parameters;
- quantized weight block format;
- KV-cache element type and layout.

Phi-4 support later should be a new model profile plus any missing kernels, not
a rewrite of the runtime.

## Fallback Rule

Every `reflex-infer` fast path needs one of:

- a runtime fallback kernel in `reflex-llm`;
- a simple reference CUDA path;
- a CPU/PyTorch/offline test reference for validation.

No fast path should be the only correctness path until it has soak coverage.

## Versioning Rule

`reflex-llm` should pin a known-good `reflex-infer` commit or release.

Benchmarks and paper results must record:

- `reflex-llm` commit;
- `reflex-infer` commit;
- JetPack/CUDA version;
- model file hash;
- quantization format;
- power mode and clock settings.
