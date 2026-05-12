# Testing

## Test Suite Overview

| Test | File | Needs model? | Needs GPU? | Tests |
|------|------|-------------|-----------|-------|
| `test_memory` | `tests/test_memory.cpp` | No | Yes (cudaMallocHost) | Budget probe, OOM guard, scratch pool |
| `test_kernels` | `tests/test_kernels.cu` | No | Yes | Softmax, RoPE, RMSNorm, FP16→INT8, SwiGLU |
| `test_model_load` | `tests/test_model_load.cpp` | Yes | Yes | Config, tokenizer, weights, memory |
| `test_plan.sh` | `scripts/test_plan.sh` | Auto-downloads | Yes | All of the above + inference + server |

## Running Tests

```bash
# Unit tests (no model needed)
./build/test_memory
./build/test_kernels

# Model loading test (needs GGUF file)
./build/test_model_load models/tinyllama.gguf

# Full automated test plan (33 tests, 7 phases)
./scripts/test_plan.sh
```

## test_memory — Memory Subsystem

**3 tests, no model needed:**

### Test 1: probe_system_memory

- Reads `/proc/meminfo`
- Asserts `total_mb > 0` and `free_mb() > 0`
- Prints memory budget table

### Test 2: OOMGuard

- Creates guard with 256 MB safety margin
- Asserts `real_free_mb() > 0`
- Prints actual free memory

### Test 3: ScratchPool

- Allocates 64 MB pool via `cudaMallocHost`
- Gets two buffers (1024 and 2048 bytes)
- Verifies they're non-null and different
- Verifies `used()` returns correct total (3072)
- Verifies `reset()` returns `used()` to 0

## test_kernels — CUDA Kernel Correctness

**5 tests, needs GPU, no model:**

### Test 1: Softmax

- Input: 1024 values, linear ramp
- Expected: all values ≥ 0, sum = 1.0 (within 1e-4)

### Test 2: RoPE (position 0 identity)

- Input: all 1.0, position = 0
- Expected: output ≈ 1.0 (cos(0)=1, sin(0)=0 → identity)

### Test 3: Fused RMSNorm

- Input: x = 1.0, residual = 0.0, weight = 1.0
- Expected: output ≈ 1.0 (RMS of all-ones = 1.0, normalized = 1.0)

### Test 4: FP16 → INT8

- Input: row of 0.5 values
- Expected: scale ≈ 0.5/127 ≈ 0.00394

### Test 5: SwiGLU

- Input: gate = 1.0, up = 2.0
- Expected: silu(1.0) × 2.0 = 0.7311 × 2.0 ≈ 1.4621

## test_model_load — Full Loading Pipeline

**8 tests, needs GGUF file:**

### Test 1: System probe
- Verifies `probe_jetson()` returns valid info
- Prints L4T version, CUDA version, SM count, RAM

### Test 2: Memory budget
- Verifies budget reads real values from `/proc/meminfo`
- Asserts `total_mb > 0`, `free_mb() > 0`

### Test 3: GGUF config parsing
- Reads model architecture from GGUF metadata
- Asserts `n_layers > 0`, `n_heads > 0`, `hidden_dim > 0`, `vocab_size > 0`
- Prints all config values

### Test 4: Weight size estimate
- Calculates estimated weight bytes
- Checks if model fits in available memory

### Test 5: KV cache context calculation
- Computes max context for FP16 and INT8 KV cache
- Asserts INT8 context ≥ FP16 context

### Test 6: Tokenizer
- Loads vocabulary from GGUF
- Asserts vocab size > 0
- Test encodes "Hello" and prints token IDs
- Test decodes back to text

### Test 7: Weight loading and mapping
- `load_and_map_weights()` — mmap + cudaHostRegister + tensor mapping
- Checks `tok_embd`, `output_norm`, `output` are non-null
- Counts layers with QKV weights mapped

### Test 8: Power and thermal
- Reads power mode and GPU frequency
- Reads temperatures
- Checks backoff recommendation

## test_plan.sh — Automated Full Test

33 tests across 7 phases, automated. See `TESTING.md` for full description.

```
Phase 0: System (6 tests)   — Jetson? CUDA? GPU? RAM? Power? Temperature?
Phase 1: Build (3 tests)    — cmake? make? binaries?
Phase 2: Unit (3 tests)     — memory, kernels, budget values
Phase 3: Model (3 tests)    — download/verify GGUF
Phase 4: Loading (6 tests)  — config, tokenizer, weights, memory
Phase 5: Inference (6 tests) — generation, tok/s, OOM, CUDA errors, stability
Phase 6: Server (4 tests)   — health, models, chat completion
Phase 7: Thermal (2 tests)  — temperature, throttling
```

## Debugging Failures

### Garbage output (random tokens)

1. Check tensor offsets: `./build/test_model_load model.gguf` — look at Test 7
2. Inspect actual tensor names vs expected patterns (see `docs/gguf.md`)
3. Profile: `nsys profile ./build/jetson-llm -m model.gguf -p "Hi" -n 5`

### Segfault

1. Check null weight pointers: Test 7 output shows `(nil)` for unmapped tensors
2. Run with: `cuda-memcheck ./build/jetson-llm -m model.gguf -p "Hi" -n 5`

### OOM

1. Check memory budget: `./build/test_memory` — is `FREE` > model size + 1 GB?
2. Disable GUI: `sudo systemctl set-default multi-user.target`
3. Reduce CMA: add `cma=256M` to kernel command line

### Wrong token count

1. Compare tokenizer output with reference:
   ```python
   from transformers import AutoTokenizer
   t = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
   print(t.encode("Hello"))
   ```
2. Check BOS/EOS IDs match GGUF metadata
