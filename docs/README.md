# jetson-llm Documentation

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System architecture, data flow, design decisions |
| [memory.md](memory.md) | Memory-first design: budget, KV cache, scratch pool, OOM guard |
| [kernels.md](kernels.md) | All 6 CUDA kernels: what they do, Orin tuning, performance notes |
| [gguf.md](gguf.md) | GGUF format parsing: config, tensors, tokenizer, weight mapping |
| [engine.md](engine.md) | Transformer forward pass, decode loop, CUDA graphs, sampling |
| [jetson-hal.md](jetson-hal.md) | Power modes, thermal management, sysfs interface, live stats |
| [server.md](server.md) | HTTP API: endpoints, request/response format, deployment |
| [build.md](build.md) | Build system, dependencies, cross-compilation notes |
| [testing.md](testing.md) | Test plan, test descriptions, expected results, debugging |
| [performance.md](performance.md) | Benchmarking, profiling, optimization targets |
