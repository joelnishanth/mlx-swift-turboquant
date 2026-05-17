# mlx-swift-turboquant

TurboQuant KV cache compression for [MLX Swift](https://github.com/ml-explore/mlx-swift) — bringing 3-bit key and 3-bit value compression to on-device LLM inference on Apple Silicon.

Based on the TurboQuant algorithm from [Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) ([Zandieh et al., ICLR 2026, arXiv 2504.19874](https://arxiv.org/abs/2504.19874)), this repository provides:

- A **benchmark CLI** (`TurboQuantBench`) that measures fp16 vs TurboQuant KV cache performance on identical workloads
- **Integration guide** for adding TurboQuant to MLX Swift LLM applications
- **Architecture documentation** explaining the compression pipeline

## What is TurboQuant?

TurboQuant compresses the KV (Key-Value) cache used during LLM autoregressive generation. During inference, the KV cache grows linearly with context length and dominates memory usage for long sequences. TurboQuant applies:

1. **PolarQuant** (3-bit) — Lloyd-Max codebook quantization with Walsh-Hadamard Transform (WHT) rotation for uniform magnitude distribution
2. **QJL (Quantized Johnson-Lindenstrauss)** — 1-bit residual correction for keys using random projection, recovering information lost in quantization

This achieves ~3.1-bit keys and 3-bit values while maintaining output quality, with a "hot window" design that keeps the most recent tokens in full precision.

### Hot-Window Architecture

```
┌─────────────────────────────────────────────────────┐
│              KV Cache (growing left to right)         │
│                                                       │
│  ┌──────────────────────┐  ┌──────────────────────┐  │
│  │   Compressed (3-bit) │  │   Hot Window (fp16)   │  │
│  │   PolarQuant + QJL   │  │   Last 256 tokens     │  │
│  │   ~5× smaller        │  │   Full precision      │  │
│  └──────────────────────┘  └──────────────────────┘  │
│  ← oldest tokens            newest tokens →           │
└─────────────────────────────────────────────────────┘
```

- Tokens in the **hot window** (configurable, default 256) remain in fp16 for maximum attention accuracy
- Tokens evicted from the hot window are compressed via PolarQuant + QJL
- During attention, compressed tokens are decoded and prepended to the hot window
- Compression activates only after `turboMinActivationTokens` (default 2048) to avoid overhead on short contexts

## Expected Performance Characteristics

Based on the [TurboQuant paper](https://arxiv.org/abs/2504.19874) (Table 1, Zandieh et al., ICLR 2026) and our MLX Swift implementation:

| Metric | fp16 Baseline | TurboQuant (3-bit) | Delta |
|--------|:------------:|:------------------:|:-----:|
| **KV Cache Size** (8K context) | 100% | ~19% | **-81%** |
| **Peak Memory** (8K context) | baseline | -40 to -60% KV portion | significant |
| **Throughput** (tok/s) | baseline | ~0.95-1.0× | minimal overhead |
| **Quality** (perplexity) | baseline | +0.1-0.3 ppl | negligible |
| **TTFT** | baseline | ~1.0× | unchanged |

> **Note:** Actual numbers vary by model architecture, hardware, and context length. Run `TurboQuantBench` on your hardware for precise measurements.

### When TurboQuant Helps Most

- **Long contexts** (4K+ tokens): Maximum KV cache savings
- **Memory-constrained devices** (8-16 GB): Enables larger contexts that wouldn't fit in fp16
- **Batch inference**: Memory savings multiply across batch dimension

### When TurboQuant Has Minimal Impact

- **Short contexts** (<2K tokens): Hot window covers the entire context, no compression occurs
- **Already-quantized models** (4-bit weights): Memory dominated by weight quantization overhead

## Benchmark Tool

### Building

The benchmark requires Xcode for Metal shader compilation:

```bash
# Open in Xcode (recommended)
open Package.swift

# Or build via xcodebuild
xcodebuild -scheme TurboQuantBench -configuration Release
```

> **Note:** `swift build` / `swift run` will compile but fail at runtime because SwiftPM doesn't compile Metal shaders. Use Xcode or `xcodebuild` instead.

### Running Benchmarks

```bash
# Default: Gemma 4 E2B, 256 max tokens, 3 measured runs
.build/release/TurboQuantBench

# Custom configuration
.build/release/TurboQuantBench \
    --model mlx-community/gemma-4-e4b-it-4bit \
    --max-tokens 512 \
    --warmup 2 \
    --runs 5 \
    --output results.json

# fp16 only (baseline)
.build/release/TurboQuantBench --fp16-only

# TurboQuant only
.build/release/TurboQuantBench --turbo-only
```

### Output

The tool produces:
1. **Console summary table** with averaged metrics per configuration
2. **JSON file** (`benchmark_results.json`) with per-run detailed results

### Benchmark Workloads

| Workload | Prompt Size | Purpose |
|----------|:-----------:|---------|
| `short-256` | ~256 tokens | Verify no overhead below activation threshold |
| `medium-2k` | ~2048 tokens | Measure compression activation point |
| `long-8k` | ~8192 tokens | Full compression, maximum memory savings |

### Metrics Collected

- **tok/s** — Generation throughput (tokens per second)
- **TTFT** — Time to first token (ms)
- **Total generation time** — End-to-end latency (ms)
- **Peak memory** — Resident set size (MB)
- **Output snippet** — First 80 chars for quality spot-check

## Integration

TurboQuant is available as an opt-in feature through forked MLX Swift dependencies:

### Dependencies

```swift
// Package.swift
.package(url: "https://github.com/joelnishanth/mlx-swift-lm", branch: "feature/turboquant"),
```

This transitively pulls `joelnishanth/mlx-swift` (feature/turboquant) which contains the C++ TurboQuant primitives.

### Enabling TurboQuant

```swift
import MLXLMCommon

let session = ChatSession(modelContainer)
session.turboQuantEnabled = true  // Opt-in to KV cache compression

for try await chunk in session.streamResponse(to: "Hello") {
    print(chunk, terminator: "")
}
```

When `turboQuantEnabled` is `true`:
- Each `KVCacheSimple` layer gets TurboQuant compression enabled
- Compression activates after 2048 tokens (configurable via `turboMinActivationTokens`)
- A 256-token hot window (configurable via `turboHotWindowSize`) stays in fp16
- Older tokens are compressed to ~3 bits using PolarQuant + QJL

### What Changes

| Component | fp16 (default) | TurboQuant |
|-----------|:-------------:|:----------:|
| `KVCacheSimple.turboQuantEnabled` | `false` | `true` |
| Cache storage | fp16 arrays | fp16 hot window + uint8 compressed |
| `attentionWithCacheUpdate()` | Direct SDPA | Decode compressed → concat → SDPA |
| Memory at 8K tokens | ~O(n) fp16 | ~O(256) fp16 + O(n) 3-bit |

## Architecture

### Repository Structure

```
mlx-swift-turboquant/
├── Package.swift            # Swift Package with MLX dependencies
├── README.md                # This file
├── Sources/
│   └── TurboQuantBench/
│       └── main.swift       # Benchmark CLI tool
└── docs/
    └── architecture.md      # Detailed compression pipeline docs
```

### Compression Pipeline

```
Input tokens → KV Cache (fp16)
                    │
                    ├─ offset < 2048 → standard fp16 path
                    │
                    └─ offset ≥ 2048 → evict cold tokens from hot window
                                          │
                                          ├─ Keys:   WHT rotate → Lloyd-Max 3-bit → QJL 1-bit residual
                                          │
                                          └─ Values: WHT rotate → Lloyd-Max 3-bit
                                          │
                                          └─ Store as uint8 packed arrays
                                          │
              During attention:           │
              Decode uint8 → fp16 → concat with hot window → SDPA
```

### Key Files (in forked repos)

| File | Repository | Purpose |
|------|:----------:|---------|
| `turbo_quant.h` | joelnishanth/mlx-swift | C++ PolarQuant + QJL algorithms |
| `turbo_quant_ops.cpp` | joelnishanth/mlx-swift | Encode/decode implementations |
| `turbo_quant_bridge.cpp` | joelnishanth/mlx-swift | C bridge for Swift FFI |
| `MLXFast.swift` | joelnishanth/mlx-swift | Swift bindings |
| `KVCache.swift` | joelnishanth/mlx-swift-lm | Hot-window eviction logic |
| `AttentionUtils.swift` | joelnishanth/mlx-swift-lm | Decode path in attention |

## Attribution

This implementation is inspired by **TurboQuant**, a vector quantization algorithm developed at **Google Research** and **Google DeepMind**. TurboQuant was presented at **ICLR 2026**.

### Research Papers

- **TurboQuant**: Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. *"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*, ICLR 2026. [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)
- **PolarQuant**: *"PolarQuant: Polar Coordinate Quantization for Vector Compression"*, AISTATS 2026. [arXiv 2502.02617](https://arxiv.org/abs/2502.02617)
- **QJL**: *"Quantized Johnson-Lindenstrauss Transform"*, AAAI 2025. [ACM DL](https://dl.acm.org/doi/10.1609/aaai.v39i24.34773)

### Google Research Blog

- [TurboQuant: Redefining AI efficiency with extreme compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — March 2026

### Acknowledgements

TurboQuant was developed by Amir Zandieh (Google Research), Majid Daliri (NYU), Majid Hadian (Google DeepMind), and Vahab Mirrokni (Google Research), with contributions from Praneeth Kacham, Insu Han, Lars Gottesbüren, and Rajesh Jayaram. This Swift implementation adapts their algorithms for Apple Silicon via the MLX framework.

## References

- [ml-explore/mlx-swift](https://github.com/ml-explore/mlx-swift) — Apple MLX framework for Swift
- [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — MLX Swift LLM library

## License

Apache 2.0, matching the upstream MLX Swift license.
