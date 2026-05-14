# TurboQuant Architecture

## Overview

TurboQuant compresses the KV (Key-Value) cache during autoregressive LLM inference. The KV cache stores previously computed key and value tensors for the attention mechanism, growing linearly with sequence length. At long contexts (8K+ tokens), it dominates GPU memory usage.

TurboQuant applies two complementary quantization techniques from Zandieh et al. (arXiv 2504.19874):

1. **PolarQuant** — 3-bit Lloyd-Max quantization with magnitude/sign decomposition
2. **QJL** — 1-bit Quantized Johnson-Lindenstrauss residual projection (keys only)

## Compression Pipeline

### Step 1: Walsh-Hadamard Transform (WHT) Rotation

Before quantization, each token vector is rotated using a normalized WHT matrix. This spreads concentrated activations across dimensions, creating a more uniform magnitude distribution that quantizes more efficiently.

```
x_rotated = WHT(x) / sqrt(D)    where D is head dimension
```

### Step 2: PolarQuant (3-bit)

PolarQuant separates each value into magnitude and sign:

```
x = |x| * sign(x)
```

- **Magnitude** is quantized using a 3-bit Lloyd-Max codebook (8 levels), which minimizes mean squared error for the expected chi-distribution of rotated vector magnitudes
- **Sign** is stored as 1 bit
- Total: 3 bits per value (2 bits magnitude index + 1 bit sign)

The codebook is pre-computed analytically for chi-distributed magnitudes (which is the distribution after WHT rotation of typically Gaussian-distributed hidden states).

### Step 3: QJL Residual Correction (Keys Only)

After PolarQuant, a residual error exists:

```
residual = x - PolarQuant_decode(PolarQuant_encode(x))
```

For keys (but not values), TurboQuant applies a 1-bit random projection:

```
qjl_bits = sign(R @ residual)    where R is a random Gaussian matrix
```

This adds ~1 bit per dimension but significantly improves key reconstruction accuracy, which matters more for attention score computation than value reconstruction.

### Step 4: Packing

The encoded data is packed into `uint8` arrays:

- **Keys**: 3 bits (PolarQuant) + 1 bit (QJL) = ~4 bits effective, packed as uint8
- **Values**: 3 bits (PolarQuant), packed as uint8

Storage per token dimension:
- fp16: 16 bits → 2 bytes
- TurboQuant keys: ~4 bits → 0.5 bytes (4× reduction)
- TurboQuant values: 3 bits → 0.375 bytes (5.3× reduction)

## Hot-Window Design

Not all tokens are compressed. TurboQuant maintains a "hot window" of the most recent tokens in full fp16 precision:

```
Total context: [compressed_tokens | hot_window_tokens]
                 (3-bit packed)     (fp16, last 256)
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `turboMinActivationTokens` | 2048 | Don't compress until this many tokens |
| `turboHotWindowSize` | 256 | Keep this many recent tokens in fp16 |
| `step` | 256 | Compression granularity (compress in chunks) |

### Eviction Logic (KVCacheSimple.update)

On each `update(keys:, values:)` call:

1. Append new tokens to the fp16 cache as normal
2. If `offset > turboMinActivationTokens`:
   - Calculate `coldEnd = offset - turboHotWindowSize`
   - Count new cold tokens: `newColdCount = coldEnd - compressedOffset`
   - If `newColdCount >= step`:
     - Extract cold slice from fp16 cache
     - Encode via `MLXFast.turboQuantEncode(keys:, values:, bits: 3)`
     - Append to compressed polar arrays
     - Rebuild fp16 cache containing only hot window + spare space

### Head Dimension Handling

- **128 or 256**: Direct encoding (standard for most LLMs)
- **512**: Split heads in half (H → H×2, D → 256), encode, merge back during decode
- **Other dimensions**: Falls back to fp16 with a warning

## Decode Path (AttentionUtils.attentionWithCacheUpdate)

During attention computation:

1. `cache.update(keys:, values:)` returns the fp16 hot window
2. If `compressedOffset > 0`:
   - Decode compressed keys: `MLXFast.turboDecodeK(packed: polarKeys)`
   - Decode compressed values: `MLXFast.turboDecodeV(packed: polarValues)`
   - Concatenate: `[decoded_history | hot_window]`
3. Apply mask slicing (compressed context changes the effective sequence length)
4. Run `MLXFast.scaledDotProductAttention` on the full (decoded + hot) context

## C++ Implementation

The core algorithms are implemented in C++ for performance:

### turbo_quant.h

Header-only implementation containing:

- `TurboQuantK` / `TurboQuantV` — Primitive structs for key/value quantization
- `turbo_quantize_k` / `turbo_quantize_v` — Encoding functions
- `turbo_dequantize_k` / `turbo_dequantize_v` — Decoding functions
- Lloyd-Max codebook constants (pre-computed for chi distribution)
- WHT rotation utilities
- QJL random projection

### turbo_quant_ops.cpp

CPU-side encode/decode operations registered in the `mlx::core::fast` namespace:

- `turbo_encode_k(keys)` — Full key encoding pipeline
- `turbo_encode_v(values)` — Full value encoding pipeline
- `turbo_decode_k(packed)` — Key decoding (PolarQuant + QJL)
- `turbo_decode_v(packed)` — Value decoding (PolarQuant only)

### turbo_quant_bridge.cpp

C bridge layer for Swift FFI:

- `mlx_fast_turbo_encode(...)` — Combined key+value encoding
- `mlx_fast_turbo_decode_k(...)` — Key decoding
- `mlx_fast_turbo_decode_v(...)` — Value decoding

## Swift Bindings

Located in `MLXFast.swift`:

```swift
public static func turboQuantEncode(
    keys: MLXArray, values: MLXArray, bits: Int = 3, stream: StreamOrDevice = .default
) -> ((MLXArray, MLXArray), (MLXArray, MLXArray))

public static func turboDecodeK(
    packed: MLXArray, stream: StreamOrDevice = .default
) -> MLXArray

public static func turboDecodeV(
    packed: MLXArray, stream: StreamOrDevice = .default
) -> MLXArray
```

## Quality Characteristics

From the TurboQuant paper (Table 1):

- **Perplexity increase**: +0.1-0.3 across model families (negligible)
- **Downstream task accuracy**: Within 1% of fp16 on most benchmarks
- **Key insight**: QJL residual correction on keys is critical for maintaining attention score accuracy; values tolerate more quantization noise
