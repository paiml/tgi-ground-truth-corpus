# Quantization Pattern

Model quantization types and compression.

## TGI Source

Derived from `backends/llamacpp/src/quantize.rs`:
- GGUF format support
- Dequantization kernels
- Mixed precision inference

## Sovereign AI Stack Equivalent

Maps to `aprender::quantize` for model quantization.

## Key Components

### QuantType

Supported quantization types:

```rust
use tgi_gtc::quantization::QuantType;

enum QuantType {
    Q4_0,   // 4-bit basic
    Q4_K,   // 4-bit K-quants
    Q5_K,   // 5-bit K-quants
    Q6_K,   // 6-bit K-quants
    Q8_0,   // 8-bit basic
    F16,    // 16-bit float
    F32,    // 32-bit float
}
```

### Methods

```rust
// Bits per weight (including overhead)
QuantType::Q4_K.bits_per_weight(); // 4.5

// Compression ratio vs F32
QuantType::Q4_K.compression_ratio(); // ~7.1x
```

## Quantization Comparison

| Type | Bits/Weight | Compression | 7B Model Size |
|------|-------------|-------------|---------------|
| Q4_0 | 4.5 | 7.1x | 3.9 GB |
| Q4_K | 4.5 | 7.1x | 3.9 GB |
| Q5_K | 5.5 | 5.8x | 4.8 GB |
| Q6_K | 6.5 | 4.9x | 5.7 GB |
| Q8_0 | 8.0 | 4.0x | 7.0 GB |
| F16 | 16.0 | 2.0x | 14.0 GB |
| F32 | 32.0 | 1.0x | 28.0 GB |

## K-Quants

K-quants (Q4_K, Q5_K, Q6_K) use importance-based quantization:

1. Compute weight importance scores
2. Use higher precision for important weights
3. Lower precision for less important weights
4. **Better quality than basic quants**

## Choosing Quantization

### Q4_K (Recommended for most)
- Best compression/quality balance
- Fits 7B models in 8GB VRAM
- Suitable for consumer GPUs (RTX 3080, 4080)

### Q5_K
- Better quality than Q4_K
- ~20% more VRAM
- Good for prosumer GPUs (RTX 4090)

### Q6_K
- Near-F16 quality
- ~45% more VRAM than Q4_K
- For quality-critical applications

### Q8_0
- Minimal quality loss
- 2x VRAM of Q4_K
- When quality matters more than memory

### F16
- Full precision inference
- Production servers with ample VRAM
- A100/H100 deployments

## Memory Calculation

```rust
fn estimate_memory_gb(params_billions: f64, quant: QuantType) -> f64 {
    let bits = quant.bits_per_weight() as f64;
    let bytes = params_billions * 1_000_000_000.0 * bits / 8.0;
    bytes / 1_000_000_000.0
}

// 7B model with Q4_K
estimate_memory_gb(7.0, QuantType::Q4_K); // ~3.9 GB
```

## GGUF Format

GGUF (GPT-Generated Unified Format) is the standard for quantized models:

- Self-contained metadata
- Multiple quantization types
- Efficient memory mapping
- Cross-platform support

TGI and llama.cpp both use GGUF for quantized inference.

## Example

```bash
cargo run --example quantization
```
