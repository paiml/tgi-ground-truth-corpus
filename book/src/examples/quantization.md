# Quantization Example

Demonstrates TGI's quantization types and compression characteristics.

## Run

```bash
cargo run --example quantization
```

## Output

```
=== Quantization Types Demo ===

GGUF Quantization Formats:
============================================================
Type          Bits/Weight     Compression Ratio   Memory (7B)
------------------------------------------------------------
Q4_0                  4.5                 7.1x       3.9 GB
Q4_K                  4.5                 7.1x       3.9 GB
Q5_K                  5.5                 5.8x       4.8 GB
Q6_K                  6.5                 4.9x       5.7 GB
Q8_0                  8.0                 4.0x       7.0 GB
F16                  16.0                 2.0x      14.0 GB
F32                  32.0                 1.0x      28.0 GB
============================================================

Memory Savings for 7B Model:
----------------------------------------
  F32 baseline: 28.0 GB
  Q4_K: 3.9 GB (86% smaller)
  Q8_0: 7.0 GB (75% smaller)
  F16: 14.0 GB (50% smaller)

Recommendations:
----------------------------------------
  • Q4_K: Best for consumer GPUs (RTX 3080/4080)
  • Q5_K: Balanced quality/size for prosumer GPUs
  • Q8_0: High quality, needs more VRAM
  • F16: Full precision, production servers

=== Demo Complete ===
```

## Code Walkthrough

### 1. Query Quantization Properties

```rust
use tgi_gtc::quantization::QuantType;

let qtype = QuantType::Q4_K;

// Bits per weight
let bits = qtype.bits_per_weight(); // 4.5

// Compression vs F32
let ratio = qtype.compression_ratio(); // ~7.1x
```

### 2. Calculate Memory

```rust
fn memory_gb(params_b: f64, qtype: QuantType) -> f64 {
    let bits = qtype.bits_per_weight() as f64;
    (params_b * 1e9 * bits / 8.0) / 1e9
}

// 7B model with Q4_K
memory_gb(7.0, QuantType::Q4_K); // ~3.9 GB
```

### 3. Compare Types

```rust
let types = [QuantType::Q4_K, QuantType::Q8_0, QuantType::F16];

for qtype in types {
    println!("{:?}: {:.1}x compression",
        qtype, qtype.compression_ratio());
}
```
