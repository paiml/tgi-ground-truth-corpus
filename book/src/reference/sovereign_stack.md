# Sovereign AI Stack

The TGI Ground Truth Corpus uses only Sovereign AI Stack crates - no external ML framework dependencies.

## Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      batuta (Orchestration)                 │
├─────────────────────────────────────────────────────────────┤
│  whisper.apr (ASR)  │  realizar (Inference)  │ pacha (Reg)  │
├─────────────────────┴────────────────────────┴──────────────┤
│   aprender (ML)   │  entrenar (Training)  │ jugar (Games)   │
├───────────────────┴───────────────────────┴─────────────────┤
│                 repartir (Distributed Compute)              │
├─────────────────────────────────────────────────────────────┤
│               trueno (SIMD/GPU Compute Primitives)          │
└─────────────────────────────────────────────────────────────┘
```

## Pattern Mapping

| TGI Pattern | Sovereign Stack Equivalent |
|-------------|---------------------------|
| Router | `realizar::serve` |
| Batching | `realizar::batch` |
| Streaming | `realizar::stream` |
| Validation | `realizar::validate` |
| Scheduling | `realizar::schedule` |
| Inference | `realizar::inference` |
| Quantization | `aprender::quantize` |

## Key Crates

### trueno

SIMD/GPU compute primitives.

- AVX2/AVX-512/NEON SIMD
- wgpu GPU compute
- LZ4/ZSTD compression

### aprender

ML algorithms and model formats.

- APR v2 model format
- Tensor operations
- Quantization kernels

### realizar

Inference engine.

- Model loading
- Batch inference
- Streaming generation

## Why Sovereign Stack?

1. **Pure Rust** - No Python, no C++ bindings
2. **Portable** - Runs everywhere Rust compiles
3. **Auditable** - Single language, clear ownership
4. **Performant** - Zero-copy, SIMD-accelerated
5. **Privacy** - No external dependencies phoning home

## Learn More

- [trueno on crates.io](https://crates.io/crates/trueno)
- [aprender on crates.io](https://crates.io/crates/aprender)
- [realizar on crates.io](https://crates.io/crates/realizar)
