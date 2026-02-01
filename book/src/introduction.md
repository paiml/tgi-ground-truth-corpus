# TGI Ground Truth Corpus

Production-ready inference serving patterns extracted from HuggingFace's Text Generation Inference (TGI), implemented using the **Sovereign AI Stack**.

## What is This?

This corpus provides battle-tested patterns for building high-performance LLM inference servers. Each pattern is:

- **Extracted from TGI** - HuggingFace's production inference server
- **Pure Rust** - Using only Sovereign AI Stack crates
- **Thoroughly tested** - 98%+ test coverage with property-based testing
- **Well documented** - With TGI source references

## Why TGI Patterns?

TGI powers inference for:
- HuggingFace Inference API
- AWS SageMaker
- Google Cloud Vertex AI
- Thousands of production deployments

The patterns in TGI represent years of production experience with:
- Continuous batching for GPU utilization
- Memory-efficient KV cache management
- Low-latency streaming responses
- Robust request validation

## Sovereign AI Stack

All patterns are implemented using only these crates:

| Crate | Purpose |
|-------|---------|
| `trueno` | SIMD/GPU compute primitives |
| `aprender` | ML algorithms, model formats |
| `realizar` | Inference engine |

No external ML framework dependencies. Pure Rust all the way down.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
tgi-gtc = "0.1"
```

Run an example:

```bash
cargo run --example basic_router
cargo run --example continuous_batching
cargo run --example streaming_sse
```

## Coverage

- **124 tests** (unit + property-based)
- **98.81% line coverage**
- **All files 95%+**

## License

Apache 2.0 - Same as TGI
