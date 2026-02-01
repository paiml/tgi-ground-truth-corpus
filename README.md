<div align="center">

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ████████╗ ██████╗ ██╗     ██████╗ ██████╗  ██████╗ ██╗   ██╗███╗   ██╗██████╗  ║
║   ╚══██╔══╝██╔════╝ ██║    ██╔════╝ ██╔══██╗██╔═══██╗██║   ██║████╗  ██║██╔══██╗ ║
║      ██║   ██║  ███╗██║    ██║  ███╗██████╔╝██║   ██║██║   ██║██╔██╗ ██║██║  ██║ ║
║      ██║   ██║   ██║██║    ██║   ██║██╔══██╗██║   ██║██║   ██║██║╚██╗██║██║  ██║ ║
║      ██║   ╚██████╔╝██║    ╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝ ║
║      ╚═╝    ╚═════╝ ╚═╝     ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝  ║
║                                                                              ║
║              ████████╗██████╗ ██╗   ██╗████████╗██╗  ██╗                      ║
║              ╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝██║  ██║                      ║
║                 ██║   ██████╔╝██║   ██║   ██║   ███████║                      ║
║                 ██║   ██╔══██╗██║   ██║   ██║   ██╔══██║                      ║
║                 ██║   ██║  ██║╚██████╔╝   ██║   ██║  ██║                      ║
║                 ╚═╝   ╚═╝  ╚═╝ ╚═════╝    ╚═╝   ╚═╝  ╚═╝                      ║
║                                                                              ║
║          Production Inference Patterns for the Sovereign AI Stack           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

# TGI Ground Truth Corpus

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.83+-orange.svg)](https://www.rust-lang.org)
[![Coverage](https://img.shields.io/badge/coverage-95%25-green.svg)](https://codecov.io)
[![Tests](https://img.shields.io/badge/tests-221%20passed-success.svg)]()

</div>

A curated collection of Rust patterns extracted from [HuggingFace Text Generation Inference](https://github.com/huggingface/text-generation-inference), adapted for the **Sovereign AI Stack**.

## Overview

This corpus provides production-ready inference serving patterns that can be directly used with the Sovereign AI Stack components:

- **trueno** - SIMD/GPU compute primitives
- **aprender** - ML algorithms and model formats
- **realizar** - Inference engine

## Why This Corpus?

TGI is the battle-tested inference server behind HuggingFace's production infrastructure. This corpus:

1. **Extracts proven patterns** from TGI's Rust codebase
2. **Documents the mapping** to Sovereign AI Stack equivalents
3. **Provides test coverage** with property-based testing
4. **Enables RAG queries** via batuta oracle

## Module Structure

```
src/
├── attention.rs     # Scaled dot-product, Flash Attention, RoPE, GQA/MQA
├── batching.rs      # Continuous batching, request queuing
├── inference.rs     # Backend inference engine patterns
├── kv_cache.rs      # Block-based KV cache (PagedAttention), CoW forking
├── profiling.rs     # Performance profiling utilities
├── quantization.rs  # GGUF, AWQ, GPTQ compression
├── router.rs        # HTTP routing, health checks, metrics
├── sampling.rs      # Temperature, top-k, top-p, penalties
├── scheduling.rs    # Priority scheduling, queue management
├── streaming.rs     # SSE streaming responses
├── tokenizer.rs     # BPE encoding/decoding, special tokens
└── validation.rs    # Request validation, token counting
```

## Pattern Mapping

| TGI Component | File | Sovereign AI Stack |
|---------------|------|-------------------|
| HTTP Router | `router/src/server.rs` | `realizar::serve` |
| Continuous Batching | `backends/v3/src/queue.rs` | `realizar::batch` |
| Block Allocator | `backends/v3/src/block_allocator.rs` | `realizar::memory` |
| KV Cache | `backends/v3/src/block_allocator.rs` | `realizar::cache` |
| Attention | `server/text_generation_server/models/` | `trueno::attention` |
| Flash Attention | `server/text_generation_server/layers/` | `trueno::flash` |
| SSE Streaming | `router/src/server.rs` | `realizar::stream` |
| Request Validation | `router/src/validation.rs` | Custom |
| Token Sampling | `server/text_generation_server/utils/` | `realizar::sample` |
| Tokenization | HuggingFace tokenizers | `aprender::tokenize` |
| GGUF Loading | `backends/llamacpp/` | `aprender::format::gguf` |
| Quantization | `backends/llamacpp/src/quantize.rs` | `aprender::quantize` |

## Quick Start

```rust
use tgi_gtc::router::{Router, RouterConfig};
use tgi_gtc::batching::ContinuousBatcher;
use tgi_gtc::streaming::SseStream;

// Create router with TGI-style configuration
let config = RouterConfig::builder()
    .max_batch_size(32)
    .max_concurrent_requests(128)
    .build();

let router = Router::new(config);

// Continuous batching (TGI's core innovation)
let batcher = ContinuousBatcher::new(32);
batcher.add_request(request).await?;

// SSE streaming response
let stream = SseStream::new(response_rx);
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
tgi-ground-truth-corpus = "0.1"
```

Or use specific modules:

```toml
[dependencies]
tgi-ground-truth-corpus = { version = "0.1", features = ["router", "batching"] }
```

## Examples

Run the examples to see patterns in action with performance profiling:

```bash
# Router with health checks and request limiting
cargo run --example basic_router --release

# Continuous batching algorithm
cargo run --example continuous_batching --release

# SSE token streaming
cargo run --example streaming_sse --release

# Priority scheduler
cargo run --example scheduler --release

# Request validation
cargo run --example request_validation --release

# SafeTensors inference (requires inference feature)
cargo run --example safetensors_inference --features inference --release
```

Each example includes performance profiling with assertions:

```
=== TGI Router Pattern Demo ===
Router Creation: 160ns
--- Performance Validation ---
  ✓ Router creation: 160ns (< 100 µs)
  ✓ Health check: 6.68 ns/op (< 1000 ns)
  ✓ Request cycle: 6.75 ns/op (< 10 µs)
  Throughput: 148056024 req/sec
=== Demo Complete - All Performance Targets Met ===
```

## Development

```bash
# Build
cargo build

# Test with coverage
cargo llvm-cov --html

# Benchmarks (Criterion)
cargo bench --bench batching_bench
cargo bench --bench attention_bench
cargo bench --bench sampling_bench

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt
```

## Quality Standards

- **95%+ test coverage** - Enforced via CI
- **80%+ mutation score** - Via cargo-mutants
- **Zero clippy warnings** - Pedantic + nursery lints
- **Property-based tests** - All pure functions

## RAG Integration

This corpus is indexed by batuta's RAG oracle:

```bash
# Index the corpus
batuta oracle --rag-index

# Query patterns
batuta oracle --rag "continuous batching implementation"
batuta oracle --rag "SSE streaming TGI"
batuta oracle --rag "block allocation KV cache"
```

## License

Apache 2.0 - Same as TGI source.

## Acknowledgments

- [HuggingFace TGI Team](https://github.com/huggingface/text-generation-inference) - Original implementation
- [Sovereign AI Stack](https://github.com/paiml) - Target runtime
