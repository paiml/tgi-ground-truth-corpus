# TGI Ground Truth Corpus - Development Guidelines

## Project Overview

This is the **TGI Ground Truth Corpus** (TGI-GTC), a curated collection of Rust patterns extracted from HuggingFace's Text Generation Inference server, adapted for the Sovereign AI Stack.

**Purpose**: Provide production-ready inference serving patterns that can be directly used with `realizar` (Sovereign AI Stack inference engine).

## Critical Rules

### Dependency Policy
- **ONLY Sovereign AI Stack crates** - trueno, aprender, realizar
- **NO Python dependencies** - Pure Rust implementation
- **NO external ML frameworks** - No PyTorch, TensorFlow bindings

### Quality Standards
- **95% minimum test coverage** - Enforced via cargo-llvm-cov
- **Zero clippy warnings** - `cargo clippy -- -D warnings`
- **80%+ mutation score** - Via cargo-mutants
- **Property-based testing** via proptest for all pure functions
- **Benchmarks required** for performance-critical code

### TDD Workflow
1. Write failing test first (RED)
2. Implement minimum code to pass (GREEN)
3. Refactor while maintaining green (REFACTOR)
4. Add proptest properties for edge cases

### Documentation Requirements
Every public function MUST have:
1. **Doc comment** with summary and examples
2. **# Errors** section documenting failure modes
3. **# Panics** section if any panic conditions exist
4. **# Examples** with runnable doctests

### Commit Format
```
feat|fix|docs|refactor|test: message (Refs TGI-XXXX)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Commands

```bash
cargo build                    # Build library
cargo test                     # Run all tests
cargo test --lib               # Unit tests only
cargo bench                    # Run benchmarks
cargo clippy -- -D warnings    # Lint check
cargo fmt                      # Format code
cargo llvm-cov --html          # Coverage report
cargo mutants --file src/      # Mutation testing
```

## Module Structure

### Pattern Categories

| Module | TGI Source | Sovereign AI Stack Target |
|--------|------------|---------------------------|
| `router` | `router/src/server.rs` | `realizar::serve` |
| `inference` | `backends/v3/src/backend.rs` | `realizar::inference` |
| `batching` | `backends/v3/src/queue.rs` | `realizar::batch` |
| `quantization` | `backends/llamacpp/src/quantize.rs` | `aprender::quantize` |
| `streaming` | `router/src/server.rs` (SSE) | `realizar::stream` |
| `validation` | `router/src/validation.rs` | Custom validation |
| `scheduling` | `backends/v3/src/block_allocator.rs` | `realizar::schedule` |

### File Template

```rust
//! Module summary.
//!
//! Detailed description of the pattern and its TGI origin.
//!
//! # TGI Source
//!
//! This pattern is derived from `text-generation-inference/router/src/server.rs`.
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::serve::HttpServer`.
//!
//! # Examples
//!
//! ```rust
//! use tgi_gtc::router::Router;
//!
//! let router = Router::new();
//! ```

use thiserror::Error;

/// Error type for this module.
#[derive(Debug, Error)]
pub enum Error {
    #[error("validation failed: {0}")]
    Validation(String),
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Main struct documentation.
///
/// # Examples
///
/// ```rust
/// # use tgi_gtc::router::Router;
/// let router = Router::builder()
///     .max_batch_size(32)
///     .build();
/// ```
pub struct Router {
    // fields
}

impl Router {
    /// Creates a new router with default settings.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use tgi_gtc::router::Router;
    /// let router = Router::new();
    /// assert!(router.is_ready());
    /// ```
    pub fn new() -> Self {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_router_creation() {
        let router = Router::new();
        assert!(router.is_ready());
    }

    proptest! {
        #[test]
        fn prop_router_handles_any_batch_size(batch_size in 1usize..1000) {
            let router = Router::builder()
                .max_batch_size(batch_size)
                .build();
            prop_assert!(router.max_batch_size() == batch_size);
        }
    }
}
```

## Pattern Mapping Guide

### Router Patterns (TGI → Sovereign AI Stack)

| TGI Pattern | Location | Sovereign Equivalent |
|-------------|----------|---------------------|
| Axum routing | `router/src/server.rs` | `realizar::serve::routes` |
| Health checks | `router/src/server.rs:health` | `realizar::serve::health` |
| Metrics | `router/src/usage_stats.rs` | `renacer::metrics` |
| SSE streaming | `router/src/server.rs:generate_stream` | `realizar::stream::sse` |

### Inference Patterns

| TGI Pattern | Location | Sovereign Equivalent |
|-------------|----------|---------------------|
| Continuous batching | `backends/v3/src/queue.rs` | `realizar::batch::continuous` |
| Block allocation | `backends/v3/src/block_allocator.rs` | `realizar::memory::blocks` |
| Radix attention | `backends/v3/src/radix.rs` | `trueno::attention::radix` |
| gRPC client | `backends/v3/src/client/` | `realizar::client::grpc` |

### Quantization Patterns

| TGI Pattern | Location | Sovereign Equivalent |
|-------------|----------|---------------------|
| GGUF loading | `backends/llamacpp/src/llamacpp.rs` | `aprender::format::gguf` |
| Quantize ops | `backends/llamacpp/src/quantize.rs` | `aprender::quantize` |

## Stack Documentation Search

Query this corpus and the entire Sovereign AI Stack using batuta's RAG Oracle:

```bash
# Index all stack documentation
batuta oracle --rag-index

# Search for inference patterns
batuta oracle --rag "continuous batching TGI"
batuta oracle --rag "SSE streaming inference"
batuta oracle --rag "block allocation KV cache"

# Check index status
batuta oracle --rag-stats
```

## Cross-Reference

- **TGI Source**: `~/src/text-generation-inference/`
- **Sovereign AI Stack**: `~/src/realizar/`, `~/src/aprender/`, `~/src/trueno/`
- **HF Corpus**: `~/src/hf-ground-truth-corpus/` (Python patterns)
