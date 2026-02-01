//! TGI Ground Truth Corpus
//!
//! Production-ready inference serving patterns extracted from HuggingFace's
//! Text Generation Inference, implemented using the Sovereign AI Stack.
//!
//! # Overview
//!
//! This crate provides battle-tested patterns for:
//!
//! - **Router**: HTTP routing, health checks, OpenAI-compatible API
//! - **Batching**: Continuous batching for optimal GPU utilization
//! - **Streaming**: Server-Sent Events for token-by-token delivery
//! - **Validation**: Request validation and token counting
//! - **Scheduling**: Fair request scheduling and queue management
//! - **Quantization**: GGUF/AWQ/GPTQ format support
//!
//! # Sovereign AI Stack
//!
//! All patterns are implemented using only Sovereign AI Stack crates:
//!
//! | Crate | Purpose |
//! |-------|---------|
//! | `trueno` | SIMD/GPU compute primitives |
//! | `aprender` | ML algorithms, model formats |
//! | `realizar` | Inference engine |
//!
//! # TGI Pattern Origin
//!
//! Each module documents the original TGI source location and the
//! corresponding Sovereign AI Stack equivalent.
//!
//! # Examples
//!
//! ```rust
//! use tgi_gtc::router::RouterConfig;
//! use tgi_gtc::batching::{ContinuousBatcher, BatchConfig};
//! use tgi_gtc::validation::{RequestValidator, ValidationConfig};
//!
//! // Configure router with TGI-style settings
//! let config = RouterConfig::builder()
//!     .max_concurrent_requests(128)
//!     .max_batch_size(32)
//!     .build();
//!
//! // Create continuous batcher
//! let batcher = ContinuousBatcher::new(BatchConfig::default());
//!
//! // Validate incoming requests
//! let validator = RequestValidator::new(ValidationConfig::default());
//! ```
//!
//! # Quality Standards
//!
//! - 95%+ test coverage
//! - 80%+ mutation score
//! - Property-based testing for all pure functions
//! - Benchmarks for performance-critical code

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod batching;
pub mod error;
pub mod inference;
pub mod quantization;
pub mod router;
pub mod scheduling;
pub mod streaming;
pub mod validation;

pub use error::{Error, Result};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// TGI version this corpus is based on
pub const TGI_VERSION: &str = "3.1.0";
