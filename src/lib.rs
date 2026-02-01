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
//! - **Attention**: Scaled dot-product, Flash Attention, RoPE
//! - **KV Cache**: Block-based memory management (PagedAttention)
//! - **Sampling**: Temperature, top-k, top-p, repetition penalties
//! - **Tokenizer**: BPE encoding/decoding, special tokens
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
// Allow common patterns that are acceptable for this ground truth corpus
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::significant_drop_tightening)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::or_fun_call)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::use_self)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::needless_lifetimes)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::option_map_or_none)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::similar_names)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::mixed_attributes_style)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::float_cmp)]
#![allow(clippy::iter_cloned_collect)]
#![allow(clippy::cloned_instead_of_copied)]
#![allow(clippy::no_effect_underscore_binding)]

pub mod attention;
pub mod batching;
pub mod error;
pub mod inference;
pub mod kv_cache;
pub mod profiling;
pub mod quantization;
pub mod router;
pub mod sampling;
pub mod scheduling;
pub mod stack;
pub mod streaming;
pub mod tokenizer;
pub mod validation;

pub use error::{Error, Result};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// TGI version this corpus is based on
pub const TGI_VERSION: &str = "3.1.0";
