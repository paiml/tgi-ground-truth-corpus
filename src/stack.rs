//! Sovereign AI Stack Integration
//!
//! This module demonstrates how TGI patterns map to the Sovereign AI Stack.
//! Each submodule shows reference implementations that can be accelerated
//! using the stack crates when available.
//!
//! # Feature Flags
//!
//! - `trueno-backend`: Enable trueno SIMD/GPU acceleration
//! - `aprender-backend`: Enable aprender ML algorithms
//! - `realizar-backend`: Enable realizar inference engine
//! - `sovereign-stack`: Enable all stack components
//!
//! # Stack API Mapping
//!
//! | TGI Pattern | Reference | Stack Equivalent |
//! |-------------|-----------|------------------|
//! | Attention | `attention::scaled_dot_product_attention` | `realizar::attention` |
//! | Sampling | `sampling::Sampler` | `realizar::sample` |
//! | Batching | `batching::ContinuousBatcher` | `realizar::batcher` |
//! | KV Cache | `kv_cache::BlockAllocator` | `realizar::cache` |
//! | Tokenizer | `tokenizer::Tokenizer` | `aprender::text::tokenize` |
//!
//! # Example
//!
//! ```toml
//! [dependencies]
//! tgi-ground-truth-corpus = { version = "0.1", features = ["sovereign-stack"] }
//! ```

/// Attention computation patterns.
///
/// # Reference vs Stack
///
/// | Operation | Reference | trueno |
/// |-----------|-----------|--------|
/// | MatMul | Nested loops | `Tensor::matmul()` |
/// | Softmax | Manual exp/sum | `Tensor::softmax()` |
/// | Scale | Scalar multiply | `Tensor::scale()` |
pub mod attention {
    use crate::attention::{scaled_dot_product_attention, AttentionConfig};

    /// Compute attention using the reference implementation.
    ///
    /// This is the portable reference implementation that works without
    /// any stack dependencies. For SIMD/GPU acceleration, use trueno
    /// tensors with the same algorithm structure.
    pub fn attention_reference(
        q: &[Vec<f32>],
        k: &[Vec<f32>],
        v: &[Vec<f32>],
        config: &AttentionConfig,
    ) -> Vec<Vec<f32>> {
        scaled_dot_product_attention(q, k, v, config.get_scale(), config.causal)
    }

    /// Convert vectors to a flat representation for stack acceleration.
    ///
    /// Stack crates like trueno and realizar use flat tensor representations.
    /// This helper shows the conversion pattern.
    pub fn flatten_for_stack(data: &[Vec<f32>]) -> (Vec<f32>, [usize; 2]) {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        let flat: Vec<f32> = data.iter().flatten().copied().collect();
        (flat, [rows, cols])
    }

    /// Convert flat stack representation back to vectors.
    pub fn unflatten_from_stack(data: &[f32], cols: usize) -> Vec<Vec<f32>> {
        data.chunks(cols).map(|row| row.to_vec()).collect()
    }
}

/// Sampling patterns with stack integration.
///
/// # Reference vs Stack
///
/// | Operation | Reference | aprender/realizar |
/// |-----------|-----------|-------------------|
/// | Softmax | Manual | `softmax_with_temperature()` |
/// | Top-k | Sort + slice | `top_k_filter()` |
/// | Top-p | Cumsum + filter | `nucleus_filter()` |
pub mod sampling {
    use crate::sampling::{Sampler, SamplingConfig};

    /// Sample using the reference implementation.
    ///
    /// The reference Sampler implements all TGI sampling strategies:
    /// - Temperature scaling
    /// - Top-k filtering
    /// - Top-p (nucleus) filtering
    /// - Min-p filtering
    /// - Repetition/frequency/presence penalties
    pub fn sample_reference(config: SamplingConfig, logits: &[f32]) -> (u32, f32) {
        let mut sampler = Sampler::new(config);
        sampler.sample(logits)
    }

    /// Apply temperature scaling to logits (stack-compatible).
    ///
    /// This pure function can be easily accelerated with SIMD.
    pub fn apply_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
        if temperature <= 0.0 {
            return logits.to_vec();
        }
        logits.iter().map(|&x| x / temperature).collect()
    }

    /// Apply top-k filtering (stack-compatible).
    pub fn top_k_filter(probs: &[f32], k: usize) -> Vec<f32> {
        if k == 0 || k >= probs.len() {
            return probs.to_vec();
        }

        let mut indexed: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let threshold = indexed[k - 1].1;
        probs
            .iter()
            .map(|&p| if p >= threshold { p } else { 0.0 })
            .collect()
    }
}

/// Batching patterns with stack integration.
///
/// # Reference vs Stack
///
/// | Operation | Reference | realizar |
/// |-----------|-----------|----------|
/// | Queue | VecDeque | `batch::Queue` |
/// | Batch formation | Manual | `batch::Batcher` |
/// | Token counting | Sum | `batch::TokenBudget` |
pub mod batching {
    use crate::batching::{BatchConfig, ContinuousBatcher};

    /// Create a batcher using the reference implementation.
    ///
    /// The reference ContinuousBatcher implements TGI's continuous
    /// batching strategy with configurable limits.
    pub fn batcher_reference(config: BatchConfig) -> ContinuousBatcher {
        ContinuousBatcher::new(config)
    }

    /// Calculate optimal batch size based on token budget.
    ///
    /// This pure function can be used by stack-accelerated batchers.
    pub fn calculate_batch_size(
        pending_tokens: usize,
        max_tokens: usize,
        max_batch_size: usize,
    ) -> usize {
        if pending_tokens == 0 {
            return 0;
        }

        let tokens_per_request = pending_tokens / max_batch_size.max(1);
        if tokens_per_request == 0 {
            return max_batch_size;
        }

        (max_tokens / tokens_per_request).min(max_batch_size)
    }
}

/// KV Cache patterns with stack integration.
///
/// # Reference vs Stack
///
/// | Operation | Reference | realizar |
/// |-----------|-----------|----------|
/// | Block alloc | VecDeque | `cache::BlockPool` |
/// | CoW fork | RefCount | `cache::CowBlock` |
/// | Memory stats | Manual | `cache::MemoryTracker` |
pub mod kv_cache {
    use crate::kv_cache::{BlockAllocator, BlockAllocatorConfig};

    /// Create allocator using reference implementation.
    ///
    /// The reference BlockAllocator implements TGI's PagedAttention-style
    /// block management with copy-on-write forking.
    pub fn allocator_reference(num_blocks: usize) -> BlockAllocator {
        BlockAllocator::new(BlockAllocatorConfig::with_blocks(num_blocks))
    }

    /// Calculate memory requirements for a given configuration.
    ///
    /// This pure function helps plan GPU memory allocation.
    pub fn calculate_memory_bytes(
        num_blocks: usize,
        block_size: usize,
        num_layers: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> usize {
        // Each block stores K and V for all layers
        // Format: [num_layers, 2 (K+V), block_size, num_kv_heads, head_dim]
        let elements_per_block = num_layers * 2 * block_size * num_kv_heads * head_dim;
        let bytes_per_element = 4; // f32
        num_blocks * elements_per_block * bytes_per_element
    }
}

/// Tokenization patterns with stack integration.
///
/// # Reference vs Stack
///
/// | Operation | Reference | aprender |
/// |-----------|-----------|----------|
/// | BPE encode | HashMap | `text::tokenize::Bpe` |
/// | Decode | Vec lookup | `text::tokenize::decode()` |
/// | Special tokens | Manual | `text::tokenize::SpecialTokens` |
pub mod tokenizer {
    use crate::tokenizer::{Tokenizer, TokenizerConfig};

    /// Create tokenizer using reference implementation.
    ///
    /// The reference Tokenizer implements BPE encoding/decoding
    /// compatible with HuggingFace tokenizers.
    pub fn tokenizer_reference() -> Tokenizer {
        Tokenizer::new(TokenizerConfig::default())
    }

    /// Estimate token count from byte length.
    ///
    /// Useful for quick token budget estimation without full encoding.
    /// Average is ~4 bytes per token for English text.
    pub fn estimate_tokens(text: &str) -> usize {
        // Conservative estimate: ~4 bytes per token for English
        (text.len() + 3) / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_reference() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let config = crate::attention::AttentionConfig::new(1, 2);
        let output = attention::attention_reference(&q, &k, &v, &config);

        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_attention_flatten_unflatten() {
        let vecs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let (flat, shape) = attention::flatten_for_stack(&vecs);
        let back = attention::unflatten_from_stack(&flat, shape[1]);

        assert_eq!(vecs, back);
    }

    #[test]
    fn test_sampling_reference() {
        let config = crate::sampling::SamplingConfig::greedy();
        let logits = vec![1.0, 2.0, 3.0, 4.0];

        let (token, _prob) = sampling::sample_reference(config, &logits);
        assert_eq!(token, 3); // Highest logit
    }

    #[test]
    fn test_sampling_temperature() {
        let logits = vec![1.0, 2.0, 3.0];
        let scaled = sampling::apply_temperature(&logits, 2.0);

        assert_eq!(scaled, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_sampling_top_k() {
        let probs = vec![0.1, 0.3, 0.2, 0.4];
        let filtered = sampling::top_k_filter(&probs, 2);

        // Only top 2 should be non-zero
        assert!(filtered[0] == 0.0); // 0.1 filtered out
        assert!(filtered[1] > 0.0); // 0.3 kept
        assert!(filtered[2] == 0.0); // 0.2 filtered out
        assert!(filtered[3] > 0.0); // 0.4 kept
    }

    #[test]
    fn test_batching_reference() {
        let config = crate::batching::BatchConfig::default();
        let batcher = batching::batcher_reference(config);

        assert_eq!(batcher.queue_len(), 0);
    }

    #[test]
    fn test_batching_calculate_size() {
        assert_eq!(batching::calculate_batch_size(1000, 4096, 32), 32);
        assert_eq!(batching::calculate_batch_size(0, 4096, 32), 0);
    }

    #[test]
    fn test_kv_cache_reference() {
        let allocator = kv_cache::allocator_reference(100);
        assert!(allocator.can_allocate(10));
    }

    #[test]
    fn test_kv_cache_memory_calculation() {
        // 100 blocks, 16 tokens/block, 32 layers, 128 head_dim, 8 kv_heads
        let bytes = kv_cache::calculate_memory_bytes(100, 16, 32, 128, 8);
        // 100 * 32 * 2 * 16 * 8 * 128 * 4 = 419,430,400 bytes = ~400 MB
        assert_eq!(bytes, 419_430_400);
    }

    #[test]
    fn test_tokenizer_reference() {
        let tokenizer = tokenizer::tokenizer_reference();
        let output = tokenizer.encode("hello");
        assert!(!output.is_empty());
    }

    #[test]
    fn test_tokenizer_estimate() {
        assert_eq!(tokenizer::estimate_tokens("hello"), 2);
        assert_eq!(tokenizer::estimate_tokens("hello world"), 3);
    }
}
