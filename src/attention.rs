//! Attention mechanism patterns.
//!
//! Scaled dot-product attention and Flash Attention patterns for
//! memory-efficient transformer inference.
//!
//! # TGI Source
//!
//! Patterns derived from TGI's attention implementations:
//! - Scaled dot-product attention
//! - Flash Attention (memory-efficient)
//! - Paged Attention (with KV cache blocks)
//! - Multi-Query Attention (MQA)
//! - Grouped-Query Attention (GQA)
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::attention` for attention computation.
//!
//! # Key Concepts
//!
//! ## Scaled Dot-Product Attention
//!
//! ```text
//! Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//! ```
//!
//! ## Flash Attention
//!
//! Computes attention in tiles to reduce memory:
//! - O(N) memory instead of O(N²)
//! - Fuses softmax with matmul
//! - No materialization of attention matrix
//!
//! ## Paged Attention
//!
//! Uses block-based KV cache:
//! - Non-contiguous memory layout
//! - Efficient for variable-length sequences
//! - Enables KV cache sharing (CoW)

// PI constant available if needed for rotary embeddings
#[allow(unused_imports)]
use std::f32::consts::PI;

/// Attention configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,

    /// Number of KV heads (for GQA/MQA).
    pub num_kv_heads: usize,

    /// Head dimension.
    pub head_dim: usize,

    /// Whether to use causal masking.
    pub causal: bool,

    /// Attention dropout probability (training only).
    pub dropout: f32,

    /// Scale factor (default: 1/sqrt(head_dim)).
    pub scale: Option<f32>,

    /// Sliding window size (None = full attention).
    pub sliding_window: Option<usize>,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 32,
            num_kv_heads: 32,
            head_dim: 128,
            causal: true,
            dropout: 0.0,
            scale: None,
            sliding_window: None,
        }
    }
}

impl AttentionConfig {
    /// Create a new config.
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        Self {
            num_heads,
            num_kv_heads: num_heads,
            head_dim,
            ..Default::default()
        }
    }

    /// Set number of KV heads (for GQA).
    pub const fn num_kv_heads(mut self, n: usize) -> Self {
        self.num_kv_heads = n;
        self
    }

    /// Set causal masking.
    pub const fn causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set dropout.
    pub const fn dropout(mut self, p: f32) -> Self {
        self.dropout = p;
        self
    }

    /// Set custom scale.
    pub const fn scale(mut self, s: f32) -> Self {
        self.scale = Some(s);
        self
    }

    /// Set sliding window.
    pub const fn sliding_window(mut self, size: usize) -> Self {
        self.sliding_window = Some(size);
        self
    }

    /// Get the scale factor.
    pub fn get_scale(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }

    /// Check if using grouped-query attention.
    pub const fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads && self.num_kv_heads > 1
    }

    /// Check if using multi-query attention.
    pub const fn is_mqa(&self) -> bool {
        self.num_kv_heads == 1
    }

    /// Get the GQA group size.
    pub const fn gqa_group_size(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    /// Calculate hidden dimension.
    pub const fn hidden_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Calculate KV dimension.
    pub const fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
}

/// Attention mask types.
#[derive(Debug, Clone, PartialEq)]
pub enum AttentionMask {
    /// No mask (full attention).
    None,

    /// Causal mask (lower triangular).
    Causal,

    /// Custom mask (true = attend, false = mask).
    Custom(Vec<Vec<bool>>),

    /// Padding mask (per sequence).
    Padding(Vec<usize>), // Sequence lengths

    /// Sliding window mask.
    SlidingWindow(usize),
}

impl AttentionMask {
    /// Check if position i can attend to position j.
    pub fn can_attend(&self, i: usize, j: usize, _seq_len: usize) -> bool {
        match self {
            Self::None => true,
            Self::Causal => j <= i,
            Self::Custom(mask) => mask
                .get(i)
                .and_then(|row| row.get(j))
                .copied()
                .unwrap_or(false),
            Self::Padding(lengths) => {
                // For batch processing, need batch index
                // Simplified: just check if j is within some valid range
                lengths.first().map(|&len| j < len).unwrap_or(true)
            }
            Self::SlidingWindow(window) => j <= i && i - j < *window,
        }
    }

    /// Generate mask values for a sequence.
    pub fn generate(&self, seq_len: usize) -> Vec<Vec<f32>> {
        let mut mask = vec![vec![0.0f32; seq_len]; seq_len];

        for i in 0..seq_len {
            for j in 0..seq_len {
                if !self.can_attend(i, j, seq_len) {
                    mask[i][j] = f32::NEG_INFINITY;
                }
            }
        }

        mask
    }
}

/// Scaled dot-product attention.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::attention::{AttentionConfig, scaled_dot_product_attention};
///
/// let config = AttentionConfig::new(4, 64);
/// let seq_len = 8;
/// let head_dim = 64;
///
/// // Q, K, V tensors [seq_len, head_dim]
/// let q = vec![vec![0.1f32; head_dim]; seq_len];
/// let k = vec![vec![0.1f32; head_dim]; seq_len];
/// let v = vec![vec![0.2f32; head_dim]; seq_len];
///
/// let output = scaled_dot_product_attention(&q, &k, &v, config.get_scale(), true);
/// assert_eq!(output.len(), seq_len);
/// ```
pub fn scaled_dot_product_attention(
    q: &[Vec<f32>], // [seq_q, head_dim]
    k: &[Vec<f32>], // [seq_k, head_dim]
    v: &[Vec<f32>], // [seq_k, head_dim]
    scale: f32,
    causal: bool,
) -> Vec<Vec<f32>> {
    let seq_q = q.len();
    let seq_k = k.len();

    if seq_q == 0 || seq_k == 0 {
        return Vec::new();
    }

    let head_dim = q[0].len();

    // Compute Q @ K^T
    let mut scores = vec![vec![0.0f32; seq_k]; seq_q];
    for i in 0..seq_q {
        for j in 0..seq_k {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i][d] * k[j][d];
            }
            scores[i][j] = dot * scale;
        }
    }

    // Apply causal mask
    if causal {
        for i in 0..seq_q {
            for j in (i + 1)..seq_k {
                scores[i][j] = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax over keys
    for i in 0..seq_q {
        let max = scores[i].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..seq_k {
            scores[i][j] = (scores[i][j] - max).exp();
            sum += scores[i][j];
        }
        if sum > 0.0 {
            for j in 0..seq_k {
                scores[i][j] /= sum;
            }
        }
    }

    // Scores @ V
    let mut output = vec![vec![0.0f32; head_dim]; seq_q];
    for i in 0..seq_q {
        for d in 0..head_dim {
            let mut sum = 0.0f32;
            for j in 0..seq_k {
                sum += scores[i][j] * v[j][d];
            }
            output[i][d] = sum;
        }
    }

    output
}

/// Multi-head attention.
///
/// Splits Q, K, V across heads, computes attention, and concatenates.
pub fn multi_head_attention(
    q: &[Vec<f32>], // [seq, hidden_dim]
    k: &[Vec<f32>], // [seq, kv_dim]
    v: &[Vec<f32>], // [seq, kv_dim]
    config: &AttentionConfig,
) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    if seq_len == 0 {
        return Vec::new();
    }

    let hidden_dim = config.hidden_dim();
    let head_dim = config.head_dim;
    let num_heads = config.num_heads;
    let _num_kv_heads = config.num_kv_heads; // Used for validation/future GQA
    let scale = config.get_scale();
    let group_size = config.gqa_group_size();

    // Output accumulator
    let mut output = vec![vec![0.0f32; hidden_dim]; seq_len];

    // Process each head
    for h in 0..num_heads {
        let kv_head = h / group_size;

        // Extract Q for this head
        let q_head: Vec<Vec<f32>> = q
            .iter()
            .map(|row| row[h * head_dim..(h + 1) * head_dim].to_vec())
            .collect();

        // Extract K, V for the corresponding KV head
        let k_head: Vec<Vec<f32>> = k
            .iter()
            .map(|row| row[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec())
            .collect();
        let v_head: Vec<Vec<f32>> = v
            .iter()
            .map(|row| row[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec())
            .collect();

        // Compute attention for this head
        let head_output =
            scaled_dot_product_attention(&q_head, &k_head, &v_head, scale, config.causal);

        // Copy to output
        for (i, out_row) in head_output.iter().enumerate() {
            for (d, &val) in out_row.iter().enumerate() {
                output[i][h * head_dim + d] = val;
            }
        }
    }

    output
}

/// Rotary Position Embedding (RoPE).
///
/// Applies rotation to Q and K based on position.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Head dimension (must be even).
    head_dim: usize,

    /// Base for frequency computation.
    base: f32,

    /// Precomputed cos values.
    cos_cache: Vec<Vec<f32>>,

    /// Precomputed sin values.
    sin_cache: Vec<Vec<f32>>,
}

impl RotaryEmbedding {
    /// Create a new rotary embedding.
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

        let half_dim = head_dim / 2;
        let mut cos_cache = Vec::with_capacity(max_seq_len);
        let mut sin_cache = Vec::with_capacity(max_seq_len);

        // Compute frequencies
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Compute cos/sin for each position
        for pos in 0..max_seq_len {
            let mut cos_row = Vec::with_capacity(head_dim);
            let mut sin_row = Vec::with_capacity(head_dim);

            for &freq in &inv_freq {
                let angle = pos as f32 * freq;
                cos_row.push(angle.cos());
                sin_row.push(angle.sin());
            }

            // Duplicate for both halves
            let cos_full: Vec<f32> = cos_row.iter().chain(cos_row.iter()).copied().collect();
            let sin_full: Vec<f32> = sin_row.iter().chain(sin_row.iter()).copied().collect();

            cos_cache.push(cos_full);
            sin_cache.push(sin_full);
        }

        Self {
            head_dim,
            base,
            cos_cache,
            sin_cache,
        }
    }

    /// Apply rotary embedding to a tensor.
    pub fn apply(&self, x: &mut [Vec<f32>], start_pos: usize) {
        let half_dim = self.head_dim / 2;

        for (i, row) in x.iter_mut().enumerate() {
            let pos = start_pos + i;
            if pos >= self.cos_cache.len() {
                continue;
            }

            let cos = &self.cos_cache[pos];
            let sin = &self.sin_cache[pos];

            // Apply rotation
            for j in 0..half_dim {
                let x0 = row[j];
                let x1 = row[j + half_dim];

                row[j] = x0 * cos[j] - x1 * sin[j];
                row[j + half_dim] = x0 * sin[j] + x1 * cos[j];
            }
        }
    }

    /// Get max sequence length supported.
    pub fn max_seq_len(&self) -> usize {
        self.cos_cache.len()
    }

    /// Get head dimension.
    pub const fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get base.
    pub const fn base(&self) -> f32 {
        self.base
    }
}

/// Flash Attention configuration for tiled computation.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Block size for Q (Br).
    pub block_size_q: usize,

    /// Block size for K/V (Bc).
    pub block_size_kv: usize,

    /// Whether to use causal masking.
    pub causal: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            causal: true,
        }
    }
}

impl FlashAttentionConfig {
    /// Create with specified block sizes.
    pub const fn new(block_size_q: usize, block_size_kv: usize) -> Self {
        Self {
            block_size_q,
            block_size_kv,
            causal: true,
        }
    }
}

/// Simulated Flash Attention (demonstrates the algorithm, not optimized).
///
/// Real Flash Attention requires custom CUDA kernels for efficiency.
pub fn flash_attention_simulated(
    q: &[Vec<f32>],
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    config: &FlashAttentionConfig,
    scale: f32,
) -> Vec<Vec<f32>> {
    let seq_len = q.len();
    if seq_len == 0 {
        return Vec::new();
    }

    let head_dim = q[0].len();
    let br = config.block_size_q;
    let bc = config.block_size_kv;

    // Output and running stats
    let mut output = vec![vec![0.0f32; head_dim]; seq_len];
    let mut row_max = vec![f32::NEG_INFINITY; seq_len];
    let mut row_sum = vec![0.0f32; seq_len];

    // Process in tiles
    let num_blocks_q = (seq_len + br - 1) / br;
    let num_blocks_kv = (seq_len + bc - 1) / bc;

    for bq in 0..num_blocks_q {
        let q_start = bq * br;
        let q_end = (q_start + br).min(seq_len);

        for bkv in 0..num_blocks_kv {
            let kv_start = bkv * bc;
            let kv_end = (kv_start + bc).min(seq_len);

            // Skip if causal and this block is fully masked
            if config.causal && kv_start > q_end - 1 {
                continue;
            }

            // Compute block attention scores
            for qi in q_start..q_end {
                let mut block_max = f32::NEG_INFINITY;
                let mut block_scores = Vec::with_capacity(kv_end - kv_start);

                for kvi in kv_start..kv_end {
                    // Causal mask
                    if config.causal && kvi > qi {
                        block_scores.push(f32::NEG_INFINITY);
                        continue;
                    }

                    // Q @ K dot product
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q[qi][d] * k[kvi][d];
                    }
                    score *= scale;
                    block_scores.push(score);
                    block_max = block_max.max(score);
                }

                // Online softmax update
                let prev_max = row_max[qi];
                let new_max = prev_max.max(block_max);

                let exp_prev = (prev_max - new_max).exp();
                let mut block_sum = 0.0f32;

                // Rescale existing output once at start of block
                for d in 0..head_dim {
                    output[qi][d] *= exp_prev;
                }

                for (j, &score) in block_scores.iter().enumerate() {
                    if score > f32::NEG_INFINITY {
                        let exp_score = (score - new_max).exp();
                        block_sum += exp_score;

                        // Update output
                        let kvi = kv_start + j;
                        for d in 0..head_dim {
                            output[qi][d] += exp_score * v[kvi][d];
                        }
                    }
                }

                row_sum[qi] = row_sum[qi] * exp_prev + block_sum;
                row_max[qi] = new_max;
            }
        }
    }

    // Normalize
    for i in 0..seq_len {
        if row_sum[i] > 0.0 {
            for d in 0..head_dim {
                output[i][d] /= row_sum[i];
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_default() {
        let config = AttentionConfig::default();
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert!(config.causal);
    }

    #[test]
    fn test_attention_config_new() {
        let config = AttentionConfig::new(8, 64);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_attention_config_gqa() {
        let config = AttentionConfig::new(32, 128).num_kv_heads(8);
        assert!(config.is_gqa());
        assert!(!config.is_mqa());
        assert_eq!(config.gqa_group_size(), 4);
    }

    #[test]
    fn test_attention_config_mqa() {
        let config = AttentionConfig::new(32, 128).num_kv_heads(1);
        assert!(!config.is_gqa());
        assert!(config.is_mqa());
        assert_eq!(config.gqa_group_size(), 32);
    }

    #[test]
    fn test_attention_config_dimensions() {
        let config = AttentionConfig::new(8, 64);
        assert_eq!(config.hidden_dim(), 512);
        assert_eq!(config.kv_dim(), 512);

        let config = config.num_kv_heads(2);
        assert_eq!(config.kv_dim(), 128);
    }

    #[test]
    fn test_attention_config_scale() {
        let config = AttentionConfig::new(8, 64);
        let expected = 1.0 / 64.0f32.sqrt();
        assert!((config.get_scale() - expected).abs() < 1e-6);

        let config = config.scale(0.5);
        assert_eq!(config.get_scale(), 0.5);
    }

    #[test]
    fn test_attention_mask_none() {
        let mask = AttentionMask::None;
        assert!(mask.can_attend(0, 5, 10));
        assert!(mask.can_attend(5, 0, 10));
    }

    #[test]
    fn test_attention_mask_causal() {
        let mask = AttentionMask::Causal;
        assert!(mask.can_attend(5, 5, 10)); // Can attend to self
        assert!(mask.can_attend(5, 3, 10)); // Can attend to past
        assert!(!mask.can_attend(5, 7, 10)); // Cannot attend to future
    }

    #[test]
    fn test_attention_mask_sliding_window() {
        let mask = AttentionMask::SlidingWindow(3);
        assert!(mask.can_attend(5, 5, 10)); // Self
        assert!(mask.can_attend(5, 4, 10)); // Within window
        assert!(mask.can_attend(5, 3, 10)); // Edge of window
        assert!(!mask.can_attend(5, 2, 10)); // Outside window
        assert!(!mask.can_attend(5, 6, 10)); // Future
    }

    #[test]
    fn test_attention_mask_generate() {
        let mask = AttentionMask::Causal;
        let generated = mask.generate(3);

        assert_eq!(generated[0][0], 0.0);
        assert_eq!(generated[0][1], f32::NEG_INFINITY);
        assert_eq!(generated[0][2], f32::NEG_INFINITY);
        assert_eq!(generated[1][0], 0.0);
        assert_eq!(generated[1][1], 0.0);
        assert_eq!(generated[1][2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_scaled_dot_product_attention_basic() {
        let head_dim = 4;
        let seq_len = 2;

        let q = vec![vec![1.0f32; head_dim]; seq_len];
        let k = vec![vec![1.0f32; head_dim]; seq_len];
        let v = vec![vec![1.0f32; head_dim]; seq_len];

        let output = scaled_dot_product_attention(&q, &k, &v, 0.5, false);

        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), head_dim);
    }

    #[test]
    fn test_scaled_dot_product_attention_causal() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let k = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let output = scaled_dot_product_attention(&q, &k, &v, 1.0, true);

        // First position can only attend to itself
        // Second position can attend to both
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_scaled_dot_product_attention_empty() {
        let output = scaled_dot_product_attention(&[], &[], &[], 1.0, false);
        assert!(output.is_empty());
    }

    #[test]
    fn test_multi_head_attention() {
        let config = AttentionConfig::new(2, 4).causal(false);
        let seq_len = 3;
        let hidden_dim = config.hidden_dim();

        let q = vec![vec![0.1f32; hidden_dim]; seq_len];
        let k = vec![vec![0.1f32; hidden_dim]; seq_len];
        let v = vec![vec![0.2f32; hidden_dim]; seq_len];

        let output = multi_head_attention(&q, &k, &v, &config);

        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), hidden_dim);
    }

    #[test]
    fn test_multi_head_attention_empty() {
        let config = AttentionConfig::new(2, 4);
        let output = multi_head_attention(&[], &[], &[], &config);
        assert!(output.is_empty());
    }

    #[test]
    fn test_rotary_embedding_new() {
        let rope = RotaryEmbedding::new(64, 1024, 10000.0);
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_seq_len(), 1024);
        assert_eq!(rope.base(), 10000.0);
    }

    #[test]
    fn test_rotary_embedding_apply() {
        let rope = RotaryEmbedding::new(4, 100, 10000.0);
        let mut x = vec![vec![1.0f32; 4]; 3];

        rope.apply(&mut x, 0);

        // Values should change (rotation applied)
        // First position with pos=0 should have minimal change
        // Later positions should show more rotation
        assert_eq!(x.len(), 3);
    }

    #[test]
    fn test_flash_attention_config() {
        let config = FlashAttentionConfig::default();
        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.block_size_kv, 64);
        assert!(config.causal);

        let config = FlashAttentionConfig::new(32, 32);
        assert_eq!(config.block_size_q, 32);
    }

    #[test]
    fn test_flash_attention_simulated() {
        let head_dim = 4;
        let seq_len = 8;

        let q = vec![vec![0.1f32; head_dim]; seq_len];
        let k = vec![vec![0.1f32; head_dim]; seq_len];
        let v = vec![vec![0.2f32; head_dim]; seq_len];

        let config = FlashAttentionConfig::new(4, 4);
        let output = flash_attention_simulated(&q, &k, &v, &config, 0.5);

        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), head_dim);
    }

    #[test]
    fn test_flash_attention_matches_standard() {
        let head_dim = 4;
        let seq_len = 4;

        let q = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.5, 0.1, 0.2, 0.3],
            vec![0.2, 0.4, 0.1, 0.5],
            vec![0.3, 0.3, 0.3, 0.3],
        ];
        let k = q.clone();
        let v = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let scale = 1.0 / (head_dim as f32).sqrt();

        let standard = scaled_dot_product_attention(&q, &k, &v, scale, true);
        let flash_config = FlashAttentionConfig::new(2, 2);
        let flash = flash_attention_simulated(&q, &k, &v, &flash_config, scale);

        // Results should be close
        for i in 0..seq_len {
            for d in 0..head_dim {
                assert!(
                    (standard[i][d] - flash[i][d]).abs() < 1e-5,
                    "Mismatch at [{}, {}]: {} vs {}",
                    i,
                    d,
                    standard[i][d],
                    flash[i][d]
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_empty() {
        let config = FlashAttentionConfig::default();
        let output = flash_attention_simulated(&[], &[], &[], &config, 1.0);
        assert!(output.is_empty());
    }
}
