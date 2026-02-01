//! Sampling patterns for token generation.
//!
//! Temperature, top-p, top-k, and repetition penalty sampling strategies.
//!
//! # TGI Source
//!
//! Patterns derived from TGI's sampling implementation:
//! - Temperature scaling
//! - Top-p (nucleus) sampling
//! - Top-k sampling
//! - Repetition penalty
//! - Frequency/presence penalties
//!
//! # Sovereign AI Stack Equivalent
//!
//! Maps to `realizar::sample` for token sampling.
//!
//! # Key Concepts
//!
//! ## Temperature
//!
//! Controls randomness by scaling logits before softmax:
//! - `T=0`: Deterministic (argmax)
//! - `T=1`: Standard softmax
//! - `T>1`: More random/creative
//! - `T<1`: More focused/deterministic
//!
//! ## Top-K Sampling
//!
//! Only consider the K most likely tokens:
//! 1. Sort logits descending
//! 2. Keep only top K
//! 3. Renormalize and sample
//!
//! ## Top-P (Nucleus) Sampling
//!
//! Keep tokens until cumulative probability reaches P:
//! 1. Sort by probability descending
//! 2. Compute cumulative probabilities
//! 3. Keep tokens until cumsum >= P
//! 4. Renormalize and sample

use std::collections::HashMap;

/// Configuration for sampling.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingConfig {
    /// Temperature for logit scaling (0.0 = deterministic).
    pub temperature: f32,

    /// Top-k value (0 = disabled).
    pub top_k: usize,

    /// Top-p (nucleus) sampling threshold (1.0 = disabled).
    pub top_p: f32,

    /// Minimum probability threshold.
    pub min_p: f32,

    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,

    /// Frequency penalty for repeated tokens (0.0 = disabled).
    pub frequency_penalty: f32,

    /// Presence penalty for any repeated token (0.0 = disabled).
    pub presence_penalty: f32,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }
}

impl SamplingConfig {
    /// Create a greedy (deterministic) config.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            ..Default::default()
        }
    }

    /// Create a creative config with higher temperature.
    pub fn creative() -> Self {
        Self {
            temperature: 0.9,
            top_p: 0.95,
            ..Default::default()
        }
    }

    /// Create a balanced config.
    pub fn balanced() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            ..Default::default()
        }
    }

    /// Set temperature.
    pub const fn temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set top-k.
    pub const fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set top-p.
    pub const fn top_p(mut self, p: f32) -> Self {
        self.top_p = p;
        self
    }

    /// Set min-p.
    pub const fn min_p(mut self, p: f32) -> Self {
        self.min_p = p;
        self
    }

    /// Set repetition penalty.
    pub const fn repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Set frequency penalty.
    pub const fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.frequency_penalty = penalty;
        self
    }

    /// Set presence penalty.
    pub const fn presence_penalty(mut self, penalty: f32) -> Self {
        self.presence_penalty = penalty;
        self
    }

    /// Set random seed.
    pub const fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Check if sampling is deterministic.
    pub fn is_deterministic(&self) -> bool {
        self.temperature == 0.0
    }
}

/// Token sampler.
///
/// # Examples
///
/// ```rust
/// use tgi_gtc::sampling::{Sampler, SamplingConfig};
///
/// let config = SamplingConfig::balanced();
/// let mut sampler = Sampler::new(config);
///
/// let logits = vec![1.0, 2.0, 3.0, 0.5, 0.1];
/// let (token_id, prob) = sampler.sample(&logits);
/// ```
#[derive(Debug)]
pub struct Sampler {
    config: SamplingConfig,
    /// Simple LCG random state.
    rng_state: u64,
    /// Token frequency counts for penalties.
    token_counts: HashMap<u32, u32>,
}

impl Sampler {
    /// Create a new sampler.
    pub fn new(config: SamplingConfig) -> Self {
        let rng_state = config.seed.unwrap_or(42);
        Self {
            config,
            rng_state,
            token_counts: HashMap::new(),
        }
    }

    /// Get configuration.
    pub const fn config(&self) -> &SamplingConfig {
        &self.config
    }

    /// Reset token counts (for new sequence).
    pub fn reset(&mut self) {
        self.token_counts.clear();
    }

    /// Record a generated token for penalty tracking.
    pub fn record_token(&mut self, token_id: u32) {
        *self.token_counts.entry(token_id).or_insert(0) += 1;
    }

    /// Sample a token from logits.
    ///
    /// Returns (token_id, probability).
    pub fn sample(&mut self, logits: &[f32]) -> (u32, f32) {
        if logits.is_empty() {
            return (0, 0.0);
        }

        // Apply all transformations
        let probs = self.process_logits(logits);

        // Sample from distribution
        if self.config.is_deterministic() {
            // Greedy: argmax
            let (idx, &prob) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            (idx as u32, prob)
        } else {
            // Stochastic sampling
            let idx = self.sample_from_probs(&probs);
            (idx as u32, probs[idx])
        }
    }

    /// Sample with explicit token IDs (for non-contiguous vocabularies).
    pub fn sample_with_ids(&mut self, logits: &[(u32, f32)]) -> (u32, f32) {
        if logits.is_empty() {
            return (0, 0.0);
        }

        let raw_logits: Vec<f32> = logits.iter().map(|(_, l)| *l).collect();
        let probs = self.process_logits(&raw_logits);

        if self.config.is_deterministic() {
            let (idx, &prob) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            (logits[idx].0, prob)
        } else {
            let idx = self.sample_from_probs(&probs);
            (logits[idx].0, probs[idx])
        }
    }

    /// Process logits through all transformations.
    fn process_logits(&self, logits: &[f32]) -> Vec<f32> {
        let mut processed = logits.to_vec();

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            self.apply_repetition_penalty(&mut processed);
        }

        // Apply frequency/presence penalties
        if self.config.frequency_penalty != 0.0 || self.config.presence_penalty != 0.0 {
            self.apply_frequency_presence_penalty(&mut processed);
        }

        // Apply temperature
        if self.config.temperature > 0.0 && self.config.temperature != 1.0 {
            for logit in &mut processed {
                *logit /= self.config.temperature;
            }
        }

        // Convert to probabilities
        let mut probs = softmax(&processed);

        // Apply top-k filtering
        if self.config.top_k > 0 && self.config.top_k < probs.len() {
            self.apply_top_k(&mut probs);
        }

        // Apply top-p filtering
        if self.config.top_p < 1.0 {
            self.apply_top_p(&mut probs);
        }

        // Apply min-p filtering
        if self.config.min_p > 0.0 {
            self.apply_min_p(&mut probs);
        }

        // Renormalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        probs
    }

    /// Apply repetition penalty to logits.
    fn apply_repetition_penalty(&self, logits: &mut [f32]) {
        let penalty = self.config.repetition_penalty;

        for (&token_id, &_count) in &self.token_counts {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                if *logit > 0.0 {
                    *logit /= penalty;
                } else {
                    *logit *= penalty;
                }
            }
        }
    }

    /// Apply frequency and presence penalties.
    fn apply_frequency_presence_penalty(&self, logits: &mut [f32]) {
        for (&token_id, &count) in &self.token_counts {
            if let Some(logit) = logits.get_mut(token_id as usize) {
                // Frequency penalty: penalize based on count
                *logit -= self.config.frequency_penalty * count as f32;

                // Presence penalty: flat penalty if token appeared at all
                *logit -= self.config.presence_penalty;
            }
        }
    }

    /// Apply top-k filtering (zero out non-top-k).
    fn apply_top_k(&self, probs: &mut [f32]) {
        let k = self.config.top_k;

        // Find k-th largest probability
        let mut sorted: Vec<f32> = probs.iter().copied().collect();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

        if k < sorted.len() {
            let threshold = sorted[k];
            for p in probs.iter_mut() {
                if *p < threshold {
                    *p = 0.0;
                }
            }
        }
    }

    /// Apply top-p (nucleus) filtering.
    fn apply_top_p(&self, probs: &mut [f32]) {
        let p = self.config.top_p;

        // Get sorted indices
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find cutoff
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed.len();

        for (i, &(_, prob)) in indexed.iter().enumerate() {
            cumsum += prob;
            if cumsum >= p {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out tokens below cutoff
        for (i, &(idx, _)) in indexed.iter().enumerate() {
            if i >= cutoff_idx {
                probs[idx] = 0.0;
            }
        }
    }

    /// Apply min-p filtering.
    fn apply_min_p(&self, probs: &mut [f32]) {
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = max_prob * self.config.min_p;

        for p in probs.iter_mut() {
            if *p < threshold {
                *p = 0.0;
            }
        }
    }

    /// Sample index from probability distribution.
    fn sample_from_probs(&mut self, probs: &[f32]) -> usize {
        let r = self.random_f32();
        let mut cumsum = 0.0;

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }

        // Fallback to last non-zero
        probs.len() - 1
    }

    /// Generate a random f32 in [0, 1).
    fn random_f32(&mut self) -> f32 {
        // Simple LCG: state = (a * state + c) mod m
        const A: u64 = 6364136223846793005;
        const C: u64 = 1442695040888963407;
        self.rng_state = self.rng_state.wrapping_mul(A).wrapping_add(C);

        // Convert to f32 in [0, 1)
        (self.rng_state >> 33) as f32 / (1u64 << 31) as f32
    }
}

/// Compute softmax of logits.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Subtract max for numerical stability
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();

    if sum == 0.0 {
        // Uniform distribution if all zeros
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exp.iter().map(|e| e / sum).collect()
    }
}

/// Compute log softmax of logits.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    let log_sum_exp = max + sum_exp.ln();

    logits.iter().map(|x| x - log_sum_exp).collect()
}

/// Compute argmax of logits.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }

    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
}

/// Compute top-k indices and values.
pub fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    indexed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
        assert_eq!(config.repetition_penalty, 1.0);
        assert!(!config.is_deterministic());
    }

    #[test]
    fn test_sampling_config_greedy() {
        let config = SamplingConfig::greedy();
        assert_eq!(config.temperature, 0.0);
        assert!(config.is_deterministic());
    }

    #[test]
    fn test_sampling_config_creative() {
        let config = SamplingConfig::creative();
        assert_eq!(config.temperature, 0.9);
        assert_eq!(config.top_p, 0.95);
    }

    #[test]
    fn test_sampling_config_balanced() {
        let config = SamplingConfig::balanced();
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.repetition_penalty, 1.1);
    }

    #[test]
    fn test_sampling_config_builder() {
        let config = SamplingConfig::default()
            .temperature(0.5)
            .top_k(10)
            .top_p(0.8)
            .repetition_penalty(1.2)
            .seed(12345);

        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_k, 10);
        assert_eq!(config.top_p, 0.8);
        assert_eq!(config.repetition_penalty, 1.2);
        assert_eq!(config.seed, Some(12345));
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values shouldn't cause overflow
        let logits = vec![1000.0, 1001.0, 1002.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);

        // All should be <= 0
        for lp in &log_probs {
            assert!(*lp <= 0.0);
        }

        // exp(log_softmax) should sum to 1
        let sum: f32 = log_probs.iter().map(|x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        let logits = vec![1.0, 3.0, 2.0];
        assert_eq!(argmax(&logits), Some(1));

        let logits = vec![5.0, 1.0, 2.0];
        assert_eq!(argmax(&logits), Some(0));

        assert_eq!(argmax(&[]), None);
    }

    #[test]
    fn test_top_k_fn() {
        let logits = vec![1.0, 4.0, 2.0, 5.0, 3.0];
        let top = top_k(&logits, 3);

        assert_eq!(top.len(), 3);
        assert_eq!(top[0], (3, 5.0)); // index 3 has value 5.0
        assert_eq!(top[1], (1, 4.0)); // index 1 has value 4.0
        assert_eq!(top[2], (4, 3.0)); // index 4 has value 3.0
    }

    #[test]
    fn test_sampler_greedy() {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);

        let logits = vec![1.0, 5.0, 2.0, 3.0];
        let (token_id, _prob) = sampler.sample(&logits);

        assert_eq!(token_id, 1); // Index of max (5.0)
    }

    #[test]
    fn test_sampler_with_seed() {
        let config = SamplingConfig::default().seed(42);

        // Same seed should give same results
        let mut sampler1 = Sampler::new(config.clone());
        let mut sampler2 = Sampler::new(config);

        let logits = vec![1.0, 1.0, 1.0, 1.0];

        let (t1, _) = sampler1.sample(&logits);
        let (t2, _) = sampler2.sample(&logits);

        assert_eq!(t1, t2);
    }

    #[test]
    fn test_sampler_record_token() {
        let config = SamplingConfig::default().repetition_penalty(2.0);
        let mut sampler = Sampler::new(config);

        sampler.record_token(5);
        sampler.record_token(5);
        sampler.record_token(3);

        // Token 5 has count 2, token 3 has count 1
        assert_eq!(sampler.token_counts.get(&5), Some(&2));
        assert_eq!(sampler.token_counts.get(&3), Some(&1));
    }

    #[test]
    fn test_sampler_reset() {
        let config = SamplingConfig::default();
        let mut sampler = Sampler::new(config);

        sampler.record_token(5);
        sampler.reset();

        assert!(sampler.token_counts.is_empty());
    }

    #[test]
    fn test_sampler_empty_logits() {
        let mut sampler = Sampler::new(SamplingConfig::default());
        let (token_id, prob) = sampler.sample(&[]);

        assert_eq!(token_id, 0);
        assert_eq!(prob, 0.0);
    }

    #[test]
    fn test_sampler_sample_with_ids() {
        let config = SamplingConfig::greedy();
        let mut sampler = Sampler::new(config);

        let logits = vec![(100, 1.0), (200, 5.0), (300, 2.0)];
        let (token_id, _prob) = sampler.sample_with_ids(&logits);

        assert_eq!(token_id, 200); // ID with max logit
    }

    #[test]
    fn test_sampler_top_k() {
        let config = SamplingConfig::default().top_k(2).temperature(0.001);
        let mut sampler = Sampler::new(config);

        let logits = vec![1.0, 5.0, 4.0, 0.5, 0.1];

        // Sample many times - should only get indices 1 or 2
        for _ in 0..100 {
            let (token_id, _) = sampler.sample(&logits);
            assert!(token_id == 1 || token_id == 2);
        }
    }

    #[test]
    fn test_sampler_min_p() {
        let config = SamplingConfig::default().min_p(0.5);
        let sampler = Sampler::new(config);

        // Only the highest probability token should remain
        let logits = vec![0.0, 10.0, 0.0, 0.0]; // Token 1 is much higher
        let probs = sampler.process_logits(&logits);

        // Token 1 should dominate after min_p filtering
        assert!(probs[1] > 0.99);
    }

    #[test]
    fn test_sampler_frequency_penalty() {
        let config = SamplingConfig::default().frequency_penalty(0.5);
        let mut sampler = Sampler::new(config);

        // Record token 1 multiple times
        sampler.record_token(1);
        sampler.record_token(1);
        sampler.record_token(1);

        let logits = vec![5.0, 5.0, 5.0];
        let probs = sampler.process_logits(&logits);

        // Token 1 should have lower probability due to frequency penalty
        assert!(probs[1] < probs[0]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_sampler_presence_penalty() {
        let config = SamplingConfig::default().presence_penalty(1.0);
        let mut sampler = Sampler::new(config);

        // Record tokens 0 and 1 (any count)
        sampler.record_token(0);
        sampler.record_token(1);

        let logits = vec![5.0, 5.0, 5.0];
        let probs = sampler.process_logits(&logits);

        // Token 2 (not recorded) should have higher probability
        assert!(probs[2] > probs[0]);
        assert!(probs[2] > probs[1]);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_softmax_sums_to_one(logits in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let probs = softmax(&logits);
            let sum: f32 = probs.iter().sum();
            prop_assert!((sum - 1.0).abs() < 1e-5);
        }

        #[test]
        fn prop_softmax_all_positive(logits in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let probs = softmax(&logits);
            for p in probs {
                prop_assert!(p >= 0.0);
            }
        }

        #[test]
        fn prop_argmax_in_range(logits in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            if let Some(idx) = argmax(&logits) {
                prop_assert!(idx < logits.len());
            }
        }

        #[test]
        fn prop_sampler_returns_valid_index(logits in prop::collection::vec(0.1f32..10.0, 1..100)) {
            let mut sampler = Sampler::new(SamplingConfig::default().seed(42));
            let (token_id, prob) = sampler.sample(&logits);

            prop_assert!((token_id as usize) < logits.len());
            prop_assert!(prob >= 0.0);
            prop_assert!(prob <= 1.0);
        }
    }
}
