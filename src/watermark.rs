//! Text Watermarking
//!
//! Implements watermarking for AI-generated text detection. The watermark is
//! embedded by biasing token probabilities based on a hash of previous tokens.
//!
//! # TGI Reference
//!
//! Based on TGI's watermarking implementation.
//! See: <https://github.com/huggingface/text-generation-inference>
//!
//! # Algorithm
//!
//! 1. Hash previous tokens to get a seed
//! 2. Partition vocabulary into "green" and "red" lists
//! 3. Add bias to green tokens before sampling
//! 4. Detection: check if generated tokens favor green list
//!
//! # Example
//!
//! ```rust
//! use tgi_gtc::watermark::{Watermarker, WatermarkConfig, WatermarkDetector};
//!
//! let config = WatermarkConfig::default();
//! let mut watermarker = Watermarker::new(config.clone());
//!
//! // Apply watermark to logits
//! let prev_tokens = vec![100, 200, 300];
//! let mut logits = vec![0.0f32; 1000];
//! watermarker.apply(&prev_tokens, &mut logits);
//!
//! // Detection
//! let detector = WatermarkDetector::new(config);
//! let tokens = vec![100, 200, 300, 400, 500];
//! let result = detector.detect(&tokens);
//! println!("Z-score: {:.2}", result.z_score);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for watermarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatermarkConfig {
    /// Bias added to green list tokens.
    pub gamma: f32,

    /// Fraction of vocabulary in green list (0.0 to 1.0).
    pub delta: f32,

    /// Number of previous tokens to hash.
    pub context_width: usize,

    /// Secret key for hash function.
    pub secret_key: u64,

    /// Vocabulary size.
    pub vocab_size: usize,

    /// Whether watermarking is enabled.
    pub enabled: bool,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            gamma: 2.0,
            delta: 0.5,
            context_width: 1,
            secret_key: 0x12345678_9ABCDEF0,
            vocab_size: 32000,
            enabled: true,
        }
    }
}

impl WatermarkConfig {
    /// Create a subtle watermark (harder to detect but less robust).
    pub fn subtle() -> Self {
        Self {
            gamma: 0.5,
            delta: 0.25,
            ..Default::default()
        }
    }

    /// Create a strong watermark (easier to detect and more robust).
    pub fn strong() -> Self {
        Self {
            gamma: 4.0,
            delta: 0.5,
            ..Default::default()
        }
    }

    /// Green list size.
    pub fn green_list_size(&self) -> usize {
        (self.vocab_size as f32 * self.delta) as usize
    }
}

/// Watermarker that modifies logits during generation.
#[derive(Debug)]
pub struct Watermarker {
    config: WatermarkConfig,
    rng_state: u64,
}

impl Watermarker {
    /// Create a new watermarker.
    pub fn new(config: WatermarkConfig) -> Self {
        Self {
            rng_state: config.secret_key,
            config,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &WatermarkConfig {
        &self.config
    }

    /// Apply watermark to logits.
    pub fn apply(&mut self, prev_tokens: &[u32], logits: &mut [f32]) {
        if !self.config.enabled || logits.len() != self.config.vocab_size {
            return;
        }

        let green_list = self.compute_green_list(prev_tokens);

        for token_id in green_list {
            if (token_id as usize) < logits.len() {
                logits[token_id as usize] += self.config.gamma;
            }
        }
    }

    /// Compute green list based on previous tokens.
    fn compute_green_list(&mut self, prev_tokens: &[u32]) -> HashSet<u32> {
        // Hash previous tokens to get seed
        let seed = self.hash_context(prev_tokens);
        self.rng_state = seed;

        // Generate green list
        let green_size = self.config.green_list_size();
        let mut green_list = HashSet::with_capacity(green_size);

        // Shuffle vocabulary and take first delta fraction
        let mut indices: Vec<u32> = (0..self.config.vocab_size as u32).collect();
        self.shuffle(&mut indices);

        for &idx in indices.iter().take(green_size) {
            green_list.insert(idx);
        }

        green_list
    }

    /// Hash context tokens to seed.
    fn hash_context(&self, tokens: &[u32]) -> u64 {
        let mut hash = self.config.secret_key;
        let context = if tokens.len() > self.config.context_width {
            &tokens[tokens.len() - self.config.context_width..]
        } else {
            tokens
        };

        for &token in context {
            hash = hash.wrapping_mul(0x5851F42D4C957F2D);
            hash ^= token as u64;
        }

        hash
    }

    /// Fisher-Yates shuffle with internal RNG.
    fn shuffle(&mut self, arr: &mut [u32]) {
        for i in (1..arr.len()).rev() {
            let j = self.random_usize(i + 1);
            arr.swap(i, j);
        }
    }

    /// Generate random usize in [0, max).
    fn random_usize(&mut self, max: usize) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(6364136223846793005);
        self.rng_state = self.rng_state.wrapping_add(1442695040888963407);
        (self.rng_state as usize) % max
    }

    /// Check if a token is in the green list.
    pub fn is_green(&mut self, prev_tokens: &[u32], token: u32) -> bool {
        let green_list = self.compute_green_list(prev_tokens);
        green_list.contains(&token)
    }
}

/// Detection result for watermark.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    /// Z-score indicating watermark strength.
    pub z_score: f32,

    /// Number of green tokens found.
    pub green_count: usize,

    /// Total tokens analyzed.
    pub total_tokens: usize,

    /// Expected green ratio under null hypothesis.
    pub expected_ratio: f32,

    /// Observed green ratio.
    pub observed_ratio: f32,

    /// Whether watermark is detected (z_score > threshold).
    pub is_watermarked: bool,

    /// Confidence level (0.0 to 1.0).
    pub confidence: f32,
}

impl DetectionResult {
    /// P-value for the detection (one-sided).
    pub fn p_value(&self) -> f32 {
        // Approximate p-value from z-score
        if self.z_score <= 0.0 {
            1.0
        } else {
            // Simplified approximation
            (-0.5 * self.z_score * self.z_score).exp()
        }
    }
}

/// Detector for watermarked text.
#[derive(Debug)]
pub struct WatermarkDetector {
    config: WatermarkConfig,
    detection_threshold: f32,
}

impl WatermarkDetector {
    /// Create a new detector.
    pub fn new(config: WatermarkConfig) -> Self {
        Self {
            config,
            detection_threshold: 4.0, // Z-score threshold
        }
    }

    /// Create detector with custom threshold.
    pub fn with_threshold(config: WatermarkConfig, threshold: f32) -> Self {
        Self {
            config,
            detection_threshold: threshold,
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &WatermarkConfig {
        &self.config
    }

    /// Detect watermark in token sequence.
    pub fn detect(&self, tokens: &[u32]) -> DetectionResult {
        if tokens.len() < 2 {
            return DetectionResult {
                z_score: 0.0,
                green_count: 0,
                total_tokens: tokens.len(),
                expected_ratio: self.config.delta,
                observed_ratio: 0.0,
                is_watermarked: false,
                confidence: 0.0,
            };
        }

        let mut watermarker = Watermarker::new(self.config.clone());
        let mut green_count = 0;

        // Check each token against its context's green list
        for i in 1..tokens.len() {
            let context = &tokens[..i];
            if watermarker.is_green(context, tokens[i]) {
                green_count += 1;
            }
        }

        let total = tokens.len() - 1;
        let expected_ratio = self.config.delta;
        let observed_ratio = green_count as f32 / total as f32;

        // Compute z-score
        let expected_green = expected_ratio * total as f32;
        let variance = total as f32 * expected_ratio * (1.0 - expected_ratio);
        let std_dev = variance.sqrt();

        let z_score = if std_dev > 0.0 {
            (green_count as f32 - expected_green) / std_dev
        } else {
            0.0
        };

        let is_watermarked = z_score > self.detection_threshold;
        let confidence = if z_score > 0.0 {
            1.0 - (-0.5 * z_score * z_score).exp()
        } else {
            0.0
        };

        DetectionResult {
            z_score,
            green_count,
            total_tokens: total,
            expected_ratio,
            observed_ratio,
            is_watermarked,
            confidence,
        }
    }

    /// Detect with sliding window for long texts.
    pub fn detect_windowed(&self, tokens: &[u32], window_size: usize) -> Vec<DetectionResult> {
        let mut results = Vec::new();

        for start in (0..tokens.len()).step_by(window_size / 2) {
            let end = (start + window_size).min(tokens.len());
            if end - start < 10 {
                break;
            }

            let window = &tokens[start..end];
            results.push(self.detect(window));
        }

        results
    }

    /// Get overall detection result from windowed analysis.
    pub fn aggregate_windowed(&self, results: &[DetectionResult]) -> DetectionResult {
        if results.is_empty() {
            return DetectionResult {
                z_score: 0.0,
                green_count: 0,
                total_tokens: 0,
                expected_ratio: self.config.delta,
                observed_ratio: 0.0,
                is_watermarked: false,
                confidence: 0.0,
            };
        }

        let total_green: usize = results.iter().map(|r| r.green_count).sum();
        let total_tokens: usize = results.iter().map(|r| r.total_tokens).sum();
        let avg_z_score: f32 =
            results.iter().map(|r| r.z_score).sum::<f32>() / results.len() as f32;

        let observed_ratio = if total_tokens > 0 {
            total_green as f32 / total_tokens as f32
        } else {
            0.0
        };

        DetectionResult {
            z_score: avg_z_score,
            green_count: total_green,
            total_tokens,
            expected_ratio: self.config.delta,
            observed_ratio,
            is_watermarked: avg_z_score > self.detection_threshold,
            confidence: results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32,
        }
    }
}

/// Statistics for watermark quality.
#[derive(Debug, Clone, Default)]
pub struct WatermarkStats {
    /// Total tokens generated.
    pub tokens_generated: u64,

    /// Green tokens generated.
    pub green_tokens: u64,

    /// Average bias applied.
    pub avg_bias: f32,
}

impl WatermarkStats {
    /// Green token ratio.
    pub fn green_ratio(&self) -> f32 {
        if self.tokens_generated == 0 {
            0.0
        } else {
            self.green_tokens as f32 / self.tokens_generated as f32
        }
    }

    /// Record a generated token.
    pub fn record(&mut self, is_green: bool, bias: f32) {
        self.tokens_generated += 1;
        if is_green {
            self.green_tokens += 1;
        }
        self.avg_bias = (self.avg_bias * (self.tokens_generated - 1) as f32 + bias)
            / self.tokens_generated as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_config_default() {
        let config = WatermarkConfig::default();
        assert_eq!(config.gamma, 2.0);
        assert_eq!(config.delta, 0.5);
        assert!(config.enabled);
    }

    #[test]
    fn test_watermark_config_green_list_size() {
        let config = WatermarkConfig {
            vocab_size: 32000,
            delta: 0.5,
            ..Default::default()
        };
        assert_eq!(config.green_list_size(), 16000);
    }

    #[test]
    fn test_watermarker_apply() {
        let config = WatermarkConfig {
            vocab_size: 100,
            ..Default::default()
        };
        let mut watermarker = Watermarker::new(config);

        let prev_tokens = vec![1, 2, 3];
        let mut logits = vec![0.0f32; 100];

        watermarker.apply(&prev_tokens, &mut logits);

        // Some logits should be biased
        let biased_count = logits.iter().filter(|&&x| x > 0.0).count();
        assert!(biased_count > 0);
        assert!(biased_count < 100);
    }

    #[test]
    fn test_watermarker_disabled() {
        let config = WatermarkConfig {
            vocab_size: 100,
            enabled: false,
            ..Default::default()
        };
        let mut watermarker = Watermarker::new(config);

        let prev_tokens = vec![1, 2, 3];
        let mut logits = vec![0.0f32; 100];

        watermarker.apply(&prev_tokens, &mut logits);

        // No logits should be biased
        assert!(logits.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_watermark_detector() {
        let config = WatermarkConfig {
            vocab_size: 100,
            ..Default::default()
        };
        let detector = WatermarkDetector::new(config);

        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result = detector.detect(&tokens);

        assert!(result.total_tokens > 0);
        assert!(result.expected_ratio > 0.0);
    }

    #[test]
    fn test_detection_result_p_value() {
        let result = DetectionResult {
            z_score: 4.0,
            green_count: 80,
            total_tokens: 100,
            expected_ratio: 0.5,
            observed_ratio: 0.8,
            is_watermarked: true,
            confidence: 0.99,
        };

        assert!(result.p_value() < 0.01);
    }

    #[test]
    fn test_watermark_stats() {
        let mut stats = WatermarkStats::default();

        stats.record(true, 2.0);
        stats.record(true, 2.0);
        stats.record(false, 2.0);

        assert_eq!(stats.tokens_generated, 3);
        assert_eq!(stats.green_tokens, 2);
        assert!((stats.green_ratio() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_detector_windowed() {
        let config = WatermarkConfig {
            vocab_size: 100,
            ..Default::default()
        };
        let detector = WatermarkDetector::new(config);

        let tokens: Vec<u32> = (0..100).collect();
        let results = detector.detect_windowed(&tokens, 20);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_detector_aggregate() {
        let config = WatermarkConfig::default();
        let detector = WatermarkDetector::new(config);

        let results = vec![
            DetectionResult {
                z_score: 3.0,
                green_count: 60,
                total_tokens: 100,
                expected_ratio: 0.5,
                observed_ratio: 0.6,
                is_watermarked: false,
                confidence: 0.8,
            },
            DetectionResult {
                z_score: 5.0,
                green_count: 70,
                total_tokens: 100,
                expected_ratio: 0.5,
                observed_ratio: 0.7,
                is_watermarked: true,
                confidence: 0.95,
            },
        ];

        let agg = detector.aggregate_windowed(&results);
        assert_eq!(agg.green_count, 130);
        assert_eq!(agg.total_tokens, 200);
        assert_eq!(agg.z_score, 4.0);
    }

    #[test]
    fn test_is_green_deterministic() {
        let config = WatermarkConfig {
            vocab_size: 100,
            secret_key: 12345,
            ..Default::default()
        };

        let mut w1 = Watermarker::new(config.clone());
        let mut w2 = Watermarker::new(config);

        let prev = vec![1, 2, 3];
        let token = 50u32;

        // Same config should give same result
        assert_eq!(w1.is_green(&prev, token), w2.is_green(&prev, token));
    }
}
